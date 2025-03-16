"""
MLX wrapper for PyTorch CSM model.
"""

import time
import os
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch

from csm.mlx_accel.mlx_ops import torch_to_mlx, mlx_to_torch, create_causal_mask, index_causal_mask
from csm.mlx_accel.mlx_embedding import MLXEmbedding
from csm.mlx_accel.mlx_generation import MLXFrameGenerator
from csm.mlx_accel.components.transformer import MLXTransformer

# Import MLX if available
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    # Create a dummy module
    class DummyMX:
        def __getattr__(self, name):
            raise ImportError("MLX is not available")
    mx = DummyMX()

class MLXModelWrapper:
    """
    Enhanced wrapper for PyTorch CSM model with MLX acceleration.
    This class handles the conversion between PyTorch and MLX formats
    and provides optimized methods for model inference.
    """
    
    def __init__(self, torch_model, args=None):
        """
        Initialize the MLX wrapper.
        
        Args:
            torch_model: PyTorch model to wrap
            args: Optional configuration arguments
        """
        self.torch_model = torch_model
        
        # Create default args if not provided
        if args is None:
            import argparse
            args = argparse.Namespace()
            if hasattr(torch_model, 'args'):
                model_args = torch_model.args
                args.audio_vocab_size = getattr(model_args, 'audio_vocab_size', 2051)
                args.audio_num_codebooks = getattr(model_args, 'audio_num_codebooks', 32)
            else:
                # Default values
                args.audio_vocab_size = 2051
                args.audio_num_codebooks = 32
            args.debug = os.environ.get("DEBUG", "0") == "1"
                
        self.args = args
        self.debug = args.debug
        
        # Initialize MLX components
        self.backbone = None
        self.decoder = None
        self.text_embeddings = None
        self.audio_embeddings = None
        self.projection = None
        self.codebook0_head = None
        self.audio_head = None
        
        # Initialize MLX helpers
        self.embedding = None
        self.frame_generator = None
        
        # Convert PyTorch model to MLX
        self._convert_from_torch()
        
        # Set up MLX caches
        self._setup_mlx_caches()
        
    def _convert_from_torch(self):
        """Convert PyTorch model to MLX format."""
        if self.debug:
            print("Converting PyTorch model to MLX format...")
            
        # Convert backbone transformer
        if hasattr(self.torch_model, 'backbone'):
            self.backbone = self._convert_transformer(self.torch_model.backbone, "backbone")
            
        # Convert decoder transformer
        if hasattr(self.torch_model, 'decoder'):
            self.decoder = self._convert_transformer(self.torch_model.decoder, "decoder")
            
        # Convert embeddings
        if hasattr(self.torch_model, 'text_embeddings') and hasattr(self.torch_model.text_embeddings, 'weight'):
            self.text_embeddings = torch_to_mlx(self.torch_model.text_embeddings.weight)
            
        if hasattr(self.torch_model, 'audio_embeddings') and hasattr(self.torch_model.audio_embeddings, 'weight'):
            self.audio_embeddings = torch_to_mlx(self.torch_model.audio_embeddings.weight)
            
        # Convert projection
        if hasattr(self.torch_model, 'projection') and hasattr(self.torch_model.projection, 'weight'):
            self.projection = torch_to_mlx(self.torch_model.projection.weight)
            
        # Convert heads
        if hasattr(self.torch_model, 'codebook0_head') and hasattr(self.torch_model.codebook0_head, 'weight'):
            # Ensure codebook0_head matches the expected audio_vocab_size
            weight = self.torch_model.codebook0_head.weight
            
            # Check if weight shape doesn't match expected vocab size
            if weight.shape[0] != self.args.audio_vocab_size:
                if self.debug:
                    print(f"WARNING: codebook0_head weight shape {weight.shape[0]} doesn't match audio_vocab_size {self.args.audio_vocab_size}")
                    print(f"Truncating or padding to match expected size")
                
                # Truncate or pad to match expected vocab size
                if weight.shape[0] > self.args.audio_vocab_size:
                    # Truncate to expected size
                    weight = weight[:self.args.audio_vocab_size, :]
                else:
                    # Need to pad with zeros
                    padding = torch.zeros((self.args.audio_vocab_size - weight.shape[0], weight.shape[1]), 
                                          dtype=weight.dtype, device=weight.device)
                    weight = torch.cat([weight, padding], dim=0)
            
            self.codebook0_head = torch_to_mlx(weight)
            
        if hasattr(self.torch_model, 'audio_head'):
            self.audio_head = []
            for head in self.torch_model.audio_head:
                # Ensure audio_head matches the expected audio_vocab_size
                weight = head.weight
                
                # Check if weight shape doesn't match expected vocab size
                if weight.shape[0] != self.args.audio_vocab_size:
                    if self.debug:
                        print(f"WARNING: audio_head weight shape {weight.shape[0]} doesn't match audio_vocab_size {self.args.audio_vocab_size}")
                        print(f"Truncating or padding to match expected size")
                    
                    # Truncate or pad to match expected vocab size
                    if weight.shape[0] > self.args.audio_vocab_size:
                        # Truncate to expected size
                        weight = weight[:self.args.audio_vocab_size, :]
                    else:
                        # Need to pad with zeros
                        padding = torch.zeros((self.args.audio_vocab_size - weight.shape[0], weight.shape[1]), 
                                              dtype=weight.dtype, device=weight.device)
                        weight = torch.cat([weight, padding], dim=0)
                
                self.audio_head.append(torch_to_mlx(weight))
                
        # Set up embedding helper
        self.embedding = MLXEmbedding(
            text_embeddings=self.text_embeddings,
            audio_embeddings=self.audio_embeddings,
            audio_vocab_size=self.args.audio_vocab_size,
            audio_num_codebooks=self.args.audio_num_codebooks,
            embed_dim=self.backbone.hidden_size if self.backbone else 2048,
            debug=self.debug
        )
        
        # Set up frame generator
        self.frame_generator = MLXFrameGenerator(
            backbone=self.backbone,
            decoder=self.decoder,
            embedding=self.embedding,
            projection_weight=self.projection,
            codebook0_head_weight=self.codebook0_head,
            audio_head_weights=self.audio_head,
            audio_vocab_size=self.args.audio_vocab_size,
            audio_num_codebooks=self.args.audio_num_codebooks,
            debug=self.debug,
            fallback_fn=self._fallback_generate
        )
        
    def _convert_transformer(self, torch_model, name="model"):
        """
        Convert a PyTorch transformer model to MLX.
        
        Args:
            torch_model: PyTorch transformer model
            name: Name of the model for logging
            
        Returns:
            MLX transformer model
        """
        if not hasattr(torch_model, 'layers'):
            raise ValueError(f"{name} does not have layers attribute")
            
        # Get configuration from first layer
        if len(torch_model.layers) == 0:
            raise ValueError(f"{name} has no layers")
            
        # Get model configuration
        import argparse
        config = argparse.Namespace()
        
        first_layer = torch_model.layers[0]
        
        # Get model dimensions from first layer
        if hasattr(first_layer, 'attn') and hasattr(first_layer.attn, 'q_proj'):
            config.hidden_size = first_layer.attn.q_proj.weight.shape[0]
            config.num_attention_heads = getattr(first_layer.attn, 'n_heads', config.hidden_size // 64)
            config.num_key_value_heads = getattr(first_layer.attn, 'n_kv_heads', config.num_attention_heads)
            config.num_layers = len(torch_model.layers)
            
            if hasattr(first_layer, 'mlp') and hasattr(first_layer.mlp, 'w1'):
                config.intermediate_size = first_layer.mlp.w1.weight.shape[0]
            else:
                config.intermediate_size = config.hidden_size * 4
        else:
            raise ValueError(f"{name} has unknown architecture")
            
        # Create MLX transformer
        mlx_model = MLXTransformer(config)
        
        # Create parameter dictionary for MLX
        params_dict = {}
        
        # Extract parameters from PyTorch model
        for i, layer in enumerate(torch_model.layers):
            prefix = f"layers.{i}"
            
            # Self-attention parameters
            if hasattr(layer, 'attn'):
                params_dict[f"{prefix}.attn.q_proj.weight"] = torch_to_mlx(layer.attn.q_proj.weight)
                params_dict[f"{prefix}.attn.k_proj.weight"] = torch_to_mlx(layer.attn.k_proj.weight)
                params_dict[f"{prefix}.attn.v_proj.weight"] = torch_to_mlx(layer.attn.v_proj.weight)
                params_dict[f"{prefix}.attn.output_proj.weight"] = torch_to_mlx(layer.attn.output_proj.weight)
                
            # Layer norm parameters
            if hasattr(layer, 'sa_norm'):
                params_dict[f"{prefix}.sa_norm.scale"] = torch_to_mlx(layer.sa_norm.scale)
                
            if hasattr(layer, 'mlp_norm'):
                params_dict[f"{prefix}.mlp_norm.scale"] = torch_to_mlx(layer.mlp_norm.scale)
                
            # MLP parameters
            if hasattr(layer, 'mlp'):
                params_dict[f"{prefix}.mlp.w1.weight"] = torch_to_mlx(layer.mlp.w1.weight)
                params_dict[f"{prefix}.mlp.w2.weight"] = torch_to_mlx(layer.mlp.w2.weight)
                params_dict[f"{prefix}.mlp.w3.weight"] = torch_to_mlx(layer.mlp.w3.weight)
                
        # Final layer norm
        if hasattr(torch_model, 'norm'):
            params_dict["model.norm.scale"] = torch_to_mlx(torch_model.norm.scale)
            
        # Load parameters into MLX model
        mlx_model.load_params(params_dict)
        
        if self.debug:
            print(f"Converted {name} to MLX with {config.num_layers} layers")
            
        return mlx_model
    
    def _setup_mlx_caches(self):
        """Set up MLX caches for inference."""
        # Caches are handled internally by the MLXTransformer
        pass
    
    def _fallback_generate(self, i=None, curr_sample=None):
        """
        Fallback method for generation that uses PyTorch.
        
        Args:
            i: Codebook index (optional)
            curr_sample: Current sample (optional)
            
        Returns:
            Generated tokens
        """
        if self.debug:
            print(f"Using PyTorch fallback for generation (i={i})")
            
        if i is not None and curr_sample is not None:
            # Codebook fallback
            try:
                ci_sample, _ = self.torch_model._generate_codebook(
                    i, mlx_to_torch(curr_sample), curr_sample.shape[1]
                )
                return ci_sample
            except Exception as e:
                if self.debug:
                    print(f"Codebook fallback failed: {e}")
                # Return zero tensor
                return torch.zeros((curr_sample.shape[0], 1), device="cpu")
        else:
            # Generic fallback
            return torch.zeros((1, self.args.audio_num_codebooks), device="cpu")
    
    def generate_frame(self, tokens, input_pos, frame_idx, topk=5, temperature=1.0):
        """
        Generate a frame using MLX acceleration.
        
        Args:
            tokens: Input tokens
            input_pos: Input positions
            frame_idx: Frame index
            topk: Top-k sampling parameter
            temperature: Temperature for sampling
            
        Returns:
            Generated frame
        """
        try:
            # Try pure MLX approach
            if self.debug:
                print("Using pure MLX approach for frame generation")
                
            # Convert inputs to MLX
            tokens_mlx = torch_to_mlx(tokens)
            pos_mlx = torch_to_mlx(input_pos)
            
            # Generate frame with pure MLX
            return self.frame_generator.generate_frame(tokens_mlx, pos_mlx, topk, temperature)
        except Exception as e:
            if self.debug:
                print(f"Pure MLX approach failed: {e}")
                
            # Fall back to hybrid approach
            return self.generate_frame_hybrid(tokens, input_pos, frame_idx, topk, temperature)
    
    def generate_frame_hybrid(self, tokens, input_pos, frame_idx, topk=5, temperature=1.0):
        """
        Generate a frame using hybrid MLX/PyTorch approach.
        
        Args:
            tokens: Input tokens
            input_pos: Input positions
            frame_idx: Frame index
            topk: Top-k sampling parameter
            temperature: Temperature for sampling
            
        Returns:
            Generated frame
        """
        if self.debug:
            print("Using hybrid MLX/PyTorch approach for frame generation")
            
        try:
            # Use PyTorch for initial processing
            with torch.no_grad():
                # Prepare inputs
                tokens_mask = torch.ones_like(tokens, dtype=torch.float)
                
                # Get hidden states from PyTorch model
                h, _, _ = self.torch_model.forward_first_stage(
                    tokens, tokens_mask, input_pos
                )
                
                # Use MLX for sampling
                mlx_success = 0
                
                # Convert to MLX
                last_h_mlx = torch_to_mlx(h[:, -1, :])
                
                # Generate c0 logits with MLX
                if self.codebook0_head is not None:
                    # Matrix multiply with MLX
                    c0_logits_mlx = mx.matmul(last_h_mlx, self.codebook0_head.T)
                    
                    # Sample with MLX
                    from csm.cli.mlx_components.sampling import mlx_categorical_sampling
                    c0_sample_mlx, success = mlx_categorical_sampling(c0_logits_mlx, temperature)
                    c0_sample = mlx_to_torch(c0_sample_mlx)
                    
                    if success:
                        mlx_success += 1
                else:
                    # Fall back to PyTorch
                    c0_logits = self.torch_model.codebook0_head(mlx_to_torch(last_h_mlx))
                    c0_sample = sample_topk(c0_logits, topk, temperature)
                
                # Initialize current sample
                curr_sample = c0_sample
                
                # Process remaining codebooks with PyTorch
                for i in range(1, self.args.audio_num_codebooks):
                    # Use PyTorch for remaining codebooks
                    ci_sample, _ = self.torch_model._generate_codebook(
                        i, curr_sample, curr_sample.shape[1]
                    )
                    curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
                
                if self.debug:
                    print(f"Hybrid frame generation: MLX sampling {mlx_success}/{self.args.audio_num_codebooks} successful")
                
                return curr_sample
        except Exception as e:
            if self.debug:
                print(f"Hybrid approach failed: {e}")
                
            # Fall back to pure PyTorch
            with torch.no_grad():
                return self.torch_model.generate_frame(tokens, tokens_mask, input_pos, temperature, topk)
    
    def reset_caches(self):
        """Reset model caches."""
        if self.backbone:
            self.backbone.reset_caches()
        if self.decoder:
            self.decoder.reset_caches()