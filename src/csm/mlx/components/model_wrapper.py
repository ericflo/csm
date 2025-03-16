"""
MLX wrapper for PyTorch CSM model.
"""

import time
import os
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch

from csm.mlx.mlx_ops import torch_to_mlx, mlx_to_torch, create_causal_mask, index_causal_mask
from csm.mlx.mlx_embedding import MLXEmbedding
from csm.mlx.mlx_generation import MLXFrameGenerator
from csm.mlx.components.transformer import MLXTransformer

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
    
    def __init__(self, torch_model_or_args, args=None):
        """
        Initialize the MLX wrapper.
        
        Args:
            torch_model_or_args: PyTorch model to wrap or model args
            args: Optional configuration arguments
        """
        # Flag for training mode
        self.train = True
        
        # Handle if args are passed instead of a model
        if isinstance(torch_model_or_args, dict):
            self.torch_model = None
            self.args = torch_model_or_args
            self.debug = os.environ.get("DEBUG", "0") == "1"
            
            # Initialize dummy structures for training
            self._init_from_args()
            return
        else:
            self.torch_model = torch_model_or_args
        
        # Create default args if not provided
        if args is None:
            import argparse
            args = argparse.Namespace()
            if hasattr(self.torch_model, 'args'):
                model_args = self.torch_model.args
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
        if hasattr(self.torch_model, 'text_embeddings'):
            if isinstance(self.torch_model.text_embeddings, torch.Tensor):
                self.text_embeddings = torch_to_mlx(self.torch_model.text_embeddings)
            elif hasattr(self.torch_model.text_embeddings, 'weight'):
                self.text_embeddings = torch_to_mlx(self.torch_model.text_embeddings.weight)
            
        if hasattr(self.torch_model, 'audio_embeddings'):
            if isinstance(self.torch_model.audio_embeddings, torch.Tensor):
                self.audio_embeddings = torch_to_mlx(self.torch_model.audio_embeddings)
            elif hasattr(self.torch_model.audio_embeddings, 'weight'):
                self.audio_embeddings = torch_to_mlx(self.torch_model.audio_embeddings.weight)
            
        # Convert projection
        if hasattr(self.torch_model, 'projection'):
            if isinstance(self.torch_model.projection, torch.Tensor):
                self.projection = torch_to_mlx(self.torch_model.projection)
            elif hasattr(self.torch_model.projection, 'weight'):
                self.projection = torch_to_mlx(self.torch_model.projection.weight)
            
        # Convert heads
        if hasattr(self.torch_model, 'codebook0_head'):
            # Get the weight tensor
            if isinstance(self.torch_model.codebook0_head, torch.Tensor):
                weight = self.torch_model.codebook0_head
            elif hasattr(self.torch_model.codebook0_head, 'weight'):
                weight = self.torch_model.codebook0_head.weight
            else:
                if self.debug:
                    print("codebook0_head is not a tensor or module with weight")
                weight = None
                
            if weight is not None:
                # Ensure codebook0_head matches the expected audio_vocab_size
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
                # Get the weight tensor
                if isinstance(head, torch.Tensor):
                    weight = head
                elif hasattr(head, 'weight'):
                    weight = head.weight
                else:
                    if self.debug:
                        print("audio_head component is not a tensor or module with weight")
                    weight = None
                    
                if weight is not None:
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
    
    def _fallback_generate(self, i=None, curr_sample=None, tokens=None, positions=None, topk=5, temperature=1.0):
        """
        Fallback method for generation that uses PyTorch.
        
        Args:
            i: Codebook index (optional)
            curr_sample: Current sample (optional)
            tokens: Input tokens for generation (optional)
            positions: Input positions for generation (optional)
            topk: Top-k sampling parameter (optional)
            temperature: Temperature for sampling (optional)
            
        Returns:
            Generated tokens
        """
        if self.debug:
            print(f"Using PyTorch fallback for generation (i={i}, tokens={tokens is not None})")
            
        # Specific codebook fallback (used from MLXFrameGenerator)
        if i is not None and curr_sample is not None:
            # Codebook fallback
            try:
                # Ensure curr_sample is a PyTorch tensor
                if not isinstance(curr_sample, torch.Tensor):
                    try:
                        curr_sample = mlx_to_torch(curr_sample)
                    except Exception as e:
                        if self.debug:
                            print(f"Error converting curr_sample to PyTorch: {e}")
                        # Create a PyTorch tensor directly
                        if hasattr(curr_sample, 'shape'):
                            curr_sample = torch.zeros(curr_sample.shape, device="cpu")
                        else:
                            curr_sample = torch.zeros((1, 1), device="cpu")
                
                # Check if torch model is available
                if self.torch_model is not None and hasattr(self.torch_model, '_generate_codebook'):
                    ci_sample, _ = self.torch_model._generate_codebook(
                        i, curr_sample, curr_sample.shape[1]
                    )
                    return ci_sample
                else:
                    # Return zero tensor if torch model is not available
                    return torch.zeros((curr_sample.shape[0] if hasattr(curr_sample, 'shape') else 1, 1), device="cpu")
            except Exception as e:
                if self.debug:
                    print(f"Codebook fallback failed: {e}")
                # Return zero tensor with appropriate shape derived from curr_sample
                if hasattr(curr_sample, 'shape') and len(curr_sample.shape) > 0:
                    return torch.zeros((curr_sample.shape[0], 1), device="cpu")
                else:
                    return torch.zeros((1, 1), device="cpu")
        
        # Full frame generation fallback (used directly from generate_frame)
        elif tokens is not None and positions is not None:
            if self.debug:
                print(f"Full frame generation fallback with tokens and positions")
            
            # Try using regular PyTorch generation with the torch model
            try:
                # Ensure tokens and positions are PyTorch tensors
                torch_tokens = tokens
                torch_positions = positions
                
                if not isinstance(tokens, torch.Tensor):
                    try:
                        torch_tokens = mlx_to_torch(tokens)
                    except Exception as e:
                        if self.debug:
                            print(f"Error converting tokens to PyTorch: {e}")
                        # Create PyTorch tensor with same shape
                        if hasattr(tokens, 'shape'):
                            torch_tokens = torch.zeros(tokens.shape, device="cpu", dtype=torch.long)
                        else:
                            torch_tokens = torch.zeros((1, 1), device="cpu", dtype=torch.long)
                
                if not isinstance(positions, torch.Tensor):
                    try:
                        torch_positions = mlx_to_torch(positions)
                    except Exception as e:
                        if self.debug:
                            print(f"Error converting positions to PyTorch: {e}")
                        # Create PyTorch tensor with same shape
                        if hasattr(positions, 'shape'):
                            torch_positions = torch.zeros(positions.shape, device="cpu", dtype=torch.long)
                        else:
                            torch_positions = torch.zeros((1, 1), device="cpu", dtype=torch.long)
                
                # Create tokens mask
                tokens_mask = torch.ones_like(torch_tokens, dtype=torch.float)
                
                # Use the torch model to generate the frame if available
                if self.torch_model is not None:
                    if hasattr(self.torch_model, 'generate_frame'):
                        # Direct generation
                        return self.torch_model.generate_frame(
                            torch_tokens, tokens_mask, torch_positions, temperature, topk
                        )
                    elif hasattr(self.torch_model, '_generate_audio_tokens'):
                        # Alternative generation method
                        return self.torch_model._generate_audio_tokens(
                            torch_tokens, torch_positions, temperature, topk
                        )
                    elif hasattr(self.torch_model, 'forward_first_stage'):
                        # This is what the hybrid approach was trying to do
                        # Get hidden states first, then generate codebooks
                        h, _, _ = self.torch_model.forward_first_stage(
                            torch_tokens, tokens_mask, torch_positions
                        )
                        # Continue with codebook generation...
                        if hasattr(self.torch_model, 'codebook0_head') and hasattr(self.torch_model, '_generate_codebook'):
                            c0_logits = self.torch_model.codebook0_head(h[:, -1, :])
                            from csm.models.model import sample_topk
                            c0_sample = sample_topk(c0_logits, topk, temperature)
                            
                            # Process remaining codebooks
                            curr_sample = c0_sample
                            for i in range(1, self.args.audio_num_codebooks):
                                ci_sample, _ = self.torch_model._generate_codebook(
                                    i, curr_sample, curr_sample.shape[1]
                                )
                                curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
                            
                            return curr_sample
            except Exception as e:
                if self.debug:
                    print(f"Full frame generation fallback failed: {e}")
                # Continue to generic fallback
                
        # Specific handler for generate_frame_direct method calls
        # This matches the signature used in the frame generator's generate_frame_direct method
        # which passes only the MLX tokens, positions, topk, and temperature
        elif isinstance(tokens, object) and isinstance(positions, object) and isinstance(topk, int) and isinstance(temperature, float):
            if self.debug:
                print(f"Direct frame generation fallback from generate_frame_direct")
            
            try:
                # Try to convert the MLX arrays to PyTorch tensors
                torch_tokens = None
                torch_positions = None
                
                # Convert tokens if possible
                try:
                    if hasattr(tokens, 'shape'):
                        # Try MLX to torch conversion
                        torch_tokens = mlx_to_torch(tokens)
                    else:
                        # Create a minimal dummy tensor
                        torch_tokens = torch.zeros((1, 1, 33), dtype=torch.long)  # batch, seq_len, codebooks+1
                except Exception as e:
                    if self.debug:
                        print(f"Error converting direct tokens to PyTorch: {e}")
                    # Create a minimal dummy tensor
                    torch_tokens = torch.zeros((1, 1, 33), dtype=torch.long)  # batch, seq_len, codebooks+1
                
                # Convert positions if possible
                try:
                    if hasattr(positions, 'shape'):
                        # Try MLX to torch conversion
                        torch_positions = mlx_to_torch(positions)
                    else:
                        # Create a minimal dummy tensor
                        torch_positions = torch.zeros((1, 1), dtype=torch.long)  # batch, seq_len
                except Exception as e:
                    if self.debug:
                        print(f"Error converting direct positions to PyTorch: {e}")
                    # Create a minimal dummy tensor
                    torch_positions = torch.zeros((1, 1), dtype=torch.long)  # batch, seq_len
                
                # Create tokens mask
                try:
                    tokens_mask = torch.ones((torch_tokens.shape[0], torch_tokens.shape[1]), dtype=torch.float)
                except Exception:
                    # Minimal fallback mask
                    tokens_mask = torch.ones((1, 1), dtype=torch.float)
                
                # Use the torch model to generate the frame
                if self.torch_model is not None:
                    if hasattr(self.torch_model, 'generate_frame'):
                        # Direct generation
                        return self.torch_model.generate_frame(
                            torch_tokens, tokens_mask, torch_positions, temperature, topk
                        )
                    elif hasattr(self.torch_model, '_generate_audio_tokens'):
                        # Alternative generation method
                        return self.torch_model._generate_audio_tokens(
                            torch_tokens, torch_positions, temperature, topk
                        )
            except Exception as e:
                if self.debug:
                    print(f"Direct frame generation fallback failed: {e}")
                # Continue to generic fallback
        
        # Generic fallback for all other cases
        # Safely get audio_num_codebooks with proper Namespace handling
        if hasattr(self, 'args'):
            if isinstance(self.args, dict):
                audio_num_codebooks = self.args.get('audio_num_codebooks', 32)
            else:
                # It's likely a Namespace object
                audio_num_codebooks = getattr(self.args, 'audio_num_codebooks', 32)
        else:
            # Default value if no args available
            audio_num_codebooks = 32
            
        return torch.zeros((1, audio_num_codebooks), device="cpu")
    
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
    
    def _init_from_args(self):
        """Initialize model structure from args without a PyTorch model."""
        # Import MLX modules
        import mlx.core as mx
        import mlx.nn as nn
        
        if self.debug:
            print("Initializing MLX model structure from args...")
        
        # Create dimensions from args
        hidden_size = self.args.get("hidden_size", 2048)
        num_layers = self.args.get("num_layers", 32)
        num_attention_heads = self.args.get("num_attention_heads", 32)
        intermediate_size = self.args.get("intermediate_size", 8192)
        
        # Create config object for transformer
        import argparse
        config = argparse.Namespace()
        config.hidden_size = hidden_size
        config.num_layers = num_layers
        config.num_attention_heads = num_attention_heads
        config.intermediate_size = intermediate_size
        config.num_key_value_heads = self.args.get("num_key_value_heads", num_attention_heads)
        
        # Initialize backbone and decoder with random weights
        self.backbone = MLXTransformer(config)
        self.decoder = MLXTransformer(config)
        
        # Initialize embeddings with random weights
        vocab_size = self.args.get("text_vocab_size", 128256)
        audio_vocab_size = self.args.get("audio_vocab_size", 2051)
        audio_num_codebooks = self.args.get("audio_num_codebooks", 32)
        
        # Create empty embeddings
        self.text_embeddings = mx.zeros((vocab_size, hidden_size))
        self.audio_embeddings = mx.zeros((audio_vocab_size * audio_num_codebooks, hidden_size))
        
        # Create projection and heads
        self.projection = mx.zeros((hidden_size, hidden_size))
        self.codebook0_head = mx.zeros((audio_vocab_size, hidden_size))
        
        # Create audio heads
        self.audio_head = []
        for i in range(audio_num_codebooks - 1):
            self.audio_head.append(mx.zeros((audio_vocab_size, hidden_size)))
        
        # Initialize embedding helper
        self.embedding = MLXEmbedding(
            text_embeddings=self.text_embeddings,
            audio_embeddings=self.audio_embeddings,
            audio_vocab_size=audio_vocab_size,
            audio_num_codebooks=audio_num_codebooks,
            embed_dim=hidden_size,
            debug=self.debug
        )
        
        # Create backbone and decoder causal masks
        self.backbone_causal_mask = create_causal_mask(2048)
        self.decoder_causal_mask = create_causal_mask(2048)
        
        if self.debug:
            print("MLX model structure initialized with random weights")
    
    def parameters(self):
        """Get all model parameters in a dictionary."""
        params = {}
        
        # Add backbone parameters
        if self.backbone:
            backbone_params = self.backbone.parameters()
            for name, param in backbone_params.items():
                params[f"backbone.{name}"] = param
        
        # Add decoder parameters
        if self.decoder:
            decoder_params = self.decoder.parameters()
            for name, param in decoder_params.items():
                params[f"decoder.{name}"] = param
        
        # Add embedding parameters
        if hasattr(self, 'text_embeddings'):
            params["text_embeddings"] = self.text_embeddings
        
        if hasattr(self, 'audio_embeddings'):
            params["audio_embeddings"] = self.audio_embeddings
        
        # Add projection and heads
        if hasattr(self, 'projection'):
            params["projection"] = self.projection
        
        if hasattr(self, 'codebook0_head'):
            params["codebook0_head"] = self.codebook0_head
        
        # Add audio heads
        if hasattr(self, 'audio_head'):
            for i, head in enumerate(self.audio_head):
                params[f"audio_head.{i}"] = head
        
        return params
    
    def update(self, params):
        """
        Update model parameters from a dictionary.
        
        Args:
            params: Dictionary of parameters
        """
        # Update backbone parameters
        backbone_params = {}
        for name, param in params.items():
            if name.startswith("backbone."):
                backbone_params[name[len("backbone."):]] = param
        
        if backbone_params and self.backbone:
            self.backbone.update(backbone_params)
        
        # Update decoder parameters
        decoder_params = {}
        for name, param in params.items():
            if name.startswith("decoder."):
                decoder_params[name[len("decoder."):]] = param
        
        if decoder_params and self.decoder:
            self.decoder.update(decoder_params)
        
        # Update embedding parameters
        if "text_embeddings" in params and hasattr(self, 'text_embeddings'):
            self.text_embeddings = params["text_embeddings"]
        
        if "audio_embeddings" in params and hasattr(self, 'audio_embeddings'):
            self.audio_embeddings = params["audio_embeddings"]
        
        # Update projection and heads
        if "projection" in params and hasattr(self, 'projection'):
            self.projection = params["projection"]
        
        if "codebook0_head" in params and hasattr(self, 'codebook0_head'):
            self.codebook0_head = params["codebook0_head"]
        
        # Update audio heads
        if hasattr(self, 'audio_head'):
            for i, _ in enumerate(self.audio_head):
                key = f"audio_head.{i}"
                if key in params:
                    self.audio_head[i] = params[key]
        
        # Reinitialize embedding helper with updated weights
        if hasattr(self, 'embedding') and (
            "text_embeddings" in params or "audio_embeddings" in params
        ):
            embed_dim = self.backbone.hidden_size if self.backbone else 2048
            audio_vocab_size = self.args.get("audio_vocab_size", 2051)
            audio_num_codebooks = self.args.get("audio_num_codebooks", 32)
            
            self.embedding = MLXEmbedding(
                text_embeddings=self.text_embeddings,
                audio_embeddings=self.audio_embeddings,
                audio_vocab_size=audio_vocab_size,
                audio_num_codebooks=audio_num_codebooks,
                embed_dim=embed_dim,
                debug=self.debug
            )
    
    def _embed_tokens(self, tokens):
        """
        Embed tokens using the embedding helper.
        
        Args:
            tokens: Input tokens
            
        Returns:
            Embedded tokens
        """
        if self.embedding:
            return self.embedding.embed_tokens(tokens)
        else:
            raise ValueError("Embedding helper not initialized")
            
    def embed_tokens(self, tokens):
        """
        Public wrapper for _embed_tokens to ensure compatibility with training code.
        
        Args:
            tokens: Input tokens
            
        Returns:
            Embedded tokens
        """
        return self._embed_tokens(tokens)
    
    def _index_causal_mask(self, mask, positions):
        """
        Index into causal mask for positions.
        
        Args:
            mask: Causal mask
            positions: Position indices
            
        Returns:
            Indexed mask
        """
        return index_causal_mask(mask, positions)