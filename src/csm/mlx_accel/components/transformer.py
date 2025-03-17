"""
Transformer model implementation for MLX acceleration.
"""

import math
from typing import Dict, List, Optional, Tuple, Union, Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch

from csm.mlx_accel.mlx_ops import torch_to_mlx, create_causal_mask, index_causal_mask

class MLXTransformerLayer:
    """MLX implementation of a Llama 3.2 transformer layer."""
    
    def __init__(self, 
                hidden_size, 
                num_heads, 
                num_kv_heads, 
                intermediate_size, 
                layer_idx,
                max_seq_len=2048, 
                dropout_prob=0.0, 
                norm_eps=1e-5,
                use_cache=True):
        """Initialize a transformer layer with MLX parameters."""
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.intermediate_size = intermediate_size
        self.layer_idx = layer_idx
        self.max_seq_len = max_seq_len
        self.dropout_prob = dropout_prob
        self.norm_eps = norm_eps
        self.use_cache = use_cache
        
        # Initialize arrays to hold parameters
        # These would be filled with actual model parameters during initialization
        self.input_layernorm_weight = None
        self.input_layernorm_bias = None
        self.q_proj_weight = None
        self.q_proj_bias = None
        self.k_proj_weight = None
        self.k_proj_bias = None
        self.v_proj_weight = None
        self.v_proj_bias = None
        self.o_proj_weight = None
        self.o_proj_bias = None
        self.post_attention_layernorm_weight = None
        self.post_attention_layernorm_bias = None
        self.gate_proj_weight = None  # SwiGLU gate
        self.gate_proj_bias = None
        self.up_proj_weight = None    # SwiGLU up
        self.up_proj_bias = None
        self.down_proj_weight = None  # SwiGLU down
        self.down_proj_bias = None
        
        # RoPE matrices
        self.cos_cached = None
        self.sin_cached = None
        
        # Will be set during model loading
        self.params_loaded = False
    
    def load_params(self, params_dict, prefix=""):
        """
        Load parameters from a dictionary of MLX arrays.
        
        Args:
            params_dict: Dictionary of parameter arrays
            prefix: Prefix for parameter names in the dictionary
        """
        # Map of parameter keys for different model architectures
        # This helps handle variations in model parameter naming
        param_map = {
            # Standard Llama 3.2 naming
            "standard": {
                "input_norm": f"{prefix}.input_layernorm.weight",
                "input_norm_bias": f"{prefix}.input_layernorm.bias",
                "q_proj": f"{prefix}.self_attn.q_proj.weight",
                "q_proj_bias": f"{prefix}.self_attn.q_proj.bias",
                "k_proj": f"{prefix}.self_attn.k_proj.weight",
                "k_proj_bias": f"{prefix}.self_attn.k_proj.bias",
                "v_proj": f"{prefix}.self_attn.v_proj.weight",
                "v_proj_bias": f"{prefix}.self_attn.v_proj.bias",
                "o_proj": f"{prefix}.self_attn.o_proj.weight",
                "o_proj_bias": f"{prefix}.self_attn.o_proj.bias",
                "post_norm": f"{prefix}.post_attention_layernorm.weight",
                "post_norm_bias": f"{prefix}.post_attention_layernorm.bias",
                "gate_proj": f"{prefix}.mlp.gate_proj.weight",
                "gate_proj_bias": f"{prefix}.mlp.gate_proj.bias",
                "up_proj": f"{prefix}.mlp.up_proj.weight",
                "up_proj_bias": f"{prefix}.mlp.up_proj.bias",
                "down_proj": f"{prefix}.mlp.down_proj.weight",
                "down_proj_bias": f"{prefix}.mlp.down_proj.bias",
            },
            # CSM naming (torchtune)
            "csm": {
                "input_norm": f"{prefix}.sa_norm.scale",
                "q_proj": f"{prefix}.attn.q_proj.weight",
                "k_proj": f"{prefix}.attn.k_proj.weight",
                "v_proj": f"{prefix}.attn.v_proj.weight",
                "o_proj": f"{prefix}.attn.output_proj.weight",
                "post_norm": f"{prefix}.mlp_norm.scale",
                "gate_proj": f"{prefix}.mlp.w1.weight",  # gate/swiGLU in CSM
                "up_proj": f"{prefix}.mlp.w3.weight",    # up-project in CSM
                "down_proj": f"{prefix}.mlp.w2.weight",  # down-project in CSM
            },
        }
        
        # Try different parameter name mappings
        failed_keys = []
        success_count = 0
        
        # First try CSM style (most likely)
        for target_key, source_key in param_map["csm"].items():
            if source_key in params_dict:
                # Process this parameter
                if target_key == "input_norm":
                    self.input_layernorm_weight = params_dict[source_key]
                    success_count += 1
                elif target_key == "q_proj":
                    self.q_proj_weight = params_dict[source_key]
                    success_count += 1
                elif target_key == "k_proj":
                    self.k_proj_weight = params_dict[source_key]
                    success_count += 1
                elif target_key == "v_proj":
                    self.v_proj_weight = params_dict[source_key]
                    success_count += 1
                elif target_key == "o_proj":
                    self.o_proj_weight = params_dict[source_key]
                    success_count += 1
                elif target_key == "post_norm":
                    self.post_attention_layernorm_weight = params_dict[source_key]
                    success_count += 1
                elif target_key == "gate_proj":
                    self.gate_proj_weight = params_dict[source_key]
                    success_count += 1
                elif target_key == "up_proj":
                    self.up_proj_weight = params_dict[source_key]
                    success_count += 1
                elif target_key == "down_proj":
                    self.down_proj_weight = params_dict[source_key]
                    success_count += 1
            else:
                failed_keys.append(source_key)
        
        # Set loaded flag if a majority of parameters were loaded
        self.params_loaded = success_count >= 7  # At least 7 of 9 parameters loaded
        
        if not self.params_loaded:
            print(f"Warning: Failed to load parameters for layer {self.layer_idx} using CSM naming")
            print(f"Missing keys: {failed_keys}")
            print(f"Available keys: {list(params_dict.keys())[:10]}...")
    
    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False):
        """Forward pass through the transformer layer."""
        residual = hidden_states
        
        # Check if we have all needed parameters
        if not self.params_loaded:
            raise ValueError(f"Layer {self.layer_idx} parameters not loaded")
        
        # Apply layer norm
        layernorm_output = self._layernorm(hidden_states, 
                                          self.input_layernorm_weight,
                                          self.input_layernorm_bias)
        
        # Self-attention
        attention_output = self._attention(
            layernorm_output,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value
        )
        
        # First residual connection
        hidden_states = residual + attention_output
        
        # Second residual connection with feedforward network
        residual = hidden_states
        
        # Apply second layer norm
        layernorm_output = self._layernorm(hidden_states,
                                         self.post_attention_layernorm_weight,
                                         self.post_attention_layernorm_bias)
        
        # Apply feedforward network
        feedforward_output = self._feedforward(layernorm_output)
        
        # Second residual connection
        hidden_states = residual + feedforward_output
        
        return hidden_states
    
    def _layernorm(self, hidden_states, weight, bias=None, eps=1e-5):
        """Apply layer normalization."""
        # Get mean and variance for layer normalization
        mean = mx.mean(hidden_states, axis=-1, keepdims=True)
        variance = mx.mean((hidden_states - mean) ** 2, axis=-1, keepdims=True)
        
        # Normalize using weight and bias
        hidden_states = (hidden_states - mean) / mx.sqrt(variance + eps)
        
        # Apply weight and bias if present
        if weight is not None:
            hidden_states = hidden_states * weight.reshape(1, 1, -1)
        
        if bias is not None:
            hidden_states = hidden_states + bias.reshape(1, 1, -1)
            
        return hidden_states
    
    def _feedforward(self, hidden_states):
        """Apply feedforward network with SwiGLU activation."""
        # Step 1: Calculate gating and up-projection
        if self.gate_proj_weight is not None:
            gate_proj = mx.matmul(hidden_states, self.gate_proj_weight.T)
            if self.gate_proj_bias is not None:
                gate_proj = gate_proj + self.gate_proj_bias
        else:
            raise ValueError("Gate projection weight is not loaded")
            
        if self.up_proj_weight is not None:
            up_proj = mx.matmul(hidden_states, self.up_proj_weight.T)
            if self.up_proj_bias is not None:
                up_proj = up_proj + self.up_proj_bias
        else:
            raise ValueError("Up projection weight is not loaded")
            
        # Step 2: Apply SwiGLU activation
        # SwiGLU = gate * swish(up_proj)
        # swish(x) = x * sigmoid(x)
        swish = up_proj * mx.sigmoid(up_proj)
        intermediate = gate_proj * swish
        
        # Step 3: Apply down projection
        if self.down_proj_weight is not None:
            output = mx.matmul(intermediate, self.down_proj_weight.T)
            if self.down_proj_bias is not None:
                output = output + self.down_proj_bias
        else:
            raise ValueError("Down projection weight is not loaded")
            
        return output
    
    def _attention(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
        """Apply multi-head attention."""
        batch_size, seq_length, hidden_size = hidden_states.shape
        
        # Step 1: Project query, key, value
        if self.q_proj_weight is not None:
            query_states = mx.matmul(hidden_states, self.q_proj_weight.T)
            if self.q_proj_bias is not None:
                query_states = query_states + self.q_proj_bias
        else:
            raise ValueError("Query projection weight is not loaded")
            
        if self.k_proj_weight is not None:
            key_states = mx.matmul(hidden_states, self.k_proj_weight.T)
            if self.k_proj_bias is not None:
                key_states = key_states + self.k_proj_bias
        else:
            raise ValueError("Key projection weight is not loaded")
            
        if self.v_proj_weight is not None:
            value_states = mx.matmul(hidden_states, self.v_proj_weight.T)
            if self.v_proj_bias is not None:
                value_states = value_states + self.v_proj_bias
        else:
            raise ValueError("Value projection weight is not loaded")
        
        # Step 2: Reshape for multi-head attention
        head_dim = hidden_size // self.num_heads
        
        query_states = query_states.reshape(batch_size, seq_length, self.num_heads, head_dim)
        key_states = key_states.reshape(batch_size, seq_length, self.num_kv_heads, head_dim)
        value_states = value_states.reshape(batch_size, seq_length, self.num_kv_heads, head_dim)
        
        # Step 3: Apply rotary embeddings if position_ids are provided
        if position_ids is not None and self.cos_cached is not None and self.sin_cached is not None:
            # Apply rotary embeddings to query and key states
            cos = mx.take(self.cos_cached, position_ids, axis=0)
            sin = mx.take(self.sin_cached, position_ids, axis=0)
            
            # NOTE: We don't need to reshape here because our _apply_rotary_pos_emb
            # method handles various input shapes including this one
            
            # Apply rotary embedding operation
            query_states = self._apply_rotary_pos_emb(query_states, cos, sin)
            key_states = self._apply_rotary_pos_emb(key_states, cos, sin)
        
        # Step 4: Transpose for matrix multiplication
        # [batch_size, num_heads, seq_length, head_dim]
        query_states = mx.transpose(query_states, (0, 2, 1, 3))
        key_states = mx.transpose(key_states, (0, 2, 1, 3))
        value_states = mx.transpose(value_states, (0, 2, 1, 3))
        
        # Step 5: Calculate attention scores
        # [batch_size, num_heads, seq_length, seq_length]
        attention_scores = mx.matmul(query_states, mx.transpose(key_states, (0, 1, 3, 2)))
        
        # Scale attention scores
        attention_scores = attention_scores / math.sqrt(head_dim)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Ensure mask has right shape for broadcasting
            # [batch_size, 1, seq_length, seq_length]
            if len(attention_mask.shape) == 3:
                attention_mask = mx.expand_dims(attention_mask, axis=1)
                
            # Apply mask by adding a large negative value to masked positions
            # Create a tensor filled with large negative values for masked positions
            neg_inf = -1e9
            attention_scores = mx.where(attention_mask, attention_scores, neg_inf)
        
        # Step 6: Apply softmax to get attention probabilities
        attention_probs = mx.softmax(attention_scores, axis=-1)
        
        # Step 7: Compute context as weighted sum of values
        # [batch_size, num_heads, seq_length, head_dim]
        context = mx.matmul(attention_probs, value_states)
        
        # Step 8: Transpose and reshape back
        # [batch_size, seq_length, num_heads, head_dim]
        context = mx.transpose(context, (0, 2, 1, 3))
        
        # [batch_size, seq_length, hidden_size]
        context = context.reshape(batch_size, seq_length, hidden_size)
        
        # Step 9: Apply output projection
        if self.o_proj_weight is not None:
            context = mx.matmul(context, self.o_proj_weight.T)
            if self.o_proj_bias is not None:
                context = context + self.o_proj_bias
        else:
            raise ValueError("Output projection weight is not loaded")
            
        return context
    
    def _apply_rotary_pos_emb(self, states, cos, sin):
        """Apply rotary position embeddings to query and key states."""
        # Extract dimensions
        batch_size, seq_len, num_heads, head_dim = states.shape
        half_dim = head_dim // 2
        
        # Reshape inputs if needed - handle various input shapes
        if cos.shape != (batch_size, seq_len, 1, half_dim):
            # Extract the core cos/sin values from the shape
            if len(cos.shape) == 2 and cos.shape[0] >= seq_len:
                # Handle case where cos/sin are [max_seq_len, head_dim//2]
                # Take the relevant positions
                cos_used = cos[:seq_len, :half_dim]
                sin_used = sin[:seq_len, :half_dim]
                
                # Now reshape to match required dimensions
                cos = mx.broadcast_to(
                    mx.reshape(cos_used, (1, seq_len, 1, half_dim)),
                    (batch_size, seq_len, 1, half_dim)
                )
                sin = mx.broadcast_to(
                    mx.reshape(sin_used, (1, seq_len, 1, half_dim)),
                    (batch_size, seq_len, 1, half_dim)
                )
            else:
                # General fallback - try to reshape directly
                try:
                    cos = cos.reshape(batch_size, seq_len, 1, half_dim)
                    sin = sin.reshape(batch_size, seq_len, 1, half_dim)
                except ValueError:
                    # If all else fails, just create new compatible tensors
                    print(f"Warning: Incompatible cos/sin shapes: {cos.shape}, {sin.shape}. Creating new ones.")
                    cos = mx.ones((batch_size, seq_len, 1, half_dim))
                    sin = mx.ones((batch_size, seq_len, 1, half_dim))
            
        # Split the tensor into even and odd dimensions
        # For rotary embeddings, we treat even and odd dimensions differently
        states_reshape = states.reshape(batch_size, seq_len, num_heads, half_dim, 2)
        
        # Separate even and odd dimensions
        x1 = states_reshape[..., 0]  # Even dimensions
        x2 = states_reshape[..., 1]  # Odd dimensions
        
        # Apply rotation formula: (cos * x1 - sin * x2, sin * x1 + cos * x2)
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        
        # Combine rotated dimensions
        states_rotated = mx.stack([y1, y2], axis=-1)
        states_rotated = states_rotated.reshape(batch_size, seq_len, num_heads, head_dim)
        
        return states_rotated

class MLXTransformer:
    """MLX implementation of a transformer model."""
    
    def __init__(self, config):
        """
        Initialize a transformer model.
        
        Args:
            config: Model configuration with parameters like hidden_size, num_layers, etc.
        """
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.num_heads = config.num_attention_heads
        self.intermediate_size = config.intermediate_size
        self.max_seq_len = getattr(config, 'max_position_embeddings', 2048)
        self.num_kv_heads = getattr(config, 'num_key_value_heads', self.num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.use_cache = True
        
        # Initialize layers
        self.layers = []
        for i in range(self.num_layers):
            layer = MLXTransformerLayer(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                intermediate_size=self.intermediate_size,
                layer_idx=i,
                max_seq_len=self.max_seq_len
            )
            self.layers.append(layer)
            
        # Final layer norm
        self.final_layernorm_weight = None
        self.final_layernorm_bias = None
        
        # RoPE embeddings
        self.cos_cached = None
        self.sin_cached = None
        
        # Caches for key-value pairs
        self.past_key_values = None
        
    def load_params(self, params_dict):
        """
        Load parameters from a dictionary of MLX arrays.
        
        Args:
            params_dict: Dictionary of parameter arrays
        """
        # Load layer parameters
        for i, layer in enumerate(self.layers):
            layer_prefix = f"layers.{i}"
            layer.load_params(params_dict, prefix=layer_prefix)
            
        # Load final layer norm
        if "model.norm.scale" in params_dict:
            self.final_layernorm_weight = params_dict["model.norm.scale"]
        elif "model.norm.weight" in params_dict:
            self.final_layernorm_weight = params_dict["model.norm.weight"]
            
        if "model.norm.bias" in params_dict:
            self.final_layernorm_bias = params_dict["model.norm.bias"]
            
        # Initialize RoPE embeddings
        self._init_rope_embeddings()
        
        # Initialize caches
        self.reset_caches()
        
    def _init_rope_embeddings(self):
        """Initialize rotary position embeddings."""
        # Configuration constants
        theta = 10000.0
        
        # Create position indices
        position = mx.arange(0, self.max_seq_len)
        
        # Create frequencies
        freqs = 1.0 / (theta ** (mx.arange(0, self.head_dim // 2) / (self.head_dim // 2)))
        
        # Outer product of positions and frequencies
        t = mx.reshape(position, (-1, 1)) * mx.reshape(freqs, (1, -1))
        
        # Create sin and cos embeddings
        self.cos_cached = mx.cos(t)
        self.sin_cached = mx.sin(t)
        
        # Set RoPE embeddings for each layer
        for layer in self.layers:
            layer.cos_cached = self.cos_cached
            layer.sin_cached = self.sin_cached
            
    def reset_caches(self):
        """Reset key-value caches for inference."""
        self.past_key_values = None
        
    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        """
        Forward pass through the transformer model.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_length, hidden_size]
            attention_mask: Optional attention mask of shape [batch_size, seq_length, seq_length]
            position_ids: Optional position indices of shape [batch_size, seq_length]
            
        Returns:
            Output tensor of shape [batch_size, seq_length, hidden_size]
        """
        # Process through each transformer layer
        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer.forward(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids
            )
            
        # Apply final layer norm if available
        if self.final_layernorm_weight is not None:
            # Get mean and variance for layer normalization
            mean = mx.mean(hidden_states, axis=-1, keepdims=True)
            variance = mx.mean((hidden_states - mean) ** 2, axis=-1, keepdims=True)
            
            # Normalize using weight and bias
            hidden_states = (hidden_states - mean) / mx.sqrt(variance + 1e-5)
            
            # Apply weight and bias if present
            hidden_states = hidden_states * self.final_layernorm_weight.reshape(1, 1, -1)
            
            if self.final_layernorm_bias is not None:
                hidden_states = hidden_states + self.final_layernorm_bias.reshape(1, 1, -1)
                
        return hidden_states