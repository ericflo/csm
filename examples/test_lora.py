#!/usr/bin/env python
"""
Simple test script for LoRA functionality.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("MLX not available. This test requires MLX.")
    sys.exit(1)

try:
    import safetensors.numpy
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    print("safetensors.numpy not available. This test requires safetensors.")
    sys.exit(1)

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Define a simple transformer layer for testing
class SimpleTransformerLayer:
    def __init__(self, hidden_size=16, num_heads=2, num_kv_heads=None):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = hidden_size // num_heads
        
        # Initialize attention weights
        self.q_proj_weight = mx.ones((hidden_size, hidden_size)) * 0.1
        self.k_proj_weight = mx.ones((hidden_size, hidden_size)) * 0.1
        self.v_proj_weight = mx.ones((hidden_size, hidden_size)) * 0.1
        self.o_proj_weight = mx.ones((hidden_size, hidden_size)) * 0.1
        
        # Initialize attention biases
        self.q_proj_bias = mx.zeros(hidden_size)
        self.k_proj_bias = mx.zeros(hidden_size)
        self.v_proj_bias = mx.zeros(hidden_size)
        self.o_proj_bias = mx.zeros(hidden_size)
        
        # Initialize MLP weights
        self.gate_proj_weight = mx.ones((hidden_size * 4, hidden_size)) * 0.1
        self.up_proj_weight = mx.ones((hidden_size * 4, hidden_size)) * 0.1
        self.down_proj_weight = mx.ones((hidden_size, hidden_size * 4)) * 0.1
        
        # Initialize MLP biases
        self.gate_proj_bias = mx.zeros(hidden_size * 4)
        self.up_proj_bias = mx.zeros(hidden_size * 4)
        self.down_proj_bias = mx.zeros(hidden_size)
        
        # Layer norms
        self.input_layernorm_weight = mx.ones(hidden_size)
        self.input_layernorm_bias = mx.zeros(hidden_size)
        self.post_attention_layernorm_weight = mx.ones(hidden_size)
        self.post_attention_layernorm_bias = mx.zeros(hidden_size)

    def _layernorm(self, x, weight, bias):
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.mean((x - mean) ** 2, axis=-1, keepdims=True)
        return weight * (x - mean) / mx.sqrt(var + 1e-5) + bias
    
    def __call__(self, hidden_states, attention_mask=None, position_ids=None):
        # Simple forward pass
        residual = hidden_states
        
        # Layer norm
        layernorm_output = self._layernorm(
            hidden_states,
            self.input_layernorm_weight,
            self.input_layernorm_bias
        )
        
        # Query projection
        q = mx.matmul(layernorm_output, self.q_proj_weight.T) + self.q_proj_bias
        
        # Key projection
        k = mx.matmul(layernorm_output, self.k_proj_weight.T) + self.k_proj_bias
        
        # Value projection
        v = mx.matmul(layernorm_output, self.v_proj_weight.T) + self.v_proj_bias
        
        # Reshape for attention
        batch_size, seq_len, _ = hidden_states.shape
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for matmul
        q = mx.transpose(q, (0, 2, 1, 3))  # [batch, heads, seq_len, head_dim]
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))
        
        # Attention weights
        scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2)))
        scores = scores / mx.sqrt(mx.array(self.head_dim, dtype=mx.float32))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            scores = mx.where(attention_mask, scores, mx.full_like(scores, -1e9))
        
        # Softmax
        attn_weights = mx.softmax(scores, axis=-1)
        
        # Apply attention
        attn_output = mx.matmul(attn_weights, v)
        
        # Reshape and transpose back
        attn_output = mx.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        
        # Output projection
        attn_output = mx.matmul(attn_output, self.o_proj_weight.T) + self.o_proj_bias
        
        # Residual connection
        hidden_states = residual + attn_output
        
        # Second residual
        residual = hidden_states
        
        # Second layer norm
        layernorm_output = self._layernorm(
            hidden_states,
            self.post_attention_layernorm_weight,
            self.post_attention_layernorm_bias
        )
        
        # Simple feedforward
        gate = mx.matmul(layernorm_output, self.gate_proj_weight.T) + self.gate_proj_bias
        up = mx.matmul(layernorm_output, self.up_proj_weight.T) + self.up_proj_bias
        
        # SwiGLU activation
        swish = up * mx.sigmoid(up)
        gate_out = gate * swish
        
        # Down projection
        ffn_output = mx.matmul(gate_out, self.down_proj_weight.T) + self.down_proj_bias
        
        # Second residual connection
        hidden_states = residual + ffn_output
        
        return hidden_states

# Simple transformer model for testing
class SimpleTransformer:
    def __init__(self, hidden_size=16, num_layers=1, num_heads=2):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.num_layers = num_layers
        
        # Create layers
        self.layers = [SimpleTransformerLayer(hidden_size, num_heads) for _ in range(num_layers)]
        
        # Final layer norm
        self.final_layernorm_weight = mx.ones(hidden_size)
        self.final_layernorm_bias = mx.zeros(hidden_size)
    
    def __call__(self, hidden_states, attention_mask=None, position_ids=None):
        # Pass through each layer
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids
            )
        
        # Final layer norm
        mean = mx.mean(hidden_states, axis=-1, keepdims=True)
        var = mx.mean((hidden_states - mean) ** 2, axis=-1, keepdims=True)
        hidden_states = (hidden_states - mean) / mx.sqrt(var + 1e-5)
        hidden_states = hidden_states * self.final_layernorm_weight.reshape(1, 1, -1)
        hidden_states = hidden_states + self.final_layernorm_bias.reshape(1, 1, -1)
        
        return hidden_states

def test_lora():
    """Run a simple test of LoRA functionality."""
    print("Testing LoRA functionality...")
    
    from csm.mlx.components.lora import apply_lora_to_model
    
    # Create a simple model
    hidden_size = 16
    batch_size = 2
    seq_len = 8
    
    # Create a model with backbone and decoder
    class SimpleCSMModel:
        def __init__(self):
            self.backbone = SimpleTransformer(hidden_size=hidden_size)
            self.decoder = SimpleTransformer(hidden_size=hidden_size)
    
    model = SimpleCSMModel()
    
    # Create input
    input_data = mx.random.normal((batch_size, seq_len, hidden_size))
    
    # Run a forward pass before applying LoRA
    print("Running forward pass with original model...")
    backbone_output = model.backbone(input_data)
    decoder_output = model.decoder(backbone_output)
    print(f"Backbone output shape: {backbone_output.shape}")
    print(f"Decoder output shape: {decoder_output.shape}")
    
    # Apply LoRA to the model
    print("\nApplying LoRA to the model...")
    lora_model = apply_lora_to_model(
        model=model,
        r=8,
        alpha=16.0,
        target_modules=["q_proj", "v_proj"]
    )
    
    # Run a forward pass with LoRA
    print("Running forward pass with LoRA model...")
    try:
        backbone_output_lora = lora_model.backbone(input_data)
        decoder_output_lora = lora_model.decoder(backbone_output_lora)
        print(f"LoRA backbone output shape: {backbone_output_lora.shape}")
        print(f"LoRA decoder output shape: {decoder_output_lora.shape}")
        
        # Check LoRA parameters
        lora_params = lora_model.get_lora_params()
        print(f"\nLoRA parameters count: {len(lora_params)}")
        
        # Compare original and LoRA outputs
        backbone_diff = mx.abs(backbone_output - backbone_output_lora).mean()
        decoder_diff = mx.abs(decoder_output - decoder_output_lora).mean()
        print(f"Mean absolute difference in backbone output: {backbone_diff}")
        print(f"Mean absolute difference in decoder output: {decoder_diff}")
        
        print("\nLoRA test successful!")
        return True
    except Exception as e:
        print(f"Error in LoRA test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Print helpful information about LoRA audio generation
    print("\n" + "="*80)
    print(" LoRA AUDIO GENERATION ".center(80, "="))
    print("="*80)
    print("\nCSM now fully supports direct LoRA inference without any extra steps!")
    print("\nYou can simply use your LoRA model directly for audio generation:")
    print("  - Just pass your LoRA model to any audio generation function")
    print("  - No need for explicit merging (though that option is still available)")
    print("\nFor better performance with repeated generations, you can optionally merge")
    print("LoRA weights with the base model using model.merge_lora_weights()")
    print("="*80 + "\n")
    
    test_lora()