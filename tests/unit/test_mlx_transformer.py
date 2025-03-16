"""Tests for MLX Transformer component implementation."""

import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock

# Skip tests only if MLX is not available
try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    
pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


class MockConfig:
    """Mock model configuration for testing."""
    def __init__(self, 
                 hidden_size=32, 
                 num_layers=2, 
                 num_attention_heads=4,
                 num_key_value_heads=2,
                 intermediate_size=64,
                 max_position_embeddings=128):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings


def test_mlx_transformer_layer_init():
    """Test MLXTransformerLayer initialization."""
    from csm.mlx_accel.components.transformer import MLXTransformerLayer
    
    # Define parameters
    hidden_size = 32
    num_heads = 4
    num_kv_heads = 2
    intermediate_size = 64
    layer_idx = 0
    max_seq_len = 128
    
    # Create transformer layer
    layer = MLXTransformerLayer(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=intermediate_size,
        layer_idx=layer_idx,
        max_seq_len=max_seq_len
    )
    
    # Check attributes
    assert layer.hidden_size == hidden_size
    assert layer.num_heads == num_heads
    assert layer.num_kv_heads == num_kv_heads
    assert layer.head_dim == hidden_size // num_heads
    assert layer.intermediate_size == intermediate_size
    assert layer.layer_idx == layer_idx
    assert layer.max_seq_len == max_seq_len
    assert layer.dropout_prob == 0.0  # Default value
    assert layer.norm_eps == 1e-5     # Default value
    assert layer.use_cache is True    # Default value
    
    # Check weight parameters (should be None)
    assert layer.input_layernorm_weight is None
    assert layer.q_proj_weight is None
    assert layer.k_proj_weight is None
    assert layer.v_proj_weight is None
    assert layer.o_proj_weight is None
    assert layer.post_attention_layernorm_weight is None
    assert layer.gate_proj_weight is None
    assert layer.up_proj_weight is None
    assert layer.down_proj_weight is None
    
    # Check params_loaded flag
    assert layer.params_loaded is False


def test_mlx_transformer_layer_load_params():
    """Test loading parameters into a transformer layer."""
    from csm.mlx_accel.components.transformer import MLXTransformerLayer
    
    # Define parameters
    hidden_size = 32
    num_heads = 4
    num_kv_heads = 2
    intermediate_size = 64
    layer_idx = 0
    
    # Create transformer layer
    layer = MLXTransformerLayer(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=intermediate_size,
        layer_idx=layer_idx
    )
    
    # Create mock parameter dictionary using CSM naming convention
    params_dict = {
        "layers.0.sa_norm.scale": mx.ones((hidden_size,)),
        "layers.0.attn.q_proj.weight": mx.ones((hidden_size, hidden_size)),
        "layers.0.attn.k_proj.weight": mx.ones((num_kv_heads * (hidden_size // num_heads), hidden_size)),
        "layers.0.attn.v_proj.weight": mx.ones((num_kv_heads * (hidden_size // num_heads), hidden_size)),
        "layers.0.attn.output_proj.weight": mx.ones((hidden_size, hidden_size)),
        "layers.0.mlp_norm.scale": mx.ones((hidden_size,)),
        "layers.0.mlp.w1.weight": mx.ones((intermediate_size, hidden_size)),  # gate_proj
        "layers.0.mlp.w3.weight": mx.ones((intermediate_size, hidden_size)),  # up_proj
        "layers.0.mlp.w2.weight": mx.ones((hidden_size, intermediate_size)),  # down_proj
    }
    
    # Load parameters
    layer.load_params(params_dict, prefix="layers.0")
    
    # Check params_loaded flag is now True
    assert layer.params_loaded is True
    
    # Check parameters are loaded correctly
    assert layer.input_layernorm_weight is not None
    assert layer.q_proj_weight is not None
    assert layer.k_proj_weight is not None
    assert layer.v_proj_weight is not None
    assert layer.o_proj_weight is not None
    assert layer.post_attention_layernorm_weight is not None
    assert layer.gate_proj_weight is not None
    assert layer.up_proj_weight is not None
    assert layer.down_proj_weight is not None


def test_mlx_transformer_layer_layernorm():
    """Test layer normalization in transformer layer."""
    from csm.mlx_accel.components.transformer import MLXTransformerLayer
    
    # Create a layer
    hidden_size = 32
    layer = MLXTransformerLayer(
        hidden_size=hidden_size,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=64,
        layer_idx=0
    )
    
    # Create input tensor
    batch_size = 2
    seq_len = 3
    hidden_states = mx.ones((batch_size, seq_len, hidden_size))
    
    # Create layernorm weights and biases
    weight = mx.ones((hidden_size,))
    
    # Apply layernorm using the private method
    output = layer._layernorm(hidden_states, weight)
    
    # Check output shape
    assert output.shape == hidden_states.shape
    
    # Convert to numpy to check values
    output_np = np.array(output.tolist())
    
    # Layernorm should normalize the values, so the mean should be close to 0
    # and the variance should be close to 1 along the last dimension before applying
    # the weight and bias
    mean = np.mean(output_np, axis=-1)
    
    # Due to the weight = 1, the output values should be as follows:
    # - If all input values are the same, the normalized values will be all zeros
    # - Then adding the weight scaling (all ones) will result in zeros
    # So we expect all values to be close to zero in this specific case
    assert np.allclose(output_np, np.zeros_like(output_np), atol=1e-6)


def test_mlx_transformer_init():
    """Test MLXTransformer initialization."""
    from csm.mlx_accel.components.transformer import MLXTransformer
    
    # Create config
    config = MockConfig()
    
    # Create transformer
    transformer = MLXTransformer(config)
    
    # Check attributes
    assert transformer.hidden_size == config.hidden_size
    assert transformer.num_layers == config.num_layers
    assert transformer.num_heads == config.num_attention_heads
    assert transformer.intermediate_size == config.intermediate_size
    assert transformer.max_seq_len == config.max_position_embeddings
    assert transformer.num_kv_heads == config.num_key_value_heads
    assert transformer.head_dim == config.hidden_size // config.num_attention_heads
    assert transformer.use_cache is True
    
    # Check layer initialization
    assert len(transformer.layers) == config.num_layers
    
    # Check other attributes
    assert transformer.final_layernorm_weight is None
    assert transformer.final_layernorm_bias is None
    assert transformer.cos_cached is None
    assert transformer.sin_cached is None
    assert transformer.past_key_values is None


def test_mlx_transformer_init_rope_embeddings():
    """Test initialization of rotary position embeddings."""
    from csm.mlx_accel.components.transformer import MLXTransformer
    
    # Create config
    config = MockConfig(
        hidden_size=32,
        num_attention_heads=4,
        max_position_embeddings=128
    )
    
    # Create transformer
    transformer = MLXTransformer(config)
    
    # Initialize RoPE embeddings
    transformer._init_rope_embeddings()
    
    # Check embeddings
    assert transformer.cos_cached is not None
    assert transformer.sin_cached is not None
    
    # Check shapes
    head_dim = config.hidden_size // config.num_attention_heads
    assert transformer.cos_cached.shape == (config.max_position_embeddings, head_dim // 2)
    assert transformer.sin_cached.shape == (config.max_position_embeddings, head_dim // 2)
    
    # Check that embeddings are set for each layer
    for layer in transformer.layers:
        assert layer.cos_cached is transformer.cos_cached
        assert layer.sin_cached is transformer.sin_cached


def test_mlx_transformer_load_params():
    """Test loading parameters into the transformer model."""
    from csm.mlx_accel.components.transformer import MLXTransformer
    
    # Create config
    config = MockConfig(
        hidden_size=32,
        num_layers=2,
        num_attention_heads=4,
        intermediate_size=64
    )
    
    # Create transformer
    transformer = MLXTransformer(config)
    
    # Create mock parameters dict (minimal for testing)
    params_dict = {
        # Layer 0 parameters
        "layers.0.sa_norm.scale": mx.ones((config.hidden_size,)),
        "layers.0.attn.q_proj.weight": mx.ones((config.hidden_size, config.hidden_size)),
        "layers.0.attn.k_proj.weight": mx.ones((config.num_key_value_heads * (config.hidden_size // config.num_attention_heads), config.hidden_size)),
        "layers.0.attn.v_proj.weight": mx.ones((config.num_key_value_heads * (config.hidden_size // config.num_attention_heads), config.hidden_size)),
        "layers.0.attn.output_proj.weight": mx.ones((config.hidden_size, config.hidden_size)),
        "layers.0.mlp_norm.scale": mx.ones((config.hidden_size,)),
        "layers.0.mlp.w1.weight": mx.ones((config.intermediate_size, config.hidden_size)),
        "layers.0.mlp.w3.weight": mx.ones((config.intermediate_size, config.hidden_size)),
        "layers.0.mlp.w2.weight": mx.ones((config.hidden_size, config.intermediate_size)),
        
        # Layer 1 parameters
        "layers.1.sa_norm.scale": mx.ones((config.hidden_size,)),
        "layers.1.attn.q_proj.weight": mx.ones((config.hidden_size, config.hidden_size)),
        "layers.1.attn.k_proj.weight": mx.ones((config.num_key_value_heads * (config.hidden_size // config.num_attention_heads), config.hidden_size)),
        "layers.1.attn.v_proj.weight": mx.ones((config.num_key_value_heads * (config.hidden_size // config.num_attention_heads), config.hidden_size)),
        "layers.1.attn.output_proj.weight": mx.ones((config.hidden_size, config.hidden_size)),
        "layers.1.mlp_norm.scale": mx.ones((config.hidden_size,)),
        "layers.1.mlp.w1.weight": mx.ones((config.intermediate_size, config.hidden_size)),
        "layers.1.mlp.w3.weight": mx.ones((config.intermediate_size, config.hidden_size)),
        "layers.1.mlp.w2.weight": mx.ones((config.hidden_size, config.intermediate_size)),
        
        # Final layernorm
        "model.norm.scale": mx.ones((config.hidden_size,)),
    }
    
    # Load parameters
    transformer.load_params(params_dict)
    
    # Check final layernorm is loaded
    assert transformer.final_layernorm_weight is not None
    
    # Check RoPE embeddings are initialized
    assert transformer.cos_cached is not None
    assert transformer.sin_cached is not None
    
    # Check layer parameters are loaded
    for layer_idx, layer in enumerate(transformer.layers):
        assert layer.params_loaded is True
        assert layer.input_layernorm_weight is not None
        assert layer.q_proj_weight is not None