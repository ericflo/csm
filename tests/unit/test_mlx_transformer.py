"""Tests for MLX Transformer component implementation."""

import pytest
import sys
import math
import numpy as np
import torch
from unittest.mock import patch, MagicMock, Mock

# Skip tests only if MLX is not available
try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

# Create mock mlx modules when not available
if not HAS_MLX:
    class MockArray:
        def __init__(self, data):
            self.data = np.array(data)
            self.shape = self.data.shape
            
        def reshape(self, *args):
            new_shape = args if len(args) > 1 else args[0]
            return MockArray(self.data.reshape(new_shape))
            
        def __getitem__(self, idx):
            return MockArray(self.data[idx])
            
        def transpose(self, *args):
            return MockArray(self.data.transpose(*args))
        
        def tolist(self):
            return self.data.tolist()
            
    class MockMX:
        @staticmethod
        def array(data, **kwargs):
            return MockArray(data)
            
        @staticmethod
        def ones(shape, **kwargs):
            return MockArray(np.ones(shape))
            
        @staticmethod
        def zeros(shape, **kwargs):
            return MockArray(np.zeros(shape))
            
        @staticmethod
        def arange(start, stop=None, **kwargs):
            if stop is None:
                stop = start
                start = 0
            return MockArray(np.arange(start, stop))
            
        @staticmethod
        def mean(x, axis=None, keepdims=False):
            if isinstance(x, MockArray):
                result = np.mean(x.data, axis=axis, keepdims=keepdims)
            else:
                result = np.mean(x, axis=axis, keepdims=keepdims)
            return MockArray(result)
            
        @staticmethod
        def sqrt(x):
            if isinstance(x, MockArray):
                return MockArray(np.sqrt(x.data))
            return MockArray(np.sqrt(x))
            
        @staticmethod
        def matmul(a, b):
            if isinstance(a, MockArray) and isinstance(b, MockArray):
                return MockArray(np.matmul(a.data, b.data))
            elif isinstance(a, MockArray):
                return MockArray(np.matmul(a.data, b))
            elif isinstance(b, MockArray):
                return MockArray(np.matmul(a, b.data))
            return MockArray(np.matmul(a, b))
            
        @staticmethod
        def transpose(x, axes):
            if isinstance(x, MockArray):
                return MockArray(np.transpose(x.data, axes))
            return MockArray(np.transpose(x, axes))
            
        @staticmethod
        def expand_dims(x, axis):
            if isinstance(x, MockArray):
                return MockArray(np.expand_dims(x.data, axis))
            return MockArray(np.expand_dims(x, axis))
            
        @staticmethod
        def where(condition, x, y):
            if isinstance(condition, MockArray):
                condition = condition.data
            if isinstance(x, MockArray):
                x = x.data
            if isinstance(y, MockArray):
                y = y.data
            return MockArray(np.where(condition, x, y))
            
        @staticmethod
        def softmax(x, axis):
            if isinstance(x, MockArray):
                x_data = x.data
            else:
                x_data = x
            # Simple softmax implementation
            exp_x = np.exp(x_data - np.max(x_data, axis=axis, keepdims=True))
            return MockArray(exp_x / np.sum(exp_x, axis=axis, keepdims=True))
            
        @staticmethod
        def sigmoid(x):
            if isinstance(x, MockArray):
                x_data = x.data
            else:
                x_data = x
            return MockArray(1 / (1 + np.exp(-x_data)))
            
        @staticmethod
        def cos(x):
            if isinstance(x, MockArray):
                return MockArray(np.cos(x.data))
            return MockArray(np.cos(x))
            
        @staticmethod
        def sin(x):
            if isinstance(x, MockArray):
                return MockArray(np.sin(x.data))
            return MockArray(np.sin(x))
            
        @staticmethod
        def take(x, indices, axis):
            if isinstance(x, MockArray):
                x_data = x.data
            else:
                x_data = x
            if isinstance(indices, MockArray):
                indices_data = indices.data
            else:
                indices_data = indices
            return MockArray(np.take(x_data, indices_data, axis=axis))
            
        @staticmethod
        def stack(arrays, axis):
            processed_arrays = []
            for arr in arrays:
                if isinstance(arr, MockArray):
                    processed_arrays.append(arr.data)
                else:
                    processed_arrays.append(arr)
            return MockArray(np.stack(processed_arrays, axis=axis))
            
        @staticmethod
        def full_like(x, fill_value):
            if isinstance(x, MockArray):
                return MockArray(np.full_like(x.data, fill_value))
            return MockArray(np.full_like(x, fill_value))
        
    # Create and install mock MLX
    mock_mlx = MagicMock()
    mock_mlx.core = MockMX()
    mock_mlx.nn = MagicMock()
    sys.modules['mlx'] = mock_mlx
    sys.modules['mlx.core'] = mock_mlx.core
    sys.modules['mlx.nn'] = mock_mlx.nn
    
    # Set flag for using the mock
    HAS_MLX = True
    USING_MOCK_MLX = True
else:
    USING_MOCK_MLX = False

# Skip tests only if we couldn't create either real or mock MLX
pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available and couldn't create mock")


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


def test_mlx_transformer_layer_load_params_missing():
    """Test loading parameters with missing keys."""
    from csm.mlx_accel.components.transformer import MLXTransformerLayer
    
    # Create a layer
    layer = MLXTransformerLayer(
        hidden_size=32,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=64,
        layer_idx=0
    )
    
    # Create an incomplete parameter dictionary
    params_dict = {
        # Only provide 3 of the required parameters
        "layers.0.sa_norm.scale": mx.ones((32,)),
        "layers.0.attn.q_proj.weight": mx.ones((32, 32)),
        "layers.0.attn.k_proj.weight": mx.ones((16, 32))
    }
    
    # Load parameters and check warning is printed
    with patch('builtins.print') as mock_print:
        layer.load_params(params_dict, prefix="layers.0")
        # Should have printed warnings
        assert mock_print.call_count >= 1
        
    # Check params_loaded is False since not enough parameters were loaded
    assert layer.params_loaded is False
    
    # Check that the provided parameters were loaded
    assert layer.input_layernorm_weight is not None
    assert layer.q_proj_weight is not None
    assert layer.k_proj_weight is not None
    
    # But other parameters are still None
    assert layer.v_proj_weight is None
    assert layer.o_proj_weight is None


def test_mlx_transformer_layer_forward_missing_params():
    """Test forward pass with missing parameters."""
    from csm.mlx_accel.components.transformer import MLXTransformerLayer
    
    # Create a layer
    layer = MLXTransformerLayer(
        hidden_size=32,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=64,
        layer_idx=0
    )
    
    # Create input tensor
    hidden_states = mx.ones((2, 3, 32))  # batch_size=2, seq_len=3, hidden_size=32
    
    # Try to run forward without loading parameters
    with pytest.raises(ValueError) as excinfo:
        layer.forward(hidden_states)
    
    # Check error message
    assert "parameters not loaded" in str(excinfo.value)


def test_mlx_transformer_layer_feedforward():
    """Test feedforward network in transformer layer."""
    from csm.mlx_accel.components.transformer import MLXTransformerLayer
    
    # Create a layer
    hidden_size = 32
    intermediate_size = 64
    layer = MLXTransformerLayer(
        hidden_size=hidden_size,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=intermediate_size,
        layer_idx=0
    )
    
    # Set up weights for testing
    layer.gate_proj_weight = mx.ones((intermediate_size, hidden_size))
    layer.up_proj_weight = mx.ones((intermediate_size, hidden_size))
    layer.down_proj_weight = mx.ones((hidden_size, intermediate_size))
    
    # Create input tensor
    hidden_states = mx.ones((2, 3, hidden_size))  # batch_size=2, seq_len=3, hidden_size=32
    
    # Run feedforward
    output = layer._feedforward(hidden_states)
    
    # Check output shape
    assert output.shape == hidden_states.shape
    
    # Test with missing weights
    layer.gate_proj_weight = None
    with pytest.raises(ValueError) as excinfo:
        layer._feedforward(hidden_states)
    assert "Gate projection weight is not loaded" in str(excinfo.value)
    
    # Restore gate weight but remove up weight
    layer.gate_proj_weight = mx.ones((intermediate_size, hidden_size))
    layer.up_proj_weight = None
    with pytest.raises(ValueError) as excinfo:
        layer._feedforward(hidden_states)
    assert "Up projection weight is not loaded" in str(excinfo.value)
    
    # Restore up weight but remove down weight
    layer.up_proj_weight = mx.ones((intermediate_size, hidden_size))
    layer.down_proj_weight = None
    with pytest.raises(ValueError) as excinfo:
        layer._feedforward(hidden_states)
    assert "Down projection weight is not loaded" in str(excinfo.value)


def test_mlx_transformer_layer_attention():
    """Test attention in transformer layer."""
    from csm.mlx_accel.components.transformer import MLXTransformerLayer
    
    # Create a layer
    hidden_size = 32
    num_heads = 4
    num_kv_heads = 2
    head_dim = hidden_size // num_heads
    layer = MLXTransformerLayer(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=64,
        layer_idx=0
    )
    
    # Set up weights for testing
    layer.q_proj_weight = mx.ones((hidden_size, hidden_size))
    layer.k_proj_weight = mx.ones((num_kv_heads * head_dim, hidden_size))
    layer.v_proj_weight = mx.ones((num_kv_heads * head_dim, hidden_size))
    layer.o_proj_weight = mx.ones((hidden_size, hidden_size))
    
    # Create input tensor
    hidden_states = mx.ones((2, 3, hidden_size))  # batch_size=2, seq_len=3, hidden_size=32
    
    # Run attention without any optional parameters
    output = layer._attention(hidden_states)
    
    # Check output shape
    assert output.shape == hidden_states.shape
    
    # Create attention mask and run with mask
    attention_mask = mx.ones((2, 3, 3))  # batch_size=2, seq_len=3, seq_len=3
    output_with_mask = layer._attention(hidden_states, attention_mask=attention_mask)
    
    # Check output shape
    assert output_with_mask.shape == hidden_states.shape
    
    # Test with missing weights
    layer.q_proj_weight = None
    with pytest.raises(ValueError) as excinfo:
        layer._attention(hidden_states)
    assert "Query projection weight is not loaded" in str(excinfo.value)
    
    # Restore q weight but remove k weight
    layer.q_proj_weight = mx.ones((hidden_size, hidden_size))
    layer.k_proj_weight = None
    with pytest.raises(ValueError) as excinfo:
        layer._attention(hidden_states)
    assert "Key projection weight is not loaded" in str(excinfo.value)
    
    # Restore k weight but remove v weight
    layer.k_proj_weight = mx.ones((num_kv_heads * head_dim, hidden_size))
    layer.v_proj_weight = None
    with pytest.raises(ValueError) as excinfo:
        layer._attention(hidden_states)
    assert "Value projection weight is not loaded" in str(excinfo.value)
    
    # Restore v weight but remove o weight
    layer.v_proj_weight = mx.ones((num_kv_heads * head_dim, hidden_size))
    layer.o_proj_weight = None
    with pytest.raises(ValueError) as excinfo:
        layer._attention(hidden_states)
    assert "Output projection weight is not loaded" in str(excinfo.value)


def test_mlx_transformer_layer_apply_rotary_pos_emb():
    """Test applying rotary position embeddings."""
    from csm.mlx_accel.components.transformer import MLXTransformerLayer
    
    # Create a layer
    hidden_size = 32
    num_heads = 4
    head_dim = hidden_size // num_heads
    layer = MLXTransformerLayer(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=2,
        intermediate_size=64,
        layer_idx=0
    )
    
    # Create state tensor
    batch_size = 2
    seq_len = 3
    states = mx.ones((batch_size, seq_len, num_heads, head_dim))
    
    # Create RoPE cos and sin tensors
    cos = mx.ones((batch_size, seq_len, 1, head_dim // 2))
    sin = mx.ones((batch_size, seq_len, 1, head_dim // 2))
    
    # Apply rotary embeddings
    output = layer._apply_rotary_pos_emb(states, cos, sin)
    
    # Check output shape
    assert output.shape == states.shape
    
    # Test with differently shaped cos/sin
    cos_alt = mx.ones((batch_size, seq_len, head_dim // 2))
    sin_alt = mx.ones((batch_size, seq_len, head_dim // 2))
    
    # This should automatically reshape the cos/sin tensors
    output_alt = layer._apply_rotary_pos_emb(states, cos_alt, sin_alt)
    
    # Check output shape
    assert output_alt.shape == states.shape


def test_mlx_transformer_layer_forward():
    """Test complete forward pass through transformer layer."""
    from csm.mlx_accel.components.transformer import MLXTransformerLayer
    
    # Create a layer
    hidden_size = 32
    num_heads = 4
    num_kv_heads = 2
    head_dim = hidden_size // num_heads
    intermediate_size = 64
    layer = MLXTransformerLayer(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=intermediate_size,
        layer_idx=0
    )
    
    # Set all required weights for full forward pass
    layer.input_layernorm_weight = mx.ones((hidden_size,))
    layer.q_proj_weight = mx.ones((hidden_size, hidden_size))
    layer.k_proj_weight = mx.ones((num_kv_heads * head_dim, hidden_size))
    layer.v_proj_weight = mx.ones((num_kv_heads * head_dim, hidden_size))
    layer.o_proj_weight = mx.ones((hidden_size, hidden_size))
    layer.post_attention_layernorm_weight = mx.ones((hidden_size,))
    layer.gate_proj_weight = mx.ones((intermediate_size, hidden_size))
    layer.up_proj_weight = mx.ones((intermediate_size, hidden_size))
    layer.down_proj_weight = mx.ones((hidden_size, intermediate_size))
    
    # Mark parameters as loaded
    layer.params_loaded = True
    
    # Create input tensor
    batch_size = 2
    seq_len = 3
    hidden_states = mx.ones((batch_size, seq_len, hidden_size))
    
    # Run forward pass
    output = layer.forward(hidden_states)
    
    # Check output shape
    assert output.shape == hidden_states.shape
    
    # Test with attention mask
    attention_mask = mx.ones((batch_size, seq_len, seq_len))
    output_with_mask = layer.forward(hidden_states, attention_mask=attention_mask)
    
    # Check output shape
    assert output_with_mask.shape == hidden_states.shape


def test_mlx_transformer_forward():
    """Test forward pass through entire transformer model."""
    from csm.mlx_accel.components.transformer import MLXTransformer
    
    # Create config
    config = MockConfig(
        hidden_size=32,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64
    )
    
    # Create transformer
    transformer = MLXTransformer(config)
    
    # Load parameters for each layer
    for layer_idx, layer in enumerate(transformer.layers):
        # Set up required weights for layer
        head_dim = config.hidden_size // config.num_attention_heads
        hidden_size = config.hidden_size
        num_kv_heads = config.num_key_value_heads
        intermediate_size = config.intermediate_size
        
        layer.input_layernorm_weight = mx.ones((hidden_size,))
        layer.q_proj_weight = mx.ones((hidden_size, hidden_size))
        layer.k_proj_weight = mx.ones((num_kv_heads * head_dim, hidden_size))
        layer.v_proj_weight = mx.ones((num_kv_heads * head_dim, hidden_size))
        layer.o_proj_weight = mx.ones((hidden_size, hidden_size))
        layer.post_attention_layernorm_weight = mx.ones((hidden_size,))
        layer.gate_proj_weight = mx.ones((intermediate_size, hidden_size))
        layer.up_proj_weight = mx.ones((intermediate_size, hidden_size))
        layer.down_proj_weight = mx.ones((hidden_size, intermediate_size))
        
        # Mark parameters as loaded
        layer.params_loaded = True
    
    # Set final layernorm
    transformer.final_layernorm_weight = mx.ones((config.hidden_size,))
    
    # Create input tensor
    batch_size = 2
    seq_len = 3
    hidden_states = mx.ones((batch_size, seq_len, config.hidden_size))
    
    # Run forward pass
    output = transformer.forward(hidden_states)
    
    # Check output shape
    assert output.shape == hidden_states.shape
    
    # Test without final layernorm
    transformer.final_layernorm_weight = None
    output_no_final_norm = transformer.forward(hidden_states)
    
    # Check output shape
    assert output_no_final_norm.shape == hidden_states.shape
    
    # Test with attention mask and position ids
    attention_mask = mx.ones((batch_size, seq_len, seq_len))
    position_ids = mx.arange(0, seq_len).reshape(1, -1).repeat(batch_size, 0)
    
    # Initialize RoPE embeddings
    transformer._init_rope_embeddings()
    
    # Run forward with attention mask and position ids
    output_with_mask_and_pos = transformer.forward(
        hidden_states, 
        attention_mask=attention_mask,
        position_ids=position_ids
    )
    
    # Check output shape
    assert output_with_mask_and_pos.shape == hidden_states.shape


def test_mlx_transformer_reset_caches():
    """Test resetting key-value caches."""
    from csm.mlx_accel.components.transformer import MLXTransformer
    
    # Create config
    config = MockConfig()
    
    # Create transformer
    transformer = MLXTransformer(config)
    
    # Set a mock cache
    transformer.past_key_values = [1, 2, 3]
    
    # Reset caches
    transformer.reset_caches()
    
    # Check that cache is reset
    assert transformer.past_key_values is None


def test_mlx_transformer_layer_attention_with_rope():
    """Test attention with rotary position embeddings."""
    from csm.mlx_accel.components.transformer import MLXTransformerLayer
    
    # Create a layer
    hidden_size = 32
    num_heads = 4
    num_kv_heads = 2
    head_dim = hidden_size // num_heads
    max_seq_len = 16
    layer = MLXTransformerLayer(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=64,
        layer_idx=0,
        max_seq_len=max_seq_len
    )
    
    # Set up weights for testing
    layer.q_proj_weight = mx.ones((hidden_size, hidden_size))
    layer.k_proj_weight = mx.ones((num_kv_heads * head_dim, hidden_size))
    layer.v_proj_weight = mx.ones((num_kv_heads * head_dim, hidden_size))
    layer.o_proj_weight = mx.ones((hidden_size, hidden_size))
    
    # Initialize RoPE embeddings
    # Create position indices
    position = mx.arange(0, max_seq_len)
    
    # Create frequencies
    theta = 10000.0
    freqs = 1.0 / (theta ** (mx.arange(0, head_dim // 2) / (head_dim // 2)))
    
    # Outer product of positions and frequencies
    t = mx.reshape(position, (-1, 1)) * mx.reshape(freqs, (1, -1))
    
    # Create sin and cos embeddings
    layer.cos_cached = mx.cos(t)
    layer.sin_cached = mx.sin(t)
    
    # Create input tensor
    batch_size = 2
    seq_len = 3
    hidden_states = mx.ones((batch_size, seq_len, hidden_size))
    
    # Create position ids
    position_ids = mx.arange(0, seq_len).reshape(1, -1).repeat(batch_size, 0)
    
    # Run attention with position ids
    output = layer._attention(hidden_states, position_ids=position_ids)
    
    # Check output shape
    assert output.shape == hidden_states.shape
    
    # Test with attention mask and position ids
    attention_mask = mx.ones((batch_size, seq_len, seq_len))
    output_with_mask_and_pos = layer._attention(
        hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids
    )
    
    # Check output shape
    assert output_with_mask_and_pos.shape == hidden_states.shape


def test_mlx_transformer_load_params_different_keys():
    """Test loading parameters with standard naming convention."""
    from csm.mlx_accel.components.transformer import MLXTransformer
    
    # Create config
    config = MockConfig(
        hidden_size=32,
        num_layers=1,
        num_attention_heads=4,
        intermediate_size=64
    )
    
    # Create transformer
    transformer = MLXTransformer(config)
    
    # Create mock parameters dict with standard Llama naming
    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = hidden_size // num_heads
    intermediate_size = config.intermediate_size
    
    params_dict = {
        # Layer 0 parameters with standard naming
        "layers.0.input_layernorm.weight": mx.ones((hidden_size,)),
        "layers.0.input_layernorm.bias": mx.ones((hidden_size,)),
        "layers.0.self_attn.q_proj.weight": mx.ones((hidden_size, hidden_size)),
        "layers.0.self_attn.q_proj.bias": mx.ones((hidden_size,)),
        "layers.0.self_attn.k_proj.weight": mx.ones((num_kv_heads * head_dim, hidden_size)),
        "layers.0.self_attn.k_proj.bias": mx.ones((num_kv_heads * head_dim,)),
        "layers.0.self_attn.v_proj.weight": mx.ones((num_kv_heads * head_dim, hidden_size)),
        "layers.0.self_attn.v_proj.bias": mx.ones((num_kv_heads * head_dim,)),
        "layers.0.self_attn.o_proj.weight": mx.ones((hidden_size, hidden_size)),
        "layers.0.self_attn.o_proj.bias": mx.ones((hidden_size,)),
        "layers.0.post_attention_layernorm.weight": mx.ones((hidden_size,)),
        "layers.0.post_attention_layernorm.bias": mx.ones((hidden_size,)),
        "layers.0.mlp.gate_proj.weight": mx.ones((intermediate_size, hidden_size)),
        "layers.0.mlp.gate_proj.bias": mx.ones((intermediate_size,)),
        "layers.0.mlp.up_proj.weight": mx.ones((intermediate_size, hidden_size)),
        "layers.0.mlp.up_proj.bias": mx.ones((intermediate_size,)),
        "layers.0.mlp.down_proj.weight": mx.ones((hidden_size, intermediate_size)),
        "layers.0.mlp.down_proj.bias": mx.ones((hidden_size,)),
        
        # Final layernorm with standard naming
        "model.norm.weight": mx.ones((hidden_size,)),
        "model.norm.bias": mx.ones((hidden_size,)),
    }
    
    # Create a modified version of load_params that uses standard naming
    # We're patching it to test the handling of standard naming paths
    # without modifying the actual code (which expects CSM naming)
    
    # Patch the layer load_params to handle standard naming
    with patch('csm.mlx_accel.components.transformer.MLXTransformerLayer.load_params') as mock_layer_load:
        # Mock layer.load_params to always return success
        mock_layer_load.side_effect = lambda params_dict, prefix: setattr(mock_layer_load, 'called_with', (params_dict, prefix))
        
        # Load parameters
        transformer.load_params(params_dict)
        
        # Verify that the layer's load_params was called
        assert mock_layer_load.called
        
        # Check final layernorm is loaded
        assert transformer.final_layernorm_weight is not None
        
        # Test ability to handle different naming for final norm
        transformer.final_layernorm_weight = None
        params_dict_alt = {
            "norm.weight": mx.ones((hidden_size,)),
        }
        transformer.load_params(params_dict_alt)
        assert transformer.final_layernorm_weight is None  # Should still be None, not recognized


def test_mlx_transformer_layer_attention_mask_handling():
    """Test attention mask handling in different formats."""
    from csm.mlx_accel.components.transformer import MLXTransformerLayer
    
    # Create a layer
    hidden_size = 32
    num_heads = 4
    num_kv_heads = 2
    head_dim = hidden_size // num_heads
    layer = MLXTransformerLayer(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=64,
        layer_idx=0
    )
    
    # Set up weights for testing
    layer.q_proj_weight = mx.ones((hidden_size, hidden_size))
    layer.k_proj_weight = mx.ones((num_kv_heads * head_dim, hidden_size))
    layer.v_proj_weight = mx.ones((num_kv_heads * head_dim, hidden_size))
    layer.o_proj_weight = mx.ones((hidden_size, hidden_size))
    
    # Create input tensor
    batch_size = 2
    seq_len = 3
    hidden_states = mx.ones((batch_size, seq_len, hidden_size))
    
    # Test with 3D attention mask (batch_size, seq_len, seq_len)
    mask_3d = mx.ones((batch_size, seq_len, seq_len))
    output_3d = layer._attention(hidden_states, attention_mask=mask_3d)
    assert output_3d.shape == hidden_states.shape
    
    # Test with 4D attention mask (batch_size, 1, seq_len, seq_len)
    # This is already expanded and doesn't need reshape
    mask_4d = mx.expand_dims(mask_3d, axis=1)
    output_4d = layer._attention(hidden_states, attention_mask=mask_4d)
    assert output_4d.shape == hidden_states.shape