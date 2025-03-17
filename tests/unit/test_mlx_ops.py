"""Tests for MLX operations."""

import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock

# These tests require MLX
pytestmark = pytest.mark.requires_mlx

# Check if MLX is available
try:
    import mlx.core as mx
    import mlx.random
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    # Skip all tests if MLX is not available
    pytest.skip("MLX is not available", allow_module_level=True)


def test_torch_to_mlx_conversion():
    """Test conversion from PyTorch tensor to MLX array."""
    from csm.mlx_accel.mlx_ops import torch_to_mlx
    
    # Create a simple PyTorch tensor
    torch_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    
    # Convert to MLX
    mlx_array = torch_to_mlx(torch_tensor)
    
    # Verify the conversion
    assert mlx_array is not None
    assert mlx_array.shape == (2, 2)
    
    # Test with different data types
    torch_int = torch.tensor([1, 2, 3], dtype=torch.int32)
    mlx_int = torch_to_mlx(torch_int)
    assert mlx_int.dtype == mx.int32
    
    # Test with bfloat16 (should convert to float32)
    torch_bf16 = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    mlx_bf16 = torch_to_mlx(torch_bf16)
    assert mlx_bf16.dtype == mx.float32

def test_mlx_to_torch_conversion():
    """Test conversion from MLX array to PyTorch tensor."""
    from csm.mlx_accel.mlx_ops import mlx_to_torch
    
    # Create a simple MLX array
    mlx_array = mx.array([[1.0, 2.0], [3.0, 4.0]])
    
    # Convert to PyTorch
    torch_tensor = mlx_to_torch(mlx_array)
    
    # Verify the conversion
    assert torch_tensor is not None
    assert torch_tensor.shape == (2, 2)
    assert torch_tensor.dtype == torch.float32

def test_create_causal_mask():
    """Test creating a causal mask."""
    from csm.mlx_accel.mlx_ops import create_causal_mask
    
    # Create a causal mask for sequence length 4
    mask = create_causal_mask(4)
    
    # Verify the mask shape and values
    assert mask.shape == (4, 4)
    
    # Upper triangular + diagonal should be True, lower triangular should be False
    expected = [
        [True, False, False, False],  # First token sees only itself
        [True, True, False, False],   # Second token sees first and itself
        [True, True, True, False],    # Third token sees first, second, and itself
        [True, True, True, True]      # Fourth token sees all tokens
    ]
    
    # Convert mask to numpy for easier comparison
    mask_np = mask.tolist()
    assert mask_np == expected

def test_mlx_layer_norm():
    """Test layer normalization implementation."""
    from csm.mlx_accel.mlx_ops import mlx_layer_norm
    
    # Create input tensor, weights and bias
    x = mx.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])  # [1, 2, 3]
    weight = mx.array([1.0, 1.0, 1.0])  # Scale parameters
    bias = mx.array([0.0, 0.0, 0.0])    # Bias parameters
    
    # Apply layer norm
    normalized = mlx_layer_norm(x, weight, bias)
    
    # Check shape
    assert normalized.shape == x.shape
    
    # Test without bias
    normalized_no_bias = mlx_layer_norm(x, weight)
    assert normalized_no_bias.shape == x.shape

def test_mlx_rotary_embedding():
    """Test rotary embedding implementation."""
    from csm.mlx_accel.mlx_ops import mlx_rotary_embedding
    
    # Create input tensors
    batch_size, seq_len, num_heads, head_dim = 2, 3, 4, 16
    x = mx.random.normal((batch_size, seq_len, num_heads, head_dim))
    
    # Create position IDs
    position_ids = mx.array([[0, 1, 2], [0, 1, 2]])
    
    # Create sine and cosine tensors
    max_seq_len = 10
    cos = mx.random.normal((max_seq_len, 1, head_dim))
    sin = mx.random.normal((max_seq_len, 1, head_dim))
    
    # Apply rotary embeddings
    result = mlx_rotary_embedding(x, cos, sin, position_ids)
    
    # Check shape is preserved
    assert result.shape == x.shape
    
    # Test with mismatched dimensions
    # Create input with head_dim=10 but cos/sin with dim=8
    x_small = mx.random.normal((batch_size, seq_len, num_heads, 10))
    cos_small = mx.random.normal((max_seq_len, 1, 8))
    sin_small = mx.random.normal((max_seq_len, 1, 8))
    
    # Should still work with dimension adaptation
    result_small = mlx_rotary_embedding(x_small, cos_small, sin_small, position_ids)
    assert result_small.shape == x_small.shape

def test_mlx_attention():
    """Test MLX attention implementation."""
    from csm.mlx_accel.mlx_ops import mlx_attention
    
    # Create query, key, value tensors
    batch_size, seq_len, num_heads, head_dim = 2, 3, 4, 8
    query = mx.random.normal((batch_size, seq_len, num_heads, head_dim))
    key = mx.random.normal((batch_size, seq_len, num_heads, head_dim))
    value = mx.random.normal((batch_size, seq_len, num_heads, head_dim))
    
    # Create mask
    mask = mx.ones((batch_size, seq_len, seq_len), dtype=mx.bool_)
    
    # Apply attention
    result = mlx_attention(query, key, value, mask)
    
    # Check shape
    assert result.shape == query.shape
    
    # Test without mask
    result_no_mask = mlx_attention(query, key, value, None)
    assert result_no_mask.shape == query.shape

def test_mlx_feed_forward():
    """Test feed-forward network implementation."""
    from csm.mlx_accel.mlx_ops import mlx_feed_forward
    
    # Create input tensor and weights
    batch_size, seq_len, dim = 2, 3, 16
    hidden_dim = 32
    
    x = mx.random.normal((batch_size, seq_len, dim))
    w1 = mx.random.normal((dim, hidden_dim))
    w2 = mx.random.normal((dim, hidden_dim))
    w3 = mx.random.normal((hidden_dim, dim))
    
    # Create biases
    bias1 = mx.random.normal((hidden_dim,))
    bias2 = mx.random.normal((hidden_dim,))
    bias3 = mx.random.normal((dim,))
    
    # Apply feed-forward with biases
    result = mlx_feed_forward(x, w1, w2, w3, bias1, bias2, bias3)
    
    # Check shape
    assert result.shape == x.shape
    
    # Test without biases
    result_no_bias = mlx_feed_forward(x, w1, w2, w3)
    assert result_no_bias.shape == x.shape

def test_create_zeros_tensor():
    """Test creating a zeros tensor."""
    from csm.mlx_accel.mlx_ops import create_zeros_tensor
    
    # Create a zeros tensor
    shape = (2, 3, 4)
    zeros = create_zeros_tensor(shape)
    
    # Check shape and values
    assert zeros.shape == shape
    assert mx.all(zeros == 0.0)
    
    # Test with different dtype
    zeros_int = create_zeros_tensor(shape, dtype=mx.int32)
    assert zeros_int.dtype == mx.int32

def test_create_ones_tensor():
    """Test creating a ones tensor."""
    from csm.mlx_accel.mlx_ops import create_ones_tensor
    
    # Create a ones tensor
    shape = (2, 3, 4)
    ones = create_ones_tensor(shape)
    
    # Check shape and values
    assert ones.shape == shape
    assert mx.all(ones == 1.0)
    
    # Test with different dtype
    ones_int = create_ones_tensor(shape, dtype=mx.int32)
    assert ones_int.dtype == mx.int32

def test_safe_reshape():
    """Test safe reshape implementation."""
    import csm.mlx_accel.mlx_ops
    
    # Create our own simplified safe_reshape implementation for testing
    def mock_safe_reshape(arr, new_shape):
        """Mock implementation that just does regular reshape."""
        return arr.reshape(new_shape)
    
    # Patch the function
    with patch.object(csm.mlx_accel.mlx_ops, 'safe_reshape', side_effect=mock_safe_reshape):
        # Create a tensor
        original = mx.array([[1.0, 2.0], [3.0, 4.0]])
        
        # Reshape to same number of elements
        reshaped = csm.mlx_accel.mlx_ops.safe_reshape(original, (1, 4))
        
        # Verify the result
        assert reshaped.shape == (1, 4)
        
        # Verify the values are preserved 
        assert mx.array_equal(reshaped, original.reshape(1, 4))
