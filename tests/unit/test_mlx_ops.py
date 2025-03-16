"""Tests for MLX operations."""

import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock

# Skip tests only if MLX is not available
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    
pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


def test_torch_to_mlx_conversion():
    """Test conversion from PyTorch tensor to MLX array."""
    from csm.mlx_accel.mlx_ops import torch_to_mlx
    
    # Create a simple PyTorch tensor
    torch_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    
    # Convert to MLX
    mlx_array = torch_to_mlx(torch_tensor)
    
    # Verify the conversion
    assert mlx_array is not None
    assert isinstance(mlx_array, mx.array)
    assert mlx_array.shape == (2, 2)
    
    # Convert back to numpy to check values
    np_array = mlx_array.tolist()
    assert np.allclose(np_array, torch_tensor.numpy())


def test_mlx_to_torch_conversion():
    """Test conversion from MLX array to PyTorch tensor."""
    from csm.mlx_accel.mlx_ops import mlx_to_torch
    
    # Create a simple MLX array
    mlx_array = mx.array([[1.0, 2.0], [3.0, 4.0]])
    
    # Convert to PyTorch
    torch_tensor = mlx_to_torch(mlx_array)
    
    # Verify the conversion
    assert torch_tensor is not None
    assert isinstance(torch_tensor, torch.Tensor)
    assert torch_tensor.shape == (2, 2)
    
    # Convert to numpy to check values
    np_array = torch_tensor.numpy()
    assert np.allclose(np_array, mlx_array.tolist())


def test_create_causal_mask():
    """Test creation of causal mask for attention."""
    from csm.mlx_accel.mlx_ops import create_causal_mask
    
    # Create a small causal mask
    seq_len = 4
    mask = create_causal_mask(seq_len)
    
    # Verify the mask
    assert mask is not None
    assert isinstance(mask, mx.array)
    assert mask.shape == (seq_len, seq_len)
    
    # Convert to numpy for checking values
    mask_np = mask.tolist()
    
    # Check that the mask is causal (lower triangular)
    for i in range(seq_len):
        for j in range(seq_len):
            if i >= j:  # Causal positions
                assert mask_np[i][j] == 1.0, f"Expected 1.0 at position ({i}, {j})"
            else:  # Non-causal positions
                assert mask_np[i][j] == 0.0, f"Expected 0.0 at position ({i}, {j})"