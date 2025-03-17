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
