"""Tests for MLX KVCache module."""

import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock

# These tests require MLX
pytestmark = pytest.mark.requires_mlx

# Check if MLX is available
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    # Skip all tests if MLX is not available
    pytest.skip("MLX is not available", allow_module_level=True)


def test_mlx_kv_cache_initialization():
    """Test MLXKVCache initialization with default parameters."""
    from csm.mlx_accel.mlx_kvcache import MLXKVCache
    
    # Define parameters
    batch_size = 2
    max_seq_len = 16
    num_layers = 3
    num_heads = 4
    num_kv_heads = 4
    head_dim = 8
