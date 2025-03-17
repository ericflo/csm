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
    
    # Initialize cache
    kv_cache = MLXKVCache(
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim
    )
    
    # Verify attributes
    assert kv_cache.batch_size == batch_size
    assert kv_cache.max_seq_len == max_seq_len
    assert kv_cache.num_layers == num_layers
    assert kv_cache.num_heads == num_heads
    assert kv_cache.num_kv_heads == num_kv_heads
    assert kv_cache.head_dim == head_dim
    
    # Verify cache initialization
    assert len(kv_cache.k_cache) == num_layers
    assert len(kv_cache.v_cache) == num_layers
    
    # Check shape of each layer's cache
    for layer_idx in range(num_layers):
        assert kv_cache.k_cache[layer_idx].shape == (batch_size, max_seq_len, num_kv_heads, head_dim)
        assert kv_cache.v_cache[layer_idx].shape == (batch_size, max_seq_len, num_kv_heads, head_dim)
    
    # Check current sequence length
    assert kv_cache.current_seq_len.shape == (batch_size,)
    assert mx.all(kv_cache.current_seq_len == 0)

def test_mlx_kv_cache_update():
    """Test updating the KV cache."""
    from csm.mlx_accel.mlx_kvcache import MLXKVCache
    
    # Define parameters
    batch_size = 2
    max_seq_len = 16
    num_layers = 2
    num_heads = 4
    num_kv_heads = 4
    head_dim = 8
    
    # Initialize cache
    kv_cache = MLXKVCache(
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim
    )
    
    # Create test keys and values to store
    seq_len = 3
    layer_idx = 0
    key = mx.random.normal((batch_size, seq_len, num_kv_heads, head_dim))
    value = mx.random.normal((batch_size, seq_len, num_kv_heads, head_dim))
    
    # Update the cache
    kv_cache.update(layer_idx, key, value)
    
    # Verify current sequence length was updated
    assert mx.all(kv_cache.current_seq_len == seq_len)
    
    # Update with specific positions
    positions = mx.array([[5, 6, 7], [8, 9, 10]])  # Custom positions
    key2 = mx.random.normal((batch_size, seq_len, num_kv_heads, head_dim))
    value2 = mx.random.normal((batch_size, seq_len, num_kv_heads, head_dim))
    
    # Update another layer with explicit positions
    layer_idx2 = 1
    kv_cache.update(layer_idx2, key2, value2, positions)
    
    # Current sequence length shouldn't change since we used explicit positions
    assert mx.all(kv_cache.current_seq_len == seq_len)

def test_mlx_kv_cache_get():
    """Test retrieving values from the KV cache."""
    from csm.mlx_accel.mlx_kvcache import MLXKVCache
    
    # Define parameters
    batch_size = 2
    max_seq_len = 16
    num_layers = 1
    num_heads = 4
    num_kv_heads = 4
    head_dim = 8
    
    # Initialize cache
    kv_cache = MLXKVCache(
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim
    )
    
    # Create test keys and values to store
    seq_len = 2
    layer_idx = 0
    
    # Create unique identifiable values
    key = mx.ones((batch_size, seq_len, num_kv_heads, head_dim))
    value = mx.full((batch_size, seq_len, num_kv_heads, head_dim), 2.0)  # All 2's
    
    # Use explicit positions
    positions = mx.array([[0, 1], [2, 3]])
    kv_cache.update(layer_idx, key, value, positions)
    
    # Retrieve the values at the same positions
    retrieved_key, retrieved_value = kv_cache.get(layer_idx, positions)
    
    # Verify shapes
    assert retrieved_key.shape == (batch_size, seq_len, num_kv_heads, head_dim)
    assert retrieved_value.shape == (batch_size, seq_len, num_kv_heads, head_dim)
    
    # The first check would be very expensive to implement with element-wise
    # comparisons, so we'll check some specific values
    assert mx.all(retrieved_key[0, 0] == 1.0)  # All ones
    assert mx.all(retrieved_value[0, 0] == 2.0)  # All twos

def test_mlx_kv_cache_reset():
    """Test resetting the KV cache."""
    from csm.mlx_accel.mlx_kvcache import MLXKVCache
    
    # Define parameters
    batch_size = 2
    max_seq_len = 16
    num_layers = 2
    num_heads = 4
    num_kv_heads = 4
    head_dim = 8
    
    # Initialize cache
    kv_cache = MLXKVCache(
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim
    )
    
    # Create test keys and values to store
    seq_len = 3
    layer_idx = 0
    key = mx.ones((batch_size, seq_len, num_kv_heads, head_dim))
    value = mx.ones((batch_size, seq_len, num_kv_heads, head_dim))
    
    # Update the cache
    kv_cache.update(layer_idx, key, value)
    
    # Verify current sequence length was updated
    assert mx.all(kv_cache.current_seq_len == seq_len)
    
    # Reset the cache
    kv_cache.reset()
    
    # Verify current sequence length was reset
    assert mx.all(kv_cache.current_seq_len == 0)
    
    # Verify cache values were reset
    for l in range(num_layers):
        assert mx.all(kv_cache.k_cache[l] == 0)
        assert mx.all(kv_cache.v_cache[l] == 0)

def test_mlx_kv_cache_with_different_num_kv_heads():
    """Test MLXKVCache with different number of key-value heads (multi-query attention)."""
    from csm.mlx_accel.mlx_kvcache import MLXKVCache
    
    # Define parameters with fewer KV heads than query heads
    batch_size = 2
    max_seq_len = 16
    num_layers = 1
    num_heads = 8  # 8 query heads
    num_kv_heads = 2  # Only 2 KV heads (multi-query attention)
    head_dim = 8
    
    # Initialize cache
    kv_cache = MLXKVCache(
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim
    )
    
    # Create test keys and values to store
    seq_len = 2
    layer_idx = 0
    key = mx.ones((batch_size, seq_len, num_kv_heads, head_dim))  # Note: Only num_kv_heads
    value = mx.ones((batch_size, seq_len, num_kv_heads, head_dim))
    
    # Update the cache
    kv_cache.update(layer_idx, key, value)
    
    # Check that the cache shapes use num_kv_heads, not num_heads
    assert kv_cache.k_cache[layer_idx].shape == (batch_size, max_seq_len, num_kv_heads, head_dim)
    assert kv_cache.v_cache[layer_idx].shape == (batch_size, max_seq_len, num_kv_heads, head_dim)
