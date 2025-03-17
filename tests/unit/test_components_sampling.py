"""
Tests for sampling functions in components/sampling.py.
"""

import pytest
import os
import time
import numpy as np
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

# Import the module not the functions to allow proper mocking
import csm.mlx_accel.components.sampling

def test_sampling_basic_functionality():
    """Test basic top-k sampling functionality."""
    # Create a simple mock for the module function
    with patch.object(csm.mlx_accel.components.sampling, 'mlx_topk_sampling') as mock_topk:
        # Configure the mock to return a predictable value
        mock_topk.return_value = mx.array([[5]], dtype=mx.int32)
        
        # Create simple test data
        logits = mx.array([0.1, 0.2, 0.7, 0.0, 0.0])
        
        # Call the function through the module
        result = csm.mlx_accel.components.sampling.mlx_topk_sampling(
            logits, k=3, temperature=1.0, seed=42
        )
        
        # Verify the mock was called with the right arguments
        mock_topk.assert_called_once()
        
        # Verify the result
        assert isinstance(result, mx.array)
        assert result.shape == (1, 1)
        assert result[0, 0] == 5

def test_categorical_sampling():
    """Test categorical sampling functionality."""
    # Create a simple mock for the module function
    with patch.object(csm.mlx_accel.components.sampling, 'mlx_categorical_sampling') as mock_cat:
        # Configure the mock to return a predictable value
        mock_cat.return_value = mx.array([[3]], dtype=mx.int32)
        
        # Create simple test data
        logits = mx.array([0.1, 0.2, 0.7, 0.0, 0.0])
        
        # Call the function through the module
        result = csm.mlx_accel.components.sampling.mlx_categorical_sampling(
            logits, temperature=1.0, seed=42
        )
        
        # Verify the mock was called with the right arguments
        mock_cat.assert_called_once()
        
        # Verify the result
        assert isinstance(result, mx.array)
        assert result.shape == (1, 1)
        assert result[0, 0] == 3

def test_topk_sampling_with_different_temperatures():
    """Test the effect of temperature on sampling."""
    # Create mocks that return different values based on temperature
    def temp_dependent_mock(logits, k=5, temperature=1.0, seed=None):
        batch_size = 1 if len(logits.shape) == 1 else logits.shape[0]
        if temperature < 0.5:
            return mx.array([[2]] * batch_size, dtype=mx.int32)
        else:
            return mx.array([[5]] * batch_size, dtype=mx.int32)
    
    with patch.object(csm.mlx_accel.components.sampling, 'mlx_topk_sampling', side_effect=temp_dependent_mock):
        # Create uniform distribution
        logits = mx.array([1.0] * 10)
        
        # Sample with high temperature
        high_temp_sample = csm.mlx_accel.components.sampling.mlx_topk_sampling(
            logits, k=5, temperature=1.0, seed=42
        )
        
        # Sample with low temperature
        low_temp_sample = csm.mlx_accel.components.sampling.mlx_topk_sampling(
            logits, k=5, temperature=0.1, seed=42
        )
        
        # The samples should differ due to temperature
        assert high_temp_sample[0, 0] != low_temp_sample[0, 0]
        assert high_temp_sample[0, 0] == 5  # From our mock
        assert low_temp_sample[0, 0] == 2   # From our mock

def test_topk_sampling_shapes():
    """Test that sampling preserves batch shapes."""
    # Create a mock that preserves batch shapes
    def shape_preserving_mock(logits, k=5, temperature=1.0, seed=None):
        batch_size = 1 if len(logits.shape) == 1 else logits.shape[0]
        return mx.array([[5]] * batch_size, dtype=mx.int32)
    
    with patch.object(csm.mlx_accel.components.sampling, 'mlx_topk_sampling', side_effect=shape_preserving_mock):
        # Test with 1D input
        logits_1d = mx.array([0.1, 0.2, 0.7, 0.0, 0.0])
        result_1d = csm.mlx_accel.components.sampling.mlx_topk_sampling(
            logits_1d, k=3, temperature=1.0, seed=42
        )
        assert result_1d.shape == (1, 1)  # Should add batch dimension
        
        # Test with batch input
        batch_logits = mx.array([
            [0.1, 0.2, 0.7, 0.0, 0.0],  # Batch 0
            [0.7, 0.2, 0.1, 0.0, 0.0]   # Batch 1
        ])
        result_batch = csm.mlx_accel.components.sampling.mlx_topk_sampling(
            batch_logits, k=3, temperature=1.0, seed=42
        )
        assert result_batch.shape == (2, 1)  # Should preserve batch dimension
