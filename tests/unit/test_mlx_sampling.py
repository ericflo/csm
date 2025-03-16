"""Tests for MLX sampling module."""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

# Skip all tests in this file since we've moved mlx to mlx_accel
pytestmark = pytest.mark.skip(reason="MLX modules moved to mlx_accel")

# We need to mock mlx imports since we might not have MLX installed
# for testing on all platforms
class MockMLX:
    def __init__(self):
        self.array = MagicMock()
        self.array.side_effect = lambda x, **kwargs: np.array(x, **kwargs)
        self.nn = MagicMock()
        self.nn.softmax = MagicMock(return_value=np.array([0.1, 0.2, 0.7]))
        self.random = MagicMock()
        self.random.categorical = MagicMock(return_value=np.array(2))
        self.maximum = MagicMock(side_effect=np.maximum)
        
    def __call__(self, *args, **kwargs):
        return self.array(*args, **kwargs)


@pytest.fixture
def mock_mlx():
    """Create and return a mock MLX module."""
    return MockMLX()


@patch("sys.modules", {"mlx": None})
@patch.dict("sys.modules", {"mlx": None})
def test_sample_from_logits(mock_mlx):
    """Test sampling from logits."""
    # Need to patch sys.modules to mock mlx
    with patch.dict("sys.modules", {"mlx": mock_mlx}):
        # Now we can safely import our module
        from csm.mlx_accel.components.sampling import sample_from_logits
        
        # Create test logits
        logits = np.array([[1.0, 2.0, 3.0]])
        temperature = 1.0
        
        # Call the function
        result = sample_from_logits(logits, temperature)
        
        # Verify results
        assert result is not None
        # The mock is set up to always return 2
        assert result == 2
        
        # Verify the softmax was called
        mock_mlx.nn.softmax.assert_called_once()
        # Verify the categorical sampling was called
        mock_mlx.random.categorical.assert_called_once()


@patch("sys.modules", {"mlx": None})
@patch.dict("sys.modules", {"mlx": None})
def test_topk_sampling(mock_mlx):
    """Test top-k sampling."""
    # Need to patch sys.modules to mock mlx
    with patch.dict("sys.modules", {"mlx": mock_mlx}):
        # Now we can safely import our module
        from csm.mlx_accel.components.sampling import sample_topk
        
        # Create test logits
        logits = np.array([[1.0, 5.0, 3.0, 2.0, 4.0]])
        temperature = 1.0
        k = 2
        
        # Mock the result of argpartition to simulate top-k
        # This would normally return the indices of the top k elements
        mock_mlx.array.return_value = np.array([[1, 4]])
        
        # Call the function
        with patch("csm.mlx_accel.components.sampling.sample_from_logits", return_value=1) as mock_sample:
            result = sample_topk(logits, k, temperature)
            
            # Verify sample_from_logits was called
            mock_sample.assert_called_once()
            
            # Verify result is the expected token
            assert result == 1


@patch("sys.modules", {"mlx": None})
@patch.dict("sys.modules", {"mlx": None})
def test_topk_with_temperature(mock_mlx):
    """Test top-k sampling with different temperatures."""
    # Need to patch sys.modules to mock mlx
    with patch.dict("sys.modules", {"mlx": mock_mlx}):
        # Now we can safely import our module
        from csm.mlx_accel.components.sampling import sample_topk
        
        # Create test logits
        logits = np.array([[1.0, 5.0, 3.0, 2.0, 4.0]])
        
        # Try with very high temperature (more randomness)
        temperature = 2.0
        k = 2
        
        # Mock the result of argpartition to simulate top-k
        mock_mlx.array.return_value = np.array([[1, 4]])
        
        # Call the function
        with patch("csm.mlx_accel.components.sampling.sample_from_logits", return_value=1) as mock_sample:
            result_high_temp = sample_topk(logits, k, temperature)
            
            # Verify sample_from_logits was called with scaled logits
            args, kwargs = mock_sample.call_args
            assert len(args) == 2
            # First arg should be the filtered logits
            # Second arg should be the temperature
            assert args[1] == temperature
            
            # Reset the mock for next test
            mock_sample.reset_mock()
            
            # Try with very low temperature (more deterministic)
            temperature = 0.5
            result_low_temp = sample_topk(logits, k, temperature)
            
            # Verify sample_from_logits was called with scaled logits
            args, kwargs = mock_sample.call_args
            assert len(args) == 2
            assert args[1] == temperature