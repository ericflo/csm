"""
Tests for utility functions in components/utils.py.
"""

import pytest
import os
import time
import platform
from unittest.mock import patch, MagicMock

# Import module paths we'll need to patch
MODULE_PATH = "csm.mlx_accel.components.utils"

# Define test constants
MLX_AVAILABLE_PATH = f"{MODULE_PATH}.MLX_AVAILABLE"


def test_is_mlx_available():
    """Test is_mlx_available function."""
    # We'll directly import the function
    from csm.mlx_accel.components.utils import is_mlx_available
    
    # Just check that it returns a boolean
    result = is_mlx_available()
    assert isinstance(result, bool)


def test_check_device_compatibility():
    """Test check_device_compatibility function."""
    from csm.mlx_accel.components.utils import check_device_compatibility
    
    # Test with MLX not available
    with patch(MLX_AVAILABLE_PATH, False):
        result = check_device_compatibility()
        assert result is False
    
    # Test with MLX available but not on Apple Silicon
    with patch(MLX_AVAILABLE_PATH, True), \
         patch('platform.system', return_value="Darwin"), \
         patch('platform.machine', return_value="x86_64"):
        result = check_device_compatibility()
        assert result is False
    
    # Test with MLX available and on Apple Silicon
    with patch(MLX_AVAILABLE_PATH, True), \
         patch('platform.system', return_value="Darwin"), \
         patch('platform.machine', return_value="arm64"):
        result = check_device_compatibility()
        assert result is True
    
    # Test with exception during platform check
    with patch(MLX_AVAILABLE_PATH, True), \
         patch('platform.system', side_effect=Exception("Test error")):
        result = check_device_compatibility()
        assert result is False


def test_measure_time():
    """Test measure_time decorator."""
    from csm.mlx_accel.components.utils import measure_time
    
    # Define a test function
    def test_func():
        time.sleep(0.01)
        return "test result"
    
    # Apply the decorator
    with patch('builtins.print') as mock_print:
        decorated = measure_time(test_func)
        result = decorated()
        
        # Check the result
        assert result == "test result"
        
        # Check that print was called with timing info
        mock_print.assert_called_once()
        call_args = mock_print.call_args[0][0]
        assert "test_func took" in call_args
        assert "seconds" in call_args


def test_setup_mlx_debug():
    """Test setup_mlx_debug function."""
    from csm.mlx_accel.components.utils import setup_mlx_debug
    
    # Test with MLX not available
    with patch(MLX_AVAILABLE_PATH, False):
        # Should not raise exception
        setup_mlx_debug(True)
    
    # Test with MLX available, enabling debug
    with patch(MLX_AVAILABLE_PATH, True), \
         patch.dict(os.environ, {}, clear=True):
        setup_mlx_debug(True)
        assert os.environ.get("MLX_DEBUG") == "1"
    
    # Test with MLX available, not enabling debug
    with patch(MLX_AVAILABLE_PATH, True), \
         patch.dict(os.environ, {}, clear=True):
        setup_mlx_debug(False)
        assert "MLX_DEBUG" not in os.environ


def test_format_dtype():
    """Test format_dtype function."""
    from csm.mlx_accel.components.utils import format_dtype
    
    # Test with various input formats
    test_cases = [
        ("torch.float32", "float32"),
        ("numpy.float64", "float64"),
        ("<class 'torch.float16'>", "float16"),
        ("mlx.core.bfloat16", "bfloat16"),
        ("test.dtype'>", "dtype")
    ]
    
    for input_str, expected in test_cases:
        assert format_dtype(input_str) == expected


def test_get_shape_info():
    """Test get_shape_info function."""
    from csm.mlx_accel.components.utils import get_shape_info
    
    # Test with None
    assert get_shape_info(None) == "None"
    
    # Test with object that has shape and dtype
    class MockTensor:
        def __init__(self, shape, dtype):
            self.shape = shape
            self.dtype = dtype
    
    tensor = MockTensor((2, 3), "torch.float32")
    info = get_shape_info(tensor)
    assert "shape=(2, 3)" in info
    assert "dtype=float32" in info
    
    # Test with object that has no shape or dtype
    class EmptyObject:
        pass
    
    obj = EmptyObject()
    info = get_shape_info(obj)
    assert "no shape" in info
    assert "no dtype" in info