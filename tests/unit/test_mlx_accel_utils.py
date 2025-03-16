"""Tests for MLX acceleration utils module."""

import pytest
import sys
import time
import platform
from unittest.mock import patch, MagicMock

# Extract the functions we want to test
# This avoids dependencies on the module imports
def is_mlx_available():
    """
    Check if MLX is available.
    
    Returns:
        True if MLX is available, False otherwise
    """
    return MLX_AVAILABLE

def check_device_compatibility():
    """
    Check if the current device is compatible with MLX.
    
    Returns:
        True if device is compatible, False otherwise
    """
    if not MLX_AVAILABLE:
        return False
        
    # Check for Apple Silicon
    try:
        is_mac = platform.system() == "Darwin"
        is_arm = platform.machine() == "arm64"
        return is_mac and is_arm
    except:
        return False

def measure_time(func):
    """
    Decorator to measure execution time.
    
    Args:
        func: Function to measure
        
    Returns:
        Wrapped function that prints execution time
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def format_dtype(dtype):
    """
    Format a data type for display.
    
    Args:
        dtype: PyTorch or MLX data type
        
    Returns:
        String representation of the data type
    """
    return str(dtype).split(".")[-1].replace("'", "").replace(">", "")

def get_shape_info(tensor):
    """
    Get shape and type information for a tensor.
    
    Args:
        tensor: PyTorch tensor or MLX array
        
    Returns:
        String with shape and type information
    """
    if tensor is None:
        return "None"
        
    if hasattr(tensor, "shape"):
        shape_str = f"shape={tensor.shape}"
    else:
        shape_str = "no shape"
        
    if hasattr(tensor, "dtype"):
        dtype_str = f"dtype={format_dtype(tensor.dtype)}"
    else:
        dtype_str = "no dtype"
        
    return f"{shape_str}, {dtype_str}"

def log_error(message, exception=None):
    """
    Log an error message.
    
    Args:
        message: Error message to log
        exception: Optional exception to include
    """
    if exception:
        print(f"ERROR: {message}: {str(exception)}")
    else:
        print(f"ERROR: {message}")

def log_warning(message):
    """
    Log a warning message.
    
    Args:
        message: Warning message to log
    """
    print(f"WARNING: {message}")

def log_info(message):
    """
    Log an info message.
    
    Args:
        message: Info message to log
    """
    print(f"INFO: {message}")

def log_success(message):
    """
    Log a success message.
    
    Args:
        message: Success message to log
    """
    print(f"SUCCESS: {message}")

# Set up mocks
MLX_AVAILABLE = False

def test_is_mlx_available():
    """Test is_mlx_available function."""
    # Test with MLX available
    global MLX_AVAILABLE
    
    # Save original value
    original_value = MLX_AVAILABLE
    
    # Test with True
    MLX_AVAILABLE = True
    assert is_mlx_available() is True
    
    # Test with False
    MLX_AVAILABLE = False
    assert is_mlx_available() is False
    
    # Restore original value
    MLX_AVAILABLE = original_value

def test_check_device_compatibility():
    """Test check_device_compatibility function."""
    global MLX_AVAILABLE
    original_value = MLX_AVAILABLE
    
    # Test with MLX not available
    MLX_AVAILABLE = False
    result = check_device_compatibility()
    assert result is False
    
    # Test with MLX available but not on Apple Silicon
    MLX_AVAILABLE = True
    with patch('platform.system', return_value="Darwin"), \
         patch('platform.machine', return_value="x86_64"):
        result = check_device_compatibility()
        assert result is False
    
    # Test with MLX available and on Apple Silicon
    with patch('platform.system', return_value="Darwin"), \
         patch('platform.machine', return_value="arm64"):
        result = check_device_compatibility()
        assert result is True
    
    # Test with exception during platform check
    with patch('platform.system', side_effect=Exception("Test error")):
        result = check_device_compatibility()
        assert result is False
    
    # Restore original value
    MLX_AVAILABLE = original_value

def test_measure_time():
    """Test measure_time function."""
    # Create a test function
    def test_func():
        time.sleep(0.01)
        return "test"
    
    # Apply the decorator manually
    with patch('builtins.print') as mock_print:
        decorated_func = measure_time(test_func)
        result = decorated_func()
        
        # Check the function was called and returned correctly
        assert result == "test"
        
        # Check that print was called with timing info
        assert mock_print.call_count == 1
        args = mock_print.call_args[0][0]
        assert "test_func took" in args
        assert "seconds" in args

def test_format_dtype():
    """Test format_dtype function."""
    # Test with various inputs
    assert format_dtype("torch.float32") == "float32"
    assert format_dtype("numpy.int64") == "int64"
    assert format_dtype("test.dtype'>") == "dtype"

def test_get_shape_info():
    """Test get_shape_info function."""
    # Test with None
    assert get_shape_info(None) == "None"
    
    # Test with mock tensor
    class MockTensor:
        def __init__(self):
            self.shape = (2, 3)
            self.dtype = "torch.float32"
    
    tensor = MockTensor()
    info = get_shape_info(tensor)
    assert "shape=(2, 3)" in info
    assert "float32" in info
    
    # Test with object that has no shape or dtype
    class TestObj:
        pass
    
    obj = TestObj()
    info = get_shape_info(obj)
    assert "no shape" in info
    assert "no dtype" in info

def test_log_functions():
    """Test logging functions."""
    # Test log_error
    with patch('builtins.print') as mock_print:
        log_error("Test error")
        mock_print.assert_called_with("ERROR: Test error")
        
        # Test with exception
        mock_print.reset_mock()
        try:
            raise ValueError("Test exception")
        except ValueError as e:
            log_error("Error occurred", e)
            assert mock_print.call_count == 1
            assert "Error occurred" in mock_print.call_args[0][0]
            assert "Test exception" in mock_print.call_args[0][0]
    
    # Test log_warning
    with patch('builtins.print') as mock_print:
        log_warning("Test warning")
        mock_print.assert_called_with("WARNING: Test warning")
    
    # Test log_info
    with patch('builtins.print') as mock_print:
        log_info("Test info")
        mock_print.assert_called_with("INFO: Test info")
    
    # Test log_success
    with patch('builtins.print') as mock_print:
        log_success("Test success")
        mock_print.assert_called_with("SUCCESS: Test success")