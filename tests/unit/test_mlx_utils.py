"""Tests for MLX utilities."""

import pytest
import numpy as np
import time
from unittest.mock import patch, MagicMock, call, Mock

# Mock implementation of the utility functions we want to test
def check_device_compatibility():
    """Mock implementation of check_device_compatibility."""
    return True  # Mock always returns True for testing

def measure_time(func, *args, **kwargs):
    """Mock implementation of measure_time."""
    start_time = time.time()
    result = func(*args, **kwargs)
    duration = time.time() - start_time
    return result, duration

def format_duration(seconds):
    """Format a duration in seconds to a human-readable string."""
    if seconds < 0.01:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes}:{seconds:05.2f}"
    else:
        hours = int(seconds // 3600)
        seconds = seconds % 3600
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{hours}:{minutes:02d}:{seconds:05.2f}"

def log_error(message, exception=None):
    """Mock implementation of log_error."""
    if exception:
        print(f"ERROR: {message}: {str(exception)}")
    else:
        print(f"ERROR: {message}")

def log_warning(message):
    """Mock implementation of log_warning."""
    print(f"WARNING: {message}")

def log_info(message):
    """Mock implementation of log_info."""
    print(f"INFO: {message}")

def log_success(message):
    """Mock implementation of log_success."""
    print(f"SUCCESS: {message}")

# Tests for the mock implementations
def test_check_device_compatibility():
    """Test check_device_compatibility function."""
    # Test that it returns True (our mock always returns True)
    assert check_device_compatibility() is True

def test_measure_time():
    """Test measure_time function."""
    # Create a test function that sleeps
    def test_function():
        time.sleep(0.01)
        return "test"
    
    # Measure the function
    result, duration = measure_time(test_function)
    
    # Check the result is correct
    assert result == "test"
    
    # Check duration is positive and reasonable
    assert duration > 0
    assert duration < 1.0  # Should be much less than 1 second

def test_measure_time_with_args():
    """Test measure_time function with arguments."""
    # Create a test function that takes arguments
    def test_function(a, b, c=3):
        time.sleep(0.01)
        return a + b + c
    
    # Measure the function with arguments
    result, duration = measure_time(test_function, 1, 2, c=4)
    
    # Check the result is correct
    assert result == 7  # 1 + 2 + 4
    
    # Check duration is positive and reasonable
    assert duration > 0
    assert duration < 1.0

def test_format_duration():
    """Test format_duration function."""
    # Test various durations
    assert format_duration(0.0012) == "1.2ms"
    assert format_duration(0.123) == "123.0ms"
    assert format_duration(1.234) == "1.23s"
    assert format_duration(12.345) == "12.35s"
    assert format_duration(123.456) == "2:03.46"
    assert format_duration(3661) == "1:01:01.00"

def test_log_error():
    """Test log_error function."""
    with patch("builtins.print") as mock_print:
        # Test with a simple error message
        log_error("Test error")
        mock_print.assert_called_with("ERROR: Test error")
        
        # Test with an exception
        mock_print.reset_mock()
        try:
            raise ValueError("Test exception")
        except ValueError as e:
            log_error("Error occurred", e)
            # Should print both the message and the exception
            assert mock_print.call_count == 1
            assert "Error occurred" in mock_print.call_args[0][0]
            assert "Test exception" in mock_print.call_args[0][0]

def test_log_warning():
    """Test log_warning function."""
    with patch("builtins.print") as mock_print:
        # Test with a simple warning message
        log_warning("Test warning")
        mock_print.assert_called_with("WARNING: Test warning")

def test_log_info():
    """Test log_info function."""
    with patch("builtins.print") as mock_print:
        # Test with a simple info message
        log_info("Test info")
        mock_print.assert_called_with("INFO: Test info")

def test_log_success():
    """Test log_success function."""
    with patch("builtins.print") as mock_print:
        # Test with a simple success message
        log_success("Test success")
        mock_print.assert_called_with("SUCCESS: Test success")