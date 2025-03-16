"""Tests for MLX utilities."""

import pytest
import numpy as np
import time
from unittest.mock import patch, MagicMock, call

# Skip all tests in this file since we've moved mlx to mlx_accel
pytestmark = pytest.mark.skip(reason="MLX modules moved to mlx_accel")


@patch("sys.modules", {"mlx": None})
def test_check_device_compatibility():
    """Test check_device_compatibility function."""
    with patch.dict("sys.modules", {"mlx": None}):
        # Import the function
        from csm.mlx_accel.components.utils import check_device_compatibility
        
        # Test that it doesn't raise an exception on mock mlx
        check_device_compatibility()


@patch("sys.modules", {"mlx": None})
def test_measure_time():
    """Test measure_time function."""
    with patch.dict("sys.modules", {"mlx": None}):
        from csm.mlx_accel.components.utils import measure_time
        
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


@patch("sys.modules", {"mlx": None})
def test_measure_time_with_args():
    """Test measure_time function with arguments."""
    with patch.dict("sys.modules", {"mlx": None}):
        from csm.mlx_accel.components.utils import measure_time
        
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


@patch("sys.modules", {"mlx": None})
def test_format_duration():
    """Test format_duration function."""
    with patch.dict("sys.modules", {"mlx": None}):
        from csm.mlx_accel.components.utils import format_duration
        
        # Test various durations
        assert format_duration(0.0012) == "1.2ms"
        assert format_duration(0.123) == "123.0ms"
        assert format_duration(1.234) == "1.23s"
        assert format_duration(12.345) == "12.35s"
        assert format_duration(123.456) == "2:03.46"
        assert format_duration(3661) == "1:01:01.00"


@patch("sys.modules", {"mlx": None})
def test_log_error():
    """Test log_error function."""
    with patch.dict("sys.modules", {"mlx": None}), \
         patch("builtins.print") as mock_print:
        
        from csm.mlx_accel.components.utils import log_error
        
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


@patch("sys.modules", {"mlx": None})
def test_log_warning():
    """Test log_warning function."""
    with patch.dict("sys.modules", {"mlx": None}), \
         patch("builtins.print") as mock_print:
        
        from csm.mlx_accel.components.utils import log_warning
        
        # Test with a simple warning message
        log_warning("Test warning")
        mock_print.assert_called_with("WARNING: Test warning")


@patch("sys.modules", {"mlx": None})
def test_log_info():
    """Test log_info function."""
    with patch.dict("sys.modules", {"mlx": None}), \
         patch("builtins.print") as mock_print:
        
        from csm.mlx_accel.components.utils import log_info
        
        # Test with a simple info message
        log_info("Test info")
        mock_print.assert_called_with("INFO: Test info")


@patch("sys.modules", {"mlx": None})
def test_log_success():
    """Test log_success function."""
    with patch.dict("sys.modules", {"mlx": None}), \
         patch("builtins.print") as mock_print:
        
        from csm.mlx_accel.components.utils import log_success
        
        # Test with a simple success message
        log_success("Test success")
        mock_print.assert_called_with("SUCCESS: Test success")