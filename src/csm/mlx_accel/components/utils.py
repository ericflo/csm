"""
Utility functions for MLX acceleration.
"""

import time
import importlib.util
from typing import Optional, Tuple, Union

import numpy as np
import torch

# Try to import MLX
MLX_AVAILABLE = False
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    # Create a dummy module
    class DummyMX:
        def __getattr__(self, name):
            raise ImportError("MLX is not available")
    mx = DummyMX()

# Constants
DEFAULT_DTYPE = torch.float32

def is_mlx_available() -> bool:
    """
    Check if MLX is available.
    
    Returns:
        True if MLX is available, False otherwise
    """
    return MLX_AVAILABLE

def check_device_compatibility() -> bool:
    """
    Check if the current device is compatible with MLX.
    
    Returns:
        True if device is compatible, False otherwise
    """
    if not MLX_AVAILABLE:
        return False
        
    # Check for Apple Silicon
    try:
        import platform
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

def setup_mlx_debug(enable: bool = False):
    """
    Setup MLX debug logging.
    
    Args:
        enable: Whether to enable debug logging
    """
    if not MLX_AVAILABLE:
        return
        
    # Set environment variables for debugging if needed
    if enable:
        import os
        os.environ["MLX_DEBUG"] = "1"

def format_dtype(dtype) -> str:
    """
    Format a data type for display.
    
    Args:
        dtype: PyTorch or MLX data type
        
    Returns:
        String representation of the data type
    """
    return str(dtype).split(".")[-1].replace("'", "").replace(">", "")

def get_shape_info(tensor) -> str:
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