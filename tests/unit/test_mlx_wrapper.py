"""
Tests for the MLX wrapper component.
"""

import sys
import argparse
import re
from unittest.mock import MagicMock, patch, ANY
import pytest
import torch
import numpy as np

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

# Create patch for torch_to_mlx and mlx_to_torch functions
# We'll apply these patches in individual tests rather than module-level
torch_to_mlx_mock = MagicMock()
torch_to_mlx_mock.return_value = mx.ones((10, 10))
mlx_to_torch_mock = MagicMock()
mlx_to_torch_mock.return_value = torch.ones((10, 10))

# Import needed modules after handling MLX availability
from csm.mlx_accel.mlx_wrapper import MLXWrapper
