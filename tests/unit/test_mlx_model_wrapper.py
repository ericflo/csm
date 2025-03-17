"""
Tests for the MLX model wrapper component.
"""

import sys
import argparse
from unittest.mock import MagicMock, patch
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

# Import needed modules after handling MLX availability
from csm.mlx_accel.components.model_wrapper import MLXModelWrapper
