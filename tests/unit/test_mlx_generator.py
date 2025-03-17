"""
Tests for the MLX generator component.
"""

import sys
import pytest
from unittest.mock import MagicMock, patch
import torch
import numpy as np

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

# Import the module under test after MLX availability check
from csm.mlx_accel.components.generator import MLXGenerator
