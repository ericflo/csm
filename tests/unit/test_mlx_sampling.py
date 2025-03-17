"""Tests for MLX sampling module."""

import sys
import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock, Mock

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

# Import components after MLX availability check
from csm.mlx_accel.components.sampling import mlx_categorical_sampling, mlx_topk_sampling
