"""Tests for MLX exact sampling."""

import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock

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


def test_mlx_sample_exact_initialization():
    """Test basic import and function access."""
    from csm.mlx_accel.mlx_sample_exact import mlx_sample_exact
    assert callable(mlx_sample_exact)
