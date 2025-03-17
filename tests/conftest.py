"""Configuration for pytest."""

import sys
import os
import pytest

# Check if MLX is available
try:
    import mlx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

# Check if user has explicitly requested to skip MLX tests via environment variable
SKIP_MLX_TESTS = os.environ.get("SKIP_MLX_TESTS", "0").lower() in ("1", "true", "yes")

def pytest_addoption(parser):
    """Add command-line options to pytest."""
    parser.addoption(
        "--skip-mlx", 
        action="store_true", 
        default=False,
        help="Skip tests that require MLX (even if MLX is available)"
    )

def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", 
        "requires_mlx: mark test as requiring MLX (skipped on systems without MLX)"
    )

def pytest_runtest_setup(item):
    """Skip tests that require MLX if MLX is not available or skip is requested."""
    # Skip MLX tests if:
    # 1. MLX is not available, or
    # 2. User has explicitly asked to skip via command line option, or
    # 3. SKIP_MLX_TESTS environment variable is set
    skip_mlx = not HAS_MLX or item.config.getoption("--skip-mlx") or SKIP_MLX_TESTS
    
    # Skip tests explicitly marked as requiring MLX
    if "requires_mlx" in item.keywords and skip_mlx:
        pytest.skip("Test requires MLX")

def pytest_collection_modifyitems(config, items):
    """Add MLX marker to appropriate tests."""
    requires_mlx_mark = pytest.mark.requires_mlx
    
    # Patterns for tests that require MLX
    mlx_patterns = [
        "test_mlx_",  # Any test starting with test_mlx_
        "mlx_accel",  # Tests in the mlx_accel module
    ]
    
    # Add marker to all MLX-related tests for better reporting
    for item in items:
        if any(pattern in item.nodeid for pattern in mlx_patterns):
            item.add_marker(requires_mlx_mark)