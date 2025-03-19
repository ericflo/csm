"""Configuration for pytest."""

import sys
import os
from unittest.mock import MagicMock
import pytest

# Store the real state of MLX availability
try:
    import mlx as real_mlx
    HAS_REAL_MLX = True
except ImportError:
    HAS_REAL_MLX = False
    real_mlx = None

# For testing purposes, we want to make it possible to run tests with mock MLX
# This mocking is controlled by environment variables and command-line flags
HAS_MLX = HAS_REAL_MLX  # Default to real MLX availability

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
    parser.addoption(
        "--mock-mlx", 
        action="store_true", 
        default=False,
        help="Use mock MLX for all tests (for testing without real MLX)"
    )

def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", 
        "requires_mlx: mark test as requiring MLX (skipped on systems without MLX)"
    )
    config.addinivalue_line(
        "markers", 
        "mock_mlx: mark test as using mocked MLX"
    )

# Create universal mock MLX class that can be used by all tests
class MockMLX:
    """Mock MLX implementation for testing."""
    def __init__(self):
        # Main module attributes
        self.core = MagicMock()
        self.random = MagicMock()
        self.nn = MagicMock()
        
        # Make test shapes configurable based on the test
        def create_mock_array(shape=(1, 3), dtype=None, **kwargs):
            """Create a mock tensor with the given shape."""
            if isinstance(shape, tuple):
                mock_tensor = MagicMock(shape=shape, dtype=dtype)
                
                # Configure common tensor methods
                mock_tensor.reshape.return_value = MagicMock(shape=shape, dtype=dtype)
                mock_tensor.__add__.return_value = MagicMock(shape=shape, dtype=dtype)
                mock_tensor.__sub__.return_value = MagicMock(shape=shape, dtype=dtype)
                mock_tensor.__mul__.return_value = MagicMock(shape=shape, dtype=dtype)
                mock_tensor.__truediv__.return_value = MagicMock(shape=shape, dtype=dtype)
                mock_tensor.__eq__.return_value = MagicMock(shape=shape, dtype=dtype)
                
                # Create a more realistic getitem implementation
                def mock_getitem(indices):
                    # Handle simple scalar indexing
                    if isinstance(indices, (int, slice)):
                        if isinstance(indices, int) and len(shape) == 1:
                            return MagicMock(dtype=dtype)  # Scalar result
                        else:
                            # For slice or higher-dimension tensors
                            new_shape = shape[1:] if isinstance(indices, int) else shape
                            return create_mock_array(shape=new_shape, dtype=dtype)
                    
                    # Handle tuple indexing
                    elif isinstance(indices, tuple):
                        # Calculate resulting shape
                        new_shape = []
                        for i, idx in enumerate(indices):
                            if i < len(shape):  # Only process valid dimensions
                                if isinstance(idx, slice):
                                    new_shape.append(shape[i])
                                # Skip dimensions with integer indexing
                        
                        # Add remaining dimensions from original shape
                        if len(indices) < len(shape):
                            new_shape.extend(shape[len(indices):])
                            
                        if not new_shape:  # If empty (scalar result)
                            return MagicMock(dtype=dtype)
                        return create_mock_array(shape=tuple(new_shape), dtype=dtype)
                    
                    return MagicMock(dtype=dtype)  # Fallback
                
                mock_tensor.__getitem__ = mock_getitem
                return mock_tensor
            else:
                # For non-tuples (like bools), just return a standard mock
                return MagicMock(shape=shape, dtype=dtype)
        
        # Make array function handle input shape
        self.core.array = create_mock_array
        
        # Make zeros, ones, etc. respect the input shape
        def tensor_factory(shape=(1, 3), dtype=None, **kwargs):
            return create_mock_array(shape=shape, dtype=dtype, **kwargs)
            
        self.core.zeros = tensor_factory
        self.core.ones = tensor_factory
        
        # Special case for MLX 'all' function
        def mock_all(a, axis=None, keepdims=False, **kwargs):
            return MagicMock(dtype=bool)
        
        self.core.all = mock_all
        
        # Make random.normal respect the input shape
        self.random.normal = tensor_factory
        
        # Set up NN module with standard mocks
        self.nn.Transformer = MagicMock()
        self.nn.Module = MagicMock()
        
        # Set up matrix multiply to respect input shapes
        def mock_matmul(a, b, **kwargs):
            if not hasattr(a, 'shape') or not hasattr(b, 'shape'):
                return create_mock_array()
                
            # Determine output shape
            if len(a.shape) == 1 and len(b.shape) == 1:
                # Vector * Vector = Scalar
                return MagicMock()
            elif len(a.shape) == 2 and len(b.shape) == 2:
                # Matrix * Matrix = Matrix with shape (a.rows, b.cols)
                return create_mock_array(shape=(a.shape[0], b.shape[1]))
            elif len(a.shape) == 1 and len(b.shape) == 2:
                # Vector * Matrix = Vector with length b.cols
                return create_mock_array(shape=(b.shape[1],))
            elif len(a.shape) == 2 and len(b.shape) == 1:
                # Matrix * Vector = Vector with length a.rows
                return create_mock_array(shape=(a.shape[0],))
            elif len(a.shape) >= 3 or len(b.shape) >= 3:
                # Handle batched matrices - very simplified version
                batch_dims = a.shape[:-2] if len(a.shape) > len(b.shape) else b.shape[:-2]
                if len(a.shape) >= 2 and len(b.shape) >= 2:
                    matrix_dims = (a.shape[-2], b.shape[-1])
                else:
                    matrix_dims = (1, 1)  # Fallback
                return create_mock_array(shape=batch_dims + matrix_dims)
            else:
                # Default fallback
                return create_mock_array()
                
        self.core.matmul = mock_matmul
        
        # Set up random key
        mock_key = MagicMock()
        self.random.key.return_value = mock_key
        
        # Set up common operations
        mock_split_result = (MagicMock(), MagicMock())
        self.random.split.return_value = mock_split_result
        
        # Add helper methods used in tests
        
        # For KV cache tests
        def create_cache_tensor(batch_size, seq_len, num_heads, head_dim):
            """Create mock tensor for KV cache with correct shape."""
            shape = (batch_size, seq_len, num_heads, head_dim)
            return create_mock_array(shape=shape)
        
        self.create_cache_tensor = create_cache_tensor
        
        # For rotary embedding tests
        def create_position_tensor(batch_size, seq_len):
            """Create mock tensor for position IDs with correct shape."""
            shape = (batch_size, seq_len)
            result = create_mock_array(shape=shape)
            return result

@pytest.fixture(scope="session")
def global_mock_mlx():
    """Create a global MockMLX instance that can be used by all tests."""
    return MockMLX()

def pytest_runtest_setup(item):
    """Handle MLX test setup based on availability and configuration."""
    # Check if using mock MLX
    mock_mlx = item.config.getoption("--mock-mlx")
    
    # Skip MLX tests if:
    # 1. MLX is not available (and not using mock), or
    # 2. User has explicitly asked to skip via command line option, or
    # 3. SKIP_MLX_TESTS environment variable is set
    skip_mlx = (not HAS_REAL_MLX and not mock_mlx) or item.config.getoption("--skip-mlx") or SKIP_MLX_TESTS
    
    # Skip tests explicitly marked as requiring MLX
    if "requires_mlx" in item.keywords and skip_mlx:
        pytest.skip("Test requires MLX")

def pytest_collection_modifyitems(config, items):
    """Add MLX marker to appropriate tests."""
    requires_mlx_mark = pytest.mark.requires_mlx
    mock_mlx_mark = pytest.mark.mock_mlx
    
    # Patterns for tests that require MLX
    mlx_patterns = [
        "test_mlx_",  # Any test starting with test_mlx_
        "mlx_accel",  # Tests in the mlx_accel module
    ]
    
    # Components tests that should use mock MLX
    mock_mlx_patterns = [
        "components/test_generator",  # Generator components tests
        "test_components_sampling",   # Sampling components tests
        "test_components_utils",      # Utils components tests
    ]
    
    # Add markers to all tests based on patterns
    for item in items:
        # Mark MLX-requiring tests
        if any(pattern in item.nodeid for pattern in mlx_patterns):
            item.add_marker(requires_mlx_mark)
            
        # Mark tests that should use mock MLX
        if any(pattern in item.nodeid for pattern in mock_mlx_patterns):
            item.add_marker(mock_mlx_mark)

@pytest.fixture(autouse=True)
def setup_mock_mlx_for_marked_tests(request, global_mock_mlx, monkeypatch):
    """Setup mock MLX for tests marked with mock_mlx."""
    # Check if this test should use mock MLX
    # (either marked directly or via --mock-mlx option)
    if "mock_mlx" in request.keywords or request.config.getoption("--mock-mlx"):
        # Save original modules
        original_modules = {}
        module_names = ['mlx', 'mlx.core', 'mlx.random', 'mlx.nn']
        
        for name in module_names:
            if name in sys.modules:
                original_modules[name] = sys.modules[name]
        
        # Install mock MLX
        mock_mlx = global_mock_mlx
        sys.modules['mlx'] = mock_mlx
        sys.modules['mlx.core'] = mock_mlx.core
        sys.modules['mlx.random'] = mock_mlx.random
        sys.modules['mlx.nn'] = mock_mlx.nn
        
        # Mock is_mlx_available to return True for tests marked with mock_mlx
        monkeypatch.setattr(
            'csm.mlx_accel.components.utils.is_mlx_available',
            lambda: True
        )
        
        yield
        
        # Restore original modules
        for name, module in original_modules.items():
            sys.modules[name] = module
    else:
        yield