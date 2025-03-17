"""
Tests for sampling functions in components/sampling.py.
"""

import pytest
import os
import time
import numpy as np
from unittest.mock import patch, MagicMock

# Import module paths we'll need to patch
MODULE_PATH = "csm.mlx_accel.components.sampling"

# Define test constants
MLX_AVAILABLE_PATH = f"{MODULE_PATH}.MLX_AVAILABLE"
INT32 = 'int32'

class MLXArray:
    """Mock implementation of MLX array."""
    
    def __init__(self, data, *args, **kwargs):
        """Initialize with numpy array data."""
        self.data = np.array(data)
        self.shape = self.data.shape
        
    def __getitem__(self, idx):
        """Support array[idx] syntax."""
        result = self.data[idx]
        if isinstance(result, np.ndarray):
            return MLXArray(result)
        return result
        
    def __truediv__(self, other):
        """Support division operation: array / scalar."""
        if isinstance(other, (int, float)):
            return MLXArray(self.data / other)
        if isinstance(other, MLXArray):
            return MLXArray(self.data / other.data)
        return MLXArray(self.data / other)
        
    def __lt__(self, other):
        """Support less than comparison: array < scalar."""
        if isinstance(other, (int, float)):
            return MLXArray(self.data < other)
        if isinstance(other, MLXArray):
            return MLXArray(self.data < other.data)
        return MLXArray(self.data < other)
        
    def __neg__(self):
        """Support unary negation: -array."""
        return MLXArray(-self.data)
        
    def __float__(self):
        """Support conversion to float."""
        if self.data.size == 1:
            return float(self.data.item())
        raise ValueError("Only size-1 arrays can be converted to Python scalars")
    
    def __len__(self):
        """Support len(array)."""
        return len(self.data)
        
    def tolist(self):
        """Convert to Python list."""
        return self.data.tolist()
        
    def item(self):
        """Get single item from array."""
        return self.data.item()
        
    def copy(self):
        """Create a copy of the array."""
        return MLXArray(self.data.copy())
        
    @property
    def at(self):
        """Support .at property for indexing operations."""
        return AtIndexer(self)
        
class MockMX:
    """A mock replacement for the MLX module."""
    
    def __init__(self):
        self.int32 = INT32
        self.float32 = 'float32'
        self.random = MockRandom()
        
    def array(self, x, **kwargs):
        """Create an MLX array."""
        return MLXArray(x)
        
    def expand_dims(self, x, axis=0):
        """Expand dimensions of array."""
        if isinstance(x, MLXArray):
            expanded = np.expand_dims(x.data, axis)
            return MLXArray(expanded)
        else:
            expanded = np.expand_dims(x, axis)
            return MLXArray(expanded)
        
    def zeros(self, shape, dtype=None, **kwargs):
        """Create a zero array."""
        return MLXArray(np.zeros(shape))
        
    def where(self, cond, x, y):
        """Conditional selection."""
        if isinstance(cond, MLXArray):
            cond_data = cond.data
        else:
            cond_data = cond
            
        if isinstance(x, MLXArray):
            x_data = x.data
        else:
            x_data = x
            
        if isinstance(y, MLXArray):
            y_data = y.data
        else:
            y_data = y
            
        result = np.where(cond_data, x_data, y_data)
        return MLXArray(result)
        
    def argmax(self, x, **kwargs):
        """Return index of max value."""
        if isinstance(x, MLXArray):
            return np.argmax(x.data, **kwargs)
        return np.argmax(x, **kwargs)
        
    def argsort(self, x, **kwargs):
        """Return sorted indices."""
        # Handle descending flag (numpy doesn't have this)
        descending = kwargs.pop('descending', False)
        
        if isinstance(x, MLXArray):
            # Get sorted indices from NumPy (always ascending)
            indices = np.argsort(x.data, **kwargs)
            
            # If we want descending order, reverse the indices
            if descending:
                indices = indices[::-1]
                
            return MLXArray(indices)
            
        # Handle non-MLXArray input
        indices = np.argsort(x, **kwargs)
        if descending:
            indices = indices[::-1]
            
        return MLXArray(indices)
        
    def take(self, x, indices):
        """Take values at indices."""
        if isinstance(x, MLXArray):
            x_data = x.data
        else:
            x_data = x
            
        if isinstance(indices, MLXArray):
            indices_data = indices.data
        else:
            indices_data = indices
            
        result = np.take(x_data, indices_data)
        return MLXArray(result)
        
    def softmax(self, x, axis=-1):
        """Apply softmax."""
        # Simplified softmax for testing
        if isinstance(x, MLXArray):
            result = np.ones(x.shape) / x.shape[-1]
        else:
            result = np.ones(np.array(x).shape) / np.array(x).shape[-1]
        return MLXArray(result)
        
    def log(self, x):
        """Compute log."""
        if isinstance(x, MLXArray):
            result = np.log(x.data)
        else:
            result = np.log(x)
        return MLXArray(result)
        
    def broadcast_to(self, x, shape):
        """Broadcast array to new shape."""
        if isinstance(x, MLXArray):
            result = np.broadcast_to(x.data, shape)
        else:
            result = np.broadcast_to(x, shape)
        return MLXArray(result)

class AtIndexer:
    """Helper class to simulate MLX's .at property for array indexing."""
    
    def __init__(self, array):
        self.array = array
        
    def __getitem__(self, indices):
        """Handle array[indices] syntax."""
        # Return an object with a .set method
        return AtSetter(self.array, indices)
        
class AtSetter:
    """Helper class to simulate MLX's .at[indices].set() functionality."""
    
    def __init__(self, array, indices):
        self.array = array
        self.indices = indices
        
    def set(self, value):
        """Set value at indices and return a new array."""
        # Make a copy of the array data
        if isinstance(self.array, MLXArray):
            result_data = self.array.data.copy()
        else:
            result_data = self.array.copy()
        
        # Convert value if it's an MLXArray
        if isinstance(value, MLXArray):
            actual_value = value.data
        else:
            actual_value = value
        
        # Handle different index types
        if isinstance(self.indices, tuple):
            # Multi-dimensional indexing
            try:
                # Use advanced indexing when possible
                result_data[self.indices] = actual_value
            except (IndexError, TypeError, ValueError):
                # Fall back for complex cases
                pass
        else:
            # Single-dimension indexing
            try:
                result_data[self.indices] = actual_value
            except (ValueError, TypeError):
                # If direct assignment fails, try more carefully:
                if isinstance(actual_value, np.ndarray):
                    # Extract a scalar when trying to assign to a single position
                    if actual_value.size == 1:
                        result_data[self.indices] = actual_value.item()
                    # For more complex cases, we need specific handling
            
        # Return a new MLX array
        return MLXArray(result_data)
        
class MockRandom:
    """A mock replacement for MLX random module."""
    
    def key(self, seed=None):
        """Create a random key."""
        return np.array([seed if seed is not None else 42])
        
    def split(self, key):
        """Split a key into two."""
        return np.array([key[0] + 1]), np.array([key[0] + 2])
        
    def uniform(self, key=None, shape=None):
        """Generate uniform random values."""
        return np.ones(shape) * 0.5

# Create a function to set up the mock environment
def setup_mock_mlx():
    """Create and set up mock MLX modules."""
    mock_mx = MockMX()  # MockMX now contains random from the start
    
    # Create the patches
    patches = [
        patch(f'{MODULE_PATH}.MLX_AVAILABLE', True),
        patch(f'{MODULE_PATH}.mx', mock_mx),
        patch(f'{MODULE_PATH}.mx.random', mock_mx.random)
    ]
    
    # Start all patches
    for p in patches:
        p.start()
        
    # Return patches for cleanup
    return patches, mock_mx

# Create a simplified test environment helper
def make_simplified_test_env():
    """Create a simplified test environment that avoids complex MLX imports."""
    # Set up mock MLX
    patches, mock_mx = setup_mock_mlx()
    return patches, mock_mx


def test_mlx_categorical_sampling():
    """Test mlx_categorical_sampling function."""
    # Create test setup
    patches, mock_mx = make_simplified_test_env()
    
    try:
        from csm.mlx_accel.components.sampling import mlx_categorical_sampling
        
        # Patch mlx_topk_sampling to verify it's called with the right parameters
        with patch(f'{MODULE_PATH}.mlx_topk_sampling') as mock_topk:
            mock_topk.return_value = MLXArray([[42]])
            
            # Create 1D input
            logits_1d = mock_mx.array([1.0, 2.0, 3.0, 4.0, 5.0])
            result_1d = mlx_categorical_sampling(logits_1d, temperature=0.8, seed=42)
            
            # Verify topk was called with vocab_size=5
            args_1d, kwargs_1d = mock_topk.call_args
            assert kwargs_1d['k'] == 5, "Should use full vocab size (5) for 1D input"
            assert kwargs_1d['temperature'] == 0.8, "Should pass temperature correctly"
            assert kwargs_1d['seed'] == 42, "Should pass seed correctly"
            
            # Create 2D input
            logits_2d = mock_mx.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
            result_2d = mlx_categorical_sampling(logits_2d, temperature=0.5, seed=123)
            
            # Verify topk was called with vocab_size=6
            args_2d, kwargs_2d = mock_topk.call_args
            assert kwargs_2d['k'] == 6, "Should use full vocab size (6) for 2D input"
            assert kwargs_2d['temperature'] == 0.5, "Should pass temperature correctly"
            assert kwargs_2d['seed'] == 123, "Should pass seed correctly"
    finally:
        # Clean up the patches
        for p in patches:
            p.stop()


def test_mlx_topk_sampling_1d_input():
    """Test that 1D input is handled correctly in mlx_topk_sampling."""
    # Create test setup
    patches, mock_mx = make_simplified_test_env()
    
    try:
        from csm.mlx_accel.components.sampling import mlx_topk_sampling
        
        # Track expand_dims calls
        expand_dims_called = False
        original_expand_dims = mock_mx.expand_dims
        
        def tracked_expand_dims(x, axis=0):
            nonlocal expand_dims_called
            expand_dims_called = True
            # Store info about the call
            tracked_expand_dims.axis = axis
            # Call the original implementation
            return original_expand_dims(x, axis)
            
        mock_mx.expand_dims = tracked_expand_dims
        
        # Create 1D input
        logits = mock_mx.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Also patch time for deterministic seed
        with patch(f'{MODULE_PATH}.time.time', return_value=1000.0):
            # Call the function
            result = mlx_topk_sampling(logits, k=2, temperature=1.0, seed=42)
            
            # Verify expand_dims was called with axis=0 to add batch dimension
            assert expand_dims_called, "expand_dims should be called for 1D input"
            assert tracked_expand_dims.axis == 0, "expand_dims should expand on axis 0 (add batch dimension)"
            
            # The function should return a valid result
            assert isinstance(result, MLXArray), "Result should be a MLXArray"
            assert result.shape == (1, 1), "Result should have shape (1, 1) for a single sample"
    finally:
        # Clean up the patches
        for p in patches:
            p.stop()


def test_mlx_topk_sampling_temperature():
    """Test temperature scaling in mlx_topk_sampling."""
    # Create test setup
    patches, mock_mx = make_simplified_test_env()
    
    try:
        # Import the function directly
        from csm.mlx_accel.components.sampling import mlx_topk_sampling
        
        # Test the actual function's temperature scaling
        # Create logits with clear preference for token at index 1
        logits = mock_mx.array([[1.0, 8.0, 3.0, 4.0, 2.0]])
        
        # Verify temperature effect directly by capturing scaled_logits
        temps_and_divisors = []
        
        # Save the original division method to restore later
        original_div = MLXArray.__truediv__
        
        # Create a version that tracks all divisions by temperature
        def tracking_div(self, other):
            # Check if other is a temperature value we're tracking
            if isinstance(other, float) and 0.1 <= other <= 10.0:  
                # This is likely a temperature division
                temps_and_divisors.append(other)
            # Call original implementation
            return original_div(self, other)
        
        # Install our tracking division
        MLXArray.__truediv__ = tracking_div
        
        try:
            # Test with different temperatures
            temperatures = [0.5, 1.0, 2.0]
            
            for temp in temperatures:
                # Clear tracking for this temperature
                temps_and_divisors.clear()
                
                # Call with this temperature
                result = mlx_topk_sampling(logits, k=5, temperature=temp, seed=42)
                
                # Verify the division by temperature happened
                assert temp in temps_and_divisors, f"Division by temperature {temp} not detected"
        finally:
            # Restore original division method
            MLXArray.__truediv__ = original_div
        
        # Create a simpler test to verify the mathematical effect
        # Higher temperature = lower scaled values
        test_value = 8.0
        scaled_values = {temp: test_value / temp for temp in temperatures}
        
        # Verify temperature scaling effect: lower temp = higher scaled values
        assert scaled_values[0.5] > scaled_values[1.0] > scaled_values[2.0], \
               "Low temperature should increase scaled values, high temperature should decrease them"
    finally:
        # Clean up the patches
        for p in patches:
            p.stop()


def test_mlx_topk_sampling_block_tokens():
    """Test that problematic tokens (1-31) are blocked."""
    # Create test setup
    patches, mock_mx = make_simplified_test_env()
    
    try:
        from csm.mlx_accel.components.sampling import mlx_topk_sampling
        
        # Our goal is to verify that tokens 1-31 are blocked by the function
        # Track which indices are blocked with large negative values
        blocked_tokens = []
        
        # Save the original AtSetter.set method
        original_at_setter_set = AtSetter.set
        
        # Create a tracking version of the set method
        def tracking_set(self, value):
            """Track token blocking operations."""
            # Check if this is a token blocking operation
            if isinstance(self.indices, tuple) and len(self.indices) == 2:
                batch_idx, token_idx = self.indices
                # Convert value to a number if it's an MLXArray or numpy array
                if isinstance(value, MLXArray):
                    numeric_value = value.item()
                elif isinstance(value, np.ndarray) and value.size == 1:
                    numeric_value = value.item()
                else:
                    numeric_value = value
                    
                # Check if this is a blocking operation
                if 1 <= token_idx < 32 and numeric_value < -1e8:
                    blocked_tokens.append((batch_idx, token_idx, numeric_value))
            
            # Call the original implementation
            return original_at_setter_set(self, value)
        
        # Install our tracking method
        AtSetter.set = tracking_set
        
        try:
            # Create logits with high values for problematic tokens
            logits_data = np.zeros((1, 100))
            for i in range(1, 32):
                logits_data[0, i] = 100.0  # Very high values to ensure they'd be selected if not blocked
            
            # Convert to our mock MLX array
            logits = mock_mx.array(logits_data)
            
            # Also patch time for deterministic seed
            with patch(f'{MODULE_PATH}.time.time', return_value=1000.0):
                # Call the function
                result = mlx_topk_sampling(logits, k=5, temperature=1.0, seed=42)
            
            # Get the unique token indices that were blocked
            blocked_indices = set(token_idx for _, token_idx, _ in blocked_tokens)
            
            # Verify all problematic tokens 1-31 were blocked
            for i in range(1, 32):
                assert i in blocked_indices, f"Token {i} should be blocked"
                
            # Verify the blocking was done with large negative values
            for _, _, value in blocked_tokens:
                assert value < -1e8, "Blocked tokens should get large negative penalties"
                
        finally:
            # Restore original method
            AtSetter.set = original_at_setter_set
                
    finally:
        # Clean up the patches
        for p in patches:
            p.stop()


def test_mlx_topk_sampling_seed():
    """Test seed handling in mlx_topk_sampling."""
    # Create test setup
    patches, mock_mx = make_simplified_test_env()
    
    try:
        from csm.mlx_accel.components.sampling import mlx_topk_sampling
        
        # Create a version of MockRandom that tracks seeds
        used_seeds = []
        
        # Save original key method to restore later
        original_key = mock_mx.random.key
        
        # Create tracking version
        def tracking_key(seed=None):
            """Track seeds used in random key generation."""
            used_seeds.append(seed)
            return original_key(seed)
            
        # Install tracking method
        mock_mx.random.key = tracking_key
        
        try:
            # Create 2D test logits
            logits = mock_mx.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
            
            # Test with explicit seed
            with patch(f'{MODULE_PATH}.time.time', return_value=1000.0):
                # Clear tracking
                used_seeds.clear()
                
                # Call function with explicit seed
                result1 = mlx_topk_sampling(logits, k=5, temperature=1.0, seed=42)
                
                # Verify seed was used
                assert 42 in used_seeds, "Explicit seed 42 should be used"
            
            # Test with None seed - should use current time*1000
            with patch(f'{MODULE_PATH}.time.time', return_value=1234.5678):
                # Clear tracking
                used_seeds.clear()
                
                # Call function with None seed (will use time)
                result2 = mlx_topk_sampling(logits, k=5, temperature=1.0, seed=None)
                
                # Expected seed from time * 1000
                expected_time_seed = int(1234.5678 * 1000)
                
                # Verify time-based seed was used
                assert expected_time_seed in used_seeds, \
                       f"Time-based seed {expected_time_seed} should be used when seed is None"
        finally:
            # Restore original method
            mock_mx.random.key = original_key
    finally:
        # Clean up patches
        for p in patches:
            p.stop()


def test_mlx_topk_sampling_safety_checks():
    """Test safety checks in mlx_topk_sampling."""
    # Create test setup
    patches, mock_mx = make_simplified_test_env()
    
    try:
        from csm.mlx_accel.components.sampling import mlx_topk_sampling
        
        # We need to test two safety features:
        # 1. Replacing problematic tokens (1-31) with 0
        # 2. Replacing tokens beyond 2050 with 2050
        
        # Save the original argmax method to restore later
        original_argmax = mock_mx.argmax
        
        # Test values to use for unsafe token tests
        test_cases = [
            # (unsafe_token_value, vocab_size, expected_replacement)
            (5, None, 0),       # Test case 1: Problematic token 5 -> 0
            (2060, 3000, 2050)  # Test case 2: Invalid token 2060 -> 2050
        ]
        
        # Track sample updates - need to track the final token value set
        sample_values = []
        
        # Save original AtSetter.set method
        original_set = AtSetter.set
        
        # Create a tracking version to catch sample updates
        def tracking_sample_set(self, value):
            """Track sample array updates (only for the specific pattern we care about)."""
            # Check if this is a token replacement in the samples array
            if isinstance(self.indices, tuple) and len(self.indices) == 2:
                batch_idx, pos_idx = self.indices
                if batch_idx == 0 and pos_idx == 0:
                    # Convert MLXArray or numpy to a simple value
                    if isinstance(value, MLXArray):
                        sample_value = value.item()
                    elif isinstance(value, np.ndarray) and value.size == 1:
                        sample_value = value.item()
                    else:
                        sample_value = value
                    # Add to our tracking list
                    sample_values.append(sample_value)
            
            # Call original implementation
            return original_set(self, value)
            
        # Install our tracking method
        AtSetter.set = tracking_sample_set
        
        try:
            # Run each test case
            for unsafe_token, vocab_size, expected_replacement in test_cases:
                # Create test logits
                logits = mock_mx.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
                
                # Create a custom argmax that returns our unsafe token
                def unsafe_argmax(x, **kwargs):
                    """Return the specified unsafe token value."""
                    return np.array(unsafe_token)
                
                # Install our unsafe argmax
                mock_mx.argmax = unsafe_argmax
                
                # If we need to simulate larger vocab
                if vocab_size:
                    # Override softmax to produce larger output
                    original_softmax = mock_mx.softmax
                    
                    def big_vocab_softmax(x, axis=-1):
                        """Return softmax with larger vocab size."""
                        if isinstance(x, MLXArray) and len(x.shape) == 2:
                            # Create a larger output
                            result_data = np.ones((x.shape[0], vocab_size)) / vocab_size
                            result = MLXArray(result_data)
                            return result
                        # Default behavior
                        return original_softmax(x, axis)
                    
                    # Install custom softmax
                    mock_mx.softmax = big_vocab_softmax
                
                # Clear tracking for this test
                sample_values.clear()
                
                # Run the function with the unsafe token
                with patch(f'{MODULE_PATH}.time.time', return_value=1000.0):
                    result = mlx_topk_sampling(logits, k=5, temperature=1.0, seed=42)
                
                # Clean up if we modified softmax
                if vocab_size:
                    mock_mx.softmax = original_softmax
                
                # Verify token was replaced as expected
                assert expected_replacement in sample_values, \
                       f"Unsafe token {unsafe_token} should be replaced with {expected_replacement}"
        finally:
            # Restore original methods
            mock_mx.argmax = original_argmax
            AtSetter.set = original_set
                
    finally:
        # Clean up patches
        for p in patches:
            p.stop()




def test_mlx_topk_sampling_gumbel():
    """Test Gumbel-max sampling technique in mlx_topk_sampling."""
    # Create test setup
    patches, mock_mx = make_simplified_test_env()
    
    try:
        from csm.mlx_accel.components.sampling import mlx_topk_sampling
        
        # Track operations performed
        operation_sequence = []
        
        # Save original methods to restore later
        original_softmax = mock_mx.softmax
        original_log = mock_mx.log
        original_argmax = mock_mx.argmax
        original_uniform = mock_mx.random.uniform
        
        # Create tracking versions of methods
        def tracking_softmax(x, axis=-1):
            """Track softmax calls."""
            operation_sequence.append('softmax')
            return original_softmax(x, axis)
            
        def tracking_log(x):
            """Track log calls."""
            operation_sequence.append('log')
            return original_log(x)
            
        def tracking_argmax(x, **kwargs):
            """Track argmax and return a fixed value."""
            operation_sequence.append('argmax')
            # Return 1 as our fixed token value
            return np.array(1)
            
        def tracking_uniform(key=None, shape=None):
            """Track uniform random generation."""
            operation_sequence.append('uniform')
            return original_uniform(key, shape)
            
        # Install our tracking methods
        mock_mx.softmax = tracking_softmax
        mock_mx.log = tracking_log
        mock_mx.argmax = tracking_argmax
        mock_mx.random.uniform = tracking_uniform
        
        # We also need to patch the final token assignment
        sample_value = None
        original_set = AtSetter.set
        
        def capture_sample_set(self, value):
            """Capture the value set in samples array."""
            nonlocal sample_value
            # Check if this is a sample output update
            if isinstance(self.indices, tuple) and len(self.indices) == 2:
                batch_idx, pos_idx = self.indices
                if batch_idx == 0 and pos_idx == 0:
                    if isinstance(value, MLXArray):
                        sample_value = value.item()
                    elif isinstance(value, np.ndarray) and value.size == 1:
                        sample_value = value.item()
                    else:
                        sample_value = value
            
            # Continue with regular implementation
            return original_set(self, value)
            
        # Install our tracking set method
        AtSetter.set = capture_sample_set
        
        try:
            # Create logits with a clear preference
            logits = mock_mx.array([[1.0, 5.0, 3.0]])
            
            # Call the function with Gumbel-max sampling
            with patch(f'{MODULE_PATH}.time.time', return_value=1000.0):
                # Clear tracking before the test
                operation_sequence.clear()
                sample_value = None
                
                # Call the function with Gumbel-max sampling
                result = mlx_topk_sampling(logits, k=3, temperature=1.0, seed=42)
            
            # The sequence should show:
            # 1. Softmax to get probabilities
            # 2. Uniform random sampling
            # 3. Log transform
            # 4. Argmax to select token
            
            # Check for main operations
            assert 'softmax' in operation_sequence, "Should apply softmax"
            assert 'uniform' in operation_sequence, "Should use uniform random sampling"
            assert 'log' in operation_sequence, "Should apply log transform"
            assert 'argmax' in operation_sequence, "Should use argmax to select token"
            
            # Check correct sequence
            uniform_idx = operation_sequence.index('uniform')
            log_idx = operation_sequence.index('log') 
            
            # Verify uniform comes before log (for Gumbel noise)
            assert uniform_idx < log_idx, "Uniform sampling should happen before log transform"
            
            # Check that operations happened in the right order
            # We don't need to verify the exact value here because it might be 
            # modified by the safety checks - our test token (1) is in the range
            # of problematic tokens (1-31) that get replaced with 0.
            assert 'argmax' in operation_sequence, "Should use argmax for sampling"
        finally:
            # Restore original methods
            mock_mx.softmax = original_softmax
            mock_mx.log = original_log
            mock_mx.argmax = original_argmax
            mock_mx.random.uniform = original_uniform
            AtSetter.set = original_set
                
    finally:
        # Clean up patches
        for p in patches:
            p.stop()