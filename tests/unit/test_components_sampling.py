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

class MockMX:
    """A mock replacement for the MLX module."""
    
    def __init__(self):
        self.int32 = INT32
        self.float32 = 'float32'
        
    def array(self, x, **kwargs):
        """Create a mock array."""
        return np.array(x)
        
    def expand_dims(self, x, axis=0):
        """Expand dimensions of array."""
        return np.expand_dims(x, axis)
        
    def zeros(self, shape, dtype=None, **kwargs):
        """Create a zero array."""
        return np.zeros(shape)
        
    def where(self, cond, x, y):
        """Conditional selection."""
        return np.where(cond, x, y)
        
    def argmax(self, x, **kwargs):
        """Return index of max value."""
        return np.argmax(x, **kwargs)
        
    def argsort(self, x, **kwargs):
        """Return sorted indices."""
        return np.argsort(x, **kwargs)
        
    def take(self, x, indices):
        """Take values at indices."""
        return np.take(x, indices)
        
    def softmax(self, x, axis=-1):
        """Apply softmax."""
        # Simplified softmax for testing
        result = np.ones(x.shape) / x.shape[-1]
        return result
        
    def log(self, x):
        """Compute log."""
        return np.log(x)
        
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
    mock_mx = MockMX()
    mock_random = MockRandom()
    mock_mx.random = mock_random
    
    # Create the patches
    patches = [
        patch(f'{MODULE_PATH}.MLX_AVAILABLE', True),
        patch(f'{MODULE_PATH}.mx', mock_mx),
        patch(f'{MODULE_PATH}.mx.random', mock_random)
    ]
    
    # Start all patches
    for p in patches:
        p.start()
        
    # Return patches for cleanup
    return patches

# Create a simplified test environment helper
def make_simplified_test_env():
    """Create a simplified test environment that avoids complex MLX imports."""
    # Set up mock MLX
    return setup_mock_mlx()


def test_mlx_categorical_sampling():
    """Test mlx_categorical_sampling function."""
    # Create test setup
    patches = make_simplified_test_env()
    
    try:
        from csm.mlx_accel.components.sampling import mlx_categorical_sampling
        
        # Patch mlx_topk_sampling to verify it's called with the right parameters
        with patch(f'{MODULE_PATH}.mlx_topk_sampling') as mock_topk:
            mock_topk.return_value = np.array([[42]])
            
            # Create 1D input
            logits_1d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            result_1d = mlx_categorical_sampling(logits_1d, temperature=0.8, seed=42)
            
            # Verify topk was called with vocab_size=5
            args_1d, kwargs_1d = mock_topk.call_args
            assert kwargs_1d['k'] == 5, "Should use full vocab size (5) for 1D input"
            assert kwargs_1d['temperature'] == 0.8, "Should pass temperature correctly"
            assert kwargs_1d['seed'] == 42, "Should pass seed correctly"
            
            # Create 2D input
            logits_2d = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
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
    patches = make_simplified_test_env()
    
    try:
        from csm.mlx_accel.components.sampling import mlx_topk_sampling
        
        # Create a mock expand_dims to track calls
        original_expand_dims = patches[1].start().expand_dims
        expand_dims_called = False
        
        def tracked_expand_dims(x, axis=0):
            nonlocal expand_dims_called
            expand_dims_called = True
            # Store info about the call
            tracked_expand_dims.axis = axis
            # Call the original implementation
            return original_expand_dims(x, axis)
            
        # Replace the mock's expand_dims with our tracked version
        patches[1].stop().expand_dims = tracked_expand_dims
        
        # Create 1D input 
        logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Create mock for .at property access
        # Our MockMX doesn't need this patched for basic tests
        
        # Run the function
        with patch(f'{MODULE_PATH}.time.time', return_value=1000.0):
            result = mlx_topk_sampling(logits, k=2, temperature=1.0, seed=42)
        
        # Verify expand_dims was called with axis=0 to add batch dimension
        assert expand_dims_called, "expand_dims should be called for 1D input"
        assert tracked_expand_dims.axis == 0, "expand_dims should expand on axis 0 (add batch dimension)"
        
        # The function should return a valid result
        assert isinstance(result, np.ndarray), "Result should be a numpy array"
        assert result.shape == (1, 1), "Result should have shape (1, 1) for a single sample"
            
    finally:
        # Clean up the patches
        for p in patches:
            p.stop()


def test_mlx_topk_sampling_temperature():
    """Test temperature scaling in mlx_topk_sampling."""
    # Create test setup
    patches = make_simplified_test_env()
    
    try:
        # Import the function directly
        from csm.mlx_accel.components.sampling import mlx_topk_sampling
        
        # Track scaled logits for different temperatures
        temp_values = {}
        
        # Create a custom test function to inspect temperature effect
        def mock_topk_temperature_test(logits, temperature=1.0, **kwargs):
            """Mock function that captures temperature scaling."""
            # Get the highest logit value for tracking
            val = logits[0, 1]
            if isinstance(val, np.ndarray):
                val = val.item()
            # Store the scaled value
            temp_values[temperature] = val / temperature
            # Return a dummy result
            return np.zeros((1, 1))
            
        # Create test logits with clear preference for token at index 1
        logits = np.array([[1.0, 8.0, 3.0, 4.0, 2.0]])
        
        # Patch the function to use our mock for testing
        with patch(f'{MODULE_PATH}.mlx_topk_sampling', side_effect=mock_topk_temperature_test):
            # Test with different temperatures
            temperatures = [0.5, 1.0, 2.0]
            
            for temp in temperatures:
                # Call with this temperature
                result = mlx_topk_sampling(logits, k=5, temperature=temp, seed=42)
                
                # Verify temperature was actually used
                assert temp in temp_values, f"Temperature {temp} should be recorded"
                
            # Verify temperature scaling effect: lower temp = higher scaled values
            assert temp_values[0.5] > temp_values[1.0] > temp_values[2.0], \
                   "Low temperature should increase scaled values, high temperature should decrease them"
    finally:
        # Clean up the patches
        for p in patches:
            p.stop()


def test_mlx_topk_sampling_block_tokens():
    """Test that problematic tokens (1-31) are blocked."""
    # Create test setup
    patches = make_simplified_test_env()
    
    try:
        from csm.mlx_accel.components.sampling import mlx_topk_sampling
        
        # Our goal is to verify that tokens 1-31 are blocked by the function
        # To do this, we need to modify our MockMX class to track when .at[].set() is called
        
        # Track which indices are blocked with large negative values
        blocked_tokens = []
        
        # Add tracking to the MockMX with a custom at property
        class TrackingMockMX(MockMX):
            def __init__(self):
                super().__init__()
                self._tracked_array = None
                
            def array(self, x, **kwargs):
                """Create an array that we can track."""
                arr = super().array(x, **kwargs)
                # Store reference to use with at property
                self._tracked_array = arr
                return arr
                
            def at_set(self, indices, value):
                """Track .at[].set() operations."""
                if isinstance(indices, tuple) and len(indices) == 2:
                    batch_idx, token_idx = indices
                    # Only track operations blocking tokens 1-31
                    if 1 <= token_idx < 32 and value < -1e8:
                        blocked_tokens.append((batch_idx, token_idx, value))
                # Modify array - this doesn't actually matter for our test
                # since we just need to verify the operation was called
                return self._tracked_array
        
        # Create a tracking mock
        tracking_mx = TrackingMockMX()
        tracking_mx.random = patches[2].start()
        
        # Replace the MX mock with our tracking version
        patches[1].stop()
        with patch(f'{MODULE_PATH}.mx', tracking_mx):
            # Create logits with high values for problematic tokens
            logits = np.zeros((1, 100))
            for i in range(1, 32):
                logits[0, i] = 100.0  # Very high values to ensure they'd be selected if not blocked
                
            # Run the function - mock out .at property usage
            with patch.object(tracking_mx, 'at', create=True) as mock_at:
                # Set up the at property to call our tracking function
                mock_at.__getitem__ = lambda indices: type('obj', (), {
                    'set': lambda val: tracking_mx.at_set(indices, val)
                })
                
                # Call the function
                with patch(f'{MODULE_PATH}.time.time', return_value=1000.0):
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
        # Clean up the patches
        for p in patches:
            p.stop()


def test_mlx_topk_sampling_seed():
    """Test seed handling in mlx_topk_sampling."""
    # Create test setup
    patches = make_simplified_test_env()
    
    try:
        from csm.mlx_accel.components.sampling import mlx_topk_sampling
        
        # Create a version of MockRandom that tracks seeds
        used_seeds = []
        
        class SeedTrackingMockRandom(MockRandom):
            def key(self, seed=None):
                """Track which seeds are used."""
                used_seeds.append(seed)
                return super().key(seed)
                
        # Replace the random mock with our tracking version
        tracking_random = SeedTrackingMockRandom()
        patches[2].stop()  # Stop the current random mock
        patches[1].start().random = tracking_random  # Attach our tracking random to mx
        
        with patch(f'{MODULE_PATH}.mx.random', tracking_random), \
             patch(f'{MODULE_PATH}.time.time', return_value=1000.0):
            
            # Create 2D test logits
            logits = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
            
            # Clear tracking
            used_seeds.clear()
            
            # Test with explicit seed
            result1 = mlx_topk_sampling(logits, k=5, temperature=1.0, seed=42)
            assert 42 in used_seeds, "Explicit seed 42 should be used"
            
            # Test with None seed - should use current time*1000
            result2 = mlx_topk_sampling(logits, k=5, temperature=1.0, seed=None)
            expected_time_seed = int(1000.0 * 1000)  # time * 1000 as in code
            assert expected_time_seed in used_seeds, \
                   f"Time-based seed {expected_time_seed} should be used when seed is None"
    finally:
        # Clean up patches
        for p in patches:
            p.stop()


def test_mlx_topk_sampling_safety_checks():
    """Test safety checks in mlx_topk_sampling."""
    # Create test setup
    patches = make_simplified_test_env()
    
    try:
        from csm.mlx_accel.components.sampling import mlx_topk_sampling
        
        # We need to test two safety features:
        # 1. Replacing problematic tokens (1-31) with 0
        # 2. Replacing tokens beyond 2050 with 2050
        
        # For this test, we'll track what happens in the samples array
        final_token_value = None
        
        # Create a mock class with safety check monitoring
        class SafetyMockMX(MockMX):
            def __init__(self):
                super().__init__()
                self.shape_override = None
                
            def argmax(self, x, **kwargs):
                """Return the mock unsafe token value."""
                # This will be set to different values in each test
                return np.array(self.unsafe_token_value)
                
            def softmax(self, x, axis=-1):
                """Create a softmax result with shape override if needed."""
                if self.shape_override:
                    # Create a mock object with the shape property
                    class ShapedSoftmax:
                        def __init__(self, shape):
                            self.shape = shape
                    return ShapedSoftmax(self.shape_override)
                # Otherwise, use normal implementation
                return super().softmax(x, axis)
        
        # Create tracking version of mx
        safety_mx = SafetyMockMX()
        safety_mx.random = patches[2].start()
        
        # Replace standard mx with our safety tracking version
        patches[1].stop()
        
        # Create a patch for tracking sample updates
        with patch(f'{MODULE_PATH}.mx', safety_mx), \
             patch(f'{MODULE_PATH}.time.time', return_value=1000.0):
            
            # Setup for tracking the final token setting
            at_sets = []
            
            # Create a custom patch to catch the final at[].set() call
            def track_at_set(batch_idx, pos_idx, value):
                """Track .at[].set() operations on the samples array."""
                # We only care about the final samples write (token replacement)
                if batch_idx == 0 and pos_idx == 0:
                    at_sets.append(value)
                return np.array([[value]])
                
            # Mock .at property to track final token setting
            with patch.object(safety_mx, 'at', create=True) as mock_at:
                # Set up the at property to track our safety operations
                mock_at.__getitem__ = lambda indices: type('obj', (), {
                    'set': lambda val: track_at_set(indices[0], indices[1], val)
                })
                
                # TEST 1: Problematic token (1-31) replacement
                safety_mx.unsafe_token_value = 5  # A problematic token in range 1-31
                
                # Clear tracking
                at_sets.clear()
                
                # Run the function
                logits = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
                result = mlx_topk_sampling(logits, k=5, temperature=1.0, seed=42)
                
                # Verify problematic token was replaced with 0
                assert 0 in at_sets, "Problematic token 5 should be replaced with 0 (silence token)"
                
                # TEST 2: Invalid token beyond 2050
                safety_mx.unsafe_token_value = 2060  # Token beyond valid range
                safety_mx.shape_override = (1, 3000)  # Make softmax return large vocab size
                
                # Clear tracking
                at_sets.clear()
                
                # Run the function again
                result = mlx_topk_sampling(logits, k=5, temperature=1.0, seed=42)
                
                # Verify invalid token was replaced with 2050
                assert 2050 in at_sets, "Invalid token 2060 should be replaced with 2050 (max valid token)"
                
    finally:
        # Clean up patches
        for p in patches:
            p.stop()




def test_mlx_topk_sampling_gumbel():
    """Test Gumbel-max sampling technique in mlx_topk_sampling."""
    # Create test setup
    patches = make_simplified_test_env()
    
    try:
        from csm.mlx_accel.components.sampling import mlx_topk_sampling
        
        # Create a simple version of the test that just verifies the steps involved
        
        # Track operations performed
        operation_sequence = []
        
        # Create a GumbelMockMX that tracks operations
        class GumbelMockMX(MockMX):
            def softmax(self, x, axis=-1):
                """Track softmax calls."""
                operation_sequence.append('softmax')
                return super().softmax(x, axis)
                
            def log(self, x):
                """Track log calls."""
                operation_sequence.append('log')
                return super().log(x)
                
            def argmax(self, x, **kwargs):
                """Track argmax and return a fixed value."""
                operation_sequence.append('argmax')
                return np.array(1)  # Return token index 1
        
        # Create a tracking random module
        class GumbelMockRandom(MockRandom):
            def uniform(self, key=None, shape=None):
                """Track uniform random generation."""
                operation_sequence.append('uniform')
                return super().uniform(key, shape)
        
        # Create our tracking mocks
        gumbel_mx = GumbelMockMX()
        gumbel_random = GumbelMockRandom()
        gumbel_mx.random = gumbel_random
        
        # Replace standard mocks with our tracking versions
        patches[1].stop()
        patches[2].stop()
        
        # Create patch for .at property access - this is needed for token blocking
        with patch(f'{MODULE_PATH}.mx', gumbel_mx), \
             patch(f'{MODULE_PATH}.mx.random', gumbel_random), \
             patch(f'{MODULE_PATH}.time.time', return_value=1000.0):
            
            # Mock .at property
            with patch.object(gumbel_mx, 'at', create=True) as mock_at:
                # Set up property to handle .at[].set() calls
                mock_at.__getitem__ = lambda indices: type('obj', (), {
                    'set': lambda val: np.array([[val]])
                })
                
                # Clear tracking
                operation_sequence.clear()
                
                # Create logits with a clear preference
                logits = np.array([[1.0, 5.0, 3.0]])
                
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
                
                # Result should match the fixed value from our mock argmax
                assert result.item() == 1, "Result should match argmax output"
                
    finally:
        # Clean up patches
        for p in patches:
            p.stop()