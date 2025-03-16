"""
Tests for the token analyzer module in MLX acceleration.
"""

import sys
import os
import pytest
from unittest.mock import MagicMock, patch, mock_open
import torch
import numpy as np
from collections import Counter

# Check if MLX is available
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    # Create a mock module
    class MockMX:
        def __init__(self):
            self.core = MagicMock()
            self.nn = MagicMock()
            self.random = MagicMock()
    mx = MockMX()
    sys.modules['mlx'] = mx
    sys.modules['mlx.core'] = mx.core
    sys.modules['mlx.nn'] = mx.nn

# Import the module under test
from csm.mlx_accel.token_analyzer import (
    distribution_similarity,
    analyze_distributions,
    save_token_analysis,
    load_token_analysis,
    capture_tokens
)

# We'll test compare_sampling_implementations separately due to its complexity


class MockPyTorchModel:
    """Mock PyTorch model for testing."""
    
    def __init__(self, return_tokens=True, raise_error=False):
        self.raise_error = raise_error
        self.return_tokens = return_tokens
        self._last_tokens = torch.tensor([[[1, 2, 3], [4, 5, 6]]], dtype=torch.long) if return_tokens else None
    
    def generate(self, text=None, speaker=0):
        if self.raise_error:
            raise ValueError("Simulated PyTorch model error")
        return torch.zeros(1, 24000)  # Simulated audio


class MockMLXModel:
    """Mock MLX model for testing."""
    
    def __init__(self, return_tokens=True, raise_error=False):
        self.raise_error = raise_error
        self.return_tokens = return_tokens
        self._last_tokens = torch.tensor([[[10, 20, 30], [40, 50, 60]]], dtype=torch.long) if return_tokens else None
    
    def generate_speech(self, text=None, speaker=0):
        if self.raise_error:
            raise ValueError("Simulated MLX model error")
        return torch.zeros(1, 24000)  # Simulated audio


def test_distribution_similarity_normal():
    """Test distribution similarity calculation with normal inputs."""
    # Create tensors with known distributions
    tokens1 = torch.tensor([1, 2, 3, 1, 2, 1])
    tokens2 = torch.tensor([1, 2, 4, 1, 2, 1])
    
    # Calculate expected similarity
    # tokens1 distribution: {1: 3/6, 2: 2/6, 3: 1/6}
    # tokens2 distribution: {1: 3/6, 2: 2/6, 4: 1/6}
    # Common tokens: 1, 2
    # Similarity: min(3/6, 3/6) + min(2/6, 2/6) = 0.5 + 0.33... = 0.833...
    
    similarity = distribution_similarity(tokens1, tokens2)
    
    # Should be close to 0.833...
    assert abs(similarity - 0.8333) < 0.01


def test_distribution_similarity_identical():
    """Test distribution similarity with identical distributions."""
    tokens = torch.tensor([1, 2, 3, 1, 2, 1])
    
    similarity = distribution_similarity(tokens, tokens)
    
    # Should be very close to 1.0 for identical distributions
    # (using almost equal due to potential floating point precision issues)
    assert abs(similarity - 1.0) < 1e-10


def test_distribution_similarity_no_overlap():
    """Test distribution similarity with no overlap."""
    tokens1 = torch.tensor([1, 2, 3])
    tokens2 = torch.tensor([4, 5, 6])
    
    similarity = distribution_similarity(tokens1, tokens2)
    
    # Should be 0.0 for no overlap
    assert similarity == 0.0


def test_distribution_similarity_non_tensor():
    """Test distribution similarity with non-tensor inputs."""
    tokens1 = [1, 2, 3]  # Not a tensor
    tokens2 = torch.tensor([1, 2, 3])
    
    similarity = distribution_similarity(tokens1, tokens2)
    
    # Should return 0 for non-tensor inputs
    assert similarity == 0.0


def test_save_token_analysis():
    """Test saving token analysis results."""
    # Create mock data
    results = {
        'pytorch': {'tokens': torch.tensor([1, 2, 3])},
        'mlx': {'tokens': torch.tensor([4, 5, 6])}
    }
    
    # Mock os.makedirs and torch.save
    with patch('os.makedirs') as mock_makedirs, \
         patch('torch.save') as mock_save, \
         patch('builtins.print') as mock_print:
        
        save_token_analysis(results, "test_results.pt")
        
        # Check if directory was created
        mock_makedirs.assert_called_once_with('token_analysis', exist_ok=True)
        
        # Check if torch.save was called with correct arguments
        mock_save.assert_called_once()
        args, kwargs = mock_save.call_args
        assert args[0] == results
        assert args[1] == os.path.join('token_analysis', 'test_results.pt')


def test_load_token_analysis_file_exists():
    """Test loading token analysis results when file exists."""
    # Create mock data
    mock_results = {
        'pytorch': {'tokens': torch.tensor([1, 2, 3])},
        'mlx': {'tokens': torch.tensor([4, 5, 6])}
    }
    
    # Mock os.path.exists to return True and torch.load to return mock_results
    with patch('os.path.exists', return_value=True), \
         patch('torch.load', return_value=mock_results):
        
        # Load the results
        results = load_token_analysis("test_results.pt")
        
        # Check if the correct results were returned
        assert results == mock_results


def test_load_token_analysis_file_not_exists():
    """Test loading token analysis results when file doesn't exist."""
    # Mock os.path.exists to return False
    with patch('os.path.exists', return_value=False), \
         patch('builtins.print') as mock_print:
        
        # Load the results
        results = load_token_analysis("nonexistent.pt")
        
        # Check if None was returned
        assert results is None
        
        # Check if a message was printed
        mock_print.assert_called_once()


def test_analyze_distributions():
    """Test analyzing token distributions."""
    # Create mock tensors with known distributions
    pt_tokens = torch.tensor([1, 2, 3, 1, 2, 1])
    mlx_tokens = torch.tensor([1, 2, 4, 1, 2, 1])
    
    # Need to patch matplotlib.pyplot directly, not the one imported in the module
    with patch('matplotlib.pyplot') as mock_plt, \
         patch('os.makedirs') as mock_makedirs, \
         patch('builtins.print') as mock_print:
        
        # We need to patch the module to use our mocked matplotlib
        with patch('csm.mlx_accel.token_analyzer.plt', mock_plt):
            # Ensure plt.figure doesn't raise an exception
            mock_plt.figure.return_value = MagicMock()
            
            # Run the analysis
            analyze_distributions(pt_tokens, mlx_tokens)
            
            # Check if a plot was created 
            assert mock_plt.figure.called
            
            # Check if directory was created
            mock_makedirs.assert_called_once_with('token_analysis', exist_ok=True)


def test_analyze_distributions_non_tensor():
    """Test analyzing distributions with non-tensor inputs."""
    # Create a non-tensor input
    pt_tokens = [1, 2, 3]  # Not a tensor
    mlx_tokens = torch.tensor([1, 2, 3])
    
    # Mock print to check for error message
    with patch('builtins.print') as mock_print:
        analyze_distributions(pt_tokens, mlx_tokens)
        
        # Check if an error message was printed
        mock_print.assert_called_once_with("Cannot analyze distributions: tokens must be PyTorch tensors")


def test_capture_tokens_success():
    """Test capturing tokens when both models succeed."""
    # Create mock models
    pt_model = MockPyTorchModel()
    mlx_model = MockMLXModel()
    
    # Mock analyze_distributions to avoid plotting
    with patch('csm.mlx_accel.token_analyzer.analyze_distributions') as mock_analyze, \
         patch('builtins.print'):
        
        # Capture tokens
        results = capture_tokens(pt_model, mlx_model, "Test text", verbose=True)
        
        # Check if analyze_distributions was called
        mock_analyze.assert_called_once()
        
        # Check if results contain expected data
        assert 'pytorch' in results
        assert 'mlx' in results
        assert 'tokens' in results['pytorch']
        assert 'tokens' in results['mlx']
        assert torch.equal(results['pytorch']['tokens'], pt_model._last_tokens)
        assert torch.equal(results['mlx']['tokens'], mlx_model._last_tokens)


def test_capture_tokens_no_verbose():
    """Test capturing tokens with verbose=False."""
    # Create mock models
    pt_model = MockPyTorchModel()
    mlx_model = MockMLXModel()
    
    # Mock analyze_distributions to verify it's not called
    with patch('csm.mlx_accel.token_analyzer.analyze_distributions') as mock_analyze, \
         patch('builtins.print') as mock_print:
        
        # Capture tokens
        results = capture_tokens(pt_model, mlx_model, "Test text", verbose=False)
        
        # Check that print was not called
        assert not mock_print.called
        
        # analyze_distributions should not be called when verbose=False
        assert not mock_analyze.called
        
        # Check if results contain expected data
        assert 'pytorch' in results
        assert 'mlx' in results
        assert 'tokens' in results['pytorch']
        assert 'tokens' in results['mlx']


def test_capture_tokens_pytorch_error():
    """Test capturing tokens when PyTorch model raises an error."""
    # Create mock models - PyTorch model raises error
    pt_model = MockPyTorchModel(raise_error=True)
    mlx_model = MockMLXModel()
    
    # Patch print to avoid cluttering the output
    with patch('builtins.print'):
        
        # Capture tokens
        results = capture_tokens(pt_model, mlx_model, "Test text", verbose=True)
        
        # Check if PyTorch result contains error
        assert 'pytorch' in results
        assert 'error' in results['pytorch']
        assert 'Simulated PyTorch model error' in results['pytorch']['error']
        
        # MLX should still have results
        assert 'mlx' in results
        assert 'tokens' in results['mlx']


def test_capture_tokens_mlx_error():
    """Test capturing tokens when MLX model raises an error."""
    # Create mock models - MLX model raises error
    pt_model = MockPyTorchModel()
    mlx_model = MockMLXModel(raise_error=True)
    
    # Patch print to avoid cluttering the output
    with patch('builtins.print'):
        
        # Capture tokens
        results = capture_tokens(pt_model, mlx_model, "Test text", verbose=True)
        
        # PyTorch should have results
        assert 'pytorch' in results
        assert 'tokens' in results['pytorch']
        
        # Check if MLX result contains error
        assert 'mlx' in results
        assert 'error' in results['mlx']
        assert 'Simulated MLX model error' in results['mlx']['error']


def test_capture_tokens_no_tokens():
    """Test capturing tokens when models don't have tokens."""
    # Create mock models that don't return tokens
    pt_model = MockPyTorchModel(return_tokens=False)
    mlx_model = MockMLXModel(return_tokens=False)
    
    # Mock analyze_distributions to verify it's not called due to lack of tokens
    with patch('csm.mlx_accel.token_analyzer.analyze_distributions') as mock_analyze, \
         patch('builtins.print'):
        
        # Capture tokens
        results = capture_tokens(pt_model, mlx_model, "Test text", verbose=True)
        
        # analyze_distributions should not be called when tokens are not available
        assert not mock_analyze.called
        
        # Check results structure
        assert 'pytorch' in results
        assert 'mlx' in results
        assert 'tokens' in results['pytorch']
        assert 'tokens' in results['mlx']
        assert results['pytorch']['tokens'] is None
        assert results['mlx']['tokens'] is None


# Special test for compare_sampling_implementations
@pytest.mark.skip(reason="This test requires extensive mocking of model loading and runtime patching")
def test_compare_sampling_implementations():
    """Test comparing sampling implementations."""
    # This would be a complex test requiring mocking of many components
    # For now, we'll skip it as indicated by the @pytest.mark.skip decorator
    
    # In a full implementation, we would mock:
    # - csm.generator.load_csm_1b
    # - csm.cli.mlx_components.generator.MLXGenerator
    # - csm.cli.mlx_components.config.MLXConfig
    # - csm.cli.mlx_embedding.mlx_sample_topk
    # - csm.cli.mlx_sample_exact.mlx_sample_exact
    # - sys.modules patching/unpatching
    
    # We would then verify:
    # - Both implementations are called
    # - Results are saved correctly
    # - Original function is properly restored
    # - Summary is calculated correctly
    pass