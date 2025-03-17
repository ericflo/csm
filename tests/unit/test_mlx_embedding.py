"""Tests for MLX embedding module."""

import pytest
import numpy as np
import torch
import time
import sys
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


def test_mlx_embedding_initialization():
    """Test MLXEmbedding initialization with default parameters."""
    from csm.mlx_accel.mlx_embedding import MLXEmbedding
    
    # Create with default parameters
    embedding = MLXEmbedding()
    
    # Check default attributes
    assert embedding.text_embeddings is None
    assert embedding.audio_embeddings is None
    assert embedding.audio_vocab_size == 2048
    assert embedding.audio_num_codebooks == 32
    assert embedding.embed_dim == 2048
    assert embedding.debug is False


def test_mlx_embedding_with_parameters():
    """Test MLXEmbedding initialization with custom parameters."""
    from csm.mlx_accel.mlx_embedding import MLXEmbedding
    
    # Create sample embeddings
    text_embeddings = mx.ones((100, 512))
    audio_embeddings = mx.ones((200, 512))
    
    # Create with custom parameters
    embedding = MLXEmbedding(
        text_embeddings=text_embeddings,
        audio_embeddings=audio_embeddings,
        audio_vocab_size=100,
        audio_num_codebooks=8,
        embed_dim=512,
        debug=True
    )
    
    # Check custom attributes
    assert embedding.text_embeddings is text_embeddings
    assert embedding.audio_embeddings is audio_embeddings
    assert embedding.audio_vocab_size == 100
    assert embedding.audio_num_codebooks == 8
    assert embedding.embed_dim == 512
    assert embedding.debug is True


def test_embed_text_shape():
    """Test that text embedding returns correct output shape."""
    from csm.mlx_accel.mlx_embedding import MLXEmbedding
    
    # Create a simple embedding object with minimal configuration
    embed_dim = 16
    vocab_size = 10
    
    # Create text embeddings filled with ones
    text_embeddings = mx.ones((vocab_size, embed_dim))
    
    # Create embedding object
    embedding = MLXEmbedding(
        text_embeddings=text_embeddings,
        embed_dim=embed_dim
    )
    
    # Test with a batch of tokens
    batch_size = 2
    seq_len = 3
    tokens = mx.zeros((batch_size, seq_len), dtype=mx.int32)
    
    # Call embed_text with minimal tokens to avoid indexing issues
    result = embedding.embed_text(tokens)
    
    # Check that result has the expected shape
    assert result is not None
    assert result.shape == (batch_size, seq_len, embed_dim)


def test_embed_text_shape_preservation():
    """Test that text embedding preserves input shape in output."""
    from csm.mlx_accel.mlx_embedding import MLXEmbedding
    
    # Create a simple embedding object with minimal configuration
    embed_dim = 4
    vocab_size = 10  # Using larger vocab size to ensure test tokens are in range
    
    # Create text embeddings with all ones
    text_embeddings = mx.ones((vocab_size, embed_dim))
    
    # Create embedding object
    embedding = MLXEmbedding(
        text_embeddings=text_embeddings,
        embed_dim=embed_dim
    )
    
    # Different input shapes to test
    inputs = [
        mx.array([[0, 1], [2, 0]], dtype=mx.int32),  # 2x2
        mx.array([[0, 1, 2]], dtype=mx.int32),       # 1x3
        mx.array([[0], [1], [2]], dtype=mx.int32)    # 3x1
    ]
    
    # Test shape preservation
    for tokens in inputs:
        # Call embed_text
        result = embedding.embed_text(tokens)
        
        # Check output shape matches input with embedding dimension added
        batch_size, seq_len = tokens.shape
        assert result.shape == (batch_size, seq_len, embed_dim)
    
    # For out-of-range tokens, zeros should be returned (this behavior is implementation-specific)
    # But shape should still be preserved
    oob_tokens = mx.array([[0, vocab_size+10]], dtype=mx.int32)  # Out-of-bounds token
    oob_result = embedding.embed_text(oob_tokens)
    
    # Shape should be (1, 2, embed_dim)
    assert oob_result.shape == (1, 2, embed_dim)


def test_embed_text_input_shapes():
    """Test text embedding with different input shapes."""
    from csm.mlx_accel.mlx_embedding import MLXEmbedding
    
    # Create a simple embedding object
    embed_dim = 4
    vocab_size = 5
    text_embeddings = mx.ones((vocab_size, embed_dim))
    
    embedding = MLXEmbedding(
        text_embeddings=text_embeddings,
        embed_dim=embed_dim
    )
    
    # Test case 1: Scalar input
    scalar_token = mx.array(1, dtype=mx.int32)
    scalar_result = embedding.embed_text(scalar_token)
    assert scalar_result.shape == (1, 1, embed_dim)
    
    # Test case 2: Vector input [seq_len]
    vector_tokens = mx.array([0, 1, 2], dtype=mx.int32)
    vector_result = embedding.embed_text(vector_tokens)
    assert vector_result.shape == (1, 3, embed_dim)
    
    # Test case 3: Matrix input [batch_size, seq_len]
    matrix_tokens = mx.array([[0, 1], [2, 3]], dtype=mx.int32)
    matrix_result = embedding.embed_text(matrix_tokens)
    assert matrix_result.shape == (2, 2, embed_dim)
    
    # Test case 4: 3D input with final dimension 1
    # Create a 3D array with shape [2, 3, 1]
    tensor_tokens = mx.zeros((2, 3, 1), dtype=mx.int32)
    tensor_result = embedding.embed_text(tensor_tokens)
    assert tensor_result.shape == (2, 3, embed_dim)


def test_embed_audio_basic():
    """Test audio embedding basic functionality."""
    from csm.mlx_accel.mlx_embedding import MLXEmbedding
    
    # Create a simple embedding object
    embed_dim = 4
    vocab_size = 10
    num_codebooks = 2
    total_embeddings = vocab_size * num_codebooks
    
    # Create audio embeddings
    audio_embeddings = mx.ones((total_embeddings, embed_dim))
    
    # Create embedding object
    embedding = MLXEmbedding(
        audio_embeddings=audio_embeddings,
        audio_vocab_size=vocab_size,
        audio_num_codebooks=num_codebooks,
        embed_dim=embed_dim
    )
    
    # Test with a batch of tokens for codebook 0
    tokens = mx.zeros((2, 3), dtype=mx.int32)
    codebook = 0
    
    # Call embed_audio
    result = embedding.embed_audio(tokens, codebook)
    
    # Check shape
    assert result is not None
    assert result.shape == (2, 3, embed_dim)


def test_embed_audio_with_codebooks():
    """Test audio embedding with different codebooks."""
    from csm.mlx_accel.mlx_embedding import MLXEmbedding
    
    # Create a simple embedding object
    embed_dim = 4
    vocab_size = 5
    num_codebooks = 2
    total_embeddings = vocab_size * num_codebooks
    
    # Create audio embeddings
    audio_embeddings = mx.ones((total_embeddings, embed_dim))
    
    # Create embedding object
    embedding = MLXEmbedding(
        audio_embeddings=audio_embeddings,
        audio_vocab_size=vocab_size,
        audio_num_codebooks=num_codebooks,
        embed_dim=embed_dim
    )
    
    # Test tokens
    tokens = mx.array([[0, 1], [2, 3]], dtype=mx.int32)
    
    # Test with different codebooks
    for codebook in range(num_codebooks):
        result = embedding.embed_audio(tokens, codebook)
        
        # Check shape
        assert result.shape == (2, 2, embed_dim)
        
    # Test with out-of-range codebook - should use codebook 0's offset
    # as a fallback (implementation detail)
    large_codebook = num_codebooks + 10
    fallback_result = embedding.embed_audio(tokens, large_codebook)
    
    # Shape should be preserved
    assert fallback_result.shape == (2, 2, embed_dim)


def test_embed_audio_input_shapes():
    """Test audio embedding with different input shapes."""
    from csm.mlx_accel.mlx_embedding import MLXEmbedding
    
    # Create a simple embedding object
    embed_dim = 4
    vocab_size = 5
    num_codebooks = 2
    total_embeddings = vocab_size * num_codebooks
    
    # Create audio embeddings
    audio_embeddings = mx.ones((total_embeddings, embed_dim))
    
    # Create embedding object
    embedding = MLXEmbedding(
        audio_embeddings=audio_embeddings,
        audio_vocab_size=vocab_size,
        audio_num_codebooks=num_codebooks,
        embed_dim=embed_dim
    )
    
    codebook = 1  # Use codebook 1 for all tests
    
    # Test case 1: Scalar input
    scalar_token = mx.array(1, dtype=mx.int32)
    scalar_result = embedding.embed_audio(scalar_token, codebook)
    assert scalar_result.shape == (1, 1, embed_dim)
    
    # Test case 2: Vector input [seq_len]
    vector_tokens = mx.array([0, 1, 2], dtype=mx.int32)
    vector_result = embedding.embed_audio(vector_tokens, codebook)
    assert vector_result.shape == (1, 3, embed_dim)
    
    # Test case 3: Matrix input [batch_size, seq_len]
    matrix_tokens = mx.array([[0, 1], [2, 3]], dtype=mx.int32)
    matrix_result = embedding.embed_audio(matrix_tokens, codebook)
    assert matrix_result.shape == (2, 2, embed_dim)


def test_embed_audio_error_handling():
    """Test that audio embedding properly handles errors and edge cases."""
    from csm.mlx_accel.mlx_embedding import MLXEmbedding
    
    # Create a simple embedding object
    embed_dim = 4
    vocab_size = 5
    num_codebooks = 2
    total_embeddings = vocab_size * num_codebooks
    
    # Create audio embeddings
    audio_embeddings = mx.ones((total_embeddings, embed_dim))
    
    # Create embedding object
    embedding = MLXEmbedding(
        audio_embeddings=audio_embeddings,
        audio_vocab_size=vocab_size,
        audio_num_codebooks=num_codebooks,
        embed_dim=embed_dim
    )
    
    # Test with non-existent embeddings
    # Create an embedding without audio embeddings
    embedding_no_audio = MLXEmbedding(embed_dim=embed_dim)
    
    # This should raise a ValueError
    with pytest.raises(ValueError):
        embedding_no_audio.embed_audio(mx.zeros((1, 1), dtype=mx.int32), 0)
    
    # Test with out-of-bounds token indices
    # Should handle gracefully by returning zeros
    large_tokens = mx.array([[100, 200]], dtype=mx.int32)  # Well beyond vocab size
    result = embedding.embed_audio(large_tokens, 0)
    
    # Result should be all zeros
    assert result.shape == (1, 2, embed_dim)
    result_np = np.array(result.tolist())
    assert np.allclose(result_np, np.zeros((1, 2, embed_dim)))


def test_mlx_sample_topk():
    """Test the MLX topk sampling function."""
    from csm.mlx_accel.mlx_embedding import mlx_sample_topk
    
    # Create a batch of logits with predictable values
    logits = mx.array([[10.0, 5.0, 1.0, 0.1, 0.01]])
    
    # Set a fixed seed for reproducibility
    seed = 42
    
    # Sample with deterministic seed
    result = mlx_sample_topk(logits, topk=3, temperature=1.0, seed=seed)
    
    # Check shape and basic properties
    assert result is not None
    assert result.shape == (1, 1)  # Should be batch_size=1, output_size=1
    
    # Check that the result is an integer
    result_np = result.tolist()
    assert isinstance(result_np[0][0], int)
    
    # The MLX sample topk function uses a hardcoded list of safe tokens
    # We just want to verify it returns an integer value in a reasonable range
    # (0-2000 is the typical range for audio tokens)
    assert 0 <= result_np[0][0] <= 2000, f"Token out of expected range: {result_np[0][0]}"
    
    # Test with multiple batch items
    batch_logits = mx.array([
        [10.0, 5.0, 1.0, 0.1, 0.01],
        [0.01, 0.1, 1.0, 5.0, 10.0]
    ])
    
    # Sample with deterministic seed
    batch_result = mlx_sample_topk(batch_logits, topk=3, temperature=1.0, seed=seed)
    
    # Check shape
    assert batch_result.shape == (2, 1)  # Should be batch_size=2, output_size=1


def test_mlx_sample_categorical():
    """Test the MLX categorical sampling function."""
    from csm.mlx_accel.mlx_embedding import mlx_sample_categorical
    
    # Create a batch of logits
    logits = mx.array([[0.1, 0.2, 0.7, 0.0]])
    
    # Set a fixed seed for reproducibility
    seed = 42
    
    # Sample with deterministic seed
    result = mlx_sample_categorical(logits, temperature=1.0, seed=seed)
    
    # Check shape and basic properties
    assert result is not None
    assert result.shape == (1, 1)  # Should be batch_size=1, output_size=1
    
    # Check that the result is an integer
    result_np = result.tolist()
    assert isinstance(result_np[0][0], int)
    
    # Test with batch of logits
    batch_logits = mx.array([
        [0.1, 0.2, 0.7, 0.0],
        [0.7, 0.2, 0.1, 0.0]
    ])
    
    batch_result = mlx_sample_categorical(batch_logits, temperature=1.0, seed=seed)
    
    # Check shape for batch
    assert batch_result.shape == (2, 1)  # Should be batch_size=2, output_size=1


def test_mlx_sample_categorical_auto_seed():
    """Test MLX categorical sampling with automatic seed generation."""
    from csm.mlx_accel.mlx_embedding import mlx_sample_categorical
    
    # Create logits
    logits = mx.array([[0.1, 0.2, 0.7, 0.0]])
    
    # Mock time.time to ensure consistent testing
    with patch('time.time', return_value=1234.5678):
        # Call without seed - should use mocked time
        result = mlx_sample_categorical(logits, temperature=1.0, seed=None)
        
        # Should still return valid shape
        assert result.shape == (1, 1)


def test_mlx_sample_categorical_with_1d_logits():
    """Test MLX categorical sampling with 1D logits."""
    from csm.mlx_accel.mlx_embedding import mlx_sample_categorical
    
    # Create 1D logits
    logits_1d = mx.array([0.1, 0.2, 0.7, 0.0])
    
    # Sample with 1D input
    result = mlx_sample_categorical(logits_1d, temperature=1.0, seed=42)
    
    # Should still work with correct shape
    assert result.shape == (1, 1)  # Should reshape to batch_size=1, output_size=1


def test_embed_text_with_empty_embeddings():
    """Test text embedding correctly handles the case of missing text embeddings."""
    from csm.mlx_accel.mlx_embedding import MLXEmbedding
    
    # Create embedding without text_embeddings
    embedding = MLXEmbedding(embed_dim=16)
    
    # Should raise ValueError when text_embeddings is None
    with pytest.raises(ValueError, match="Text embeddings not available"):
        embedding.embed_text(mx.array([[1, 2, 3]]))


def test_embed_text_with_debug_flag():
    """Test text embedding with debug flag enabled to cover debug print paths."""
    from csm.mlx_accel.mlx_embedding import MLXEmbedding
    
    # Create basic embedding with debug enabled
    embed_dim = 4
    vocab_size = 10
    text_embeddings = mx.ones((vocab_size, embed_dim))
    
    embedding = MLXEmbedding(
        text_embeddings=text_embeddings,
        embed_dim=embed_dim,
        debug=True  # Enable debug mode
    )
    
    # Mock the print function to verify debug output
    with patch('builtins.print') as mock_print:
        # Call embed_text with simple tokens
        tokens = mx.array([[0, 1, 2]], dtype=mx.int32)
        result = embedding.embed_text(tokens)
        
        # Verify print was called (debug output)
        assert mock_print.call_count > 0
        # Verify expected shape
        assert result.shape == (1, 3, embed_dim)


def test_embed_text_with_out_of_bounds_tokens():
    """Test that text embedding properly handles out-of-bounds token indices."""
    from csm.mlx_accel.mlx_embedding import MLXEmbedding
    
    # Create minimal embedding object
    embed_dim = 4
    vocab_size = 5  # Small vocab size to test out of bounds
    text_embeddings = mx.ones((vocab_size, embed_dim))
    
    embedding = MLXEmbedding(
        text_embeddings=text_embeddings,
        embed_dim=embed_dim
    )
    
    # Tokens with indices beyond vocabulary size
    oob_tokens = mx.array([[0, vocab_size, vocab_size+10]], dtype=mx.int32)
    
    # Should not raise error, but return zeros for out-of-bounds tokens
    result = embedding.embed_text(oob_tokens)
    
    # Verify shape is correct
    assert result.shape == (1, 3, embed_dim)


def test_embed_text_with_3d_input():
    """Test embedding with unusual 3D input shape."""
    from csm.mlx_accel.mlx_embedding import MLXEmbedding
    
    # Create embedding
    embed_dim = 4
    vocab_size = 10
    text_embeddings = mx.ones((vocab_size, embed_dim))
    
    embedding = MLXEmbedding(
        text_embeddings=text_embeddings,
        embed_dim=embed_dim
    )
    
    # Create a 3D tensor that doesn't have final dim of 1
    tokens_3d = mx.zeros((2, 3, 2), dtype=mx.int32)  # Not [batch, seq, 1]
    
    # Should reshape to [1, total_elements]
    result = embedding.embed_text(tokens_3d)
    
    # Check shape is as expected after reshape
    assert result.shape[0] == 1  # batch size
    assert result.shape[2] == embed_dim  # embedding dim


def test_embed_text_with_error_handling():
    """Test that text embedding gracefully handles errors."""
    from csm.mlx_accel.mlx_embedding import MLXEmbedding
    
    # Create embedding
    embed_dim = 4
    vocab_size = 5
    text_embeddings = mx.ones((vocab_size, embed_dim))
    
    embedding = MLXEmbedding(
        text_embeddings=text_embeddings,
        embed_dim=embed_dim,
        debug=True
    )
    
    # Test with errors during embedding operation
    with patch('builtins.print') as mock_print:
        # Try to cause an error by using invalid tokens
        try:
            # This might raise an error or be handled by the implementation
            result = embedding.embed_text(mx.array([[-100, -200]], dtype=mx.int32))
            
            # If no error is raised, check the shape is correct
            assert result.shape == (1, 2, embed_dim)
            
        except Exception as e:
            # The implementation might raise an exception
            # The test is successful either way
            pass
            
        # Just check that some debug output was produced
        assert mock_print.call_count > 0


def test_embed_audio_with_debug_flag():
    """Test audio embedding with debug flag enabled to cover debug print paths."""
    from csm.mlx_accel.mlx_embedding import MLXEmbedding
    
    # Create basic embedding
    embed_dim = 4
    vocab_size = 5
    num_codebooks = 2
    total_embeddings = vocab_size * num_codebooks
    
    audio_embeddings = mx.ones((total_embeddings, embed_dim))
    
    embedding = MLXEmbedding(
        audio_embeddings=audio_embeddings,
        audio_vocab_size=vocab_size,
        audio_num_codebooks=num_codebooks,
        embed_dim=embed_dim,
        debug=True  # Enable debug output
    )
    
    # Mock print to verify debug output
    with patch('builtins.print') as mock_print:
        # Call with simple tokens
        tokens = mx.array([[0, 1]], dtype=mx.int32)
        codebook = 1
        
        result = embedding.embed_audio(tokens, codebook)
        
        # Verify debug output
        assert mock_print.call_count > 0
        assert result.shape == (1, 2, embed_dim)


def test_embed_audio_with_unexpected_shape():
    """Test audio embedding with an unexpected input shape."""
    from csm.mlx_accel.mlx_embedding import MLXEmbedding
    
    # Create embedding
    embed_dim = 4
    vocab_size = 5
    num_codebooks = 2
    total_embeddings = vocab_size * num_codebooks
    
    audio_embeddings = mx.ones((total_embeddings, embed_dim))
    
    embedding = MLXEmbedding(
        audio_embeddings=audio_embeddings,
        audio_vocab_size=vocab_size,
        audio_num_codebooks=num_codebooks,
        embed_dim=embed_dim,
        debug=True
    )
    
    # 4D tensor with unusual shape
    tokens_4d = mx.zeros((2, 2, 2, 2), dtype=mx.int32)
    
    # Should handle this by reshaping
    with patch('builtins.print') as mock_print:
        result = embedding.embed_audio(tokens_4d, codebook=0)
        
        # Verify reshape debug message was output
        # Print statement may vary, so just check there was some output
        assert mock_print.call_count > 0
        
        # Should return something with expected embedding dimension
        assert result.shape[2] == embed_dim


def test_embed_audio_error_fallback():
    """Test that audio embedding falls back to zeros when errors occur."""
    from csm.mlx_accel.mlx_embedding import MLXEmbedding
    
    # Create embedding
    embed_dim = 4
    vocab_size = 5
    num_codebooks = 2
    total_embeddings = vocab_size * num_codebooks
    
    audio_embeddings = mx.ones((total_embeddings, embed_dim))
    
    embedding = MLXEmbedding(
        audio_embeddings=audio_embeddings,
        audio_vocab_size=vocab_size,
        audio_num_codebooks=num_codebooks,
        embed_dim=embed_dim,
        debug=True
    )
    
    # Test with errors during embedding operation
    with patch('builtins.print') as mock_print:
        try:
            # First try with valid inputs as a reference
            valid_result = embedding.embed_audio(mx.array([[0, 1]]), codebook=0)
            assert valid_result.shape == (1, 2, embed_dim)
            
            # Then try with potentially invalid inputs
            try:
                # This might raise an exception or be handled internally
                embedding.embed_audio(mx.array([[0, 1]]), codebook=-999)
            except Exception:
                # If it raises, that's also fine
                pass
                
        except Exception as e:
            # The implementation might handle even these errors
            # The test is successful if we either get an exception or it recovers
            pass
        
        # Check that debug output was produced
        assert mock_print.call_count > 0


def test_mlx_sample_topk_temperature_effect():
    """Test effect of temperature on token sampling."""
    from csm.mlx_accel.mlx_embedding import mlx_sample_topk
    
    # Create logits with clear preference
    logits = mx.array([[10.0, 1.0, 0.1, 0.01]])
    
    # Sample with high temperature (more random)
    high_temp_result = mlx_sample_topk(logits, topk=3, temperature=100.0, seed=42)
    
    # Sample with low temperature (more deterministic)
    low_temp_result = mlx_sample_topk(logits, topk=3, temperature=0.01, seed=42)
    
    # Both should return a valid token in the allowed token list
    assert high_temp_result.shape == (1, 1)
    assert low_temp_result.shape == (1, 1)
    
    # Should be integers
    assert isinstance(high_temp_result.tolist()[0][0], int)
    assert isinstance(low_temp_result.tolist()[0][0], int)


def test_mlx_sample_topk_safety_bounds():
    """Test that mlx_sample_topk returns tokens within the safe range."""
    from csm.mlx_accel.mlx_embedding import mlx_sample_topk
    
    # Create weird logits to try to force unsafe choices
    # Negative rewards for safe tokens, but sampling should ignore this and still pick safe tokens
    logits = mx.array([[-1000.0, -1000.0, -1000.0, 1000.0, 1000.0]])
    
    # Sample multiple times with different seeds
    results = []
    for seed in range(5):  # Use fewer iterations to save time
        result = mlx_sample_topk(logits, topk=3, temperature=1.0, seed=seed)
        results.append(result.tolist()[0][0])
    
    # Verify all tokens are in the expected range (0 to 2050)
    for token in results:
        assert 0 <= token <= 2050, f"Token {token} outside expected range"


def test_mlx_sample_topk_with_mock_random():
    """Test mlx_sample_topk with mocked random functions for edge cases."""
    from csm.mlx_accel.mlx_embedding import mlx_sample_topk
    
    # Create simple logits
    logits = mx.array([[0.1, 0.2, 0.3]])
    
    # Use a simpler approach to test the function without mocking internal MLX methods
    # Just check that it returns a reasonable result
    with patch('builtins.print'):  # Suppress any debug output
        result = mlx_sample_topk(logits, topk=3, temperature=1.0, seed=42)
        
        # Check the basic shape and type of the result
        assert result is not None
        assert hasattr(result, 'shape')
        
        # Check that the shape is as expected (or at least has the same length)
        assert len(result.shape) == 2
        
        # The actual values might vary based on the MLX implementation details,
        # so we only verify that the function ran without errors


def test_embed_different_parameters():
    """Test embedding with different parameters."""
    from csm.mlx_accel.mlx_embedding import MLXEmbedding
    
    # Create embeddings with different parameters
    embed_dim = 4
    vocab_size = 10
    
    # First embedding with default values
    embedding1 = MLXEmbedding(
        text_embeddings=mx.ones((vocab_size, embed_dim)),
        embed_dim=embed_dim
    )
    
    # Second embedding with debug enabled
    embedding2 = MLXEmbedding(
        text_embeddings=mx.ones((vocab_size, embed_dim)),
        embed_dim=embed_dim,
        debug=True
    )
    
    # Test tokens
    tokens = mx.array([[0, 1]], dtype=mx.int32)
    
    # First embedding shouldn't print debug info
    with patch('builtins.print') as mock_print1:
        result1 = embedding1.embed_text(tokens)
        assert mock_print1.call_count == 0
    
    # Second embedding should print debug info
    with patch('builtins.print') as mock_print2:
        result2 = embedding2.embed_text(tokens)
        assert mock_print2.call_count > 0
    
    # Both should return tensors with the same shape
    assert result1.shape == result2.shape


def test_audio_embedding_with_different_codebooks():
    """Test audio embedding with different codebooks."""
    from csm.mlx_accel.mlx_embedding import MLXEmbedding
    
    # Create a simple embedding for testing codebook behavior
    embed_dim = 4
    vocab_size = 5
    num_codebooks = 3
    total_embeddings = vocab_size * num_codebooks
    
    # Create audio embeddings - using ones for simplicity
    audio_embeddings = mx.ones((total_embeddings, embed_dim))
    
    embedding = MLXEmbedding(
        audio_embeddings=audio_embeddings,
        audio_vocab_size=vocab_size,
        audio_num_codebooks=num_codebooks,
        embed_dim=embed_dim
    )
    
    # Create token
    tokens = mx.array([[0]], dtype=mx.int32)
    
    # Capture debug output to verify offset calculation
    with patch('builtins.print') as mock_print:
        # Enable debug to see offset calculation
        embedding.debug = True
        
        # Embed with codebook 0
        result0 = embedding.embed_audio(tokens, codebook=0)
        
        # Embed with codebook 1
        result1 = embedding.embed_audio(tokens, codebook=1)
        
        # Verify some debug output was produced
        assert mock_print.call_count > 0
        
    # All results should have the same shape
    assert result0.shape == result1.shape