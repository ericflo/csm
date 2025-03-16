"""Tests for MLX embedding module."""

import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock

# Skip tests only if MLX is not available
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    
pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


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