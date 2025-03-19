"""
Conftest module for the components unit tests.
"""

import pytest
from unittest.mock import MagicMock

# We now use the global fixture and marker from the root conftest.py
# No need to manually mock MLX modules here anymore

@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = MagicMock()
    model.args = MagicMock()
    model.args.audio_vocab_size = 2051
    model.args.audio_num_codebooks = 32
    return model

@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3, 4]
    return tokenizer

@pytest.fixture
def mock_mlx_wrapper():
    """Create a mock MLX wrapper for testing."""
    wrapper = MagicMock()
    wrapper.generate_tokens.return_value = MagicMock()
    return wrapper