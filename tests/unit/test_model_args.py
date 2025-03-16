"""Tests for Model and ModelArgs classes."""

import pytest

from csm.models.model import ModelArgs


def test_model_args_initialization():
    """Test that ModelArgs initializes correctly with expected values."""
    args = ModelArgs(
        backbone_flavor="llama-1B",
        decoder_flavor="llama-100M",
        text_vocab_size=128256,
        audio_vocab_size=2051,
        audio_num_codebooks=32,
    )
    
    # Verify all attributes are correctly set
    assert args.backbone_flavor == "llama-1B"
    assert args.decoder_flavor == "llama-100M"
    assert args.text_vocab_size == 128256
    assert args.audio_vocab_size == 2051
    assert args.audio_num_codebooks == 32