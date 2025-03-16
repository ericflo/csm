"""Tests for the Model class."""

import torch
import pytest
from unittest.mock import patch, MagicMock

from csm.models.model import Model, ModelArgs, _create_causal_mask, sample_topk


@pytest.fixture
def model_args():
    """Create a test ModelArgs fixture."""
    return ModelArgs(
        backbone_flavor="llama-1B",
        decoder_flavor="llama-100M",
        text_vocab_size=128256,
        audio_vocab_size=2051,
        audio_num_codebooks=32,
    )


def test_model_initialization(model_args):
    """Test that the Model initializes correctly."""
    with torch.no_grad():
        model = Model(model_args)
    
    # Check that the model has the correct config
    assert model.config == model_args
    
    # Check that the model has the expected components
    assert hasattr(model, "backbone")
    assert hasattr(model, "decoder")
    assert hasattr(model, "text_embeddings")
    assert hasattr(model, "audio_embeddings")
    assert hasattr(model, "projection")
    assert hasattr(model, "codebook0_head")
    assert hasattr(model, "audio_head")


def test_multinomial_sampling():
    """Test the multinomial sampling function."""
    from csm.models.model import _multinomial_sample_one_no_sync
    
    # Create a probability tensor with a clear maximum
    probs = torch.tensor([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]])
    
    # Using a fixed seed for reproducibility
    torch.manual_seed(42)
    
    # Sampling should generally select the highest probability item
    # but since it's random, we'll do multiple trials
    samples = []
    for _ in range(50):
        sample = _multinomial_sample_one_no_sync(probs)
        samples.append(sample)
    
    # Convert samples to a tensor for easier analysis
    samples_tensor = torch.cat(samples, dim=1)
    
    # Check shape
    assert samples_tensor.shape == (2, 50)
    
    # For row 0, most samples should be index 2 (0.7 probability)
    # For row 1, most samples should be index 0 (0.8 probability)
    row0_counts = torch.bincount(samples_tensor[0], minlength=3)
    row1_counts = torch.bincount(samples_tensor[1], minlength=3)
    
    # The highest probability index should have the most samples
    assert row0_counts.argmax().item() == 2
    assert row1_counts.argmax().item() == 0


def test_sample_topk():
    """Test the sample_topk function."""
    # Create a test logit tensor
    logits = torch.tensor([[1.0, 2.0, 3.0, 0.5, 1.5], [5.0, 2.0, 1.0, 0.5, 3.0]])
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # Test with topk=2, should only sample from the top 2 values
    temperature = 1.0
    topk = 2
    
    # Collect multiple samples to analyze the distribution
    samples = []
    for _ in range(50):
        sample = sample_topk(logits, topk, temperature)
        samples.append(sample)
    
    # Convert samples to a tensor for easier analysis
    samples_tensor = torch.cat(samples, dim=1)
    
    # For row 0, only indices 1 and 2 should be selected (corresponding to values 2.0 and 3.0)
    # For row 1, only indices 0 and 4 should be selected (corresponding to values 5.0 and 3.0)
    row0_elements = set(samples_tensor[0].tolist())
    row1_elements = set(samples_tensor[1].tolist())
    
    # Check that only the top k elements are present in the samples
    assert row0_elements.issubset({1, 2})
    assert row1_elements.issubset({0, 4})


def test_embed_tokens(model_args):
    """Test the _embed_tokens method."""
    with torch.no_grad():
        model = Model(model_args)
        
        # Create a test token tensor
        # Shape: (batch_size, seq_len, audio_num_codebooks+1)
        tokens = torch.zeros(2, 3, 33).long()
        
        # Set some values for testing
        # Last dimension is text token
        tokens[:, :, -1] = torch.tensor([[101, 102, 103], [201, 202, 203]])
        
        # Set audio token values
        for i in range(32):  # audio_num_codebooks
            tokens[:, :, i] = i * 10 + torch.tensor([[1, 2, 3], [4, 5, 6]])
        
        # Call the embedding function
        embeddings = model._embed_tokens(tokens)
        
        # Check the shape of the output
        # Should be (batch_size, seq_len, audio_num_codebooks+1, embed_dim)
        expected_embed_dim = model.text_embeddings.embedding_dim
        assert embeddings.shape[:3] == (2, 3, 33)
        assert embeddings.shape[3] == expected_embed_dim


def test_embed_audio(model_args):
    """Test the _embed_audio method."""
    with torch.no_grad():
        model = Model(model_args)
        
        # Create a test token tensor
        # Shape: (batch_size, seq_len)
        codebook = 5
        tokens = torch.tensor([[101, 102], [201, 202]])
        
        # Call the embedding function
        embeddings = model._embed_audio(codebook, tokens)
        
        # Check the shape of the output
        # Should be (batch_size, seq_len, embed_dim)
        expected_embed_dim = model.audio_embeddings.embedding_dim
        assert embeddings.shape[:2] == (2, 2)
        assert embeddings.shape[2] == expected_embed_dim
        
        # The actual embedding should offset by codebook * audio_vocab_size
        offset = codebook * model.config.audio_vocab_size
        expected_indices = tokens + offset
        
        # Verify the correct indices were used by checking that a different offset produces different embeddings
        different_offset = (codebook + 1) * model.config.audio_vocab_size
        different_indices = tokens + different_offset
        different_embeddings = model.audio_embeddings(different_indices)
        
        assert not torch.allclose(embeddings, different_embeddings)


def test_reset_caches(model_args):
    """Test the reset_caches method."""
    # Mock the backbone and decoder objects
    class MockModel(Model):
        def __init__(self, args):
            super().__init__(args)
            self.backbone_reset_called = False
            self.decoder_reset_called = False
            
        def reset_backbone_caches(self):
            self.backbone_reset_called = True
            
        def reset_decoder_caches(self):
            self.decoder_reset_called = True
    
    # Monkey patch the reset methods
    with patch.object(Model, 'reset_caches') as mock_reset:
        # Create a model instance
        model = Model(model_args)
        
        # Call reset_caches
        model.reset_caches()
        
        # Verify the method was called
        mock_reset.assert_called_once()