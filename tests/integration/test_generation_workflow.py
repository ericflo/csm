"""Integration tests for generation workflow.

These tests simulate the basic workflow of the text-to-speech generation
process without requiring actual model files or computation.
"""

import pytest
import torch
from unittest.mock import patch, MagicMock

# Test fixture to simulate a model
@pytest.fixture
def mock_model():
    """Create a mock Model instance."""
    mock = MagicMock()
    # Mock the generate_frame method to return a tensor of shape [batch_size, audio_num_codebooks]
    mock.generate_frame.return_value = torch.zeros(1, 32).long()
    # Mock the reset_caches method
    mock.reset_caches = MagicMock()
    # Add a config attribute
    mock.config = MagicMock()
    mock.config.audio_num_codebooks = 32
    mock.config.audio_vocab_size = 2051
    
    return mock


# Test fixture to simulate a tokenizer
@pytest.fixture
def mock_text_tokenizer():
    """Create a mock text tokenizer."""
    mock = MagicMock()
    # Mock the encode method to return a list of token IDs
    mock.encode.return_value = [101, 102, 103, 104, 105]
    return mock


# Test fixture to simulate an audio tokenizer
@pytest.fixture
def mock_audio_tokenizer():
    """Create a mock audio tokenizer."""
    mock = MagicMock()
    # Mock the encode method to return a tensor of shape [codebooks, sequence_length]
    mock.encode.return_value = [torch.zeros(32, 10)]
    # Mock the decode method to return a tensor of shape [batch_size, audio_length]
    mock.decode.return_value = torch.zeros(1, 24000)
    # Add sample_rate attribute
    mock.sample_rate = 24000
    return mock


# Test fixture to simulate a watermarker
@pytest.fixture
def mock_watermarker():
    """Create a mock watermarker."""
    mock = MagicMock()
    return mock


@pytest.mark.skip(reason="Requires mocked dependencies")
def test_generation_workflow(mock_model, mock_text_tokenizer, mock_audio_tokenizer, mock_watermarker):
    """Test the basic generation workflow."""
    # Mock the watermark function to return the input audio and a sample rate
    mock_watermark.return_value = (torch.zeros(24000), 24000)
    
    # Import Generator here to avoid issues with actual dependencies
    with patch("csm.generator.load_llama3_tokenizer", return_value=mock_text_tokenizer), \
         patch("huggingface_hub.hf_hub_download", return_value="mock_path"), \
         patch("moshi.models.loaders.get_mimi", return_value=mock_audio_tokenizer), \
         patch("csm.watermarking.utils.load_watermarker", return_value=mock_watermarker), \
         patch("torchaudio.functional.resample", return_value=torch.zeros(24000)):
        
        from csm.generator import Generator, Segment
        
        # Create a Generator instance
        generator = Generator(mock_model)
        
        # Verify that generator attributes are correctly set
        assert generator._model == mock_model
        assert generator._text_tokenizer == mock_text_tokenizer
        assert generator._audio_tokenizer == mock_audio_tokenizer
        assert generator._watermarker == mock_watermarker
        assert generator.sample_rate == 24000
        
        # Test generate method with minimal context
        text = "Hello, world!"
        speaker = 0
        context = []
        
        # Generate audio
        audio = generator.generate(text=text, speaker=speaker, context=context)
        
        # Verify the model's reset_caches was called
        mock_model.reset_caches.assert_called_once()
        
        # Verify the text tokenizer was called with the correct text
        mock_text_tokenizer.encode.assert_called_with(f"[{speaker}]{text}")
        
        # Verify the model's generate_frame was called at least once
        assert mock_model.generate_frame.call_count > 0
        
        # Verify the audio tokenizer's decode was called
        mock_audio_tokenizer.decode.assert_called_once()
        
        # Verify the watermark function was called
        mock_watermark.assert_called_once()
        
        # Verify the result is a tensor
        assert isinstance(audio, torch.Tensor)


@pytest.mark.skip(reason="Requires mocked dependencies")
def test_generation_with_context(mock_model, mock_text_tokenizer, mock_audio_tokenizer, mock_watermarker):
    """Test generation with conversation context."""
    # Mock the watermark function to return the input audio and a sample rate
    mock_watermark.return_value = (torch.zeros(24000), 24000)
    
    # Import Generator here to avoid issues with actual dependencies
    with patch("csm.generator.load_llama3_tokenizer", return_value=mock_text_tokenizer), \
         patch("huggingface_hub.hf_hub_download", return_value="mock_path"), \
         patch("moshi.models.loaders.get_mimi", return_value=mock_audio_tokenizer), \
         patch("csm.watermarking.utils.load_watermarker", return_value=mock_watermarker), \
         patch("torchaudio.functional.resample", return_value=torch.zeros(24000)):
        
        from csm.generator import Generator, Segment
        
        # Create a Generator instance
        generator = Generator(mock_model)
        
        # Create context segments
        context = [
            Segment(
                text="Hi there, how are you?",
                speaker=0,
                audio=torch.zeros(12000)
            ),
            Segment(
                text="I'm doing well, thanks!",
                speaker=1,
                audio=torch.zeros(10000)
            )
        ]
        
        # Test generate method with context
        text = "That's great to hear!"
        speaker = 0
        
        # Reset mocks for this test
        mock_model.reset_caches.reset_mock()
        mock_text_tokenizer.encode.reset_mock()
        mock_audio_tokenizer.encode.reset_mock()
        mock_audio_tokenizer.decode.reset_mock()
        mock_watermark.reset_mock()
        
        # Generate audio
        audio = generator.generate(text=text, speaker=speaker, context=context)
        
        # Verify the model's reset_caches was called
        mock_model.reset_caches.assert_called_once()
        
        # Verify the text tokenizer was called multiple times (for context and current text)
        assert mock_text_tokenizer.encode.call_count == 3
        
        # Verify the audio tokenizer's encode was called multiple times (for context segments)
        assert mock_audio_tokenizer.encode.call_count == 2
        
        # Verify the audio tokenizer's decode was called
        mock_audio_tokenizer.decode.assert_called_once()
        
        # Verify the watermark function was called
        mock_watermark.assert_called_once()
        
        # Verify the result is a tensor
        assert isinstance(audio, torch.Tensor)