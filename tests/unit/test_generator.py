"""Tests for the Generator class."""

import pytest
import torch
from unittest.mock import patch, MagicMock

from csm.generator import Generator, Segment
from csm.models.model import Model, ModelArgs


@pytest.fixture
def mock_model():
    """Create a mock Model instance."""
    mock = MagicMock(spec=Model)
    # Set up mocks for the model's methods
    mock.reset_caches = MagicMock()
    mock.generate_frame = MagicMock()
    mock.generate_frame.return_value = torch.zeros(1, 32).long()
    mock.config = MagicMock()
    mock.config.audio_num_codebooks = 32
    mock.config.audio_vocab_size = 2051
    return mock


@pytest.fixture
def mock_text_tokenizer():
    """Create a mock text tokenizer."""
    mock = MagicMock()
    # Mock the encode method to return a list of token IDs
    mock.encode.return_value = [101, 102, 103, 104, 105]
    return mock


@pytest.fixture
def mock_audio_tokenizer():
    """Create a mock audio tokenizer."""
    mock = MagicMock()
    # Mock the encode method
    mock.encode.return_value = [torch.zeros(32, 10)]
    # Mock the decode method
    mock.decode.return_value = torch.zeros(1, 24000)
    # Add sample_rate attribute
    mock.sample_rate = 24000
    return mock


@pytest.fixture
def mock_watermarker():
    """Create a mock watermarker."""
    mock = MagicMock()
    return mock


@pytest.mark.skip(reason="Requires mocked dependencies")
def test_generator_initialization(mock_model, mock_text_tokenizer, mock_audio_tokenizer, mock_watermarker):
    """Test initialization of the Generator class."""
    # Set up mocks
    mock_load_tokenizer.return_value = mock_text_tokenizer
    mock_hf_hub_download.return_value = "mock_path"
    mock_get_mimi.return_value = mock_audio_tokenizer
    mock_load_watermarker.return_value = mock_watermarker
    mock_watermark.return_value = (torch.zeros(24000), 24000)
    mock_resample.return_value = torch.zeros(24000)
    
    # Initialize the generator
    generator = Generator(mock_model)
    
    # Verify that generator attributes are correctly set
    assert generator._model == mock_model
    assert generator._text_tokenizer == mock_text_tokenizer
    assert generator._audio_tokenizer == mock_audio_tokenizer
    assert generator._watermarker == mock_watermarker
    assert generator.sample_rate == 24000


@pytest.mark.skip(reason="Requires mocked dependencies")
def test_tokenize_text_segment(mock_model, mock_text_tokenizer, mock_audio_tokenizer, mock_watermarker):
    """Test tokenizing a text segment."""
    # Set up mocks
    mock_load_tokenizer.return_value = mock_text_tokenizer
    mock_hf_hub_download.return_value = "mock_path"
    mock_get_mimi.return_value = mock_audio_tokenizer
    mock_load_watermarker.return_value = mock_watermarker
    mock_watermark.return_value = (torch.zeros(24000), 24000)
    mock_resample.return_value = torch.zeros(24000)
    
    # Initialize the generator
    generator = Generator(mock_model)
    
    # Call the tokenize method
    text = "Hello, world!"
    speaker = 0
    tokens, masks = generator._tokenize_text_segment(text, speaker)
    
    # Verify tokenizer was called correctly
    mock_text_tokenizer.encode.assert_called_with(f"[{speaker}]{text}")
    
    # Check the shapes of tokens and masks
    assert tokens.shape[1] == 33  # audio_num_codebooks + 1
    assert masks.shape[1] == 33
    
    # Check that masks are set correctly - text should be at the last dimension
    assert torch.all(masks[:, -1] == True)
    assert torch.all(masks[:, :-1] == False)


@pytest.mark.skip(reason="Requires mocked dependencies")
def test_tokenize_audio(mock_model, mock_text_tokenizer, mock_audio_tokenizer, mock_watermarker):
    """Test tokenizing an audio segment."""
    # Set up mocks
    mock_load_tokenizer.return_value = mock_text_tokenizer
    mock_hf_hub_download.return_value = "mock_path"
    mock_get_mimi.return_value = mock_audio_tokenizer
    mock_load_watermarker.return_value = mock_watermarker
    mock_watermark.return_value = (torch.zeros(24000), 24000)
    mock_resample.return_value = torch.zeros(24000)
    
    # Initialize the generator
    generator = Generator(mock_model)
    
    # Set up test audio
    audio = torch.zeros(24000)
    
    # Call the tokenize method
    tokens, masks = generator._tokenize_audio(audio)
    
    # Verify audio tokenizer was called correctly
    mock_audio_tokenizer.encode.assert_called_once()
    
    # Check the shapes of tokens and masks
    assert tokens.shape[1] == 33  # audio_num_codebooks + 1
    assert masks.shape[1] == 33
    
    # Check that masks are set correctly - audio should be at all but the last dimension
    assert torch.all(masks[:, :-1] == True)
    assert torch.all(masks[:, -1] == False)


@pytest.mark.skip(reason="Requires mocked dependencies")
def test_tokenize_segment(mock_model, mock_text_tokenizer, mock_audio_tokenizer, mock_watermarker):
    """Test tokenizing a complete segment."""
    # Set up mocks
    mock_load_tokenizer.return_value = mock_text_tokenizer
    mock_hf_hub_download.return_value = "mock_path"
    mock_get_mimi.return_value = mock_audio_tokenizer
    mock_load_watermarker.return_value = mock_watermarker
    mock_watermark.return_value = (torch.zeros(24000), 24000)
    mock_resample.return_value = torch.zeros(24000)
    
    # Initialize the generator
    generator = Generator(mock_model)
    
    # Create a test segment
    segment = Segment(
        text="Hello, world!",
        speaker=0,
        audio=torch.zeros(24000)
    )
    
    # Mock the individual tokenize methods
    text_tokens = torch.zeros(5, 33)
    text_masks = torch.zeros(5, 33)
    text_masks[:, -1] = 1
    
    audio_tokens = torch.zeros(10, 33)
    audio_masks = torch.zeros(10, 33)
    audio_masks[:, :-1] = 1
    
    generator._tokenize_text_segment = MagicMock(return_value=(text_tokens, text_masks))
    generator._tokenize_audio = MagicMock(return_value=(audio_tokens, audio_masks))
    
    # Call the tokenize_segment method
    tokens, masks = generator._tokenize_segment(segment)
    
    # Verify the individual tokenize methods were called correctly
    generator._tokenize_text_segment.assert_called_with(segment.text, segment.speaker)
    generator._tokenize_audio.assert_called_with(segment.audio)
    
    # Check the shapes of tokens and masks
    assert tokens.shape == (15, 33)  # 5 + 10
    assert masks.shape == (15, 33)


@pytest.mark.skip(reason="Requires mocked dependencies")
def test_generate(mock_model, mock_text_tokenizer, mock_audio_tokenizer, mock_watermarker):
    """Test the generate method."""
    # Set up mocks
    mock_load_tokenizer.return_value = mock_text_tokenizer
    mock_hf_hub_download.return_value = "mock_path"
    mock_get_mimi.return_value = mock_audio_tokenizer
    mock_load_watermarker.return_value = mock_watermarker
    mock_watermark.return_value = (torch.zeros(24000), 24000)
    mock_resample.return_value = torch.zeros(24000)
    
    # Initialize the generator
    generator = Generator(mock_model)
    
    # Set up test parameters
    text = "Hello, world!"
    speaker = 0
    context = []
    
    # Mocks for the tokenize method
    text_tokens = torch.zeros(5, 33)
    text_masks = torch.zeros(5, 33)
    text_masks[:, -1] = 1
    
    generator._tokenize_text_segment = MagicMock(return_value=(text_tokens, text_masks))
    
    # Call the generate method
    audio = generator.generate(
        text=text,
        speaker=speaker,
        context=context,
        max_audio_length_ms=1000,
        temperature=0.9,
        topk=50
    )
    
    # Verify model reset_caches was called
    mock_model.reset_caches.assert_called_once()
    
    # Verify generate_frame was called at least once
    assert mock_model.generate_frame.call_count > 0
    
    # Verify audio decoding was performed
    mock_audio_tokenizer.decode.assert_called_once()
    
    # Verify watermarking was applied
    mock_watermark.assert_called_once()
    
    # Verify resampling was performed
    mock_resample.assert_called_once()
    
    # Check that the function returned a tensor
    assert isinstance(audio, torch.Tensor)