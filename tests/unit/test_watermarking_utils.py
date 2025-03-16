"""Tests for watermarking utilities."""

import pytest
import torch
from unittest.mock import patch, MagicMock

from csm.watermarking.utils import verify, load_audio, check_audio_from_file, cli


@patch("csm.watermarking.utils.verify")
@patch("csm.watermarking.utils.load_audio")
@patch("csm.watermarking.utils.load_watermarker")
def test_check_audio_from_file(mock_load_watermarker, mock_load_audio, mock_verify):
    """Test checking audio from file."""
    # Set up mocks
    mock_watermarker = MagicMock()
    mock_load_watermarker.return_value = mock_watermarker
    
    mock_audio = torch.zeros(1000)
    mock_sample_rate = 44100
    mock_load_audio.return_value = (mock_audio, mock_sample_rate)
    
    mock_verify.return_value = True
    
    # Call the function
    with patch("builtins.print") as mock_print:
        check_audio_from_file("test_audio.wav")
        
        # Verify the mocks were called correctly
        mock_load_watermarker.assert_called_once()
        mock_load_audio.assert_called_with("test_audio.wav")
        mock_verify.assert_called_once()
        
        # Verify the output
        mock_print.assert_called_once()
        assert "Watermarked" in mock_print.call_args[0][0]
        
    # Test with not watermarked
    mock_verify.return_value = False
    
    with patch("builtins.print") as mock_print:
        check_audio_from_file("test_audio.wav")
        
        # Verify the output
        assert "Not watermarked" in mock_print.call_args[0][0]


@patch("torchaudio.load")
def test_load_audio(mock_torchaudio_load):
    """Test loading audio."""
    # Set up mocks
    mock_audio = torch.ones(2, 1000)  # 2 channels, 1000 samples
    mock_sample_rate = 44100
    mock_torchaudio_load.return_value = (mock_audio, mock_sample_rate)
    
    # Call the function
    audio, sample_rate = load_audio("test_audio.wav")
    
    # Verify the mocks were called correctly
    mock_torchaudio_load.assert_called_with("test_audio.wav")
    
    # Verify the output
    assert sample_rate == mock_sample_rate
    assert audio.shape == (1000,)  # Should be mono now (mean of channels)
    assert torch.all(audio == 1.0)  # All ones since we used ones for the mock


@patch("argparse.ArgumentParser.parse_args")
@patch("csm.watermarking.utils.check_audio_from_file")
def test_cli(mock_check_audio, mock_parse_args):
    """Test the CLI function."""
    # Set up mocks
    args = MagicMock()
    args.audio_path = "test_audio.wav"
    mock_parse_args.return_value = args
    
    # Call the function
    cli()
    
    # Verify the mocks were called correctly
    mock_check_audio.assert_called_with("test_audio.wav")


@patch("torchaudio.functional.resample")
def test_verify(mock_resample):
    """Test the verify function."""
    # Set up mocks
    mock_watermarker = MagicMock()
    mock_watermarker.decode_wav.return_value = {
        "status": True,
        "messages": [[212, 211, 146, 56, 201]]  # Matches CSM_1B_GH_WATERMARK
    }
    
    mock_audio = torch.zeros(1000)
    mock_sample_rate = 44100
    
    # Call the function
    result = verify(
        watermarker=mock_watermarker,
        watermarked_audio=mock_audio,
        sample_rate=mock_sample_rate,
        watermark_key=[212, 211, 146, 56, 201]
    )
    
    # Verify the mocks were called correctly
    mock_resample.assert_called_once()
    mock_watermarker.decode_wav.assert_called_once()
    
    # Verify the result
    assert result is True
    
    # Test with different message
    mock_watermarker.decode_wav.return_value = {
        "status": True,
        "messages": [[1, 2, 3, 4, 5]]  # Different message
    }
    
    result = verify(
        watermarker=mock_watermarker,
        watermarked_audio=mock_audio,
        sample_rate=mock_sample_rate,
        watermark_key=[212, 211, 146, 56, 201]
    )
    
    # Should be False now
    assert result is False
    
    # Test with no watermark
    mock_watermarker.decode_wav.return_value = {
        "status": False,
        "messages": []
    }
    
    result = verify(
        watermarker=mock_watermarker,
        watermarked_audio=mock_audio,
        sample_rate=mock_sample_rate,
        watermark_key=[212, 211, 146, 56, 201]
    )
    
    # Should be False
    assert result is False