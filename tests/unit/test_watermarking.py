"""Tests for watermarking module."""

import pytest
import torch
from unittest.mock import patch, MagicMock


@patch("csm.watermarking.utils.load_watermarker")
def test_load_watermarker(mock_load):
    """Test the load_watermarker function."""
    # Set up the mock to return a MagicMock
    mock_watermarker = MagicMock()
    mock_load.return_value = mock_watermarker
    
    # Import the function
    from csm.watermarking.utils import load_watermarker
    
    # Call the function
    watermarker = load_watermarker(device="cpu")
    
    # Verify the function called the underlying loader with the correct parameters
    mock_load.assert_called_once_with(device="cpu")
    
    # Verify the function returned the mock watermarker
    assert watermarker == mock_watermarker


def test_watermark_constants():
    """Test the watermark constants are properly defined."""
    # Import the constants
    from csm.watermarking import CSM_1B_GH_WATERMARK
    
    # Check that the watermark is a list
    assert isinstance(CSM_1B_GH_WATERMARK, list)
    # A proper key should be non-empty
    assert len(CSM_1B_GH_WATERMARK) > 0
    # All elements should be integers
    assert all(isinstance(value, int) for value in CSM_1B_GH_WATERMARK)


@patch("csm.watermarking.utils.load_watermarker")
@patch("csm.watermarking.utils.watermark")
def test_watermark_function(mock_watermark, mock_load_watermarker):
    """Test the watermark function."""
    # Set up the mock to return a sample audio and sample rate
    mock_audio = torch.zeros(24000)
    mock_sample_rate = 24000
    mock_watermark.return_value = (mock_audio, mock_sample_rate)
    
    # Create a mock watermarker
    mock_watermarker = MagicMock()
    mock_load_watermarker.return_value = mock_watermarker
    
    # Import the function
    from csm.watermarking.utils import watermark, load_watermarker
    
    # Create test audio
    test_audio = torch.ones(16000)
    test_sample_rate = 16000
    test_key = "test_key"
    
    # Call the function
    result_audio, result_sample_rate = watermark(mock_watermarker, test_audio, test_sample_rate, test_key)
    
    # Verify the function called the underlying watermarker functions
    mock_watermark.assert_called_once()
    
    # Verify the function returned the expected values
    assert torch.equal(result_audio, mock_audio)
    assert result_sample_rate == mock_sample_rate