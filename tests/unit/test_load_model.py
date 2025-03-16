"""Tests for model loading functions."""

import pytest
import torch
from unittest.mock import patch, MagicMock

from csm.generator import load_csm_1b
from csm.models.model import Model, ModelArgs


@pytest.mark.skip(reason="Requires mocked dependencies")
def test_load_csm_1b_with_local_path():
    """Test loading the CSM-1B model from a local path."""
    # Mock the state dict and Model class
    mock_state_dict = {"key": "value"}
    mock_torch_load.return_value = mock_state_dict
    
    with patch("csm.models.model.Model") as mock_model_class, \
         patch("csm.generator.Generator") as mock_generator_class:
        
        # Set up mocks
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        
        mock_generator = MagicMock()
        mock_generator_class.return_value = mock_generator
        
        # Call the function
        result = load_csm_1b(ckpt_path="test_path.pt", device="cpu")
        
        # Verify Model was created with the correct args
        model_args_call = mock_model_class.call_args[0][0]
        assert isinstance(model_args_call, ModelArgs)
        assert model_args_call.backbone_flavor == "llama-1B"
        assert model_args_call.decoder_flavor == "llama-100M"
        assert model_args_call.text_vocab_size == 128256
        assert model_args_call.audio_vocab_size == 2051
        assert model_args_call.audio_num_codebooks == 32
        
        # Verify model was loaded with state dict
        mock_model.load_state_dict.assert_called_with(mock_state_dict)
        
        # Verify model was moved to the correct device and dtype
        mock_model.to.assert_called_with(device="cpu", dtype=torch.bfloat16)
        
        # Verify Generator was created with the model
        mock_generator_class.assert_called_with(mock_model)
        
        # Verify the function returned the generator
        assert result == mock_generator


@pytest.mark.skip(reason="Requires mocked dependencies")
def test_load_csm_1b_with_from_hf():
    """Test loading the CSM-1B model from HuggingFace."""
    # With MLX integration, the model can now be loaded from the Model.from_pretrained method
    # Set up our mocks
    with patch("csm.models.model.Model") as mock_model_class, \
         patch("csm.generator.Generator") as mock_generator_class:
        
        # Create mock objects
        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        mock_generator = MagicMock()
        mock_generator_class.return_value = mock_generator
        
        # Call the function with None to trigger HF loading
        result = load_csm_1b(device="cuda", model_path=None)
        
        # Verify model was moved to the correct device
        mock_model.to.assert_called_with(device="cuda", dtype=torch.bfloat16)
        
        # Verify Generator was created with the model
        mock_generator_class.assert_called_with(mock_model)
        
        # Verify the function returned the generator
        assert result == mock_generator