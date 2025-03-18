"""
Tests for the MLX model wrapper component.
"""

import sys
import argparse
from unittest.mock import MagicMock, patch, PropertyMock
import pytest
import torch
import numpy as np
import os

# These tests require MLX
pytestmark = pytest.mark.requires_mlx

# Check if MLX is available
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    # Skip all tests if MLX is not available
    pytest.skip("MLX is not available", allow_module_level=True)

# Import needed modules after handling MLX availability
from csm.mlx_accel.components.model_wrapper import MLXModelWrapper


class MockTorchModel:
    """Mock PyTorch model for testing the MLX wrapper."""
    
    def __init__(self, with_args=True, has_audio_head=True, has_codebook0=True):
        # Model configuration
        if with_args:
            self.args = argparse.Namespace()
            self.args.audio_vocab_size = 2051
            self.args.audio_num_codebooks = 32
        
        # Set up backbone
        self.backbone = MagicMock()
        self.backbone.layers = [self._create_layer() for _ in range(3)]
        self.backbone.norm = MagicMock()
        self.backbone.norm.scale = torch.ones(1024)
        
        # Set up decoder
        self.decoder = MagicMock()
        self.decoder.layers = [self._create_layer() for _ in range(2)]
        self.decoder.norm = MagicMock()
        self.decoder.norm.scale = torch.ones(1024)
        
        # Set up embeddings
        self.text_embeddings = MagicMock()
        self.text_embeddings.weight = torch.ones(32000, 1024)
        
        self.audio_embeddings = MagicMock()
        self.audio_embeddings.weight = torch.ones(2051, 1024)
        
        # Set up projection
        self.projection = MagicMock()
        self.projection.weight = torch.ones(2048, 1024)
        
        # Set up codebook0_head and audio_head
        if has_codebook0:
            self.codebook0_head = MagicMock()
            self.codebook0_head.weight = torch.ones(2051, 1024)
        
        if has_audio_head:
            self.audio_head = []
            for _ in range(32):
                head = MagicMock()
                head.weight = torch.ones(2051, 1024)
                self.audio_head.append(head)
    
    def _create_layer(self):
        """Create a mock transformer layer."""
        layer = MagicMock()
        
        # Add attention component
        layer.attn = MagicMock()
        layer.attn.q_proj = MagicMock()
        layer.attn.k_proj = MagicMock()
        layer.attn.v_proj = MagicMock()
        layer.attn.output_proj = MagicMock()
        
        # Add weights
        layer.attn.q_proj.weight = torch.ones(1024, 1024)
        layer.attn.k_proj.weight = torch.ones(1024, 1024)
        layer.attn.v_proj.weight = torch.ones(1024, 1024)
        layer.attn.output_proj.weight = torch.ones(1024, 1024)
        
        # Add attributes
        layer.attn.n_heads = 16
        layer.attn.n_kv_heads = 16
        
        # Add layer norms
        layer.sa_norm = MagicMock()
        layer.sa_norm.scale = torch.ones(1024)
        
        layer.mlp_norm = MagicMock()
        layer.mlp_norm.scale = torch.ones(1024)
        
        # Add MLP
        layer.mlp = MagicMock()
        layer.mlp.w1 = MagicMock()
        layer.mlp.w2 = MagicMock()
        layer.mlp.w3 = MagicMock()
        
        # Add weights
        layer.mlp.w1.weight = torch.ones(4096, 1024)
        layer.mlp.w2.weight = torch.ones(4096, 1024)
        layer.mlp.w3.weight = torch.ones(1024, 4096)
        
        return layer
    
    def forward_first_stage(self, tokens, tokens_mask, input_pos):
        """Mock for the first-stage forward pass."""
        batch_size = tokens.shape[0]
        return torch.ones(batch_size, tokens.shape[1], 1024), None, None
    
    def _generate_codebook(self, i, curr_sample, seq_len):
        """Mock for generating codebook tokens."""
        batch_size = curr_sample.shape[0]
        return torch.ones(batch_size, 1), None
    
    def generate_frame(self, tokens, tokens_mask, input_pos, temperature, topk):
        """Mock for frame generation."""
        batch_size = tokens.shape[0]
        return torch.ones(batch_size, 32)


@pytest.fixture
def mock_torch_model():
    """Create a mock PyTorch model for testing."""
    return MockTorchModel()


@pytest.fixture
def mock_torch_model_no_args():
    """Create a mock PyTorch model without args."""
    return MockTorchModel(with_args=False)


@pytest.fixture
def mock_torch_model_no_audio_head():
    """Create a mock PyTorch model without audio_head."""
    return MockTorchModel(has_audio_head=False)


@pytest.fixture
def mock_torch_model_no_codebook():
    """Create a mock PyTorch model without codebook0_head."""
    return MockTorchModel(has_codebook0=False)


@patch('csm.mlx_accel.components.model_wrapper.MLXEmbedding')
@patch('csm.mlx_accel.components.model_wrapper.MLXFrameGenerator')
@patch('csm.mlx_accel.components.model_wrapper.MLXTransformer')
def test_mlx_model_wrapper_initialization(mock_transformer, mock_frame_generator, mock_embedding, mock_torch_model):
    """Test initializing the MLX model wrapper."""
    # Arrange
    mock_transformer_instance = MagicMock()
    mock_transformer.return_value = mock_transformer_instance
    mock_transformer_instance.hidden_size = 1024
    
    # Act
    wrapper = MLXModelWrapper(mock_torch_model)
    
    # Assert
    assert wrapper.torch_model is mock_torch_model
    assert wrapper.args is not None
    assert wrapper.args.audio_vocab_size == 2051
    assert wrapper.args.audio_num_codebooks == 32
    assert wrapper.backbone is not None
    assert wrapper.decoder is not None
    assert wrapper.text_embeddings is not None
    assert wrapper.audio_embeddings is not None
    assert wrapper.projection is not None
    assert wrapper.codebook0_head is not None
    assert wrapper.audio_head is not None
    assert isinstance(wrapper.audio_head, list)
    assert wrapper.embedding is not None
    assert wrapper.frame_generator is not None


@patch('csm.mlx_accel.components.model_wrapper.MLXEmbedding')
@patch('csm.mlx_accel.components.model_wrapper.MLXFrameGenerator')
@patch('csm.mlx_accel.components.model_wrapper.MLXTransformer')
def test_mlx_model_wrapper_no_args(mock_transformer, mock_frame_generator, mock_embedding, mock_torch_model_no_args):
    """Test initializing the MLX model wrapper without model args."""
    # Arrange
    mock_transformer_instance = MagicMock()
    mock_transformer.return_value = mock_transformer_instance
    mock_transformer_instance.hidden_size = 1024
    
    # Act
    wrapper = MLXModelWrapper(mock_torch_model_no_args)
    
    # Assert
    assert wrapper.args is not None
    assert wrapper.args.audio_vocab_size == 2051
    assert wrapper.args.audio_num_codebooks == 32


@patch('csm.mlx_accel.components.model_wrapper.MLXEmbedding')
@patch('csm.mlx_accel.components.model_wrapper.MLXFrameGenerator')
@patch('csm.mlx_accel.components.model_wrapper.MLXTransformer')
def test_mlx_model_wrapper_custom_args(mock_transformer, mock_frame_generator, mock_embedding, mock_torch_model):
    """Test initializing the MLX model wrapper with custom args."""
    # Arrange
    custom_args = argparse.Namespace()
    custom_args.audio_vocab_size = 4000
    custom_args.audio_num_codebooks = 64
    custom_args.debug = True
    
    mock_transformer_instance = MagicMock()
    mock_transformer.return_value = mock_transformer_instance
    mock_transformer_instance.hidden_size = 1024
    
    # Act
    wrapper = MLXModelWrapper(mock_torch_model, args=custom_args)
    
    # Assert
    assert wrapper.args is custom_args
    assert wrapper.args.audio_vocab_size == 4000
    assert wrapper.args.audio_num_codebooks == 64
    assert wrapper.debug is True


@patch('csm.mlx_accel.components.model_wrapper.MLXEmbedding')
@patch('csm.mlx_accel.components.model_wrapper.MLXFrameGenerator')
@patch('csm.mlx_accel.components.model_wrapper.MLXTransformer')
def test_mlx_model_wrapper_no_audio_head(mock_transformer, mock_frame_generator, mock_embedding, mock_torch_model_no_audio_head):
    """Test initializing the MLX model wrapper without audio_head."""
    # Arrange
    mock_transformer_instance = MagicMock()
    mock_transformer.return_value = mock_transformer_instance
    mock_transformer_instance.hidden_size = 1024
    
    # Act
    wrapper = MLXModelWrapper(mock_torch_model_no_audio_head)
    
    # Assert
    assert wrapper.audio_head is None


@patch('csm.mlx_accel.components.model_wrapper.MLXEmbedding')
@patch('csm.mlx_accel.components.model_wrapper.MLXFrameGenerator')
@patch('csm.mlx_accel.components.model_wrapper.MLXTransformer')
def test_mlx_model_wrapper_no_codebook(mock_transformer, mock_frame_generator, mock_embedding, mock_torch_model_no_codebook):
    """Test initializing the MLX model wrapper without codebook0_head."""
    # Arrange
    mock_transformer_instance = MagicMock()
    mock_transformer.return_value = mock_transformer_instance
    mock_transformer_instance.hidden_size = 1024
    
    # Act
    wrapper = MLXModelWrapper(mock_torch_model_no_codebook)
    
    # Assert
    assert wrapper.codebook0_head is None


@patch('csm.mlx_accel.components.model_wrapper.MLXEmbedding')
@patch('csm.mlx_accel.components.model_wrapper.MLXFrameGenerator')
@patch('csm.mlx_accel.components.model_wrapper.MLXTransformer')
def test_mlx_model_wrapper_debug_mode(mock_transformer, mock_frame_generator, mock_embedding, mock_torch_model):
    """Test MLXModelWrapper with debug mode enabled."""
    # Arrange
    custom_args = argparse.Namespace()
    custom_args.audio_vocab_size = 2051
    custom_args.audio_num_codebooks = 32
    custom_args.debug = True
    
    mock_transformer_instance = MagicMock()
    mock_transformer.return_value = mock_transformer_instance
    mock_transformer_instance.hidden_size = 1024
    
    # Act
    with patch('builtins.print') as mock_print:
        wrapper = MLXModelWrapper(mock_torch_model, args=custom_args)
        
        # Assert
        assert wrapper.debug is True
        mock_print.assert_called()


@patch('csm.mlx_accel.components.model_wrapper.MLXEmbedding')
@patch('csm.mlx_accel.components.model_wrapper.MLXFrameGenerator')
@patch('csm.mlx_accel.components.model_wrapper.MLXTransformer')
def test_reset_caches(mock_transformer, mock_frame_generator, mock_embedding, mock_torch_model):
    """Test resetting MLX caches."""
    # Arrange
    mock_transformer_instance = MagicMock()
    mock_transformer.return_value = mock_transformer_instance
    mock_transformer_instance.hidden_size = 1024
    
    wrapper = MLXModelWrapper(mock_torch_model)
    
    # Act
    wrapper.reset_caches()
    
    # Assert
    mock_transformer_instance.reset_caches.assert_called()


@patch('csm.mlx_accel.components.model_wrapper.MLXEmbedding')
@patch('csm.mlx_accel.components.model_wrapper.MLXFrameGenerator')
@patch('csm.mlx_accel.components.model_wrapper.MLXTransformer')
@patch('csm.mlx_accel.components.model_wrapper.torch_to_mlx')
@patch('csm.mlx_accel.components.model_wrapper.mlx_to_torch')
def test_generate_frame_pure_mlx(mock_mlx_to_torch, mock_torch_to_mlx, mock_transformer, 
                               mock_frame_generator, mock_embedding, mock_torch_model):
    """Test generate_frame with pure MLX path."""
    # Arrange
    mock_transformer_instance = MagicMock()
    mock_transformer.return_value = mock_transformer_instance
    mock_transformer_instance.hidden_size = 1024
    
    mock_frame_generator_instance = MagicMock()
    mock_frame_generator.return_value = mock_frame_generator_instance
    mock_frame_generator_instance.generate_frame.return_value = mx.array([[1, 2, 3]])
    
    mock_torch_to_mlx.return_value = mx.array([[1, 2, 3]])
    mock_mlx_to_torch.return_value = torch.tensor([[1, 2, 3]])
    
    wrapper = MLXModelWrapper(mock_torch_model)
    
    # Input tensors
    tokens = torch.ones((1, 10), dtype=torch.long)
    input_pos = torch.arange(10).unsqueeze(0)
    
    # Act
    result = wrapper.generate_frame(tokens, input_pos, 0, topk=5, temperature=1.0)
    
    # Assert
    mock_frame_generator_instance.generate_frame.assert_called_once()
    assert result is not None


@patch('csm.mlx_accel.components.model_wrapper.MLXEmbedding')
@patch('csm.mlx_accel.components.model_wrapper.MLXFrameGenerator')
@patch('csm.mlx_accel.components.model_wrapper.MLXTransformer')
@patch('csm.mlx_accel.components.model_wrapper.torch_to_mlx')
@patch('csm.mlx_accel.components.model_wrapper.mlx_to_torch')
def test_generate_frame_hybrid_fallback(mock_mlx_to_torch, mock_torch_to_mlx, mock_transformer, 
                                      mock_frame_generator, mock_embedding, mock_torch_model):
    """Test generate_frame with hybrid fallback path."""
    # Arrange
    mock_transformer_instance = MagicMock()
    mock_transformer.return_value = mock_transformer_instance
    mock_transformer_instance.hidden_size = 1024
    
    mock_frame_generator_instance = MagicMock()
    mock_frame_generator.return_value = mock_frame_generator_instance
    mock_frame_generator_instance.generate_frame.side_effect = Exception("MLX error")
    
    mock_torch_to_mlx.return_value = mx.array([[1, 2, 3]])
    mock_mlx_to_torch.return_value = torch.tensor([[1, 2, 3]])
    
    wrapper = MLXModelWrapper(mock_torch_model)
    
    # Patch generate_frame_hybrid to return a known value
    with patch.object(wrapper, 'generate_frame_hybrid') as mock_hybrid:
        mock_hybrid.return_value = torch.tensor([[4, 5, 6]])
        
        # Input tensors
        tokens = torch.ones((1, 10), dtype=torch.long)
        input_pos = torch.arange(10).unsqueeze(0)
        
        # Act
        result = wrapper.generate_frame(tokens, input_pos, 0, topk=5, temperature=1.0)
        
        # Assert
        mock_hybrid.assert_called_once()
        assert result is not None
        assert torch.all(result == torch.tensor([[4, 5, 6]]))


@patch('csm.mlx_accel.components.model_wrapper.MLXEmbedding')
@patch('csm.mlx_accel.components.model_wrapper.MLXFrameGenerator')
@patch('csm.mlx_accel.components.model_wrapper.MLXTransformer')
@patch('csm.mlx_accel.components.model_wrapper.torch_to_mlx')
@patch('csm.mlx_accel.components.model_wrapper.mlx_to_torch')
def test_generate_frame_hybrid(mock_mlx_to_torch, mock_torch_to_mlx, 
                             mock_transformer, mock_frame_generator, mock_embedding, mock_torch_model):
    """Test generate_frame_hybrid function."""
    # Arrange
    mock_transformer_instance = MagicMock()
    mock_transformer.return_value = mock_transformer_instance
    mock_transformer_instance.hidden_size = 1024
    
    # Set up mocks for conversion
    mock_torch_to_mlx.return_value = mx.array([[1, 2, 3]])
    mock_mlx_to_torch.return_value = torch.tensor([[1]])
    
    # We'll use a module-level patch for the sampling function
    with patch('csm.mlx_accel.components.sampling.mlx_categorical_sampling') as mock_categorical:
        mock_categorical.return_value = (mx.array([[0]]), True)
        
        wrapper = MLXModelWrapper(mock_torch_model)
        
        # Input tensors
        tokens = torch.ones((1, 10), dtype=torch.long)
        input_pos = torch.arange(10).unsqueeze(0)
        
        # Act
        result = wrapper.generate_frame_hybrid(tokens, input_pos, 0, topk=5, temperature=1.0)
        
        # Assert
        assert result is not None
        assert result.shape[1] == wrapper.args.audio_num_codebooks


@patch('csm.mlx_accel.components.model_wrapper.MLXEmbedding')
@patch('csm.mlx_accel.components.model_wrapper.MLXFrameGenerator')
@patch('csm.mlx_accel.components.model_wrapper.MLXTransformer')
def test_fallback_generate(mock_transformer, mock_frame_generator, mock_embedding, mock_torch_model):
    """Test _fallback_generate function."""
    # Arrange
    mock_transformer_instance = MagicMock()
    mock_transformer.return_value = mock_transformer_instance
    mock_transformer_instance.hidden_size = 1024
    
    wrapper = MLXModelWrapper(mock_torch_model)
    
    # Act - Test codebook fallback
    curr_sample = mx.array([[1, 2, 3]])
    result1 = wrapper._fallback_generate(i=1, curr_sample=curr_sample)
    
    # Act - Test generic fallback
    result2 = wrapper._fallback_generate()
    
    # Assert
    assert result1 is not None
    assert result2 is not None
    assert result1.shape[1] == 1
    assert result2.shape[1] == 32


@patch('csm.mlx_accel.components.model_wrapper.MLXEmbedding')
@patch('csm.mlx_accel.components.model_wrapper.MLXFrameGenerator')
@patch('csm.mlx_accel.components.model_wrapper.MLXTransformer')
def test_convert_from_torch_transformer_error(mock_transformer, mock_frame_generator, mock_embedding, mock_torch_model):
    """Test _convert_transformer with errors."""
    # Arrange
    mock_transformer_instance = MagicMock()
    mock_transformer.return_value = mock_transformer_instance
    mock_transformer_instance.hidden_size = 1024
    
    # Make a model with an empty layer list
    bad_model = MagicMock()
    bad_model.layers = []
    
    wrapper = MLXModelWrapper(mock_torch_model)
    
    # Act & Assert - Test with empty layers
    with pytest.raises(ValueError, match="has no layers"):
        wrapper._convert_transformer(bad_model, "empty_model")
    
    # Make a model without a layers attribute
    no_layers_model = MagicMock()
    delattr(no_layers_model, 'layers')
    
    # Act & Assert - Test without layers attribute
    with pytest.raises(ValueError, match="does not have layers attribute"):
        wrapper._convert_transformer(no_layers_model, "no_layers_model")
    
    # Make a model with unknown architecture
    unknown_model = MagicMock()
    unknown_model.layers = [MagicMock()]  # Layer without attn attribute
    # Make sure there's no 'attn' attribute
    unknown_layer = unknown_model.layers[0]
    # Explicitly test if 'attn' is in the mock's attributes and remove it if needed
    if hasattr(unknown_layer, 'attn'):
        delattr(unknown_layer, 'attn')
    
    # Act & Assert - Test with unknown architecture
    with pytest.raises(ValueError, match="has unknown architecture"):
        wrapper._convert_transformer(unknown_model, "unknown_model")


@patch('csm.mlx_accel.components.model_wrapper.MLXEmbedding')
@patch('csm.mlx_accel.components.model_wrapper.MLXFrameGenerator')
@patch('csm.mlx_accel.components.model_wrapper.MLXTransformer')
def test_convert_mismatched_vocab_size(mock_transformer, mock_frame_generator, mock_embedding):
    """Test converting model with mismatched vocabulary sizes."""
    # Arrange - Create a model with larger vocab size than args
    model_big_vocab = MockTorchModel()
    model_big_vocab.codebook0_head.weight = torch.ones(3000, 1024)  # Larger than audio_vocab_size
    
    # Create args with smaller vocab size
    custom_args = argparse.Namespace()
    custom_args.audio_vocab_size = 2000  # Smaller than weight shape
    custom_args.audio_num_codebooks = 32
    custom_args.debug = True
    
    mock_transformer_instance = MagicMock()
    mock_transformer.return_value = mock_transformer_instance
    mock_transformer_instance.hidden_size = 1024
    
    # Act - Initialize with debug mode to trigger prints
    with patch('builtins.print') as mock_print:
        wrapper = MLXModelWrapper(model_big_vocab, args=custom_args)
        
        # Assert
        assert wrapper.codebook0_head.shape[0] == 2000  # Should be truncated to match args
        mock_print.assert_any_call("WARNING: codebook0_head weight shape 3000 doesn't match audio_vocab_size 2000")
    
    # Arrange - Create a model with smaller vocab size than args
    model_small_vocab = MockTorchModel()
    model_small_vocab.codebook0_head.weight = torch.ones(1000, 1024)  # Smaller than audio_vocab_size
    
    # Create args with larger vocab size
    custom_args = argparse.Namespace()
    custom_args.audio_vocab_size = 2000  # Larger than weight shape
    custom_args.audio_num_codebooks = 32
    custom_args.debug = True
    
    # Act - Initialize with debug mode to trigger prints
    with patch('builtins.print') as mock_print:
        wrapper = MLXModelWrapper(model_small_vocab, args=custom_args)
        
        # Assert
        assert wrapper.codebook0_head.shape[0] == 2000  # Should be padded to match args
        mock_print.assert_any_call("WARNING: codebook0_head weight shape 1000 doesn't match audio_vocab_size 2000")
