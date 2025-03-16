"""
Tests for the MLX model wrapper component.
"""

import sys
import argparse
from unittest.mock import MagicMock, patch
import pytest
import torch
import numpy as np

# Check if MLX is available
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    # Create a mock module
    class MockMX:
        def __init__(self):
            self.core = MagicMock()
            self.nn = MagicMock()
    mx = MockMX()
    sys.modules['mlx'] = mx
    sys.modules['mlx.core'] = mx.core
    sys.modules['mlx.nn'] = mx.nn

# Skip tests if MLX not available
pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")

# Import the module under test
from csm.mlx_accel.components.model_wrapper import MLXModelWrapper


class MockTransformerLayer:
    """Mock transformer layer for testing."""
    
    def __init__(self, hidden_size=512, n_heads=8, n_kv_heads=8):
        self.attn = MagicMock()
        self.attn.q_proj = MagicMock()
        self.attn.q_proj.weight = torch.randn(hidden_size, hidden_size)
        self.attn.k_proj = MagicMock()
        self.attn.k_proj.weight = torch.randn(hidden_size, hidden_size)
        self.attn.v_proj = MagicMock()
        self.attn.v_proj.weight = torch.randn(hidden_size, hidden_size)
        self.attn.output_proj = MagicMock()
        self.attn.output_proj.weight = torch.randn(hidden_size, hidden_size)
        self.attn.n_heads = n_heads
        self.attn.n_kv_heads = n_kv_heads
        
        self.sa_norm = MagicMock()
        self.sa_norm.scale = torch.ones(hidden_size)
        
        self.mlp = MagicMock()
        self.mlp.w1 = MagicMock()
        self.mlp.w1.weight = torch.randn(hidden_size * 4, hidden_size)
        self.mlp.w2 = MagicMock()
        self.mlp.w2.weight = torch.randn(hidden_size, hidden_size * 4)
        self.mlp.w3 = MagicMock()
        self.mlp.w3.weight = torch.randn(hidden_size * 4, hidden_size)
        
        self.mlp_norm = MagicMock()
        self.mlp_norm.scale = torch.ones(hidden_size)


class MockTransformer:
    """Mock transformer for testing."""
    
    def __init__(self, num_layers=2, hidden_size=512, n_heads=8, n_kv_heads=8):
        self.layers = [
            MockTransformerLayer(hidden_size, n_heads, n_kv_heads) 
            for _ in range(num_layers)
        ]
        self.norm = MagicMock()
        self.norm.scale = torch.ones(hidden_size)


class MockModelParameters:
    """Mock model parameters for testing."""
    
    def __init__(self, hidden_size=512, vocab_size=2051, num_codebooks=32):
        self.text_embeddings = MagicMock()
        self.text_embeddings.weight = torch.randn(30000, hidden_size)
        
        self.audio_embeddings = MagicMock()
        self.audio_embeddings.weight = torch.randn(vocab_size * num_codebooks, hidden_size)
        
        self.projection = MagicMock()
        self.projection.weight = torch.randn(hidden_size, hidden_size)
        
        self.codebook0_head = MagicMock()
        self.codebook0_head.weight = torch.randn(vocab_size, hidden_size)
        
        self.audio_head = [MagicMock() for _ in range(num_codebooks - 1)]
        for head in self.audio_head:
            head.weight = torch.randn(vocab_size, hidden_size)


class MockCSMModel:
    """Mock CSM model for testing."""
    
    def __init__(self, hidden_size=512, vocab_size=2051, num_codebooks=32):
        self.backbone = MockTransformer(num_layers=2, hidden_size=hidden_size)
        self.decoder = MockTransformer(num_layers=2, hidden_size=hidden_size)
        
        # Create model parameters
        model_params = MockModelParameters(hidden_size, vocab_size, num_codebooks)
        
        # Assign parameters
        self.text_embeddings = model_params.text_embeddings
        self.audio_embeddings = model_params.audio_embeddings
        self.projection = model_params.projection
        self.codebook0_head = model_params.codebook0_head
        self.audio_head = model_params.audio_head
        
        # Create model args
        self.args = argparse.Namespace()
        self.args.audio_vocab_size = vocab_size
        self.args.audio_num_codebooks = num_codebooks
        
        # Mock methods
        self._generate_codebook = MagicMock()
        self._generate_codebook.return_value = (torch.zeros((1, 1)), None)
        
        self.generate_frame = MagicMock()
        self.generate_frame.return_value = torch.zeros((1, num_codebooks))
        
        self.forward_first_stage = MagicMock()
        self.forward_first_stage.return_value = (
            torch.randn(1, 10, hidden_size),  # hidden states
            None,  # attentions
            None   # past_key_values
        )


def test_mlx_model_wrapper_init():
    """Test initialization of MLXModelWrapper."""
    # Create mock model
    mock_model = MockCSMModel()
    
    # Create wrapper
    wrapper = MLXModelWrapper(mock_model)
    
    # Check if wrapper was initialized correctly
    assert wrapper.torch_model == mock_model
    assert wrapper.args.audio_vocab_size == 2051
    assert wrapper.args.audio_num_codebooks == 32
    
    # Check if components were created
    assert wrapper.backbone is not None
    assert wrapper.decoder is not None
    assert wrapper.text_embeddings is not None
    assert wrapper.audio_embeddings is not None
    assert wrapper.projection is not None
    assert wrapper.codebook0_head is not None
    assert wrapper.audio_head is not None
    assert wrapper.embedding is not None
    assert wrapper.frame_generator is not None


def test_convert_from_torch():
    """Test conversion from PyTorch to MLX."""
    # Create mock model
    mock_model = MockCSMModel()
    
    # Create wrapper with patch to isolate _convert_transformer
    with patch('csm.mlx_accel.components.model_wrapper.MLXModelWrapper._convert_transformer') as mock_convert:
        # Mock return value
        mock_convert.return_value = MagicMock()
        mock_convert.return_value.hidden_size = 512
        
        # Create wrapper
        wrapper = MLXModelWrapper(mock_model)
        
        # Check if conversion methods were called
        assert mock_convert.call_count == 2
        mock_convert.assert_any_call(mock_model.backbone, "backbone")
        mock_convert.assert_any_call(mock_model.decoder, "decoder")


def test_convert_transformer():
    """Test conversion of transformer from PyTorch to MLX."""
    # Create mock model and transformer
    mock_model = MockCSMModel()
    mock_transformer = MockTransformer()
    
    # Create wrapper
    wrapper = MLXModelWrapper(mock_model)
    
    # Test conversion
    with patch('csm.mlx_accel.components.model_wrapper.torch_to_mlx') as mock_torch_to_mlx:
        # Mock return value
        mock_torch_to_mlx.return_value = mx.zeros((10, 10))
        
        # Convert transformer
        mlx_transformer = wrapper._convert_transformer(mock_transformer, "test_transformer")
        
        # Check conversion results
        assert mlx_transformer is not None
        assert mlx_transformer.hidden_size == 512
        assert mlx_transformer.num_heads == 8
        assert mlx_transformer.num_kv_heads == 8
        assert mlx_transformer.num_layers == 2
        
        # Check parameter conversion
        assert mock_torch_to_mlx.call_count >= 14  # Expect at least 14 conversions


def test_convert_transformer_exceptions():
    """Test exceptions during transformer conversion."""
    # Create mock model
    mock_model = MockCSMModel()
    
    # Create wrapper
    wrapper = MLXModelWrapper(mock_model)
    
    # Test with empty layers
    empty_transformer = MagicMock()
    empty_transformer.layers = []
    
    with pytest.raises(ValueError, match="has no layers"):
        wrapper._convert_transformer(empty_transformer, "empty_transformer")
    
    # Test with no layers attribute
    no_layers_transformer = MagicMock()
    del no_layers_transformer.layers
    
    with pytest.raises(ValueError, match="does not have layers attribute"):
        wrapper._convert_transformer(no_layers_transformer, "no_layers_transformer")
    
    # Test with unknown architecture
    unknown_transformer = MagicMock()
    unknown_transformer.layers = [MagicMock()]
    del unknown_transformer.layers[0].attn
    
    with pytest.raises(ValueError, match="has unknown architecture"):
        wrapper._convert_transformer(unknown_transformer, "unknown_transformer")


def test_fallback_generate():
    """Test fallback generation method."""
    # Create mock model
    mock_model = MockCSMModel()
    
    # Create wrapper
    wrapper = MLXModelWrapper(mock_model)
    
    # Test fallback with i and curr_sample
    curr_sample = mx.zeros((1, 10))
    result = wrapper._fallback_generate(i=1, curr_sample=curr_sample)
    
    # Check result
    assert isinstance(result, torch.Tensor)
    assert result.shape == (1, 1)
    
    # Check if torch model method was called
    mock_model._generate_codebook.assert_called_once()
    
    # Test fallback without parameters
    result = wrapper._fallback_generate()
    
    # Check result
    assert isinstance(result, torch.Tensor)
    assert result.shape == (1, 32)  # mock_model.args.audio_num_codebooks = 32


def test_generate_frame():
    """Test frame generation."""
    # Create mock model
    mock_model = MockCSMModel()
    
    # Create wrapper
    wrapper = MLXModelWrapper(mock_model)
    
    # Mock frame_generator
    wrapper.frame_generator.generate_frame = MagicMock()
    wrapper.frame_generator.generate_frame.return_value = torch.zeros((1, 32))
    
    # Test generation
    tokens = torch.zeros((1, 10), dtype=torch.long)
    input_pos = torch.zeros((1, 10), dtype=torch.long)
    result = wrapper.generate_frame(tokens, input_pos, 0)
    
    # Check if frame_generator was called
    wrapper.frame_generator.generate_frame.assert_called_once()
    
    # Test fallback when MLX fails
    wrapper.frame_generator.generate_frame.side_effect = Exception("MLX error")
    
    with patch('csm.mlx_accel.components.model_wrapper.MLXModelWrapper.generate_frame_hybrid') as mock_hybrid:
        # Mock return value
        mock_hybrid.return_value = torch.zeros((1, 32))
        
        # Generate with fallback
        result = wrapper.generate_frame(tokens, input_pos, 0)
        
        # Check if hybrid approach was used
        mock_hybrid.assert_called_once()


def test_generate_frame_hybrid():
    """Test hybrid frame generation."""
    # Create mock model
    mock_model = MockCSMModel()
    
    # Create wrapper
    wrapper = MLXModelWrapper(mock_model)
    
    # Mock tensor conversion
    with patch('csm.mlx_accel.components.model_wrapper.torch_to_mlx') as mock_to_mlx, \
         patch('csm.mlx_accel.components.model_wrapper.mlx_to_torch') as mock_to_torch:
        
        # Mock return values
        mock_to_mlx.return_value = mx.zeros((1, 512))
        mock_to_torch.return_value = torch.zeros((1, 1))
        
        # Mock mx.matmul
        with patch.object(mx, 'matmul') as mock_matmul:
            mock_matmul.return_value = mx.zeros((1, 2051))
            
            # Mock categorical sampling
            with patch('csm.mlx_accel.components.sampling.mlx_categorical_sampling') as mock_sampling:
                mock_sampling.return_value = (mx.zeros((1, 1), dtype=mx.int32), True)
                
                # Test hybrid generation
                tokens = torch.zeros((1, 10), dtype=torch.long)
                input_pos = torch.zeros((1, 10), dtype=torch.long)
                result = wrapper.generate_frame_hybrid(tokens, input_pos, 0)
                
                # Check result
                assert isinstance(result, torch.Tensor)
                assert mock_model.forward_first_stage.call_count == 1
                assert mock_model._generate_codebook.call_count > 0


def test_reset_caches():
    """Test cache reset."""
    # Create mock model
    mock_model = MockCSMModel()
    
    # Create wrapper
    wrapper = MLXModelWrapper(mock_model)
    
    # Mock backbone and decoder
    wrapper.backbone = MagicMock()
    wrapper.decoder = MagicMock()
    
    # Reset caches
    wrapper.reset_caches()
    
    # Check if reset_caches was called on both components
    wrapper.backbone.reset_caches.assert_called_once()
    wrapper.decoder.reset_caches.assert_called_once()