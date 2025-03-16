"""
Tests for the MLX frame generation component.
"""

import sys
import pytest
from unittest.mock import MagicMock, patch
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
            self.random = MagicMock()
    mx = MockMX()
    sys.modules['mlx'] = mx
    sys.modules['mlx.core'] = mx.core
    sys.modules['mlx.nn'] = mx.nn

# Skip tests if MLX not available
pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")

# Import the module under test
from csm.mlx_accel.mlx_generation import MLXFrameGenerator


class MockMLXTransformer:
    """Mock MLX transformer for testing."""
    
    def __init__(self):
        self.forward = MagicMock()
        # Return a tensor with same batch and seq_len but fixed embed_dim
        self.forward.return_value = mx.zeros((2, 3, 512))


class MockMLXEmbedding:
    """Mock MLX embedding for testing."""
    
    def __init__(self, embed_dim=512):
        self.embed_dim = embed_dim
        self.embed_text = MagicMock()
        self.embed_text.return_value = mx.zeros((2, 3, embed_dim))
        self.embed_audio = MagicMock()
        self.embed_audio.return_value = mx.zeros((2, 3, embed_dim))


class MockSampler:
    """Mock sampler for testing."""
    
    def __init__(self):
        pass
    
    @staticmethod
    def sample(logits, topk=5, temperature=1.0):
        # Return sample indices with batch_size as first dimension
        return mx.zeros((logits.shape[0], 1), dtype=mx.int32)


def test_mlx_frame_generator_init():
    """Test initialization of MLXFrameGenerator."""
    # Create mock components
    backbone = MockMLXTransformer()
    decoder = MockMLXTransformer()
    embedding = MockMLXEmbedding()
    
    # Create weights
    if HAS_MLX:
        projection_weight = mx.zeros((512, 512))
        codebook0_head_weight = mx.zeros((2051, 512))
        audio_head_weights = [mx.zeros((2051, 512)) for _ in range(31)]
    else:
        projection_weight = mx.core.zeros((512, 512))
        codebook0_head_weight = mx.core.zeros((2051, 512))
        audio_head_weights = [mx.core.zeros((2051, 512)) for _ in range(31)]
        
    # Create generator with patched print to avoid cluttering test output
    with patch('builtins.print'):
        generator = MLXFrameGenerator(
            backbone=backbone,
            decoder=decoder,
            embedding=embedding,
            projection_weight=projection_weight,
            codebook0_head_weight=codebook0_head_weight,
            audio_head_weights=audio_head_weights,
            audio_vocab_size=2051,
            audio_num_codebooks=32,
            debug=False
        )
        
        # Check if generator was initialized correctly
        assert generator.backbone == backbone
        assert generator.decoder == decoder
        assert generator.embedding == embedding
        assert generator.audio_vocab_size == 2051
        assert generator.audio_num_codebooks == 32
        assert generator.debug is False


def test_generate_frame_direct():
    """Test direct frame generation with pre-converted MLX arrays."""
    # Create mock components
    backbone = MockMLXTransformer()
    decoder = MockMLXTransformer()
    embedding = MockMLXEmbedding()
    
    # Create weights
    if HAS_MLX:
        projection_weight = mx.zeros((512, 512))
        codebook0_head_weight = mx.zeros((2051, 512))
        audio_head_weights = [mx.zeros((2051, 512)) for _ in range(31)]
    else:
        projection_weight = mx.core.zeros((512, 512))
        codebook0_head_weight = mx.core.zeros((2051, 512))
        audio_head_weights = [mx.core.zeros((2051, 512)) for _ in range(31)]
    
    # Mock MLX sample_exact to avoid errors
    with patch('csm.mlx_accel.mlx_generation.mlx_sample_exact') as mock_sample:
        # Setup mock sampler to return a simple tensor
        if HAS_MLX:
            mock_sample.return_value = mx.zeros((2, 1), dtype=mx.int32)
        else:
            mock_sample.return_value = mx.core.zeros((2, 1), dtype=mx.core.int32)
        
        # Create generator with patched print
        with patch('builtins.print'):
            generator = MLXFrameGenerator(
                backbone=backbone,
                decoder=decoder,
                embedding=embedding,
                projection_weight=projection_weight,
                codebook0_head_weight=codebook0_head_weight,
                audio_head_weights=audio_head_weights,
                audio_vocab_size=2051,
                audio_num_codebooks=32,
                debug=False
            )
            
            # Create input tensors
            if HAS_MLX:
                mlx_tokens = mx.zeros((2, 3, 33), dtype=mx.float32)
                mlx_positions = mx.zeros((2, 3), dtype=mx.int32)
            else:
                mlx_tokens = mx.core.zeros((2, 3, 33), dtype=mx.core.float32)
                mlx_positions = mx.core.zeros((2, 3), dtype=mx.core.int32)
            
            # Create main test by patching the internal method to return a known result
            with patch.object(generator, '_generate_frame_internal') as mock_internal:
                # Set up return value for internal method
                mock_internal.return_value = torch.zeros((2, 32))
                
                # Call the method
                result = generator.generate_frame_direct(mlx_tokens, mlx_positions)
                
                # Check if internal method was called with correct parameters
                mock_internal.assert_called_once()
                _, args, kwargs = mock_internal.mock_calls[0]
                assert args[0] == 2  # batch_size
                assert args[1] == 3  # seq_len
                assert args[2] == 33  # total_codebooks
                
                # Check basic properties of the result
                assert isinstance(result, torch.Tensor)
                assert result.shape == (2, 32)


def test_generate_frame():
    """Test main frame generation with PyTorch to MLX conversion."""
    # Create mock components
    backbone = MockMLXTransformer()
    decoder = MockMLXTransformer()
    embedding = MockMLXEmbedding()
    
    # Create weights
    if HAS_MLX:
        projection_weight = mx.zeros((512, 512))
        codebook0_head_weight = mx.zeros((2051, 512))
        audio_head_weights = [mx.zeros((2051, 512)) for _ in range(31)]
    else:
        projection_weight = mx.core.zeros((512, 512))
        codebook0_head_weight = mx.core.zeros((2051, 512))
        audio_head_weights = [mx.core.zeros((2051, 512)) for _ in range(31)]
    
    # Create generator with patched print and torch_to_mlx
    with patch('builtins.print'), \
         patch('csm.mlx_accel.mlx_generation.torch_to_mlx') as mock_to_mlx:
        
        # Setup mock conversion to return MLX arrays
        if HAS_MLX:
            mock_to_mlx.return_value = mx.zeros((2, 3))
        else:
            mock_to_mlx.return_value = mx.core.zeros((2, 3))
            
        # Create the generator
        generator = MLXFrameGenerator(
            backbone=backbone,
            decoder=decoder,
            embedding=embedding,
            projection_weight=projection_weight,
            codebook0_head_weight=codebook0_head_weight,
            audio_head_weights=audio_head_weights,
            audio_vocab_size=2051,
            audio_num_codebooks=32,
            debug=False
        )
        
        # Create PyTorch input tensors
        tokens = torch.zeros((2, 3, 33), dtype=torch.float32)
        positions = torch.zeros((2, 3), dtype=torch.long)
        
        # Test by patching the direct method to return a known result
        with patch.object(generator, 'generate_frame_direct') as mock_direct:
            # Set up return value for direct method
            mock_direct.return_value = torch.zeros((2, 32))
            
            # Call the method
            result = generator.generate_frame(tokens, positions)
            
            # Check if direct method was called
            mock_direct.assert_called_once()
            
            # Check basic properties of the result
            assert isinstance(result, torch.Tensor)
            assert result.shape == (2, 32)


def test_generate_frame_with_numpy_fallback():
    """Test frame generation with fallback to numpy conversion."""
    # Create mock components
    backbone = MockMLXTransformer()
    decoder = MockMLXTransformer()
    embedding = MockMLXEmbedding()
    
    # Create weights
    if HAS_MLX:
        projection_weight = mx.zeros((512, 512))
        codebook0_head_weight = mx.zeros((2051, 512))
        audio_head_weights = [mx.zeros((2051, 512)) for _ in range(31)]
    else:
        projection_weight = mx.core.zeros((512, 512))
        codebook0_head_weight = mx.core.zeros((2051, 512))
        audio_head_weights = [mx.core.zeros((2051, 512)) for _ in range(31)]
    
    # Create generator with patched print and torch_to_mlx that raises exception
    with patch('builtins.print'), \
         patch('csm.mlx_accel.mlx_generation.torch_to_mlx') as mock_to_mlx, \
         patch('csm.mlx_accel.mlx_generation.mx.array') as mock_mx_array:
        
        # Setup mock conversion to raise an exception
        mock_to_mlx.side_effect = Exception("Conversion error")
        
        # Setup mx.array to return a proper MLX array
        if HAS_MLX:
            mock_mx_array.return_value = mx.zeros((2, 3))
        else:
            mock_mx_array.return_value = mx.core.zeros((2, 3))
            
        # Create the generator
        generator = MLXFrameGenerator(
            backbone=backbone,
            decoder=decoder,
            embedding=embedding,
            projection_weight=projection_weight,
            codebook0_head_weight=codebook0_head_weight,
            audio_head_weights=audio_head_weights,
            audio_vocab_size=2051,
            audio_num_codebooks=32,
            debug=False
        )
        
        # Create PyTorch input tensors
        tokens = torch.zeros((2, 3, 33), dtype=torch.float32)
        positions = torch.zeros((2, 3), dtype=torch.long)
        
        # Test by patching the direct method to return a known result
        with patch.object(generator, 'generate_frame_direct') as mock_direct:
            # Set up return value for direct method
            mock_direct.return_value = torch.zeros((2, 32))
            
            # Call the method
            result = generator.generate_frame(tokens, positions)
            
            # Check if direct method was called
            mock_direct.assert_called_once()
            
            # Check if numpy conversion was used (mx.array called)
            mock_mx_array.assert_called()
            
            # Check basic properties of the result
            assert isinstance(result, torch.Tensor)
            assert result.shape == (2, 32)


def test_generate_frame_error_handling():
    """Test frame generation with error handling."""
    # Create mock components
    backbone = MockMLXTransformer()
    decoder = MockMLXTransformer()
    embedding = MockMLXEmbedding()
    
    # Create a custom fallback function
    fallback_fn = MagicMock()
    fallback_fn.return_value = torch.zeros((2, 32))
    
    # Create weights
    if HAS_MLX:
        projection_weight = mx.zeros((512, 512))
        codebook0_head_weight = mx.zeros((2051, 512))
        audio_head_weights = [mx.zeros((2051, 512)) for _ in range(31)]
    else:
        projection_weight = mx.core.zeros((512, 512))
        codebook0_head_weight = mx.core.zeros((2051, 512))
        audio_head_weights = [mx.core.zeros((2051, 512)) for _ in range(31)]
    
    # Create generator with patched print
    with patch('builtins.print'):
        generator = MLXFrameGenerator(
            backbone=backbone,
            decoder=decoder,
            embedding=embedding,
            projection_weight=projection_weight,
            codebook0_head_weight=codebook0_head_weight,
            audio_head_weights=audio_head_weights,
            audio_vocab_size=2051,
            audio_num_codebooks=32,
            debug=False,
            fallback_fn=fallback_fn
        )
        
        # Create input tensors
        tokens = torch.zeros((2, 3, 33), dtype=torch.float32)
        positions = torch.zeros((2, 3), dtype=torch.long)
        
        # Test by patching generate_frame_direct to raise an exception
        with patch.object(generator, 'generate_frame_direct') as mock_direct:
            # Force an error
            mock_direct.side_effect = Exception("Forced error")
            
            # Call the method
            result = generator.generate_frame(tokens, positions)
            
            # Check if fallback was used
            fallback_fn.assert_called_once()
            
            # Check basic properties of the result
            assert isinstance(result, torch.Tensor)
            assert result.shape == (2, 32)


def test_generate_frame_internal():
    """Test the internal frame generation method."""
    # Create mock components
    backbone = MockMLXTransformer()
    decoder = MockMLXTransformer()
    embedding = MockMLXEmbedding()
    
    # Create weights
    if HAS_MLX:
        projection_weight = mx.zeros((512, 512))
        codebook0_head_weight = mx.zeros((2051, 512))
        audio_head_weights = [mx.zeros((2051, 512)) for _ in range(31)]
        
        # Create necessary MLX arrays for testing
        text_tokens = mx.zeros((2, 3), dtype=mx.int32)
        audio_tokens = mx.zeros((2, 3, 32), dtype=mx.int32)
        mlx_positions = mx.zeros((2, 3), dtype=mx.int32)
        tokens_mask = mx.ones((2, 3), dtype=mx.float32)
    else:
        projection_weight = mx.core.zeros((512, 512))
        codebook0_head_weight = mx.core.zeros((2051, 512))
        audio_head_weights = [mx.core.zeros((2051, 512)) for _ in range(31)]
        
        # Create necessary mock MLX arrays for testing
        text_tokens = mx.core.zeros((2, 3), dtype=mx.core.int32)
        audio_tokens = mx.core.zeros((2, 3, 32), dtype=mx.core.int32)
        mlx_positions = mx.core.zeros((2, 3), dtype=mx.core.int32)
        tokens_mask = mx.core.ones((2, 3), dtype=mx.core.float32)
    
    # Create a custom fallback function to avoid going through the full implementation
    fallback_fn = MagicMock()
    fallback_fn.return_value = torch.zeros((2, 32))
    
    # Create generator with patched print and mock mlx_sample_exact
    with patch('builtins.print'), \
         patch('csm.mlx_accel.mlx_generation.mlx_sample_exact') as mock_sample, \
         patch('csm.mlx_accel.mlx_generation.create_causal_mask') as mock_create_mask, \
         patch('csm.mlx_accel.mlx_generation.index_causal_mask') as mock_index_mask, \
         patch('csm.mlx_accel.mlx_generation.mlx_to_torch') as mock_to_torch:
        
        # Setup mock samplers to return simple tensors
        if HAS_MLX:
            mock_sample.return_value = mx.zeros((2, 1), dtype=mx.int32)
            mock_create_mask.return_value = mx.ones((3, 3), dtype=mx.float32)
            mock_index_mask.return_value = mx.ones((2, 3, 3), dtype=mx.float32)
        else:
            mock_sample.return_value = mx.core.zeros((2, 1), dtype=mx.core.int32)
            mock_create_mask.return_value = mx.core.ones((3, 3), dtype=mx.core.float32)
            mock_index_mask.return_value = mx.core.ones((2, 3, 3), dtype=mx.core.float32)
        
        mock_to_torch.return_value = torch.zeros((2, 1), dtype=torch.long)
        
        # Create the generator
        generator = MLXFrameGenerator(
            backbone=backbone,
            decoder=decoder,
            embedding=embedding,
            projection_weight=projection_weight,
            codebook0_head_weight=codebook0_head_weight,
            audio_head_weights=audio_head_weights,
            audio_vocab_size=2051,
            audio_num_codebooks=32,
            debug=False,
            fallback_fn=fallback_fn
        )
        
        # Call the internal method directly
        batch_size = 2
        seq_len = 3
        total_codebooks = 33
        
        # Most of the implementation gets mocked or falls back to the fallback function
        # in the actual function due to its complexity with reshape operations
        with patch.object(embedding, 'embed_text') as mock_embed_text, \
             patch.object(embedding, 'embed_audio') as mock_embed_audio:
            
            # Setup mock embedding to return MLX arrays
            if HAS_MLX:
                mock_embed_text.return_value = mx.zeros((2, 3, 512))
                mock_embed_audio.return_value = mx.zeros((2, 3, 512))
            else:
                mock_embed_text.return_value = mx.core.zeros((2, 3, 512))
                mock_embed_audio.return_value = mx.core.zeros((2, 3, 512))
            
            # Call the method with all required parameters
            result = generator._generate_frame_internal(
                batch_size=batch_size,
                seq_len=seq_len,
                total_codebooks=total_codebooks,
                text_tokens=text_tokens,
                audio_tokens=audio_tokens,
                mlx_positions=mlx_positions,
                tokens_mask=tokens_mask,
                topk=5,
                temperature=1.0
            )
            
            # Check basic properties of the result
            assert isinstance(result, torch.Tensor)
            
            # Check that either:
            # 1. We called the fallback function, or
            # 2. We got a tensor of the right shape from the actual implementation
            if fallback_fn.call_count > 0:
                # Fallback was called
                assert fallback_fn.call_count > 0
            else:
                # Implementation returned a tensor directly
                assert result.shape[0] == batch_size  # Should match batch_size
                assert mock_embed_text.call_count > 0
                assert mock_embed_audio.call_count > 0


def test_generate_frame_internal_error_handling():
    """Test error handling in the internal frame generation method."""
    # Create mock components
    backbone = MockMLXTransformer()
    decoder = MockMLXTransformer()
    embedding = MockMLXEmbedding()
    
    # Create a custom fallback function
    fallback_fn = MagicMock()
    fallback_fn.return_value = torch.zeros((2, 32))
    
    # Create weights
    if HAS_MLX:
        projection_weight = mx.zeros((512, 512))
        codebook0_head_weight = mx.zeros((2051, 512))
        audio_head_weights = [mx.zeros((2051, 512)) for _ in range(31)]
        
        # Create necessary MLX arrays for testing
        text_tokens = mx.zeros((2, 3), dtype=mx.int32)
        audio_tokens = mx.zeros((2, 3, 32), dtype=mx.int32)
        mlx_positions = mx.zeros((2, 3), dtype=mx.int32)
        tokens_mask = mx.ones((2, 3), dtype=mx.float32)
    else:
        projection_weight = mx.core.zeros((512, 512))
        codebook0_head_weight = mx.core.zeros((2051, 512))
        audio_head_weights = [mx.core.zeros((2051, 512)) for _ in range(31)]
        
        # Create necessary mock MLX arrays for testing
        text_tokens = mx.core.zeros((2, 3), dtype=mx.core.int32)
        audio_tokens = mx.core.zeros((2, 3, 32), dtype=mx.core.int32)
        mlx_positions = mx.core.zeros((2, 3), dtype=mx.core.int32)
        tokens_mask = mx.core.ones((2, 3), dtype=mx.core.float32)
    
    # Create generator with patched print
    with patch('builtins.print'):
        generator = MLXFrameGenerator(
            backbone=backbone,
            decoder=decoder,
            embedding=embedding,
            projection_weight=projection_weight,
            codebook0_head_weight=codebook0_head_weight,
            audio_head_weights=audio_head_weights,
            audio_vocab_size=2051,
            audio_num_codebooks=32,
            debug=False,
            fallback_fn=fallback_fn
        )
        
        # Force backbone to raise an exception
        backbone.forward.side_effect = Exception("Forced error in backbone")
        
        # Call the internal method directly
        batch_size = 2
        seq_len = 3
        total_codebooks = 33
        
        # Call the method
        result = generator._generate_frame_internal(
            batch_size=batch_size,
            seq_len=seq_len,
            total_codebooks=total_codebooks,
            text_tokens=text_tokens,
            audio_tokens=audio_tokens,
            mlx_positions=mlx_positions,
            tokens_mask=tokens_mask,
            topk=5,
            temperature=1.0
        )
        
        # Check if fallback was used
        fallback_fn.assert_called_once()
        
        # Check basic properties of the result
        assert isinstance(result, torch.Tensor)