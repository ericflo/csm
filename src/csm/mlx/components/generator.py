"""
MLX Generator for CSM with exact PyTorch-matching sampling.

This implementation uses a pure MLX approach with exact PyTorch-matching
sampling that achieves high audio quality without relying on PyTorch for
token generation.
"""

import time
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import numpy as np
import torch

from csm.mlx.mlx_wrapper import MLXWrapper
from csm.mlx.components.config import MLXConfig
from csm.mlx.components.utils import measure_time, is_mlx_available
from csm.mlx.mlx_layers import torch_to_mlx

# Import MLX if available
try:
    import mlx.core as mx
    import mlx.random
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    # Create a dummy module
    class DummyMX:
        def __getattr__(self, name):
            raise ImportError("MLX is not available")
    mx = DummyMX()

class MLXGenerator:
    """
    MLX-accelerated speech generator that handles the entire generation process
    using exact PyTorch-matching sampling for high-quality audio.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        device: Optional[torch.device] = None,
        debug: bool = False
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.debug = debug
        self.speaker = None  # Initialize speaker attribute
        self.text = None   # Initialize text attribute
        self.sample_rate = 24000  # Default sample rate
        self._last_audio = None  # Store last audio for direct access
        self._last_samples = None  # Store raw samples from generate_frame
        self._last_tokens = None  # Store last tokens for debugging
        
        # Always use exact MLX sampling for high quality
        self.sampling_mode = 'exact'
        
        # Check if MLX is available
        self.mlx_available = is_mlx_available()
        if self.mlx_available:
            # Initialize MLX wrapper
            try:
                # Create argument holder
                import argparse
                args = argparse.Namespace()
                
                # Copy any arguments from the model if available
                if hasattr(model, 'args'):
                    model_args = model.args
                    args.audio_vocab_size = getattr(model_args, 'audio_vocab_size', 2051)
                    args.audio_num_codebooks = getattr(model_args, 'audio_num_codebooks', 32)
                else:
                    # Default values
                    args.audio_vocab_size = 2051
                    args.audio_num_codebooks = 32
                    
                # Set debug flag
                args.debug = debug
                
                # Always use exact sampling with the optimized implementation
                args.use_exact_sampling = True
                args.use_pytorch_tokens = False
                
                # We're now using the optimized exact implementation by default
                # Since our cleanup, this is the only implementation available
                if debug:
                    print("Using optimized MLX sampling implementation")
                
                # Create MLX wrapper
                self.mlx_wrapper = MLXWrapper(model, args)
                if self.debug:
                    print("MLX wrapper initialized successfully")
                    print("Using exact PyTorch-matching sampling for high quality audio")
            except Exception as e:
                print(f"Error initializing MLX wrapper: {e}")
                self.mlx_wrapper = None
                self.mlx_available = False
        else:
            self.mlx_wrapper = None
            
    @measure_time
    def generate_speech(
        self,
        text: str,
        speaker: int = 0,
        temperature: float = 1.0,
        topk: int = 50,
        seed: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> torch.Tensor:
        """
        Generate speech audio from text using MLX acceleration.
        
        Args:
            text: Text to generate speech for
            speaker: Speaker ID (default: 0)
            temperature: Temperature for sampling
            topk: Top-k value for sampling
            seed: Random seed for reproducible generation
            progress_callback: Optional callback for progress updates
            
        Returns:
            Generated speech audio as a tensor
        """
        # Store the text for later reference
        self.text = text
        
        # Store the speaker ID
        self.speaker = speaker
        
        # Tokenize the input text
        text_tokens = self.tokenize(text)
        
        # Generate audio tokens
        audio_tokens = self.generate_audio_tokens(
            text_tokens=text_tokens,
            temperature=temperature,
            topk=topk,
            seed=seed,
            progress_callback=progress_callback
        )
        
        # Decode tokens to audio
        audio = self.decode_audio_tokens(audio_tokens)
        
        # Store the audio for reference
        self._last_audio = audio
        
        return audio
    
    def tokenize(self, text: str) -> torch.Tensor:
        """
        Tokenize text input.
        
        Args:
            text: Text to tokenize
            
        Returns:
            Tokenized text as a tensor
        """
        # Try different tokenization approaches
        tokens = None
        
        # Try direct tokenizer if available
        if self.tokenizer is not None:
            try:
                if hasattr(self.tokenizer, 'encode'):
                    # SentencePiece style
                    tokens = torch.tensor(self.tokenizer.encode(text))
                elif hasattr(self.tokenizer, 'tokenize'):
                    # Custom tokenizer
                    tokens = self.tokenizer.tokenize(text)
            except Exception as e:
                if self.debug:
                    print(f"Tokenizer failed: {e}")
                tokens = None
        
        # Try model's tokenize method as fallback
        if tokens is None and hasattr(self.model, 'tokenize'):
            try:
                tokens = self.model.tokenize(text)
            except Exception as e:
                if self.debug:
                    print(f"Model tokenize failed: {e}")
                raise ValueError(f"Failed to tokenize text: {text}")
        
        # Convert to tensor if needed
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens)
        
        # Add batch dimension if needed
        if len(tokens.shape) == 1:
            tokens = tokens.unsqueeze(0)
            
        if self.debug:
            print(f"Tokenized shape: {tokens.shape}")
            
        return tokens
        
    def generate_audio_tokens(
        self,
        text_tokens: torch.Tensor,
        temperature: float = 1.0,
        topk: int = 25,
        seed: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> torch.Tensor:
        """
        Generate audio tokens from text tokens.
        
        Args:
            text_tokens: Tokenized text input
            temperature: Temperature for sampling
            topk: Top-k value for sampling
            seed: Random seed for reproducible token generation
            progress_callback: Optional callback for progress updates
            
        Returns:
            Generated audio tokens
        """
        with torch.no_grad():
            if self.mlx_available and self.mlx_wrapper is not None:
                # Use MLX acceleration
                if self.debug:
                    print("Using MLX acceleration for audio token generation")
                
                # Pass all arguments including seed if provided
                mlx_args = {
                    "text_tokens": text_tokens,
                    "temperature": temperature,
                    "topk": topk,
                    "progress_callback": progress_callback
                }
                
                if seed is not None:
                    mlx_args["seed"] = seed
                    
                return self.generate_audio_tokens_mlx(**mlx_args)
            else:
                # Fall back to pure PyTorch
                if self.debug:
                    print("Using PyTorch for audio token generation")
                
                # Pass all arguments including seed if provided
                torch_args = {
                    "text_tokens": text_tokens,
                    "temperature": temperature,
                    "topk": topk,
                    "progress_callback": progress_callback
                }
                
                if seed is not None:
                    torch_args["seed"] = seed
                    
                return self.generate_audio_tokens_torch(**torch_args)
    
    def generate_audio_tokens_mlx(
        self,
        text_tokens: torch.Tensor,
        temperature: float = 1.0,
        topk: int = 25,
        seed: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> torch.Tensor:
        """
        Generate audio tokens using MLX acceleration with enhanced reliability.
        
        Args:
            text_tokens: Tokenized text input
            temperature: Temperature for sampling
            topk: Top-k value for sampling
            seed: Random seed for reproducible token generation
            progress_callback: Optional callback for progress updates
            
        Returns:
            Generated audio tokens
        """
        # Always use MLX with exact PyTorch-matching sampling
        import inspect
        
        if hasattr(self.model, 'generate'):
            try:
                sig = inspect.signature(self.model.generate)
                param_names = [param for param in sig.parameters]
                
                if self.debug:
                    print(f"MLX Model.generate parameters: {param_names}")
                
                # Create parameters based on what the model's generate method accepts
                generate_kwargs = {}
                
                # Handle text parameter
                if 'text' in param_names and self.text is not None:
                    # Always prioritize actual text over tokenized text
                    generate_kwargs['text'] = self.text
                elif 'prompt' in param_names and self.text is not None:
                    # Some models might use 'prompt' instead of 'text'
                    generate_kwargs['prompt'] = self.text
                    
                # Handle text_tokens or tokens
                if 'text_tokens' in param_names:
                    generate_kwargs['text_tokens'] = text_tokens
                elif 'tokens' in param_names:
                    generate_kwargs['tokens'] = text_tokens
                    
                # Handle temperature parameter
                if 'temperature' in param_names:
                    generate_kwargs['temperature'] = temperature
                    
                # Handle topk/top_k difference
                if 'topk' in param_names:
                    generate_kwargs['topk'] = topk
                elif 'top_k' in param_names:
                    generate_kwargs['top_k'] = topk
                
                # Pass seed parameter if provided
                if seed is not None:
                    if 'seed' in param_names:
                        generate_kwargs['seed'] = seed
                    # Also set the MLX random seed for all other sampling operations
                    import mlx
                    mlx.random.seed(seed)
                    if self.debug:
                        print(f"Set MLX random seed to {seed}")
                    
                # Handle callback/progress_callback
                if 'callback' in param_names and progress_callback is not None:
                    generate_kwargs['callback'] = progress_callback
                elif 'progress_callback' in param_names and progress_callback is not None:
                    generate_kwargs['progress_callback'] = progress_callback
                    
                # Handle other common parameters
                if 'use_mlx' in param_names:
                    generate_kwargs['use_mlx'] = True
                    
                # Handle speaker parameter
                if 'speaker' in param_names:
                    # Use speaker ID directly
                    generate_kwargs['speaker'] = self.speaker
                        
                # Handle context parameter
                if 'context' in param_names:
                    generate_kwargs['context'] = []
                    
                # Handle max_audio_length_ms parameter
                if 'max_audio_length_ms' in param_names:
                    generate_kwargs['max_audio_length_ms'] = 10000  # 10 seconds default
                        
                if self.debug:
                    print(f"Calling MLX generate with kwargs: {generate_kwargs}")
                
                try:
                    # Call the generate method with the appropriate arguments
                    raw_output = self.model.generate(**generate_kwargs)
                    
                    # Process the output based on what was returned
                    if isinstance(raw_output, list) and len(raw_output) > 0:
                        # List of segments
                        from ..generator import Segment
                        if isinstance(raw_output[0], Segment):
                            # Extract audio tokens from segments
                            self._last_tokens = raw_output[0].tokens if hasattr(raw_output[0], 'tokens') else None
                            if self._last_tokens is None and hasattr(raw_output[0], 'audio_tokens'):
                                self._last_tokens = raw_output[0].audio_tokens
                            
                            # Return the tokens if available
                            if self._last_tokens is not None:
                                return self._last_tokens
                    elif isinstance(raw_output, dict) and 'tokens' in raw_output:
                        # Dictionary with tokens
                        self._last_tokens = raw_output['tokens']
                        return self._last_tokens
                    elif isinstance(raw_output, torch.Tensor):
                        # Direct tensor output
                        self._last_tokens = raw_output
                        return raw_output
                    else:
                        if self.debug:
                            print(f"Unknown output format from model.generate: {type(raw_output)}")
                            
                    # Try to find tokens in model attributes if output format wasn't recognized
                    if hasattr(self.model, '_last_tokens'):
                        self._last_tokens = self.model._last_tokens
                        return self._last_tokens
                    elif hasattr(self.model, 'audio_tokens'):
                        self._last_tokens = self.model.audio_tokens
                        return self._last_tokens
                        
                except Exception as e:
                    if self.debug:
                        print(f"Error in MLX model.generate: {e}")
                    raise
            except Exception as e:
                if self.debug:
                    print(f"Error using model.generate: {e}")
                raise
        
        # If we get here, try using MLX wrapper directly
        if self.mlx_wrapper is not None:
            try:
                # Convert tokens to MLX format
                mlx_text_tokens = torch_to_mlx(text_tokens)
                
                if self.debug:
                    print(f"Using MLX wrapper with text tokens shape: {mlx_text_tokens.shape}")
                    
                # Create audio tokens using MLX wrapper with frame generator
                audio_tokens = self.mlx_wrapper.generate_tokens(
                    text_tokens=mlx_text_tokens,
                    temperature=temperature,
                    topk=topk,
                    seed=seed,
                    progress_callback=progress_callback
                )
                
                # Store tokens for debug and return
                if audio_tokens is not None:
                    self._last_tokens = audio_tokens
                    return audio_tokens
            except Exception as e:
                if self.debug:
                    print(f"Error using MLX wrapper: {e}")
                raise
        
        # If all MLX approaches failed, try fallback to PyTorch
        if self.debug:
            print("All MLX approaches failed, falling back to PyTorch")
            
        return self.generate_audio_tokens_torch(
            text_tokens=text_tokens,
            temperature=temperature,
            topk=topk,
            seed=seed,
            progress_callback=progress_callback
        )
    
    def generate_audio_tokens_torch(
        self,
        text_tokens: torch.Tensor,
        temperature: float = 1.0,
        topk: int = 25,
        seed: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> torch.Tensor:
        """
        Generate audio tokens using PyTorch as a fallback.
        
        Args:
            text_tokens: Tokenized text input
            temperature: Temperature for sampling
            topk: Top-k value for sampling
            seed: Random seed for reproducible generation
            progress_callback: Optional callback for progress updates
            
        Returns:
            Generated audio tokens
        """
        if self.debug:
            print("Using PyTorch fallback for audio token generation")
            
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        # Create keyword arguments for generate method
        kwargs = {
            "temperature": temperature,
        }
        
        # Handle parameter naming differences (some models use top_k, others use topk)
        # Try both parameter names for compatibility
        try:
            import inspect
            # Inspect the function signature of the model's generate method
            if hasattr(self.model, 'generate'):
                sig = inspect.signature(self.model.generate)
                param_names = list(sig.parameters.keys())
                
                # Use the correct parameter name based on what the function expects
                if 'top_k' in param_names:
                    kwargs['top_k'] = topk
                elif 'topk' in param_names:
                    kwargs['topk'] = topk
                else:
                    # Default to top_k as most models use this
                    kwargs['top_k'] = topk
            else:
                # Default to both for maximum compatibility
                kwargs['top_k'] = topk
                kwargs['topk'] = topk
        except Exception:
            # If inspection fails, include both parameter names
            kwargs['top_k'] = topk
            kwargs['topk'] = topk
        
        if progress_callback is not None:
            kwargs["callback"] = progress_callback
            
        # Handle speaker parameter
        if self.speaker is not None:
            kwargs["speaker"] = self.speaker
            
        with torch.no_grad():
            try:
                # Try calling generate with text
                if self.text is not None:
                    segments = self.model.generate(text=self.text, **kwargs)
                    
                    # Extract tokens from the first segment
                    if hasattr(segments[0], 'tokens'):
                        # Most recent approach stores tokens directly on the segment
                        tokens = segments[0].tokens
                    elif hasattr(segments[0], 'audio_tokens'):
                        # Alternative attribute name
                        tokens = segments[0].audio_tokens
                    else:
                        # Try to find tokens on the model
                        tokens = getattr(self.model, '_last_tokens', None)
                        
                    # Store tokens for debugging
                    self._last_tokens = tokens
                    
                    return tokens
                else:
                    # If text isn't available, use tokens directly
                    segments = self.model.generate(tokens=text_tokens, **kwargs)
                    
                    # Extract tokens from the first segment
                    if hasattr(segments[0], 'tokens'):
                        tokens = segments[0].tokens
                    elif hasattr(segments[0], 'audio_tokens'):
                        tokens = segments[0].audio_tokens
                    else:
                        tokens = getattr(self.model, '_last_tokens', None)
                        
                    # Store tokens for debugging
                    self._last_tokens = tokens
                    
                    return tokens
            except Exception as e:
                print(f"Error generating audio tokens with PyTorch: {e}")
                if hasattr(self.model, '_last_tokens'):
                    return self.model._last_tokens
                raise ValueError(f"Failed to generate audio tokens: {e}")
    
    def decode_audio_tokens(self, audio_tokens: torch.Tensor) -> torch.Tensor:
        """
        Decode audio tokens to audio waveform.
        
        Args:
            audio_tokens: Audio tokens to decode
            
        Returns:
            Audio waveform
        """
        if self.debug:
            print(f"Decoding audio tokens with shape: {audio_tokens.shape}")
            
        # Try different approaches to decode audio
        audio = None
        
        try:
            # Try using model's decode_audio method if available
            if hasattr(self.model, 'decode_audio'):
                audio = self.model.decode_audio(audio_tokens)
                if audio is not None:
                    if self.debug:
                        print(f"Used model.decode_audio, shape: {audio.shape}")
                    return audio
            
            # Try using model's decoder if available
            if hasattr(self.model, 'decoder') and hasattr(self.model.decoder, 'decode'):
                audio = self.model.decoder.decode(audio_tokens)
                if audio is not None:
                    if self.debug:
                        print(f"Used model.decoder.decode, shape: {audio.shape}")
                    return audio
                    
            # Check if tokens is already audio (some implementations store audio in '_last_audio')
            if audio_tokens.shape[-1] > 100:  # Heuristic to detect audio waveform vs. tokens
                if self.debug:
                    print(f"Tokens appear to already be audio waveform")
                return audio_tokens
        except Exception as e:
            print(f"Error decoding audio: {e}")
            
        # If we get here, try using model's _last_audio if available
        if hasattr(self.model, '_last_audio') and self.model._last_audio is not None:
            audio = self.model._last_audio
            if self.debug:
                print(f"Used model._last_audio, shape: {audio.shape}")
            return audio
            
        # If we get here, try using model's last_samples attribute if available
        if hasattr(self.model, 'last_samples') and self.model.last_samples is not None:
            audio = self.model.last_samples
            if self.debug:
                print(f"Used model.last_samples, shape: {audio.shape}")
            return audio
            
        # If we get here and still no audio, raise an error
        if audio is None:
            raise ValueError("Failed to decode audio tokens")
            
        return audio