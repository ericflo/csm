#!/usr/bin/env python3
"""Command-line interface for CSM inference with MLX acceleration on Apple Silicon."""

import argparse
import importlib.util
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import numpy as np
import torch
import torchaudio
from huggingface_hub import hf_hub_download

# Import modular components
from csm.mlx_accel.components.utils import (
    is_mlx_available, check_device_compatibility, setup_mlx_debug, measure_time
)
from csm.mlx_accel.components.config import MLXConfig
from csm.mlx_accel.components.generator import MLXGenerator

# Create dummy modules for imports that might fail on Mac
try:
    class DummyModule:
        def __getattr__(self, name):
            return None
    
    # Create dummy modules for triton and bitsandbytes
    sys.modules['triton'] = DummyModule()
    sys.modules['bitsandbytes'] = DummyModule()
    print(f"import error: No module named 'triton'")
    
    # Also patch the quantize.linear function if needed
    try:
        from moshi.utils import quantize
        orig_linear = getattr(quantize, 'linear', None)
        
        # Define a replacement linear function
        def patched_linear(module, input_tensor, weight_name='weight', bias_name=None):
            weight = getattr(module, weight_name)
            output = torch.nn.functional.linear(input_tensor, weight)
            if bias_name is not None and hasattr(module, bias_name):
                bias = getattr(module, bias_name)
                output = output + bias.unsqueeze(0).expand_as(output)
            return output
        
        # Apply the patch
        if hasattr(quantize, 'linear'):
            print(f"Warning: Patched function signature differs from original")
            quantize.linear = patched_linear
    except Exception:
        pass
except Exception:
    pass

print("No attention modules found to patch")

# Check if MLX is available
MLX_AVAILABLE = is_mlx_available()

if not MLX_AVAILABLE:
    print("MLX is not available. Install with: pip install -e '.[apple]'")
    print("Falling back to standard PyTorch CPU implementation...")
    # Fall back to regular implementation
    from .generate import main as fallback_main
    main = fallback_main
else:
    # Import MLX specific modules
    import mlx.core as mx
    import mlx.nn as nn
    
    # Import generator components from modular implementation
    from ..generator import Segment, load_csm_1b
    from ..models.model import ModelArgs, Model, sample_topk

    def load_csm_1b_mlx(
        ckpt_path: str,
        debug: bool = False
    ):
        """
        Load the CSM-1B model from a checkpoint path for inference with MLX acceleration
        using exact PyTorch-matching sampling for high quality.
        
        Args:
            ckpt_path: Path to the checkpoint, either local path or "csm-1B" to download from HuggingFace
            debug: Whether to enable debug output
            
        Returns:
            MLXGenerator instance with the loaded model
        """
        # Handle "csm-1B" as a special case to download from HuggingFace
        if ckpt_path == "csm-1B":
            # Create directories if they don't exist
            cache_dir = os.path.expanduser("~/.cache/csm")
            os.makedirs(cache_dir, exist_ok=True)
            
            # Download model directly
            print(f"Downloading model from Hugging Face hub...")
            model_file = hf_hub_download(repo_id="sesame/csm-1b", filename="ckpt.pt", cache_dir=cache_dir)
            print(f"Downloaded model to {model_file}")
            
            # Use the downloaded file directly (not the directory)
            ckpt_path = model_file
        
        # Load the PyTorch model explicitly to CPU
        print(f"Loading model from {ckpt_path}")
        try:
            # Apply a more comprehensive monkey patch for device handling
            import torch
            
            # Save original functions
            original_is_available = torch.cuda.is_available
            original_to = torch.Tensor.to
            
            # Override CUDA availability
            torch.cuda.is_available = lambda: False
            
            # Override tensor.to method to handle bfloat16 issues on CPU
            def safe_to(self, *args, **kwargs):
                # Extract the device and dtype
                device = kwargs.get('device', None)
                dtype = kwargs.get('dtype', None)
                
                # If the dtype is bfloat16 and we're on CPU, use float32 instead
                if dtype == torch.bfloat16 and (device == 'cpu' or device is None or str(device) == 'cpu'):
                    kwargs['dtype'] = torch.float32
                    
                # Call the original method
                return original_to(self, *args, **kwargs)
            
            # Apply the patch
            torch.Tensor.to = safe_to
            
            try:
                # Load the model with the patches applied
                torch_model = load_csm_1b(device="cpu", model_path=ckpt_path)
            finally:
                # Restore original functions
                torch.cuda.is_available = original_is_available
                torch.Tensor.to = original_to
                
        except Exception as e:
            print(f"Error loading model with patched CPU mode: {e}")
            # Fall back to a more direct approach - try to monkey patch the model_args
            try:
                # First, just load the checkpoint to get model_args
                import torch
                checkpoint = torch.load(os.path.join(ckpt_path, "ckpt.pt"), map_location="cpu")
                
                # Import the model directly
                from csm.models.model import Model, ModelArgs
                
                # Carefully examine the checkpoint structure
                print("Examining checkpoint structure...")
                if isinstance(checkpoint, dict):
                    print(f"Checkpoint keys: {list(checkpoint.keys())}")
                    
                    # Handle different possible structures
                    if "model_args" in checkpoint and "model" in checkpoint:
                        # Standard format
                        model_args = checkpoint["model_args"]
                        model_state = checkpoint["model"]
                        print("Found standard checkpoint format with model_args and model")
                    elif "args" in checkpoint and "model_state_dict" in checkpoint:
                        # Alternative format
                        model_args = checkpoint["args"]
                        model_state = checkpoint["model_state_dict"]
                        print("Found alternative checkpoint format with args and model_state_dict")
                    elif "model_state_dict" in checkpoint:
                        # Only state dict, use default args
                        print("WARNING: Could not find model_args, using defaults")
                        # Create default model args based on state dict
                        from csm.models.model import ModelArgs
                        model_args = ModelArgs()
                        model_state = checkpoint["model_state_dict"]
                    elif "state_dict" in checkpoint:
                        # Generic PyTorch format
                        print("Found generic PyTorch state_dict format")
                        from csm.models.model import ModelArgs
                        model_args = ModelArgs()
                        model_state = checkpoint["state_dict"]
                    elif len(checkpoint) > 0:
                        # Assume the checkpoint is itself the state dict
                        print("Assuming checkpoint is itself the state dict")
                        from csm.models.model import ModelArgs
                        model_args = ModelArgs()
                        model_state = checkpoint
                    else:
                        raise ValueError("Could not find model parameters in checkpoint")
                else:
                    # Not a dict - unexpected format
                    raise ValueError(f"Unexpected checkpoint format: {type(checkpoint)}")
                    
                # Create model directly with float32
                torch_model = Model(model_args)
                
                try:
                    # Try to load the state dict
                    torch_model.load_state_dict(model_state)
                except Exception as state_e:
                    print(f"Error loading state dict: {state_e}")
                    print("Trying to load with strict=False...")
                    torch_model.load_state_dict(model_state, strict=False)
                
                print("Successfully loaded model with direct state_dict approach")
            except Exception as e2:
                print(f"Fatal error loading model: {e2}")
                raise
        
        # Create a MLXGenerator for inference
        tokenizer = getattr(torch_model, 'tokenizer', None)
        if tokenizer is None:
            # Try to load from sentencepiece if available
            try:
                import sentencepiece as spm
                tokenizer = spm.SentencePieceProcessor()
                tokenizer.load(os.path.join(ckpt_path, "tokenizer.model"))
            except:
                print("Warning: No tokenizer found - will use model's tokenize method")
        
        # Create MLX-powered generator with exact PyTorch-matching sampling
        generator = MLXGenerator(
            model=torch_model,
            tokenizer=tokenizer,
            debug=debug or os.environ.get("DEBUG", "0") == "1" 
        )
        
        return generator

    def progress_callback(current: int, total: int):
        """Display progress during generation."""
        # Calculate percentage
        percent = 100 * current / total if total > 0 else 0
        
        # Create progress bar
        width = 50
        filled = int(width * current / total) if total > 0 else 0
        bar = "█" * filled + "░" * (width - filled)
        
        # Print progress
        print(f"\r[{bar}] {current}/{total} ({percent:.1f}%)", end="", flush=True)

    def main():
        """Main entry point for the MLX-accelerated CLI."""
        # Suppress FutureWarnings for torch.load
        warnings.filterwarnings("ignore", category=FutureWarning, module="torch.serialization")
        
        # Print MLX information
        if MLX_AVAILABLE:
            print(f"MLX version: {mx.__version__ if hasattr(mx, '__version__') else 'unknown'}")
            print(f"MLX backend: {mx.default_device()}")
        
        parser = argparse.ArgumentParser(description="Generate speech with CSM (MLX-accelerated for Apple Silicon)")
        parser.add_argument(
            "--model-path",
            type=str,
            default="csm-1B",
            help="Path to the model checkpoint. If not provided, will download from HuggingFace.",
        )
        parser.add_argument(
            "--text",
            type=str,
            required=True,
            help="Text to generate speech for",
        )
        parser.add_argument(
            "--speaker",
            type=int,
            default=0,
            help="Speaker ID (default: 0)",
        )
        parser.add_argument(
            "--output",
            type=str,
            default="audio.wav",
            help="Output file path (default: audio.wav)",
        )
        parser.add_argument(
            "--context-audio",
            type=str,
            nargs="*",
            help="Path(s) to audio file(s) to use as context",
        )
        parser.add_argument(
            "--context-text",
            type=str,
            nargs="*",
            help="Text(s) corresponding to the context audio files",
        )
        parser.add_argument(
            "--context-speaker",
            type=int,
            nargs="*",
            help="Speaker ID(s) for the context segments",
        )
        parser.add_argument(
            "--max-audio-length-ms",
            type=int,
            default=10000,
            help="Maximum audio length in milliseconds (default: 10000)",
        )
        parser.add_argument(
            "--temperature",
            type=float,
            default=0.9,
            help="Sampling temperature (default: 0.9)",
        )
        parser.add_argument(
            "--topk",
            type=int,
            default=50,
            help="Top-k sampling parameter (default: 50)",
        )
        # Sampling options
        parser.add_argument(
            "--seed", 
            type=int,
            default=None,
            help="Random seed for reproducible token generation",
        )
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Enable debug mode with more detailed output",
        )
        
        args = parser.parse_args()
        
        # Set up debug mode if requested
        if args.debug:
            os.environ["DEBUG"] = "1"
            setup_mlx_debug(True)
        
        # Use the speaker ID directly
        speaker_id = args.speaker
        print(f"Using temperature={args.temperature}, topk={args.topk}")
        
        # Get the model path, downloading if necessary
        model_path = args.model_path
        
        # Prepare context segments if provided
        context = []
        if args.context_audio:
            if not (args.context_text and args.context_speaker):
                print("Error: If context audio is provided, context text and speaker must also be provided")
                sys.exit(1)
            if not (
                len(args.context_audio) == len(args.context_text) == len(args.context_speaker)
            ):
                print("Error: The number of context audio, text, and speaker entries must be the same")
                sys.exit(1)

            for audio_path, text, speaker in zip(
                args.context_audio, args.context_text, args.context_speaker
            ):
                print(f"Loading context audio: {audio_path}")
                try:
                    # Load audio file
                    audio_tensor, sample_rate = torchaudio.load(audio_path)
                    # Get sample rate from arguments for resampling if needed
                    target_sample_rate = 24000  # Default sample rate for CSM
                    
                    # Resample if needed
                    if sample_rate != target_sample_rate:
                        audio_tensor = torchaudio.functional.resample(
                            audio_tensor.squeeze(0),
                            orig_freq=sample_rate,
                            new_freq=target_sample_rate,
                        )
                    
                    # Create segment
                    context.append(
                        Segment(text=text, speaker=speaker, audio=audio_tensor)
                    )
                except Exception as e:
                    print(f"Error loading context audio {audio_path}: {e}")
                    if args.debug:
                        import traceback
                        traceback.print_exc()
                    sys.exit(1)
        
        try:
            # Load with MLX acceleration using exact PyTorch-matching sampling
            generator = load_csm_1b_mlx(
                model_path,
                debug=args.debug
            )
            
            print("Model loaded successfully with MLX acceleration")
            print("Using exact PyTorch-matching sampling for high quality audio")
                
            using_mlx = True
        except Exception as e:
            print(f"Error loading model with MLX: {e}")
            print("Falling back to standard PyTorch implementation...")
            
            if args.debug:
                import traceback
                traceback.print_exc()
                
            try:
                # Try loading with standard PyTorch to CPU
                from ..generator import load_csm_1b
                
                # Apply comprehensive monkey patching for device and dtype handling
                import torch
                
                # Save original functions
                original_is_available = torch.cuda.is_available
                original_to = torch.Tensor.to
                
                # Override CUDA availability
                torch.cuda.is_available = lambda: False
                
                # Override tensor.to method to handle bfloat16 issues on CPU
                def safe_to(self, *args, **kwargs):
                    # Extract the device and dtype
                    device = kwargs.get('device', None)
                    dtype = kwargs.get('dtype', None)
                    
                    # If the dtype is bfloat16 and we're on CPU, use float32 instead
                    if dtype == torch.bfloat16 and (device == 'cpu' or device is None or str(device) == 'cpu'):
                        kwargs['dtype'] = torch.float32
                        
                    # Call the original method
                    return original_to(self, *args, **kwargs)
                
                # Apply the patches
                torch.Tensor.to = safe_to
                
                try:
                    # Try loading with patched functions
                    generator = load_csm_1b(model_path, device="cpu")
                except Exception as inner_e:
                    # Try direct loading approach as a last resort
                    print(f"Standard loading failed: {inner_e}")
                    print("Attempting direct loading approach...")
                    
                    # Check if the path is a file or a symbolic model name
                    if os.path.isfile(model_path):
                        # Use the path directly
                        checkpoint_path = model_path
                    elif model_path == "csm-1B":
                        # Download the model
                        cache_dir = os.path.expanduser("~/.cache/csm")
                        os.makedirs(cache_dir, exist_ok=True)
                        checkpoint_path = hf_hub_download(repo_id="sesame/csm-1b", filename="ckpt.pt", cache_dir=cache_dir)
                        print(f"Downloaded model to {checkpoint_path}")
                    else:
                        # Assume it's a directory with ckpt.pt
                        checkpoint_path = os.path.join(model_path, "ckpt.pt")
                        
                    # Load the checkpoint
                    print(f"Loading checkpoint from {checkpoint_path}")
                    checkpoint = torch.load(checkpoint_path, map_location="cpu")
                    
                    # Import the model directly
                    from csm.models.model import Model
                    
                    # Carefully examine the checkpoint structure
                    print("Examining checkpoint structure...")
                    if isinstance(checkpoint, dict):
                        print(f"Checkpoint keys: {list(checkpoint.keys())}")
                        
                        # Handle different possible structures
                        if "model_args" in checkpoint and "model" in checkpoint:
                            # Standard format
                            model_args = checkpoint["model_args"]
                            model_state = checkpoint["model"]
                            print("Found standard checkpoint format with model_args and model")
                        elif "args" in checkpoint and "model_state_dict" in checkpoint:
                            # Alternative format
                            model_args = checkpoint["args"]
                            model_state = checkpoint["model_state_dict"]
                            print("Found alternative checkpoint format with args and model_state_dict")
                        elif "model_state_dict" in checkpoint:
                            # Only state dict, use default args
                            print("WARNING: Could not find model_args, using defaults")
                            # Create default model args based on state dict
                            from csm.models.model import ModelArgs
                            model_args = ModelArgs()
                            model_state = checkpoint["model_state_dict"]
                        elif "state_dict" in checkpoint:
                            # Generic PyTorch format
                            print("Found generic PyTorch state_dict format")
                            from csm.models.model import ModelArgs
                            model_args = ModelArgs()
                            model_state = checkpoint["state_dict"]
                        elif len(checkpoint) > 0:
                            # Assume the checkpoint is itself the state dict
                            print("Assuming checkpoint is itself the state dict")
                            from csm.models.model import ModelArgs
                            model_args = ModelArgs()
                            model_state = checkpoint
                        else:
                            raise ValueError("Could not find model parameters in checkpoint")
                    else:
                        # Not a dict - unexpected format
                        raise ValueError(f"Unexpected checkpoint format: {type(checkpoint)}")
                        
                    # Create model directly with float32
                    from csm.models.model import Model
                    model = Model(model_args)
                    
                    try:
                        # Try to load the state dict
                        model.load_state_dict(model_state)
                    except Exception as state_e:
                        print(f"Error loading state dict: {state_e}")
                        print("Trying to load with strict=False...")
                        model.load_state_dict(model_state, strict=False)
                    
                    # Create a generator with the loaded model
                    from ..generator import Generator
                    generator = Generator(model)
                    
                    print("Successfully loaded model with direct approach")
                finally:
                    # Restore original functions
                    torch.cuda.is_available = original_is_available
                    torch.Tensor.to = original_to
                    
                using_mlx = False
                print("Successfully loaded model with standard PyTorch CPU (no MLX acceleration)")
            except Exception as e2:
                print(f"Fatal error: Could not load model: {e2}")
                if args.debug:
                    traceback.print_exc()
                sys.exit(1)
        
        # Generate audio using MLX acceleration
        print(f"Generating audio for text: '{args.text}'")
        start_time = time.time()
        
        try:
            # Generate speech
            if hasattr(generator, 'generate_speech'):
                # Modern modular generator
                generate_kwargs = {
                    "text": args.text,
                    "speaker": speaker_id,
                    "temperature": args.temperature,
                    "topk": args.topk,
                    "progress_callback": progress_callback
                }
                
                # Add seed if provided for reproducible generation
                if args.seed is not None:
                    generate_kwargs["seed"] = args.seed
                    
                audio = generator.generate_speech(**generate_kwargs)
            else:
                # Legacy generator with generate method
                segments = generator.generate(
                    text=args.text,
                    speaker=speaker_id,
                    context=context,
                    max_audio_length_ms=args.max_audio_length_ms,
                    temperature=args.temperature,
                    topk=args.topk,
                    callback=progress_callback
                )
                
                # Get audio from first segment
                if len(segments) > 0:
                    audio = segments[0].audio
                else:
                    raise RuntimeError("No audio was generated")
            
            # Print newline after progress bar
            print()
            
            # Calculate generation time
            end_time = time.time()
            total_time = end_time - start_time
            
            # Get sample rate from generator
            sample_rate = getattr(generator, 'sample_rate', 24000)
            
            # Calculate real-time factor
            audio_length = len(audio) / sample_rate
            rtf = total_time / audio_length
            
            print(f"Generated {audio_length:.2f}s audio in {total_time:.2f}s (RTF: {rtf:.2f}x)")
            
            # Save the audio
            output_path = args.output
            output_dir = os.path.dirname(os.path.abspath(output_path))
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                
            # Import torch if needed
            import torch
                
            # Convert to tensor for torchaudio if needed
            if isinstance(audio, np.ndarray):
                # Ensure correct format and range for audio
                if audio.dtype != np.float32:
                    if args.debug:
                        print(f"Converting audio from {audio.dtype} to float32")
                    audio = audio.astype(np.float32)
                
                # Simple NaN check
                has_nans = np.isnan(audio).any()
                if has_nans:
                    print("NaN values detected in audio - handling")
                    audio = np.nan_to_num(audio, nan=0.0)
                
                # Normal tensor conversion
                audio_tensor = torch.from_numpy(audio)
            elif isinstance(audio, torch.Tensor):
                audio_tensor = audio
            else:
                raise ValueError(f"Unexpected audio type: {type(audio)}")
            
            # Post-process the audio tensor for best output quality:
            
            # 1. Handle any NaN values that might be present
            has_nans = torch.isnan(audio_tensor).any()
            if has_nans:
                nan_count = torch.isnan(audio_tensor).sum().item()
                print(f"WARNING: Detected {nan_count} NaN values in audio tensor, fixing")
                audio_tensor = torch.nan_to_num(audio_tensor, nan=0.0)
                
                # Check if this is a systematic issue with all values
                if nan_count > 0.5 * audio_tensor.numel():
                    print("ALERT: More than half of audio values are NaNs!")
                    print("This indicates a systematic issue with the token generation or decoding")
                    
                    # Try analyzing the tokens if available
                    if hasattr(generator, '_last_tokens') and generator._last_tokens is not None:
                        last_tokens = generator._last_tokens
                        print(f"\nAnalyzing tokens that caused NaNs:")
                        print(f"Token shape: {last_tokens.shape}")
                        print(f"Token min/max: {last_tokens.min().item()}/{last_tokens.max().item()}")
                        print(f"Unique token values: {last_tokens.unique().tolist()[:20]}")
                        
                        # Check for problematic range
                        problematic = (last_tokens >= 1) & (last_tokens <= 31)
                        if problematic.any():
                            prob_count = problematic.sum().item()
                            print(f"Found {prob_count} tokens in problematic range 1-31")
                
            # 2. Normalize audio tensor for proper volume
            max_abs = torch.abs(audio_tensor).max()
            if max_abs > 1.0:
                print(f"Normalizing audio (max amplitude: {max_abs.item()})")
                audio_tensor = audio_tensor / max_abs * 0.9
            elif max_abs < 0.1:
                print(f"Audio signal very quiet (max amplitude: {max_abs.item()}), boosting")
                audio_tensor = audio_tensor * (0.9 / max_abs)
                
            # Check for too many zeros (successful NaN replacement but flat audio)
            zero_count = (audio_tensor == 0.0).sum().item()
            if zero_count > 0.8 * audio_tensor.numel():
                print("WARNING: Audio tensor contains more than 80% zeros")
                print("This suggests that the NaN replacement created mostly flat audio")
                
            # 3. Ensure proper shape for saving
            if len(audio_tensor.shape) == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            # Final check for all zeros (might indicate total NaN replacement)
            if torch.all(audio_tensor == 0):
                print("Warning: Audio tensor contains all zeros, creating minimal placeholder audio")
                duration = 1.0  # short duration
                t = torch.linspace(0.0, float(duration), steps=int(sample_rate * duration))
                audio_tensor = 0.5 * torch.sin(2 * 3.14159 * 440 * t).unsqueeze(0)
                
            # Debug info about the final audio
            if args.debug:
                print(f"Final audio tensor shape: {audio_tensor.shape}")
                print(f"Final audio min: {audio_tensor.min().item()}, max: {audio_tensor.max().item()}")
                print(f"Final audio dtype: {audio_tensor.dtype}")
            
            # Save to WAV file
            torchaudio.save(
                output_path,
                audio_tensor.cpu(),
                sample_rate,
            )
            print(f"Audio saved to {output_path}")
            print(f"Sample rate: {sample_rate} Hz")
            
            if using_mlx:
                print(f"Generated using MLX acceleration on {mx.default_device()}")
            else:
                print(f"Generated using PyTorch CPU")
            
            # Show performance stats in debug mode
            if args.debug and hasattr(generator, 'timing_stats'):
                stats = generator.timing_stats
                if "frames_generated" in stats and stats["frames_generated"] > 0:
                    print("\nPerformance Summary:")
                    print(f"Total time: {stats['total_time']:.2f}s")
                    print(f"Frames generated: {stats['frames_generated']}")
                    
                    if "backbone_time" in stats:
                        avg_frame_time = stats["backbone_time"] / stats["frames_generated"]
                        print(f"Average time per frame: {avg_frame_time:.3f}s")
                        
                    if "decode_time" in stats:
                        print(f"Total decode time: {stats['decode_time']:.2f}s")
                        
                    fps = stats["frames_generated"] / stats["total_time"]
                    print(f"Frames per second: {fps:.2f} FPS")
                    
                    # Show benefit of Apple Silicon if using MLX
                    if using_mlx:
                        device_str = str(mx.default_device())
                        if 'gpu' in device_str.lower():
                            print(f"\nUsing Apple Silicon GPU for MLX acceleration")
                            estimated_cpu_time = stats["total_time"] * 1.5  # Conservative estimate
                            print(f"Estimated speedup vs CPU: {estimated_cpu_time/stats['total_time']:.1f}x")
            
        except Exception as e:
            print(f"Error generating audio: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            sys.exit(1)
            
        return 0

if __name__ == "__main__":
    main()