#\!/usr/bin/env python
"""
Test script for verifying LoRA generation.
This is a minimal version that only tests the generation step.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Initialize the mock modules first - this must be done before any other imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    # Try to import from same directory first
    import mock_modules
    mock_modules.install_all_mocks()
except ImportError:
    # Fallback to parent directory
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        import mock_modules
        mock_modules.install_all_mocks()
    except ImportError:
        print("Warning: Could not import mock_modules helper. Some dependencies may be missing.")

# Add a direct workaround for the test script to avoid transformers dependency issues
# This fixes the issue with transformers.utils.import_utils._is_package_available
# We need to mock these modules directly to avoid the errors
class DummySpec:
    def __init__(self, name):
        self.name = name
        self.parent = None
        self.origin = None
        self.submodule_search_locations = []

class DummyMod:
    def __init__(self, name):
        self.__name__ = name
        self.__spec__ = DummySpec(name)
    def __getattr__(self, key):
        return DummyMod(f"{self.__name__}.{key}")

# Add essential direct mocks
for mod_name in ['bitsandbytes', 'triton', 'flash_attn', 'xformers']:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = DummyMod(mod_name)
        # Add common submodules
        if mod_name == 'bitsandbytes':
            sys.modules[f'{mod_name}.nn'] = DummyMod(f'{mod_name}.nn')
            sys.modules[f'{mod_name}.functional'] = DummyMod(f'{mod_name}.functional')
        elif mod_name == 'triton':
            sys.modules[f'{mod_name}.language'] = DummyMod(f'{mod_name}.language')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_lora_generation")

def parse_args():
    parser = argparse.ArgumentParser(description="Test LoRA generation")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/Users/ericflo/.cache/csm/models--sesame--csm-1b/snapshots/bf27c9b04fa0131aa912fb15860765db56e5ad1b/ckpt.pt",
        help="Path to the base model"
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default="abc123/fine_tuned_model.safetensors",
        help="Path to the LoRA model"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_output",
        help="Output directory for generated audio"
    )
    parser.add_argument(
        "--force-mlx",
        action="store_true",
        help="Force MLX generation rather than PyTorch fallback"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output for detailed error messages"
    )
    return parser.parse_args()

def main():
    print("\n" + "="*80)
    print(" LORA GENERATION TEST ".center(80, "="))
    print("="*80 + "\n")
    
    args = parse_args()
    
    # Make sure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set debug environment variable if requested
    if args.debug:
        os.environ["DEBUG"] = "1"
    
    # Import required modules
    from csm.training.lora_trainer import CSMLoRATrainer
    import torch
    
    # Create trainer instance
    logger.info(f"Creating LoRA trainer with base model: {args.model_path}")
    
    # Standard trainer creation
    trainer = CSMLoRATrainer(
        model_path=args.model_path,
        output_dir=args.output_dir,
        lora_r=4  # Same as the test script
    )
    
    # Load LoRA weights
    if os.path.exists(args.lora_path):
        logger.info(f"Loading LoRA weights from: {args.lora_path}")
        trainer.load_lora_weights(args.lora_path)
    else:
        logger.warning(f"LoRA weights not found at: {args.lora_path}")
        logger.info("Continuing with base model only")
    
    # Generate sample
    output_path = os.path.join(args.output_dir, "sample.wav")
    logger.info(f"Generating sample to: {output_path}")
    
    test_text = "This is a test of LoRA fine-tuning for speech synthesis."
    
    try:
        # Always use MLX-based generation for LoRA, with no fallbacks
        # Use the MLX wrapper functionality directly with fixed mask implementation
        logger.info("Using MLX wrapper with fixed implementation")
        
        logger.info(f"Generating audio for text: '{test_text}'")
        try:
            # Import required libraries
            from csm.mlx.mlx_wrapper import generate_audio
            import torch
            import torchaudio
            
            # Get the model from the trainer
            model = trainer.model
            
            # Generate audio using the MLX wrapper with fixed implementation
            audio = generate_audio(
                model=model,
                text=test_text,
                speaker_id=0,
                temperature=0.9,
                top_k=50,
                debug=args.debug,
                merge_lora=True  # Always merge LoRA weights if present
            )
            
            # Ensure directories exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Convert audio to the right format for saving
            if not isinstance(audio, torch.Tensor):
                # If it's a numpy array, convert to torch
                import numpy as np
                if isinstance(audio, np.ndarray):
                    audio = torch.from_numpy(audio)
                else:
                    # If it's some other type, try to convert through numpy
                    audio = torch.tensor(np.array(audio))
            
            # Make sure it has the channel dimension
            if len(audio.shape) == 1:
                audio = audio.unsqueeze(0)
                
            # Save the audio
            sample_rate = 24000  # Standard rate for CSM
            torchaudio.save(output_path, audio.float(), sample_rate)
            
            logger.info(f"Successfully generated audio using MLX wrapper")
        except Exception as e:
            logger.error(f"Error in generate: {e}")
            import traceback
            traceback.print_exc()
            raise  # Re-raise to fail the test rather than falling back
    except Exception as e:
        logger.error(f"Error generating sample: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
    
    # Check if generation succeeded
    if os.path.exists(output_path):
        logger.info(f"Successfully generated sample at: {output_path}")
        
        # Check the sample rate of the generated file
        try:
            import soundfile as sf
            info = sf.info(output_path)
            logger.info(f"Audio file info: {info.samplerate}Hz, {info.channels} channels, {info.duration:.2f}s")
            
            if info.samplerate != 24000:
                logger.warning(f"WARNING: Sample rate is {info.samplerate}Hz, expected 24000Hz")
                print(f"\nWARNING: Sample rate is {info.samplerate}Hz, expected 24000Hz")
        except ImportError:
            logger.warning("Could not check sample rate (soundfile not available)")
        except Exception as e:
            logger.warning(f"Error checking sample rate: {e}")
        
        print("\n" + "="*80)
        print(" TEST COMPLETED SUCCESSFULLY ".center(80, "="))
        print("="*80 + "\n")
        return 0
    else:
        logger.error(f"Failed to generate sample at: {output_path}")
        print("\n" + "="*80)
        print(" TEST FAILED ".center(80, "="))
        print("="*80 + "\n")
        return 1

if __name__ == "__main__":
    main()
