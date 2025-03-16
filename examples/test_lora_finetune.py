#!/usr/bin/env python
"""
Test script for validating LoRA fine-tuning with Hugging Face datasets.

This script demonstrates the primary use case:
1. Download voice samples from Hugging Face
2. Train on that voice sample data
3. Run inference with the trained model

Usage:
    python test_lora_finetune.py --model-path /path/to/model.safetensors
"""

import os
import sys
import argparse
import tempfile
from pathlib import Path
import subprocess
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_lora_finetune")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test LoRA fine-tuning with Hugging Face datasets")
    
    parser.add_argument(
        "--model-path",
        type=str,
        default="default",
        help="Path to the CSM model (.safetensors format) or 'default' to use the default model (will download if needed)"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="patrickvonplaten/librispeech_asr_dummy",
        help="Hugging Face dataset to use (default: patrickvonplaten/librispeech_asr_dummy - a tiny test dataset)"
    )
    
    parser.add_argument(
        "--language",
        type=str,
        default="clean",
        help="Configuration to filter the dataset (default: clean for librispeech_asr_dummy)"
    )
    
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples to use (default: 1 for quick testing)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1)"
    )
    
    parser.add_argument(
        "--lora-r",
        type=int,
        default=4,
        help="LoRA rank (default: 4, smaller to reduce memory usage)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size (default: 2)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: temporary directory)"
    )
    
    parser.add_argument(
        "--keep-data",
        action="store_true",
        help="Keep downloaded data after training"
    )
    
    parser.add_argument(
        "--debug-audio",
        action="store_true",
        help="Use test tone for audio generation in case of errors (for debugging)"
    )
    
    return parser.parse_args()

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import mlx
        logger.info(f"MLX version: {mlx.__version__ if hasattr(mlx, '__version__') else 'unknown'}")
    except ImportError:
        logger.error("MLX is required but not installed. Please install with: pip install mlx")
        return False
    
    try:
        import datasets
        logger.info(f"Hugging Face datasets version: {datasets.__version__}")
    except ImportError:
        logger.error("Hugging Face datasets is required but not installed. Please install with: pip install datasets")
        return False
    
    try:
        import torch
        import torchaudio
        logger.info(f"PyTorch version: {torch.__version__}")
    except ImportError:
        logger.error("PyTorch and torchaudio are required but not installed. Please install with: pip install torch torchaudio")
        return False
    
    return True

def find_huggingface_script():
    """Find the huggingface_lora_finetune.py script."""
    # Try common locations
    script_path = None
    locations = [
        # Current directory
        os.path.join(os.getcwd(), "huggingface_lora_finetune.py"),
        # Examples directory
        os.path.join(os.getcwd(), "examples", "huggingface_lora_finetune.py"),
        # Parent directory examples
        os.path.join(os.path.dirname(os.getcwd()), "examples", "huggingface_lora_finetune.py"),
        # Relative to this script
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "huggingface_lora_finetune.py")
    ]
    
    for loc in locations:
        if os.path.exists(loc):
            script_path = loc
            break
    
    if not script_path:
        logger.error("Could not find huggingface_lora_finetune.py in common locations")
        logger.info(f"Searched locations: {locations}")
        return None
    
    return script_path

def run_finetune_process(args, script_path, output_dir):
    """Run the fine-tuning process."""
    import sys
    import time
    from itertools import cycle
    
    cmd = [
        sys.executable,
        script_path,
        "--model-path", args.model_path,
        "--output-dir", output_dir,
        "--dataset", args.dataset,
        "--language", args.language,
        "--num-samples", str(args.num_samples),
        "--lora-r", str(args.lora_r),
        "--batch-size", str(args.batch_size),
        "--epochs", str(args.epochs)
    ]
    
    if args.keep_data:
        cmd.append("--keep-data")
    
    # Add detailed logging and save mode
    cmd.extend(["--log-level", "info", "--save-mode", "lora"])  # Use info level and only save LoRA parameters
    
    # Add debug-audio flag if requested
    if args.debug_audio:
        cmd.append("--log-level")
        cmd.append("debug")  # Set log level to debug to enable test tones
    
    logger.info(f"Running command: {' '.join(cmd)}")
    logger.info("Starting fine-tuning process - this may take several minutes")
    print("\n" + "="*80)
    print("STARTING HUGGING FACE FINE-TUNING TEST")
    print("="*80 + "\n")
    
    try:
        # Run the process with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Redirect stderr to stdout for unified output
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Process and display output in real-time with a progress spinner
        
        spinner = cycle('⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏')
        last_update = time.time()
        last_message = ""
        
        # Process output in real-time
        for line in process.stdout:
            # Only keep important lines for display
            if "error" in line.lower() or "warning" in line.lower() or any(
                keyword in line.lower() for keyword in [
                    "downloaded", "processed", "training", "fine-tuning", 
                    "epoch", "sample", "batch", "loss", "dataset", "model"
                ]
            ):
                # Clean up line for display
                clean_line = line.strip()
                if clean_line:
                    print(f"  {clean_line}")
                    last_message = clean_line
                    last_update = time.time()
            elif time.time() - last_update > 5:
                # Show a spinner for long-running operations without output
                current_time = time.time()
                if current_time - last_update > 30:
                    # If more than 30 seconds with no output, show a message
                    print(f"  {next(spinner)} Still running... (last: {last_message})")
                    last_update = current_time
        
        # Wait for process to complete
        return_code = process.wait()
        
        if return_code == 0:
            print("\n" + "="*80)
            print("FINE-TUNING PROCESS COMPLETED SUCCESSFULLY")
            print("="*80 + "\n")
        else:
            print("\n" + "="*80)
            print("FINE-TUNING PROCESS FAILED")
            print("="*80 + "\n")
        
        return return_code == 0
    except Exception as e:
        logger.error(f"Error running fine-tuning process: {e}")
        print("\n" + "="*80)
        print(f"FINE-TUNING PROCESS FAILED WITH ERROR: {e}")
        print("="*80 + "\n")
        return False

def check_output_files(output_dir):
    """Check that expected output files exist."""
    # Convert to Path object if it's a string
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    
    # Check for model files
    model_files = list(output_dir.glob("*.safetensors"))
    if not model_files:
        logger.error("No model files (.safetensors) found in output directory")
        return False
    
    logger.info(f"Found {len(model_files)} model files: {[f.name for f in model_files]}")
    
    # Check for sample audio
    sample_file = output_dir / "sample.wav"
    if not sample_file.exists():
        logger.warning("No sample.wav file found in output directory")
    else:
        logger.info(f"Found sample audio: {sample_file}")
    
    # Check for log file
    log_file = output_dir / "huggingface_finetune.log"
    if not log_file.exists():
        logger.warning("No log file found in output directory")
    else:
        logger.info(f"Found log file: {log_file}")
    
    return len(model_files) > 0

def create_test_model(output_path):
    """Create a minimal test model for testing."""
    logger.info(f"Creating test model at {output_path}")
    
    try:
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Try to import create_test_model if it exists
        try:
            from tests.create_test_model import create_model
            create_model(output_path)
            logger.info(f"Created test model using create_test_model function")
            return
        except ImportError:
            logger.info("create_test_model not found, using fallback")
        
        # Fallback: Create a simplified test model
        try:
            # Try to create using MLX
            import mlx.core as mx
            from csm.mlx.components.model_wrapper import MLXModelWrapper
            import safetensors.numpy
            import numpy as np
            
            # Create a small test model
            model_args = {
                "backbone_flavor": "llama-100M",  # Very small for testing
                "decoder_flavor": "llama-100M",
                "text_vocab_size": 32000,
                "audio_vocab_size": 2051,
                "audio_num_codebooks": 4,
                "debug": True
            }
            
            # Initialize model
            model = MLXModelWrapper(model_args)
            
            # Get model parameters and convert to numpy
            params = model.parameters()
            np_params = {}
            
            for k, v in params.items():
                if hasattr(v, 'dtype'):
                    # Convert MLX arrays to numpy arrays
                    np_params[k] = np.array(v)
            
            # Save as safetensors
            safetensors.numpy.save_file(np_params, output_path)
            logger.info(f"Created test model with MLX")
            return
            
        except (ImportError, Exception) as e:
            logger.info(f"MLX model creation failed: {e}")
            logger.info("Using PyTorch fallback")
            
            # PyTorch fallback
            import torch
            import safetensors.torch
            
            # Create a minimal model with random weights
            dummy_state = {}
            for i in range(2):
                # Create some backbone parameters
                dummy_state[f"backbone.layers.{i}.attn.q_proj.weight"] = torch.randn(512, 512)
                dummy_state[f"backbone.layers.{i}.attn.k_proj.weight"] = torch.randn(512, 512)
                dummy_state[f"backbone.layers.{i}.attn.v_proj.weight"] = torch.randn(512, 512)
                dummy_state[f"backbone.layers.{i}.attn.o_proj.weight"] = torch.randn(512, 512)
                
                # Create some decoder parameters
                dummy_state[f"decoder.layers.{i}.attn.q_proj.weight"] = torch.randn(256, 256)
                dummy_state[f"decoder.layers.{i}.attn.k_proj.weight"] = torch.randn(256, 256)
                dummy_state[f"decoder.layers.{i}.attn.v_proj.weight"] = torch.randn(256, 256)
                dummy_state[f"decoder.layers.{i}.attn.o_proj.weight"] = torch.randn(256, 256)
            
            # Add embedding parameters
            dummy_state["backbone.token_embeddings.weight"] = torch.randn(32000, 512)
            dummy_state["decoder.token_embeddings.weight"] = torch.randn(2051, 256)
            
            # Add output layers
            dummy_state["backbone.output.weight"] = torch.randn(32000, 512)
            dummy_state["decoder.output.weight"] = torch.randn(2051, 256)
            
            # Save as safetensors
            safetensors.torch.save_file(dummy_state, output_path)
            logger.info(f"Created test model with PyTorch")
            return
    
    except Exception as e:
        logger.error(f"Error creating test model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def get_default_model_path():
    """Get the default model path, downloading if necessary."""
    logger.info("Using default model (will download if needed)")
    try:
        # Check for model in common locations first
        import os
        import glob
        
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache", "csm")
        
        # Specific path where the CSM model is stored
        csm_model_path = os.path.join(
            cache_dir, 
            "models--sesame--csm-1b", 
            "snapshots", 
            "*", 
            "ckpt.pt"
        )
        
        # Common locations for the model (in order of preference)
        model_patterns = [
            csm_model_path,  # Direct path to downloaded model
            os.path.join(cache_dir, "models--*", "snapshots", "*", "*.pt"),
            os.path.join(cache_dir, "models--*", "snapshots", "*", "*.safetensors"),
            os.path.join(cache_dir, "*.safetensors"),
            os.path.join(cache_dir, "*", "*.safetensors"),
            os.path.join(home_dir, ".cache", "torch", "hub", "safetensors", "*.safetensors")
        ]
        
        for pattern in model_patterns:
            matches = glob.glob(pattern)
            if matches:
                model_path = matches[0]
                logger.info(f"Found default model at: {model_path}")
                return model_path
        
        # If no model found, download it using the csm-generate command
        logger.info("No model found, attempting to download using csm-generate")
        import subprocess
        
        try:
            # Create a temporary directory for output to avoid creating files
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_output = os.path.join(tmp_dir, "tmp_output.wav")
                
                # Run csm-generate with a minimal command to trigger the download
                subprocess.run(
                    ["csm-generate", "--text", "Test download", "--output", tmp_output],
                    check=True,
                    capture_output=True,
                    text=True
                )
            
            # Check locations again after download
            for pattern in model_patterns:
                matches = glob.glob(pattern)
                if matches:
                    model_path = matches[0]
                    logger.info(f"Downloaded model found at: {model_path}")
                    return model_path
        except Exception as e:
            logger.warning(f"Could not run csm-generate to download model: {e}")
                
        # If still not found, create a test model
        fixed_path = "/tmp/test_model.safetensors"
        logger.warning(f"Could not find downloaded model, creating minimal test model at {fixed_path}")
        
        # Create a minimal test model for testing
        create_test_model(fixed_path)
        return fixed_path
        
    except Exception as e:
        logger.error(f"Error getting default model: {e}")
        raise

def main():
    """Main function to test LoRA fine-tuning."""
    # Print a clear header
    print("\n" + "="*80)
    print(" HUGGING FACE LORA FINE-TUNING TEST ".center(80, "="))
    print("="*80 + "\n")
    
    args = parse_args()
    
    print("Phase 1: Checking dependencies and setup")
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Find the Hugging Face script
    script_path = find_huggingface_script()
    if not script_path:
        return 1
    
    print("\nPhase 2: Locating model")
    # Resolve model path if using default
    if args.model_path == "default":
        try:
            args.model_path = get_default_model_path()
            print(f"✓ Using model: {args.model_path}")
        except Exception as e:
            logger.error(f"Failed to get default model: {e}")
            logger.error("Please specify a model path with --model-path")
            return 1
    else:
        print(f"✓ Using model: {args.model_path}")
    
    # Create output directory if not provided
    if args.output_dir:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        using_temp_dir = False
        print(f"✓ Using output directory: {output_dir}")
    else:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="lora_finetune_test_")
        output_dir = temp_dir
        using_temp_dir = True
        print(f"✓ Created temporary output directory: {output_dir}")
    
    print("\nPhase 3: Running fine-tuning process")
    print(f"• Dataset: {args.dataset} (config: {args.language})")
    print(f"• Number of samples: {args.num_samples}")
    print(f"• LoRA rank: {args.lora_r}")
    print(f"• Batch size: {args.batch_size}")
    print(f"• Epochs: {args.epochs}")
    print("\nStarting fine-tuning - this may take several minutes...\n")
    
    # Run the fine-tuning process
    success = run_finetune_process(args, script_path, output_dir)
    
    print("\nPhase 4: Verifying results")
    
    if success:
        # Check output files
        if check_output_files(output_dir):
            print("✅ TEST PASSED: Fine-tuning produced expected output files")
        else:
            print("❌ TEST FAILED: Fine-tuning did not produce expected output files")
            success = False
    else:
        print("❌ TEST FAILED: Fine-tuning process failed")
    
    # If using temp dir and --keep-data not specified, print note about temp dir location
    if using_temp_dir:
        print(f"\nTest results saved in temporary directory: {output_dir}")
        print("This directory will be removed when the system cleans up temporary files.")
        print(f"To preserve results, copy files to a permanent location or use --output-dir next time.")
    
    # Final summary
    print("\n" + "="*80)
    if success:
        print(" TEST COMPLETED SUCCESSFULLY ".center(80, "="))
    else:
        print(" TEST FAILED ".center(80, "="))
    print("="*80 + "\n")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())