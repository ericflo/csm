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
        required=True,
        help="Path to the CSM model (.safetensors format)"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="mozilla-foundation/common_voice_16_0",
        help="Hugging Face dataset to use (default: mozilla-foundation/common_voice_16_0)"
    )
    
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language to filter the dataset (default: en)"
    )
    
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of samples to use (default: 20)"
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
        default=8,
        help="LoRA rank (default: 8)"
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
    
    # Add detailed logging
    cmd.extend(["--log-level", "debug"])
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run the process and capture output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1  # Line buffered
        )
        
        # Process output in real-time
        for line in process.stdout:
            print(line, end='')
        
        # Wait for process to complete
        return_code = process.wait()
        
        # Get any remaining stderr
        stderr = process.stderr.read()
        if stderr:
            print("\nError output:")
            print(stderr)
        
        return return_code == 0
    except Exception as e:
        logger.error(f"Error running fine-tuning process: {e}")
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

def main():
    """Main function to test LoRA fine-tuning."""
    args = parse_args()
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Find the Hugging Face script
    script_path = find_huggingface_script()
    if not script_path:
        return 1
    
    # Create output directory if not provided
    if args.output_dir:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        using_temp_dir = False
    else:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="lora_finetune_test_")
        output_dir = temp_dir
        using_temp_dir = True
    
    logger.info(f"Output directory: {output_dir}")
    
    # Run the fine-tuning process
    success = run_finetune_process(args, script_path, output_dir)
    
    if success:
        logger.info("Fine-tuning process completed successfully")
        
        # Check output files
        if check_output_files(output_dir):
            logger.info("✅ TEST PASSED: Fine-tuning produced expected output files")
        else:
            logger.error("❌ TEST FAILED: Fine-tuning did not produce expected output files")
            success = False
    else:
        logger.error("❌ TEST FAILED: Fine-tuning process failed")
    
    # If using temp dir and --keep-data not specified, print note about temp dir location
    if using_temp_dir:
        logger.info(f"Test results saved in temporary directory: {output_dir}")
        logger.info("This directory will be removed when the system cleans up temporary files.")
        logger.info(f"To preserve results, copy files to a permanent location or use --output-dir next time.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())