#!/usr/bin/env python
"""
Example script for fine-tuning a CSM model with LoRA using data from Hugging Face Hub.

This script demonstrates how to:
1. Download a voice dataset from Hugging Face Hub
2. Prepare the data for fine-tuning
3. Fine-tune a CSM model using LoRA
4. Generate samples with the fine-tuned model

Usage:
    python huggingface_lora_finetune.py --model-path /path/to/base/model.safetensors
                                        --output-dir ./fine_tuned_model
                                        --dataset mozilla-foundation/common_voice_16_0
                                        --language en
                                        --num-samples 100
"""

import os
import sys
import argparse
import logging
import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("MLX not available. This example requires MLX.")
    sys.exit(1)

try:
    import torch
    import torchaudio
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch and torchaudio are required for audio processing.")
    sys.exit(1)

try:
    from datasets import load_dataset, Audio
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Hugging Face datasets library is required.")
    sys.exit(1)

# Add the project root to the path if needed
if os.path.exists(os.path.join(os.path.dirname(__file__), '..', 'src')):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def setup_logger(name, log_file=None, level_name="info"):
    """Set up logger with specified level."""
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL
    }
    level = level_map.get(level_name.lower(), logging.INFO)
    
    # Configure logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def download_dataset(dataset_name: str, language: str, num_samples: int, output_dir: Path, logger: logging.Logger, audio_dir=None, transcript_dir=None):
    """
    Download and prepare dataset from Hugging Face Hub.
    
    Args:
        dataset_name: Name of the dataset on Hugging Face Hub
        language: Language code to filter the dataset
        num_samples: Number of samples to download
        output_dir: Directory to save the processed data
        logger: Logger instance
        audio_dir: Optional directory with existing audio files
        transcript_dir: Optional directory with existing transcript files
        
    Returns:
        Tuple of (audio_dir, transcript_dir) with the processed data
    """
    # Create directories if not provided
    if audio_dir is None:
        audio_dir = output_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
    else:
        audio_dir = Path(audio_dir)
        
    if transcript_dir is None:
        transcript_dir = output_dir / "transcripts"
        transcript_dir.mkdir(parents=True, exist_ok=True)
    else:
        transcript_dir = Path(transcript_dir)
    
    # Check if we're using a local dataset
    if dataset_name.lower() == "local":
        logger.info(f"Using local dataset from {audio_dir} and {transcript_dir}")
        
        if not audio_dir.exists() or not transcript_dir.exists():
            logger.error(f"Local dataset directories not found: {audio_dir}, {transcript_dir}")
            raise ValueError("Local dataset directories not found")
        
        # For local dataset, just validate that the files exist and match
        audio_files = list(audio_dir.glob("*.wav"))
        transcript_files = list(transcript_dir.glob("*.txt"))
        
        logger.info(f"Found {len(audio_files)} audio files and {len(transcript_files)} transcript files")
        
        # Check matching files
        audio_basenames = {f.stem for f in audio_files}
        transcript_basenames = {f.stem for f in transcript_files}
        matching = audio_basenames.intersection(transcript_basenames)
        
        logger.info(f"Found {len(matching)} matching audio-transcript pairs")
        
        # Take a subset if requested
        if num_samples and num_samples < len(matching):
            # Limit to requested number of samples
            matching = list(matching)[:num_samples]
            logger.info(f"Using {len(matching)} samples for fine-tuning")
        
        # Return the same directories since we're using local files
        return audio_dir, transcript_dir
    
    # Otherwise, download from Hugging Face
    logger.info(f"Downloading dataset {dataset_name} ({language}) with {num_samples} samples")
    
    try:
        # Load dataset
        # Try to load train split or fallback to validation/test
        try:
            dataset = load_dataset(
                dataset_name,
                language,
                split="train",
                streaming=False,
                trust_remote_code=True
            )
        except ValueError as e:
            if "Unknown split" in str(e):
                # Try validation split
                logger.info("Train split not found, trying validation split...")
                dataset = load_dataset(
                    dataset_name,
                    language,
                    split="validation",
                    streaming=False,
                    trust_remote_code=True
                )
            else:
                raise
        
        logger.info(f"Dataset loaded with {len(dataset)} samples")
        
        # Take a subset if requested
        if num_samples and num_samples < len(dataset):
            dataset = dataset.select(range(min(num_samples, len(dataset))))
            logger.info(f"Using {len(dataset)} samples for fine-tuning")
        
        # Make sure we have the audio column in the right format
        if "audio" not in dataset.column_names:
            logger.error("Dataset does not contain audio column")
            raise ValueError("Dataset does not contain audio column")
        
        # Check for transcript/sentence column
        transcript_col = None
        for col in ["transcript", "transcription", "sentence", "text", "english_transcription"]:
            if col in dataset.column_names:
                transcript_col = col
                break
        
        if transcript_col is None:
            logger.error("Dataset does not contain transcript column")
            logger.error(f"Available columns: {dataset.column_names}")
            raise ValueError("Dataset does not contain transcript column")
        
        # Process and save each sample
        logger.info("Processing and saving samples...")
        
        for idx, item in enumerate(dataset):
            try:
                # Extract audio
                audio_data = item["audio"]
                audio_array = audio_data["array"]
                sample_rate = audio_data["sampling_rate"]
                
                # Extract transcript
                transcript = item[transcript_col]
                
                # Skip empty or very short samples
                if len(transcript.strip()) < 5 or len(audio_array) < sample_rate * 0.5:  # Shorter than 0.5s
                    continue
                
                # Save audio file
                file_id = f"sample_{idx:04d}"
                audio_path = audio_dir / f"{file_id}.wav"
                
                torchaudio.save(
                    audio_path,
                    torch.tensor(audio_array).unsqueeze(0),
                    sample_rate,
                    encoding="PCM_S",
                    bits_per_sample=16
                )
                
                # Save transcript
                transcript_path = transcript_dir / f"{file_id}.txt"
                with open(transcript_path, "w", encoding="utf-8") as f:
                    f.write(transcript)
                
                if idx % 10 == 0:
                    logger.info(f"Processed {idx+1}/{len(dataset)} samples")
                    
            except Exception as e:
                logger.warning(f"Error processing sample {idx}: {e}")
                continue
        
        logger.info(f"Finished processing dataset. Files saved to {output_dir}")
        return audio_dir, transcript_dir
    
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def run_lora_finetuning(
    model_path: str,
    output_dir: Path,
    audio_dir: Path,
    transcript_dir: Path,
    speaker_id: int = 0,
    lora_r: int = 8,
    lora_alpha: float = 16.0,
    batch_size: int = 2,
    epochs: int = 5,
    save_mode: str = "both",
    logger: logging.Logger = None,
    **kwargs
):
    """
    Run LoRA fine-tuning on the prepared dataset.
    
    Args:
        model_path: Path to the base CSM model
        output_dir: Directory to save the fine-tuned model
        audio_dir: Directory with audio files
        transcript_dir: Directory with transcript files
        speaker_id: Speaker ID to use
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling factor
        batch_size: Batch size for training
        epochs: Number of training epochs
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger("lora_finetuning")
    
    logger.info(f"Starting LoRA fine-tuning with rank={lora_r}, alpha={lora_alpha}")
    
    # Import here to avoid import errors if CSM is not installed
    try:
        from csm.training.lora_trainer import CSMLoRATrainer
        
        # Initialize trainer
        trainer = CSMLoRATrainer(
            model_path=model_path,
            output_dir=str(output_dir),
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],  # Default efficient choices
            learning_rate=1e-4
        )
        
        # Prepare optimizer
        logger.info("Preparing optimizer")
        trainer.prepare_optimizer()
        
        # Prepare data
        logger.info("Preparing training data")
        from csm.cli.finetune_lora import prepare_data
        
        train_dataset, val_dataset = prepare_data(
            audio_dir=str(audio_dir),
            transcript_dir=str(transcript_dir),
            speaker_id=speaker_id,
            val_split=0.1,
            max_seq_len=2048,
            context_turns=2,
            batch_size=batch_size
        )
        
        # Train the model
        logger.info(f"Starting training for {epochs} epochs")
        best_loss = trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=batch_size,
            epochs=epochs,
            val_every=10,
            save_every=50
        )
        
        logger.info(f"Fine-tuning completed with best loss: {best_loss:.6f}")
        
        # Save the model
        logger.info(f"Saving fine-tuned model in {save_mode} mode")
        model_save_path = output_dir / "fine_tuned_model.safetensors"
        trainer.save_model(str(model_save_path), save_mode=save_mode)
        
        # Generate a sample
        logger.info("Generating sample with fine-tuned model")
        sample_path = output_dir / "sample.wav"
        try:
            # Use debug mode for debugging, but not for production
            debug_mode = kwargs.get('debug', False)
            trainer.generate_sample(
                text="This is an example of fine-tuned speech using LoRA adaptation.",
                speaker_id=speaker_id,
                output_path=str(sample_path),
                debug_mode=debug_mode
            )
            logger.info(f"Sample generated at {sample_path}")
            
            # Check for error file and warn if present
            error_file = sample_path.with_suffix('.error.txt')
            if error_file.exists():
                logger.warning("Audio generation encountered errors. Check the .error.txt file for details.")
                logger.warning(f"Error details in: {error_file}")
                # Read the first few lines of the error file to give a hint
                try:
                    with open(error_file, 'r') as f:
                        error_content = f.read(500)  # First 500 chars
                        logger.warning(f"Error preview: {error_content[:200]}...")
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Error generating sample: {e}")
        
        logger.info("LoRA fine-tuning process completed")
        return str(model_save_path)
    
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure you have CSM installed properly")
        raise
    except Exception as e:
        logger.error(f"Error during fine-tuning: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def main():
    """Main function for the example script."""
    parser = argparse.ArgumentParser(description="Fine-tune CSM model with LoRA using HuggingFace data")
    
    # Required arguments
    parser.add_argument("--model-path", type=str, required=True,
                     help="Path to pretrained CSM model (safetensors format)")
    parser.add_argument("--output-dir", type=str, default="./hf_finetuned_model",
                     help="Directory to save fine-tuned model and outputs")
    
    # Dataset options
    parser.add_argument("--dataset", type=str, default="mozilla-foundation/common_voice_16_0",
                     help="HuggingFace dataset to use (or 'local' to use local files)")
    parser.add_argument("--language", type=str, default="en",
                     help="Language to filter the dataset (default: en)")
    parser.add_argument("--num-samples", type=int, default=100,
                     help="Number of samples to use (default: 100)")
    
    # Local dataset options
    parser.add_argument("--audio-dir", type=str, default=None,
                     help="Directory containing audio files (for local dataset)")
    parser.add_argument("--transcript-dir", type=str, default=None,
                     help="Directory containing transcript files (for local dataset)")
    
    # Fine-tuning options
    parser.add_argument("--speaker-id", type=int, default=0,
                     help="Speaker ID to use for training (default: 0)")
    parser.add_argument("--lora-r", type=int, default=8,
                     help="LoRA rank (default: 8)")
    parser.add_argument("--lora-alpha", type=float, default=16.0,
                     help="LoRA alpha scaling factor (default: 16.0)")
    parser.add_argument("--batch-size", type=int, default=2,
                     help="Batch size (default: 2)")
    parser.add_argument("--epochs", type=int, default=5,
                     help="Number of epochs (default: 5)")
    
    # Misc options
    parser.add_argument("--keep-data", action="store_true",
                     help="Keep downloaded data after training")
    parser.add_argument("--log-level", type=str, default="info",
                     choices=["debug", "info", "warning", "error", "critical"],
                     help="Logging level (default: info)")
    parser.add_argument("--save-mode", type=str, default="both",
                     choices=["lora", "full", "both"],
                     help="How to save the fine-tuned model (default: both)")
    
    args = parser.parse_args()
    
    # Check MLX
    if not HAS_MLX:
        print("Error: MLX is required but not installed.")
        print("Install it with: pip install mlx")
        return 1
    
    # Set up main logger
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / "huggingface_finetune.log"
    logger = setup_logger("hf_finetune", log_file=log_file, level_name=args.log_level)
    
    logger.info("Starting HuggingFace dataset fine-tuning example")
    
    # Log all arguments
    logger.info("Configuration:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    try:
        # For local dataset, use the provided directories
        if args.dataset.lower() == "local":
            if args.audio_dir is None or args.transcript_dir is None:
                logger.error("When using a local dataset, --audio-dir and --transcript-dir must be provided")
                return 1
                
            logger.info(f"Using local dataset from {args.audio_dir} and {args.transcript_dir}")
            audio_dir = Path(args.audio_dir)
            transcript_dir = Path(args.transcript_dir)
            
            # Validate directories
            audio_dir, transcript_dir = download_dataset(
                dataset_name="local",
                language=args.language,
                num_samples=args.num_samples,
                output_dir=output_dir,
                logger=logger,
                audio_dir=audio_dir,
                transcript_dir=transcript_dir
            )
        else:
            # Create a temp directory for downloaded data
            if args.keep_data:
                data_dir = output_dir / "data"
            else:
                data_dir = Path(tempfile.mkdtemp(prefix="csm_hf_data_"))
            
            data_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Data will be stored in {data_dir}")
            
            # Download and prepare dataset
            audio_dir, transcript_dir = download_dataset(
                dataset_name=args.dataset,
                language=args.language,
                num_samples=args.num_samples,
                output_dir=data_dir,
                logger=logger
            )
        
        # Run LoRA fine-tuning
        fine_tuned_model_path = run_lora_finetuning(
            model_path=args.model_path,
            output_dir=output_dir,
            audio_dir=audio_dir,
            transcript_dir=transcript_dir,
            speaker_id=args.speaker_id,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            batch_size=args.batch_size,
            epochs=args.epochs,
            save_mode=args.save_mode,
            logger=logger,
            debug=args.log_level.lower() == "debug"  # Pass debug flag based on log level
        )
        
        # Clean up temporary data if not keeping
        if not args.dataset.lower() == "local" and not args.keep_data:
            # Variable data_dir only exists in the HuggingFace dataset path
            if 'data_dir' in locals() and not str(data_dir).startswith(str(output_dir)):
                logger.info(f"Cleaning up temporary data directory: {data_dir}")
                shutil.rmtree(data_dir)
        
        logger.info(f"Fine-tuning complete! Model saved to: {fine_tuned_model_path}")
        logger.info(f"Sample audio available at: {output_dir}/sample.wav")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error in HuggingFace fine-tuning example: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())