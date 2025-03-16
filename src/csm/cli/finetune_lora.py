#!/usr/bin/env python
"""
CLI for fine-tuning CSM models with LoRA using MLX.

This script provides a command-line interface for fine-tuning CSM models
using LoRA (Low-Rank Adaptation) with MLX on Apple Silicon.
"""

import os
import sys
import time
import argparse
import logging
from typing import List, Optional
from pathlib import Path

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

class SmartFormatter(argparse.HelpFormatter):
    """Help formatter that respects newlines in help text."""
    
    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()
        return argparse.HelpFormatter._split_lines(self, text, width)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune CSM models with LoRA using MLX",
        formatter_class=SmartFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to pretrained CSM model (safetensors or PyTorch format)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save fine-tuned model and logs"
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        required=True,
        help="Directory containing audio files for fine-tuning"
    )
    parser.add_argument(
        "--transcript-dir",
        type=str,
        required=True,
        help="Directory containing transcript files for fine-tuning"
    )
    
    # Optional dataset arguments
    parser.add_argument(
        "--alignment-dir",
        type=str,
        default=None,
        help="Directory containing alignment files (optional)"
    )
    parser.add_argument(
        "--speaker-id",
        type=int,
        default=0,
        help="Speaker ID to use for training (default: 0)"
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)"
    )
    parser.add_argument(
        "--context-turns",
        type=int,
        default=2,
        help="Number of context turns to include (default: 2)"
    )
    
    # LoRA configuration
    parser.add_argument(
        "--lora-r",
        type=int,
        default=8,
        help="LoRA rank (default: 8)"
    )
    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=16.0,
        help="LoRA alpha scaling factor (default: 16.0)"
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.0,
        help="LoRA dropout probability (default: 0.0)"
    )
    parser.add_argument(
        "--target-modules",
        type=str,
        nargs="+",
        default=None,
        help=("R|LoRA target modules to adapt (default: ['q_proj', 'v_proj'])\n"
              "Options: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj")
    )
    parser.add_argument(
        "--target-layers",
        type=int,
        nargs="+",
        default=None,
        help="Layer indices to apply LoRA to (default: all layers)"
    )
    parser.add_argument(
        "--lora-bias",
        action="store_true",
        help="Use bias in LoRA layers"
    )
    
    # Training parameters
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--semantic-weight",
        type=float,
        default=100.0,
        help="Weight for semantic token loss (default: 100.0)"
    )
    parser.add_argument(
        "--acoustic-weight",
        type=float,
        default=1.0,
        help="Weight for acoustic token loss (default: 1.0)"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay (default: 0.01)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size (default: 2)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs (default: 5)"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1)"
    )
    parser.add_argument(
        "--val-every",
        type=int,
        default=100,
        help="Validate every N steps (default: 100)"
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=500,
        help="Save checkpoint every N steps (default: 500)"
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping (default: 1.0)"
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume training from checkpoint"
    )
    
    # Output options
    parser.add_argument(
        "--save-mode",
        type=str,
        choices=["lora", "full", "both"],
        default="lora",
        help=("R|How to save the fine-tuned model (default: lora)\n"
              "  lora: Save only LoRA parameters\n"
              "  full: Save the full model with merged weights\n"
              "  both: Save both LoRA parameters and merged model")
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Logging level (default: info)"
    )
    
    # Additional options
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    parser.add_argument(
        "--generate-samples",
        action="store_true",
        help="Generate audio samples during validation"
    )
    parser.add_argument(
        "--sample-prompt",
        type=str,
        default="This is a test of the fine-tuned voice model.",
        help="Prompt for sample generation"
    )
    
    args = parser.parse_args()
    return args


def setup_logger(level_name):
    """Set up root logger with specified level."""
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL
    }
    level = level_map.get(level_name.lower(), logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    return logging.getLogger(__name__)


def prepare_data(
    audio_dir,
    transcript_dir,
    alignment_dir=None,
    speaker_id=0,
    val_split=0.1,
    max_seq_len=2048,
    context_turns=2,
    batch_size=2
):
    """
    Prepare data for fine-tuning.
    
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    from csm.data import CSMDataProcessor, ContextualExampleGenerator
    
    logger = logging.getLogger("prepare_data")
    logger.info(f"Preparing data from {audio_dir} and {transcript_dir}")
    
    # Create processor and generate examples
    processor = CSMDataProcessor()
    
    # Get a list of all audio files
    import glob
    audio_files = glob.glob(os.path.join(audio_dir, "**/*.wav"), recursive=True)
    logger.info(f"Found {len(audio_files)} audio files")
    
    # Process each file and collect examples
    all_examples = []
    for audio_file in audio_files:
        # Get matching transcript file
        base_name = os.path.basename(audio_file).replace(".wav", "")
        transcript_file = os.path.join(transcript_dir, f"{base_name}.txt")
        
        # Get alignment file if available
        alignment_file = None
        if alignment_dir:
            alignment_file = os.path.join(alignment_dir, f"{base_name}.json")
            if not os.path.exists(alignment_file):
                alignment_file = None
        
        # Check if transcript exists
        if not os.path.exists(transcript_file):
            logger.warning(f"No transcript found for {audio_file}, skipping")
            continue
        
        # Process the file
        try:
            # Call with or without alignment depending on availability
            if alignment_file:
                examples = processor.prepare_from_audio_file(
                    audio_file,
                    transcript_file,
                    alignment_file=alignment_file,
                    speaker_id=speaker_id
                )
            else:
                examples = processor.prepare_from_audio_file(
                    audio_file,
                    transcript_file,
                    speaker_id=speaker_id
                )
            
            all_examples.extend(examples)
        except Exception as e:
            logger.error(f"Error processing {audio_file}: {e}")
    
    logger.info(f"Processed {len(all_examples)} examples")
    
    # Apply context turns
    if context_turns > 0:
        logger.info(f"Generating contextual examples with {context_turns} context turns")
        generator = ContextualExampleGenerator(max_context_turns=context_turns)
        contextual_examples = generator.create_contextual_examples(all_examples)
    else:
        # No context, just wrap in the expected format
        contextual_examples = [{"context": [], "target": ex} for ex in all_examples]
    
    # Split into train and validation
    import random
    random.shuffle(contextual_examples)
    
    val_size = int(len(contextual_examples) * val_split)
    train_examples = contextual_examples[val_size:]
    val_examples = contextual_examples[:val_size]
    
    logger.info(f"Split into {len(train_examples)} train and {len(val_examples)} validation examples")
    
    # Create MLX datasets
    from csm.training.data import MLXDataset
    
    train_dataset = MLXDataset(train_examples, max_seq_len=max_seq_len)
    val_dataset = MLXDataset(val_examples, max_seq_len=max_seq_len)
    
    logger.info(f"Created train dataset with {len(train_dataset)} batches")
    logger.info(f"Created validation dataset with {len(val_dataset)} batches")
    
    return train_dataset, val_dataset


def main():
    """Main function for the LoRA fine-tuning CLI."""
    # Check for MLX
    if not HAS_MLX:
        print("Error: MLX is required but not installed.")
        print("Install it with: pip install mlx")
        return 1
    
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    logger = setup_logger(args.log_level)
    logger.info(f"Starting CSM LoRA fine-tuning with MLX")
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up file logger
    log_file = os.path.join(args.output_dir, "training.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Log all arguments
    logger.info("Training arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # Check if running on Apple Silicon
    try:
        import platform
        is_apple_silicon = platform.processor() == 'arm'
        logger.info(f"Running on Apple Silicon: {is_apple_silicon}")
    except:
        logger.warning("Could not determine if running on Apple Silicon")
    
    try:
        # Prepare data
        logger.info("Preparing datasets")
        train_dataset, val_dataset = prepare_data(
            audio_dir=args.audio_dir,
            transcript_dir=args.transcript_dir,
            alignment_dir=args.alignment_dir,
            speaker_id=args.speaker_id,
            val_split=args.val_split,
            max_seq_len=args.max_seq_len,
            context_turns=args.context_turns,
            batch_size=args.batch_size
        )
        
        # Initialize LoRA trainer
        logger.info("Initializing LoRA trainer")
        from csm.training.lora_trainer import CSMLoRATrainer
        
        trainer = CSMLoRATrainer(
            model_path=args.model_path,
            output_dir=args.output_dir,
            log_file=log_file,
            learning_rate=args.learning_rate,
            semantic_weight=args.semantic_weight,
            acoustic_weight=args.acoustic_weight,
            weight_decay=args.weight_decay,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
            target_layers=args.target_layers,
            lora_use_bias=args.lora_bias
        )
        
        # Prepare optimizer
        logger.info("Preparing optimizer")
        trainer.prepare_optimizer()
        
        # Fine-tune the model
        logger.info("Starting fine-tuning")
        best_loss = trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=args.batch_size,
            epochs=args.epochs,
            val_every=args.val_every,
            save_every=args.save_every,
            max_grad_norm=args.max_grad_norm,
            resume_from=args.resume_from
        )
        
        logger.info(f"Fine-tuning completed with best validation loss: {best_loss:.6f}")
        
        # Save the fine-tuned model
        logger.info(f"Saving fine-tuned model in {args.save_mode} mode")
        save_path = os.path.join(args.output_dir, "fine_tuned_model.safetensors")
        trainer.save_model(save_path, save_mode=args.save_mode)
        
        # Generate sample if requested
        if args.generate_samples:
            logger.info("Generating sample audio")
            sample_path = os.path.join(args.output_dir, "sample.wav")
            try:
                # Generate sample with debug mode based on args.debug flag
                trainer.generate_sample(
                    text=args.sample_prompt,
                    speaker_id=args.speaker_id,
                    output_path=sample_path,
                    debug_mode=args.debug
                )
                logger.info(f"Sample audio saved to {sample_path}")
                
                # Check for error file and warn if present
                error_file = sample_path.replace(".wav", ".error.txt")
                if os.path.exists(error_file):
                    logger.warning("Audio generation encountered errors. Check the .error.txt file for details.")
                    logger.warning(f"Error details in: {error_file}")
            except Exception as e:
                logger.error(f"Error generating sample: {e}")
        
        logger.info("LoRA fine-tuning completed successfully")
        return 0
    
    except Exception as e:
        logger.error(f"Error during fine-tuning: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())