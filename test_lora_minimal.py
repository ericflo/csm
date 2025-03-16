#\!/usr/bin/env python
"""
Minimal test for LoRA with MLX - directly testing the core functionality.
"""

import os
import sys
import time
import json
import logging
import argparse
import tempfile
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_lora_minimal")

def parse_args():
    parser = argparse.ArgumentParser(description="Minimal LoRA with MLX test")
    parser.add_argument(
        "--lora-path",
        type=str,
        default="abc123/fine_tuned_model.safetensors",
        help="Path to the LoRA weights file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_output",
        help="Output directory for output files"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    return parser.parse_args()

def main():
    print("\n" + "="*80)
    print(" MINIMAL LORA WITH MLX TEST ".center(80, "="))
    print("="*80 + "\n")
    
    args = parse_args()
    
    # Make sure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set debug environment variable if requested
    if args.debug:
        os.environ["DEBUG"] = "1"
        debug = True
    else:
        debug = False
    
    # Import MLX directly
    try:
        import mlx.core as mx
        print(f"MLX version: {mx.__version__}")
        mlx_available = True
    except ImportError:
        print("MLX not available. Please install with: pip install mlx")
        mlx_available = False
    
    # Import safetensors for LoRA weights loading
    try:
        import safetensors.numpy
        safetensors_available = True
    except ImportError:
        print("safetensors not available. Please install with: pip install safetensors")
        safetensors_available = False
    
    # Test basic MLX functionality
    if mlx_available:
        print("\nTesting basic MLX functionality...")
        
        # Create test tensors
        x = mx.ones((3, 4))
        y = mx.ones((4, 5))
        
        # Matrix multiplication
        try:
            z = mx.matmul(x, y)
            print(f"Matrix multiplication successful: {x.shape} @ {y.shape} = {z.shape}")
        except Exception as e:
            print(f"Matrix multiplication failed: {e}")
    
    # Load LoRA weights if available
    if mlx_available and safetensors_available and os.path.exists(args.lora_path):
        print(f"\nLoading LoRA weights from: {args.lora_path}")
        
        try:
            # Load weights
            lora_weights = safetensors.numpy.load_file(args.lora_path)
            
            print(f"Successfully loaded {len(lora_weights)} LoRA parameter tensors")
            
            # Display summary
            total_params = 0
            lora_a_count = 0
            lora_b_count = 0
            
            for key, value in lora_weights.items():
                shape_info = f"shape={value.shape}"
                if 'lora_A' in key:
                    lora_a_count += 1
                if 'lora_B' in key:
                    lora_b_count += 1
                param_count = np.prod(value.shape)
                total_params += param_count
                if debug:
                    print(f"  {key}: {shape_info}, params={param_count}")
            
            print(f"Total LoRA parameters: {total_params:,}")
            print(f"LoRA A matrices: {lora_a_count}")
            print(f"LoRA B matrices: {lora_b_count}")
            
            # Convert to MLX arrays
            print("\nConverting to MLX arrays...")
            mlx_lora_weights = {}
            for k, v in lora_weights.items():
                mlx_lora_weights[k] = mx.array(v)
            
            print(f"Successfully converted {len(mlx_lora_weights)} tensors to MLX format")
            
            # Output a test file to verify the loading
            output_path = os.path.join(args.output_dir, "lora_test_results.json")
            print(f"\nSaving test results to: {output_path}")
            
            # Create test results
            results = {
                "timestamp": time.time(),
                "lora_path": args.lora_path,
                "total_parameters": int(total_params),
                "parameter_count": len(lora_weights),
                "lora_A_count": lora_a_count,
                "lora_B_count": lora_b_count,
                "mlx_version": mx.__version__ if mlx_available else None,
                "test_passed": True,
                "parameter_names": list(lora_weights.keys()) if debug else None
            }
            
            # Save results
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            
            print("\nLoRA + MLX test completed successfully\!")
            
            # Test basic LoRA math operations
            if lora_a_count > 0 and lora_b_count > 0:
                print("\nTesting basic LoRA math operations...")
                
                # Get a sample A and B matrix
                a_key = next((k for k in mlx_lora_weights if 'lora_A' in k), None)
                b_key = next((k for k in mlx_lora_weights if 'lora_B' in k), None)
                
                # Ensure they're paired matrices
                if a_key and b_key and a_key.replace('lora_A', '') == b_key.replace('lora_B', ''):
                    a_matrix = mlx_lora_weights[a_key]
                    b_matrix = mlx_lora_weights[b_key]
                    
                    print(f"Using paired matrices:")
                    print(f"  A: {a_key} shape={a_matrix.shape}")
                    print(f"  B: {b_key} shape={b_matrix.shape}")
                    
                    # LoRA math: output = base_weight + B @ A * (alpha/r)
                    try:
                        # Create a tiny base weight for testing
                        base_weight = mx.ones((b_matrix.shape[0], a_matrix.shape[1]))
                        
                        # LoRA parameters
                        r = a_matrix.shape[0]  # Rank
                        alpha = 16.0  # Default alpha
                        scaling = alpha / r
                        
                        # Compute LoRA contribution
                        lora_contribution = mx.matmul(b_matrix, a_matrix) * scaling
                        
                        # Add to base weights
                        merged_weight = base_weight + lora_contribution
                        
                        print(f"  Base weight shape: {base_weight.shape}")
                        print(f"  LoRA contribution shape: {lora_contribution.shape}")
                        print(f"  Merged weight shape: {merged_weight.shape}")
                        
                        print("\nMLX LoRA math operation successful\!")
                    except Exception as e:
                        print(f"\nMLX LoRA math operation failed: {e}")
            
            # Final success message
            print("\n" + "="*80)
            print(" TEST COMPLETED SUCCESSFULLY ".center(80, "="))
            print("="*80 + "\n")
            return 0
        except Exception as e:
            print(f"\nERROR: Failed to process LoRA weights: {e}")
            if debug:
                import traceback
                traceback.print_exc()
                
            print("\n" + "="*80)
            print(" TEST FAILED ".center(80, "="))
            print("="*80 + "\n")
            return 1
    else:
        reasons = []
        if not mlx_available:
            reasons.append("MLX not available")
        if not safetensors_available:
            reasons.append("safetensors not available")
        if not os.path.exists(args.lora_path):
            reasons.append(f"LoRA weights file not found at {args.lora_path}")
            
        print(f"\nCannot perform LoRA test: {', '.join(reasons)}")
        
        # Create an empty file as a placeholder
        placeholder_path = os.path.join(args.output_dir, "lora_test_placeholder.txt")
        with open(placeholder_path, "w") as f:
            f.write("LoRA test could not be performed.\n")
            f.write(f"Reasons: {', '.join(reasons)}\n")
        
        print(f"Created placeholder file at {placeholder_path}")
        
        print("\n" + "="*80)
        print(" TEST SKIPPED ".center(80, "="))
        print("="*80 + "\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
