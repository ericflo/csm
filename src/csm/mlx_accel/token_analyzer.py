"""
Token analysis utilities for comparing PyTorch and MLX token generation.

This module provides utilities to capture and analyze token generation from
both PyTorch and MLX implementations of the CSM model, helping to identify
differences in their distributions and sampling behaviors.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os
from typing import Dict, List, Optional, Tuple, Union, Any

def capture_tokens(pt_model, mlx_model, text, verbose=True):
    """
    Capture token generation from both PyTorch and MLX implementations.
    
    Args:
        pt_model: PyTorch CSM generator
        mlx_model: MLX CSM generator
        text: Text to generate speech from
        verbose: Whether to print analysis information
        
    Returns:
        Dictionary containing tokens and audio from both implementations
    """
    results = {}
    
    # Generate with PyTorch
    if verbose:
        print("Generating with PyTorch...")
    
    try:
        pt_audio = pt_model.generate(text=text, speaker=0)
        
        # Try to access tokens if they're stored
        pt_tokens = None
        if hasattr(pt_model, '_last_tokens'):
            pt_tokens = pt_model._last_tokens
        elif hasattr(pt_model, '_last_samples'):
            # Convert samples to tokens if needed
            samples = pt_model._last_samples
            if isinstance(samples, list) and len(samples) > 0 and isinstance(samples[0], torch.Tensor):
                pt_tokens = torch.stack(samples).permute(1, 2, 0)  # Standard format [batch, codebooks, seq]
        
        # Store results
        results['pytorch'] = {
            'tokens': pt_tokens,
            'audio': pt_audio
        }
        
        if verbose:
            print(f"PyTorch generation complete")
            if pt_tokens is not None:
                print(f"PyTorch tokens shape: {pt_tokens.shape}")
            else:
                print("PyTorch tokens not captured")
    
    except Exception as e:
        if verbose:
            print(f"PyTorch generation failed: {e}")
        
        # Store error information
        results['pytorch'] = {
            'error': str(e)
        }
    
    # Generate with MLX hybrid (our current working solution)
    if verbose:
        print("\nGenerating with MLX hybrid...")
    
    try:
        mlx_audio = mlx_model.generate_speech(text=text, speaker=0)
        
        # Try to access tokens if they're stored
        mlx_tokens = None
        if hasattr(mlx_model, '_last_tokens'):
            mlx_tokens = mlx_model._last_tokens
        
        # Store results
        results['mlx'] = {
            'tokens': mlx_tokens,
            'audio': mlx_audio
        }
        
        if verbose:
            print(f"MLX generation complete")
            if mlx_tokens is not None:
                print(f"MLX tokens shape: {mlx_tokens.shape}")
            else:
                print("MLX tokens not captured")
                
    except Exception as e:
        if verbose:
            print(f"MLX generation failed: {e}")
        
        # Store error information
        results['mlx'] = {
            'error': str(e)
        }
    
    # Analyze distributions if tokens are available
    if verbose and 'pytorch' in results and 'mlx' in results:
        pt_tokens = results['pytorch'].get('tokens')
        mlx_tokens = results['mlx'].get('tokens')
        
        if pt_tokens is not None and mlx_tokens is not None:
            analyze_distributions(pt_tokens, mlx_tokens)
    
    return results

def analyze_distributions(pt_tokens, mlx_tokens):
    """
    Analyze token distributions between PyTorch and MLX implementations.
    
    Args:
        pt_tokens: PyTorch generated tokens
        mlx_tokens: MLX generated tokens
    """
    # Ensure we have tensors
    if not isinstance(pt_tokens, torch.Tensor) or not isinstance(mlx_tokens, torch.Tensor):
        print("Cannot analyze distributions: tokens must be PyTorch tensors")
        return
    
    # Flatten tensors for analysis
    pt_flat = pt_tokens.flatten().cpu().tolist()
    mlx_flat = mlx_tokens.flatten().cpu().tolist()
    
    # Count token occurrences
    pt_counter = Counter(pt_flat)
    mlx_counter = Counter(mlx_flat)
    
    # Basic statistics
    print(f"\nPyTorch: {len(pt_counter)} unique tokens out of {len(pt_flat)} total")
    print(f"MLX: {len(mlx_counter)} unique tokens out of {len(mlx_flat)} total")
    
    # Top tokens
    print("\nPyTorch top tokens:")
    for token, count in pt_counter.most_common(10):
        print(f"  {token}: {count} times ({count/len(pt_flat)*100:.2f}%)")
        
    print("\nMLX top tokens:")
    for token, count in mlx_counter.most_common(10):
        print(f"  {token}: {count} times ({count/len(mlx_flat)*100:.2f}%)")
        
    # Calculate overlap
    pt_set = set(pt_counter.keys())
    mlx_set = set(mlx_counter.keys())
    common = pt_set.intersection(mlx_set)
    
    print(f"\nToken overlap: {len(common)} tokens ({len(common)/len(pt_set)*100:.2f}% of PyTorch tokens)")
    
    # Distribution similarity (Jaccard similarity for distributions)
    similarity = 0
    for token in common:
        pt_prob = pt_counter[token] / len(pt_flat)
        mlx_prob = mlx_counter[token] / len(mlx_flat)
        # Use min to measure overlap
        similarity += min(pt_prob, mlx_prob)
    
    print(f"Distribution similarity: {similarity*100:.2f}%")
    
    # Token range analysis
    pt_min, pt_max = min(pt_flat), max(pt_flat)
    mlx_min, mlx_max = min(mlx_flat), max(mlx_flat)
    
    print(f"\nPyTorch token range: {pt_min} to {pt_max}")
    print(f"MLX token range: {mlx_min} to {mlx_max}")
    
    # Check problematic token ranges (1-31)
    pt_problematic = [t for t in pt_flat if 1 <= t <= 31]
    mlx_problematic = [t for t in mlx_flat if 1 <= t <= 31]
    
    print(f"\nPyTorch problematic tokens (1-31): {len(pt_problematic)} ({len(pt_problematic)/len(pt_flat)*100:.2f}%)")
    print(f"MLX problematic tokens (1-31): {len(mlx_problematic)} ({len(mlx_problematic)/len(mlx_flat)*100:.2f}%)")
    
    # Distribution visualization
    try:
        # Plot histograms
        plt.figure(figsize=(15, 10))
        
        # Set bins to cover the full range with reasonable granularity
        max_token = max(pt_max, mlx_max)
        bins = min(100, max_token)  # Don't use too many bins
        
        # Plot PyTorch distribution
        plt.subplot(2, 1, 1)
        plt.hist(pt_flat, bins=bins, alpha=0.7, color='blue')
        plt.title('PyTorch Token Distribution')
        plt.xlabel('Token Value')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Plot MLX distribution
        plt.subplot(2, 1, 2)
        plt.hist(mlx_flat, bins=bins, alpha=0.7, color='green')
        plt.title('MLX Token Distribution')
        plt.xlabel('Token Value')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        os.makedirs('token_analysis', exist_ok=True)
        plt.savefig('token_analysis/token_distribution.png')
        print("\nToken distribution plot saved to token_analysis/token_distribution.png")
        
    except Exception as e:
        print(f"Error generating distribution visualization: {e}")

def distribution_similarity(tokens1, tokens2):
    """
    Calculate similarity between two token distributions.
    
    Args:
        tokens1: First token tensor
        tokens2: Second token tensor
        
    Returns:
        Similarity score between 0 and 1
    """
    # Handle non-tensor inputs
    if not isinstance(tokens1, torch.Tensor) or not isinstance(tokens2, torch.Tensor):
        return 0
    
    # Flatten tensors
    flat1 = tokens1.flatten().cpu().tolist()
    flat2 = tokens2.flatten().cpu().tolist()
    
    # Count tokens
    counter1 = Counter(flat1)
    counter2 = Counter(flat2)
    
    # Get token sets
    tokens_set1 = set(counter1.keys())
    tokens_set2 = set(counter2.keys())
    
    # Find common tokens
    common = tokens_set1.intersection(tokens_set2)
    
    # Calculate distribution similarity
    similarity = 0
    for token in common:
        prob1 = counter1[token] / len(flat1)
        prob2 = counter2[token] / len(flat2)
        # Use min to measure overlap
        similarity += min(prob1, prob2)
    
    return similarity

def save_token_analysis(results, filename="token_analysis_results.pt"):
    """
    Save token analysis results to a file.
    
    Args:
        results: Dictionary of token generation results
        filename: Filename to save results to
    """
    os.makedirs('token_analysis', exist_ok=True)
    torch.save(results, os.path.join('token_analysis', filename))
    print(f"Analysis saved to token_analysis/{filename}")

def load_token_analysis(filename="token_analysis_results.pt"):
    """
    Load token analysis results from a file.
    
    Args:
        filename: Filename to load results from
        
    Returns:
        Dictionary of token generation results
    """
    path = os.path.join('token_analysis', filename)
    if os.path.exists(path):
        return torch.load(path)
    else:
        print(f"File not found: {path}")
        return None

def compare_sampling_implementations(text="Hello, this is a test of the CSM speech model."):
    """
    Compare different sampling implementations.
    
    Args:
        text: Text to generate speech from
        
    Returns:
        Dictionary containing results from hybrid and exact implementations
    """
    # Load models
    from csm.generator import load_csm_1b
    from csm.cli.mlx_components.generator import MLXGenerator
    from csm.cli.mlx_components.config import MLXConfig
    
    # Load PyTorch model
    pt_model = load_csm_1b(device="cpu")
    
    # Load MLX model
    mlx_config = MLXConfig()
    mlx_model = MLXGenerator(model=pt_model._model, tokenizer=pt_model._tokenizer, debug=True)
    
    # Generate with current implementation (hybrid)
    print("=== Current hybrid implementation ===")
    hybrid_results = capture_tokens(pt_model, mlx_model, text)
    
    # Save the current tokens
    save_token_analysis(hybrid_results, "hybrid_tokens.pt")
    
    # Now patch the MLX model to use our exact sampling implementation
    from csm.cli.mlx_embedding import mlx_sample_topk
    from csm.cli.mlx_sample_exact import mlx_sample_exact  # Import the exact implementation
    
    # Temporarily backup original function
    original_sample_topk = mlx_sample_topk
    
    # Patch with our exact implementation
    import sys
    sys.modules['csm.cli.mlx_embedding'].mlx_sample_topk = mlx_sample_exact
    
    print("\n=== Exact matching implementation ===")
    exact_results = capture_tokens(pt_model, mlx_model, text)
    
    # Save the exact tokens
    save_token_analysis(exact_results, "exact_tokens.pt")
    
    # Restore original function
    sys.modules['csm.cli.mlx_embedding'].mlx_sample_topk = original_sample_topk
    
    # Print summary
    print("\n=== Summary ===")
    
    hybrid_similarity = 0
    exact_similarity = 0
    
    if ('pytorch' in hybrid_results and 'tokens' in hybrid_results['pytorch'] and
        'mlx' in hybrid_results and 'tokens' in hybrid_results['mlx']):
        hybrid_similarity = distribution_similarity(
            hybrid_results['pytorch']['tokens'],
            hybrid_results['mlx']['tokens']
        )
    
    if ('pytorch' in exact_results and 'tokens' in exact_results['pytorch'] and
        'mlx' in exact_results and 'tokens' in exact_results['mlx']):
        exact_similarity = distribution_similarity(
            exact_results['pytorch']['tokens'],
            exact_results['mlx']['tokens']
        )
    
    print(f"Hybrid implementation similarity: {hybrid_similarity*100:.2f}%")
    print(f"Exact implementation similarity: {exact_similarity*100:.2f}%")
    
    # Return all results for further analysis
    return {
        'hybrid': hybrid_results,
        'exact': exact_results
    }