"""
Sampling functions for MLX acceleration.
"""

from typing import Optional, Tuple, Union
import time

import numpy as np
import torch

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

def mlx_topk_sampling(
    logits: mx.array, 
    k: int = 5, 
    temperature: float = 1.0,
    seed: Optional[int] = None
) -> mx.array:
    """
    Sample from logits using top-k sampling with MLX.
    
    This implementation exactly matches PyTorch's sample_topk function
    with special handling to avoid tokens that cause MIMI codec errors.
    
    Args:
        logits: Logits to sample from [batch_size, vocab_size]
        k: Number of top candidates to sample from
        temperature: Temperature for sampling
        seed: Random seed for reproducibility
        
    Returns:
        Sampled indices with shape [batch_size, 1]
    """
    if not MLX_AVAILABLE:
        raise ImportError("MLX is not available for sampling")
    
    # Get dimensions first
    if len(logits.shape) == 1:
        # [vocab_size] -> [1, vocab_size]
        batch_size = 1
        vocab_size = logits.shape[0]
        logits = mx.expand_dims(logits, axis=0)
    else:
        batch_size, vocab_size = logits.shape
        
    # CRITICAL: Block problematic tokens (1-31) right from the start
    # These tokens cause fatal errors in the MIMI codec
    for i in range(1, 32):
        if i < vocab_size:
            # Apply an extreme penalty to these tokens to prevent selection
            for b in range(batch_size):
                # Update logits with mx.where
                logits = mx.where(
                    (mx.arange(logits.shape[0])[:,None] == b) & 
                    (mx.arange(logits.shape[1])[None,:] == i),
                    -1e9,
                    logits
                )
    
    # Apply temperature
    scaled_logits = logits / temperature
    
    # Apply top-k filtering
    filtered_logits = scaled_logits.copy()
    
    # Process batch by batch
    for b in range(batch_size):
        # Get this batch's logits
        batch_logits = scaled_logits[b]
        
        # Get top-k values and indices
        sorted_indices = mx.argsort(batch_logits, descending=True)
        k_val = min(k, vocab_size)
        topk_indices = sorted_indices[:k_val]
        topk_values = mx.take(batch_logits, topk_indices)
        
        # Get kth largest value as threshold
        threshold = topk_values[-1]
        
        # Create mask for values below threshold
        below_threshold = batch_logits < threshold
        
        # Set values below threshold to negative infinity
        batch_filtered = mx.where(below_threshold, mx.array(-float('inf')), batch_logits)
        # Update filtered_logits with mx.where
        filtered_logits = mx.where(
            mx.arange(filtered_logits.shape[0])[:,None] == b,
            batch_filtered[None,:],
            filtered_logits
        )
    
    # Apply softmax to filtered logits
    probs = mx.softmax(filtered_logits, axis=-1)
    
    # Sample using Gumbel-max trick (exactly like PyTorch's _multinomial_sample_one_no_sync)
    samples = mx.zeros((batch_size, 1), dtype=mx.int32)
    
    # Use deterministic seed if provided
    key = mx.random.key(seed if seed is not None else int(time.time() * 1000))
    
    for b in range(batch_size):
        # Get fresh key for this batch
        key, subkey = mx.random.split(key)
        
        # Generate uniform random values
        uniform = mx.random.uniform(subkey, shape=probs[b].shape)
        
        # Transform to exponential distribution: -log(u) ~ Exp(1)
        exponential = -mx.log(uniform + 1e-10)  # Add epsilon for numerical stability
        
        # Apply Gumbel-max trick: probs / exponential is equivalent to torch implementation
        gumbel_probs = probs[b] / exponential
        
        # ADDITIONAL SAFETY: Explicitly zero out problematic tokens 1-31
        for i in range(1, 32):
            if i < gumbel_probs.shape[0]:
                # Update with mx.where
                gumbel_probs = mx.where(
                    mx.arange(gumbel_probs.shape[0]) == i,
                    0.0,
                    gumbel_probs
                )
        
        # ADDITIONAL SAFETY: Check for invalid token ranges
        vocab_size = probs.shape[-1]  # Get vocab size from probs tensor
        if vocab_size > 2051:  # Standard CSM audio vocab size is 2051
            # Any value beyond 2050 is likely invalid for the MIMI codec
            for i in range(2051, vocab_size):
                # Update with mx.where
                gumbel_probs = mx.where(
                    mx.arange(gumbel_probs.shape[0]) == i,
                    0.0,
                    gumbel_probs
                )
        
        # Get argmax (sample with highest probability)
        sample_idx = mx.argmax(gumbel_probs)
        
        # FINAL SAFETY CHECK: Never return values in problematic range
        if 1 <= sample_idx < 32:
            sample_idx = mx.array(0)  # Use silence token instead
        # FINAL SAFETY CHECK: Never return values beyond valid audio vocab range
        elif sample_idx >= 2051:
            sample_idx = mx.array(2050)  # Use max valid token instead
        
        # Store the result
        # Update samples with mx.where
        samples = mx.where(
            (mx.arange(samples.shape[0])[:,None] == b) &
            (mx.arange(samples.shape[1])[None,:] == 0),
            sample_idx.item(),
            samples
        )
    
    return samples

def mlx_categorical_sampling(
    logits: mx.array, 
    temperature: float = 1.0,
    seed: Optional[int] = None
) -> mx.array:
    """
    Categorical sampling from logits with MLX.
    
    This is a convenience wrapper around mlx_topk_sampling that uses the
    full vocabulary size (no filtering).
    
    Args:
        logits: Logits to sample from [batch_size, vocab_size]
        temperature: Temperature for sampling
        seed: Random seed for reproducibility
        
    Returns:
        Sampled indices with shape [batch_size, 1]
    """
    # Determine vocab size
    if len(logits.shape) == 1:
        vocab_size = logits.shape[0]
    else:
        vocab_size = logits.shape[1]
    
    # Use top-k with k = vocab_size
    return mlx_topk_sampling(logits, k=vocab_size, temperature=temperature, seed=seed)