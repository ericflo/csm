"""
MLX implementation of Key-Value cache for transformer inference.
"""

import mlx.core as mx
import numpy as np

class MLXKVCache:
    """Key-Value cache for MLX-based transformer models."""
    
    def __init__(self, batch_size, max_seq_len, num_layers, num_heads, num_kv_heads, head_dim, dtype=mx.float32):
        """Initialize an empty KV cache."""
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        
        # For multi-query attention, kv_heads may be less than query heads
        # Create empty caches for each layer
        self.k_cache = [
            mx.zeros((batch_size, max_seq_len, num_kv_heads, head_dim), dtype=dtype)
            for _ in range(num_layers)
        ]
        self.v_cache = [
            mx.zeros((batch_size, max_seq_len, num_kv_heads, head_dim), dtype=dtype)
            for _ in range(num_layers)
        ]
        
        # Track the current sequence length for each batch item
        self.current_seq_len = mx.zeros((batch_size,), dtype=mx.int32)
        
    def update(self, layer_idx, key, value, positions=None):
        """
        Update the KV cache for a specific layer.
        Reimplemented to avoid .set() operations.
        
        Args:
            layer_idx: The layer index
            key: [batch_size, seq_len, num_kv_heads, head_dim]
            value: [batch_size, seq_len, num_kv_heads, head_dim]
            positions: Optional positions to update, if None, uses sequential positions
        """
        batch_size, seq_len = key.shape[:2]
        
        # If positions are not provided, use the current sequence length
        if positions is None:
            positions = mx.expand_dims(self.current_seq_len, axis=1) + mx.arange(seq_len)
            # Update the current sequence length
            self.current_seq_len = self.current_seq_len + seq_len
        
        # Ensure positions are within bounds
        positions = mx.clip(positions, 0, self.max_seq_len - 1)
        
        # Create completely new caches - this is less efficient but avoids .set()
        new_k_cache = mx.zeros_like(self.k_cache[layer_idx])
        new_v_cache = mx.zeros_like(self.v_cache[layer_idx])
        
        # Copy old cache data (except where we're updating)
        for b in range(batch_size):
            # Create mask for positions that are NOT being updated
            update_positions = positions[b, :].reshape(-1)
            
            # First copy all existing cache values
            new_k_cache = new_k_cache + self.k_cache[layer_idx]
            new_v_cache = new_v_cache + self.v_cache[layer_idx]
            
            # Then overwrite with new values where needed
            for s in range(seq_len):
                pos = positions[b, s].item()  # get as integer
                
                # Create masks for the source position
                b_mask = mx.array([i == b for i in range(batch_size)])
                pos_mask = mx.array([i == pos for i in range(self.max_seq_len)])
                
                # Create combined mask of shape (batch_size, max_seq_len, 1, 1)
                combined_mask = mx.outer(b_mask, pos_mask).reshape(batch_size, self.max_seq_len, 1, 1)
                
                # Use mask to blend new values with old cache
                # Create full cache-shaped tensors with the new value expanded
                new_k_value = mx.broadcast_to(
                    key[b, s].reshape(1, 1, self.num_kv_heads, self.head_dim),
                    (batch_size, self.max_seq_len, self.num_kv_heads, self.head_dim)
                )
                new_v_value = mx.broadcast_to(
                    value[b, s].reshape(1, 1, self.num_kv_heads, self.head_dim),
                    (batch_size, self.max_seq_len, self.num_kv_heads, self.head_dim)
                )
                
                # Update only where mask is True
                k_update = mx.where(combined_mask, new_k_value, new_k_cache)
                v_update = mx.where(combined_mask, new_v_value, new_v_cache)
                
                new_k_cache = k_update
                new_v_cache = v_update
        
        # Assign the new caches
        self.k_cache[layer_idx] = new_k_cache
        self.v_cache[layer_idx] = new_v_cache
    
    def get(self, layer_idx, positions):
        """
        Get values from the KV cache for a specific layer.
        Reimplemented to avoid .set() operations.
        
        Args:
            layer_idx: The layer index
            positions: [batch_size, seq_len] positions to retrieve
            
        Returns:
            (keys, values) tuple of cached values for given positions
        """
        batch_size, seq_len = positions.shape
        k_out = mx.zeros((batch_size, seq_len, self.num_kv_heads, self.head_dim), dtype=self.dtype)
        v_out = mx.zeros((batch_size, seq_len, self.num_kv_heads, self.head_dim), dtype=self.dtype)
        
        # Create an indexing structure for gathering values
        for b in range(batch_size):
            for s in range(seq_len):
                pos = positions[b, s].item()  # get as integer
                if pos < self.max_seq_len:
                    # Extract the correct cache values
                    k_value = self.k_cache[layer_idx][b, pos]
                    v_value = self.v_cache[layer_idx][b, pos]
                    
                    # Create masks for batch and sequence position
                    b_mask = mx.array([i == b for i in range(batch_size)])
                    s_mask = mx.array([i == s for i in range(seq_len)])
                    
                    # Combine masks to create target mask
                    combined_mask = mx.outer(b_mask, s_mask).reshape(batch_size, seq_len, 1, 1)
                    
                    # Expand the value to the output shape
                    expanded_k = mx.broadcast_to(
                        k_value.reshape(1, 1, self.num_kv_heads, self.head_dim),
                        (batch_size, seq_len, self.num_kv_heads, self.head_dim)
                    )
                    expanded_v = mx.broadcast_to(
                        v_value.reshape(1, 1, self.num_kv_heads, self.head_dim),
                        (batch_size, seq_len, self.num_kv_heads, self.head_dim)
                    )
                    
                    # Update output with the new values where mask is True
                    k_out = mx.where(combined_mask, expanded_k, k_out)
                    v_out = mx.where(combined_mask, expanded_v, v_out)
        
        return k_out, v_out
    
    def reset(self):
        """Reset the KV cache to empty state."""
        for layer_idx in range(self.num_layers):
            self.k_cache[layer_idx] = mx.zeros_like(self.k_cache[layer_idx])
            self.v_cache[layer_idx] = mx.zeros_like(self.v_cache[layer_idx])
        self.current_seq_len = mx.zeros_like(self.current_seq_len)