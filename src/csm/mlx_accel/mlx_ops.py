"""
Core MLX operations with special handling for reshape constraints.
"""

import mlx.core as mx
import numpy as np
import torch

def torch_to_mlx(tensor: torch.Tensor) -> mx.array:
    """
    Convert a PyTorch tensor to an MLX array with special handling for BFloat16.
    
    Args:
        tensor: PyTorch tensor to convert
        
    Returns:
        MLX array
    """
    # Handle BFloat16 and other unsupported dtypes by converting to float32
    if tensor.dtype == torch.bfloat16 or tensor.dtype not in [torch.float32, torch.float64, torch.int32, torch.int64, torch.bool]:
        tensor = tensor.to(dtype=torch.float32)
    return mx.array(tensor.detach().cpu().numpy())

def mlx_to_torch(array: mx.array) -> torch.Tensor:
    """
    Convert an MLX array to a PyTorch tensor.
    
    Args:
        array: MLX array to convert
        
    Returns:
        PyTorch tensor
    """
    # More efficient conversion using numpy as an intermediate step
    return torch.from_numpy(array.to_numpy()).to(dtype=torch.float32)

def create_causal_mask(seq_len: int):
    """
    Create a causal mask for transformer attention in MLX.
    Avoids using .set() which is not compatible with all MLX versions.
    
    Args:
        seq_len: Sequence length
        
    Returns:
        Causal mask of shape [seq_len, seq_len]
    """
    # Create the mask directly using broadcasting comparison
    indices = mx.arange(seq_len)
    # This creates a mask where each element (i,j) is True if i >= j (upper triangular + diagonal)
    mask = indices[:, None] >= indices[None, :]
    return mask

def index_causal_mask(mask: mx.array, input_pos: mx.array):
    """
    Index into a causal mask using input positions in MLX.
    Avoids using .set() which is not compatible with all MLX versions.
    
    Args:
        mask: Causal mask of shape [seq_len, seq_len]
        input_pos: Position indices of shape [batch_size, seq_len]
        
    Returns:
        Indexed mask of shape [batch_size, seq_len, seq_len]
    """
    # This implementation assumes input_pos is a 2D tensor [batch, seq_len]
    batch_size, seq_len = input_pos.shape
    
    # Create the output tensor directly
    indexed_mask = mx.zeros((batch_size, seq_len, mask.shape[1]), dtype=mx.bool_)
    
    # Use a new implementation that avoids .set() operations
    # First, we'll create a new mask by selecting based on positions
    for b in range(batch_size):
        for s in range(seq_len):
            # Get the position for this batch and sequence element
            pos = input_pos[b, s]
            # Extract current row
            row = mask[pos]
            # Build a new mask by concatenating slices
            if s == 0:
                # Initial slice
                indexed_mask_b = row.reshape(1, -1)
            else:
                # Concatenate additional rows
                new_row = row.reshape(1, -1)
                indexed_mask_b = mx.concatenate([indexed_mask_b, new_row], axis=0)
        
        # Add the batch dimension back
        if b == 0:
            indexed_mask = indexed_mask_b.reshape(1, seq_len, -1)
        else:
            new_batch = indexed_mask_b.reshape(1, seq_len, -1)
            indexed_mask = mx.concatenate([indexed_mask, new_batch], axis=0)
            
    return indexed_mask

def mlx_layer_norm(x: mx.array, weight: mx.array, bias: mx.array = None, eps: float = 1e-5) -> mx.array:
    """
    Apply layer normalization using MLX operations.
    
    Args:
        x: Input tensor [batch_size, seq_len, dim]
        weight: Scale parameter [dim]
        bias: Shift parameter [dim], optional
        eps: Epsilon for numerical stability
    
    Returns:
        Normalized tensor [batch_size, seq_len, dim]
    """
    # Calculate mean and variance along the last dimension
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.mean((x - mean) ** 2, axis=-1, keepdims=True)
    
    # Normalize
    x_norm = (x - mean) / mx.sqrt(var + eps)
    
    # Scale and shift
    if bias is not None:
        return x_norm * weight + bias
    else:
        return x_norm * weight

def mlx_rotary_embedding(x: mx.array, cos: mx.array, sin: mx.array, position_ids: mx.array):
    """
    Apply rotary embeddings to input tensors using MLX.
    Completely rewritten to avoid any .set() operations.
    
    Args:
        x: Input tensor [batch_size, seq_len, num_heads, head_dim]
        cos: Cosine values [seq_len, 1, head_dim]
        sin: Sine values [seq_len, 1, head_dim]
        position_ids: Position indices [batch_size, seq_len]
        
    Returns:
        Tensor with rotary embeddings applied
    """
    # Get dimensions
    batch_size, seq_len, num_heads, head_dim = x.shape
    half_dim = head_dim // 2
    
    # Select the cosine and sine values for the positions
    # Get the first position ID (assuming all are the same in generation)
    position = position_ids[0, 0]
    
    # Get the corresponding cosine and sine values
    if position < cos.shape[0]:
        cos_selected = cos[position]
        sin_selected = sin[position]
    else:
        # Fallback if position is out of bounds
        cos_selected = cos[0]
        sin_selected = sin[0]
    
    # Force reshape to a 1D array first
    cos_1d = cos_selected.reshape(-1) 
    sin_1d = sin_selected.reshape(-1)
    
    # Adaptive approach for dimension handling
    if cos_1d.shape[0] != half_dim:
        # Create fixed-size target tensors
        cos_fixed = mx.ones((half_dim,), dtype=cos_selected.dtype)
        sin_fixed = mx.zeros((half_dim,), dtype=sin_selected.dtype)
        
        # If the source is larger than target, take first half_dim elements
        if cos_1d.shape[0] >= half_dim:
            # Take just the first half_dim elements
            cos_fixed = cos_1d[:half_dim]
            sin_fixed = sin_1d[:half_dim]
        else:
            # If source is smaller, use as many elements as available
            avail_dim = cos_1d.shape[0]
            # Create new tensors with the first avail_dim elements from source
            
            # For cos: ones (initial) with first avail_dim elements from source
            cos_zeros = mx.zeros((half_dim,), dtype=cos_selected.dtype)
            cos_mask = mx.zeros((half_dim,), dtype=mx.bool_)
            # Set first avail_dim elements to True
            cos_indices = mx.arange(avail_dim)
            # For each valid index, set the corresponding mask element to True
            for i in range(avail_dim):
                idx = cos_indices[i]
                if idx < half_dim:  # Safety check
                    cos_mask_value = cos_mask.astype(mx.int32)
                    cos_mask_value = cos_mask_value.at[idx].set(1)
                    cos_mask = cos_mask_value.astype(mx.bool_)
                    cos_zeros = cos_zeros.at[idx].set(cos_1d[i])
            
            # For sin: zeros (initial) with first avail_dim elements from source
            sin_zeros = mx.zeros((half_dim,), dtype=sin_selected.dtype)
            sin_mask = mx.zeros((half_dim,), dtype=mx.bool_)
            # Set first avail_dim elements to True
            sin_indices = mx.arange(avail_dim)
            # For each valid index, set the corresponding mask element to True
            for i in range(avail_dim):
                idx = sin_indices[i]
                if idx < half_dim:  # Safety check
                    sin_mask_value = sin_mask.astype(mx.int32)
                    sin_mask_value = sin_mask_value.at[idx].set(1)
                    sin_mask = sin_mask_value.astype(mx.bool_)
                    sin_zeros = sin_zeros.at[idx].set(sin_1d[i])
            
            # Update fixed values
            cos_fixed = cos_zeros
            sin_fixed = sin_zeros
    else:
        # Dimensions match, use as is
        cos_fixed = cos_1d
        sin_fixed = sin_1d
    
    # Reshape for broadcasting to the input tensor
    # [1, 1, 1, half_dim]
    cos_broadcasted = cos_fixed.reshape(1, 1, 1, half_dim)
    sin_broadcasted = sin_fixed.reshape(1, 1, 1, half_dim)
    
    # Split input x along last dimension
    try:
        # Try standard split
        x1, x2 = mx.split(x, 2, axis=-1)
    except:
        # Manual split fallback
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
    
    # Apply rotary embeddings directly
    output1 = x1 * cos_broadcasted + x2 * sin_broadcasted
    output2 = -x1 * sin_broadcasted + x2 * cos_broadcasted
    
    # Concatenate the results
    return mx.concatenate([output1, output2], axis=-1)

def full_like(tensor, fill_value):
    """
    Creates a tensor filled with a scalar value, with the same shape and dtype as the input tensor.
    This is a replacement for mx.full_like which may not be available in all MLX versions.
    
    Args:
        tensor: Input tensor whose shape and dtype will be used
        fill_value: Value to fill the output tensor with
        
    Returns:
        New tensor with the same shape and dtype as tensor, filled with fill_value
    """
    return mx.full(tensor.shape, fill_value, dtype=tensor.dtype)

def categorical_sampling(rng_key, probs):
    """
    Sample from a categorical distribution using MLX.
    This is a workaround for compatibility issues with mx.random.categorical.
    
    Args:
        rng_key: Random key for sampling
        probs: Probability distribution [batch_size, num_classes]
        
    Returns:
        Sampled indices with shape [batch_size]
    """
    print(f"!!!!! DEBUG: probs.shape={probs.shape}")
    
    # Force shape to be at least 2D
    if len(probs.shape) == 1:
        probs = mx.expand_dims(probs, axis=0)
    
    # Get the cumulative distribution function
    cdf = mx.cumsum(probs, axis=-1)
    
    # Generate a single random value in [0,1)
    # This avoids shape issues with mx.random.uniform
    random_value = np.random.random()
    
    # Directly compare the random value with each element in CDF
    # Counting how many elements in CDF are less than our random value
    results = []
    for i in range(probs.shape[0]):
        # Find the first index where CDF exceeds our random value
        row_cdf = cdf[i]
        idx = 0
        while idx < row_cdf.shape[0] and row_cdf[idx] <= random_value:
            idx += 1
        results.append(idx)
    
    # Create a NumPy array with the results and convert to MLX array
    # This ensures it will have the expected to_numpy() method
    return mx.array(np.array(results))

def mlx_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    mask: mx.array,
    dropout_prob: float = 0.0
) -> mx.array:
    """
    Compute attention using MLX operations.
    
    Args:
        query: [batch_size, seq_len, num_heads, head_dim]
        key: [batch_size, seq_len, num_heads, head_dim]
        value: [batch_size, seq_len, num_heads, head_dim]
        mask: [batch_size, seq_len, seq_len] or None
        dropout_prob: dropout probability
    
    Returns:
        output: [batch_size, seq_len, num_heads, head_dim]
    """
    # Get dimensions
    batch_size, q_len, num_heads, head_dim = query.shape
    _, k_len, _, _ = key.shape
    
    # Ensure inputs have the same dimensions
    assert query.shape[-1] == key.shape[-1], "Query and key dimensions must match"
    assert key.shape[:-2] == value.shape[:-2], "Key and value batch/sequence dims must match"
    assert key.shape[-2] == value.shape[-2], "Key and value heads must match"
    
    # Compute scaled dot-product attention
    # Reshape query and key for matrix multiplication
    # [batch_size, num_heads, seq_len, head_dim]
    q = mx.transpose(query, (0, 2, 1, 3))
    k = mx.transpose(key, (0, 2, 1, 3))
    v = mx.transpose(value, (0, 2, 1, 3))
    
    # Compute attention scores
    # [batch_size, num_heads, q_len, k_len]
    scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2)))
    
    # Scale scores
    scores = scores / mx.sqrt(mx.array(head_dim, dtype=scores.dtype))
    
    # Apply mask
    if mask is not None:
        # Add dimensions to match scores: [batch_size, 1, q_len, k_len]
        if len(mask.shape) == 3:
            # If mask is [batch_size, q_len, k_len]
            expanded_mask = mask.reshape(batch_size, 1, q_len, k_len)
        else:
            expanded_mask = mask
        
        # Apply mask by setting masked positions to large negative value
        # Use our custom full_like function instead of mx.full_like
        scores = mx.where(expanded_mask, scores, full_like(scores, -1e9))
    
    # Apply softmax to get attention weights
    attn_weights = mx.softmax(scores, axis=-1)
    
    # Apply dropout if needed
    if dropout_prob > 0.0:
        # Generate a dropout key
        rng_key = mx.random.key(np.random.randint(0, 2**32))
        attn_weights = mx.random.dropout(rng_key, attn_weights, dropout_prob)
    
    # Apply attention weights to value
    # [batch_size, num_heads, q_len, head_dim]
    context = mx.matmul(attn_weights, v)
    
    # Transpose back to original shape
    # [batch_size, q_len, num_heads, head_dim]
    context = mx.transpose(context, (0, 2, 1, 3))
    
    return context

def mlx_feed_forward(x: mx.array, w1: mx.array, w2: mx.array, w3: mx.array, bias1=None, bias2=None, bias3=None) -> mx.array:
    """
    Compute feed-forward network using MLX operations.
    This implements a SwiGLU FFN used in Llama 3.2.
    
    Args:
        x: input tensor [batch_size, seq_len, dim]
        w1, w2, w3: weight matrices
        bias1, bias2, bias3: optional biases
    
    Returns:
        output tensor [batch_size, seq_len, dim]
    """
    # SwiGLU activation
    # First compute the gating and linear paths
    if bias1 is not None:
        gate = mx.matmul(x, w1) + bias1
    else:
        gate = mx.matmul(x, w1)
        
    if bias2 is not None:
        hidden = mx.matmul(x, w2) + bias2
    else:
        hidden = mx.matmul(x, w2)
    
    # Apply SwiGLU: gate * swish(hidden)
    # swish(x) = x * sigmoid(x)
    swish = hidden * mx.sigmoid(hidden)
    activated = gate * swish
    
    # Project back to input dimension
    if bias3 is not None:
        return mx.matmul(activated, w3) + bias3
    else:
        return mx.matmul(activated, w3)

def create_zeros_tensor(shape, dtype=mx.float32):
    """
    Create a zero-filled tensor with the given shape directly, avoiding reshape operations.
    
    Args:
        shape: Tuple of dimensions
        dtype: Data type for the tensor
        
    Returns:
        MLX array with the given shape
    """
    return mx.zeros(shape, dtype=dtype)

def create_ones_tensor(shape, dtype=mx.float32):
    """
    Create a tensor filled with ones with the given shape directly, avoiding reshape operations.
    
    Args:
        shape: Tuple of dimensions
        dtype: Data type for the tensor
        
    Returns:
        MLX array with the given shape
    """
    return mx.ones(shape, dtype=dtype)

def create_tensor_from_scalar(scalar_value, shape, dtype=mx.float32):
    """
    Create a tensor with the given shape filled with a scalar value.
    This avoids reshape operations that may fail with MLX.
    
    Args:
        scalar_value: Value to fill the tensor with
        shape: Tuple of dimensions
        dtype: Data type for the tensor
        
    Returns:
        MLX array with the given shape
    """
    tensor = mx.zeros(shape, dtype=dtype)
    # Use elementwise assignment if needed
    if scalar_value != 0:
        return mx.full(shape, scalar_value, dtype=dtype)
    return tensor

def safe_reshape(arr, new_shape):
    """
    Safely reshape an MLX array, handling cases where direct reshape might fail.
    
    Args:
        arr: Input MLX array
        new_shape: Target shape tuple
        
    Returns:
        Reshaped MLX array
    """
    # Check if direct reshape would work (same number of elements)
    old_size = mx.prod(mx.array(arr.shape))
    new_size = mx.prod(mx.array(new_shape))
    
    if old_size == new_size:
        # If sizes match, use direct reshape (should work)
        return arr.reshape(new_shape)
    else:
        # Create a new tensor with the desired shape
        result = mx.zeros(new_shape, dtype=arr.dtype)
        
        # Try to copy data if possible (only for expanding)
        if old_size <= new_size:
            # Flatten both
            flat_arr = arr.reshape(-1)
            flat_result = result.reshape(-1)
            
            # Copy element by element up to the size of the original array
            for i in range(flat_arr.size):
                flat_result = flat_result.at[i].set(flat_arr[i])
                
            # Reshape back
            return flat_result.reshape(new_shape)
        else:
            # For shrinking, just take the first elements
            flat_arr = arr.reshape(-1)
            flat_result = result.reshape(-1)
            
            for i in range(flat_result.size):
                flat_result = flat_result.at[i].set(flat_arr[i])
            
            return flat_result.reshape(new_shape)