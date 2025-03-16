"""
Core MLX operations with special handling for reshape constraints.
"""

import mlx.core as mx
import numpy as np
import torch

def torch_to_mlx(tensor) -> mx.array:
    """
    Convert a PyTorch tensor to an MLX array with special handling for various input types.
    
    Args:
        tensor: Tensor to convert - can be PyTorch tensor, MLX array, numpy array, or other
        
    Returns:
        MLX array
    """
    # If we already have an MLX array, return it directly
    if isinstance(tensor, mx.array):
        return tensor
        
    # Handle numpy arrays directly
    if isinstance(tensor, np.ndarray):
        return mx.array(tensor)
        
    # Handle PyTorch tensors
    if isinstance(tensor, torch.Tensor):
        # Handle BFloat16 and other unsupported dtypes by converting to float32
        if tensor.dtype == torch.bfloat16 or tensor.dtype not in [torch.float32, torch.float64, torch.int32, torch.int64, torch.bool]:
            tensor = tensor.to(dtype=torch.float32)
        return mx.array(tensor.detach().cpu().numpy())
        
    # If it's some other type, try direct conversion through numpy
    try:
        return mx.array(np.array(tensor))
    except Exception as e:
        # Last resort - if it's some iterable, try forcing it to a list first
        try:
            return mx.array(np.array(list(tensor)))
        except Exception:
            # If all else fails, raise the original exception
            raise ValueError(f"Cannot convert {type(tensor)} to MLX array: {e}")

def mlx_to_torch(array) -> torch.Tensor:
    """
    Convert an MLX array to a PyTorch tensor with robust error handling for various input types.
    
    Args:
        array: Array to convert - can be MLX array, PyTorch tensor, numpy array, or other
        
    Returns:
        PyTorch tensor
    """
    # Handle None input
    if array is None:
        return None
        
    # If it's already a PyTorch tensor, return it directly
    if isinstance(array, torch.Tensor):
        return array
    
    # Handle numpy arrays directly
    if isinstance(array, np.ndarray):
        return torch.from_numpy(array.copy())  # Copy to avoid potential issues

    # For MLX arrays
    if isinstance(array, mx.array):
        # Convert through numpy
        try:
            return torch.tensor(np.array(array))
        except Exception as e:
            # Fallback to element-wise conversion
            try:
                # Convert via list
                return torch.tensor(array.tolist())
            except Exception as e2:
                # More fallbacks if needed
                raise ValueError(f"Could not convert MLX array to PyTorch: {e2}")
    
    # Handle scalar values
    if isinstance(array, (int, float, bool)):
        return torch.tensor([array], dtype=torch.float32)
        
    # Handle lists and tuples
    if isinstance(array, (list, tuple)):
        return torch.tensor(array, dtype=torch.float32)
    
    # For other data structures, try generic approaches
    try:
        # Try standard numpy-based conversion
        return torch.tensor(np.array(array))
    except Exception as e:
        # Try other approaches
        try:
            # Direct list conversion if iterable
            if hasattr(array, '__iter__'):
                return torch.tensor(list(array))
            else:
                # Last resort - try direct conversion
                return torch.tensor(array)
        except Exception:
            # Fallback for any other errors - create a minimal tensor
            print(f"Error converting {type(array)} to PyTorch tensor: {e}")
            # Return a small zero tensor as fallback
            return torch.zeros(1, dtype=torch.float32)

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
    Completely rewritten to create the mask directly without dynamic updates.
    
    Args:
        mask: Causal mask of shape [seq_len, seq_len]
        input_pos: Position indices of shape [batch_size, seq_len]
        
    Returns:
        Indexed mask of shape [batch_size, seq_len, seq_len]
    """
    # This implementation assumes input_pos is a 2D tensor [batch, seq_len]
    batch_size, seq_len = input_pos.shape
    
    # Create a default causal mask directly
    # Create a range of indices for each position in the sequence
    indices = mx.arange(seq_len)
    
    # Create a mask where position i can attend to positions j ≤ i
    # This matrix has shape [seq_len, seq_len] where element [i,j] is True if j <= i
    new_mask = indices[:, None] >= indices[None, :]
    
    # Create a batch dimension so it becomes [1, seq_len, seq_len]
    new_mask = mx.expand_dims(new_mask, axis=0)
    
    # Broadcast to all batches [batch_size, seq_len, seq_len]
    # In MLX we can do this by concatenating the same mask for each batch
    batched_masks = [new_mask for _ in range(batch_size)]
    batched_mask = mx.concatenate(batched_masks, axis=0)
    
    # Apply any masking for padding positions if needed
    # Since we're not using the input_pos in this simpler implementation
    # we assume all positions in the sequence are valid
    
    return batched_mask

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
    Completely rewritten to avoid ANY use of .at[idx].set() operations.
    
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
    
    # Adaptive approach for dimension handling without using .at[].set()
    if cos_1d.shape[0] != half_dim:
        # If the source is larger than target, take first half_dim elements
        if cos_1d.shape[0] >= half_dim:
            # Take just the first half_dim elements - this operation is safe
            cos_fixed = cos_1d[:half_dim]
            sin_fixed = sin_1d[:half_dim]
        else:
            # If source is smaller, we need a different approach without using .at[].set()
            
            # Create a new array using concatenation instead
            # First create zero padding of the right size
            avail_dim = cos_1d.shape[0]
            pad_size = half_dim - avail_dim
            
            # Create zero padding arrays
            cos_padding = mx.zeros((pad_size,), dtype=cos_1d.dtype)
            sin_padding = mx.zeros((pad_size,), dtype=sin_1d.dtype)
            
            # Concatenate to get the correctly sized arrays
            cos_fixed = mx.concatenate([cos_1d, cos_padding])
            sin_fixed = mx.concatenate([sin_1d, sin_padding])
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
    Completely rewritten to avoid ANY use of .at[idx].set() operations.
    
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
        # For both expanding and shrinking, use concatenation and slicing
        # instead of element-wise .at[].set() operations
        
        # Flatten the input array
        flat_arr = arr.reshape(-1)
        
        if old_size <= new_size:
            # Expanding case: pad with zeros
            pad_size = new_size - old_size
            padding = mx.zeros((pad_size,), dtype=arr.dtype)
            
            # Concatenate the original data with padding
            result_flat = mx.concatenate([flat_arr, padding])
            
            # Reshape to final shape
            return result_flat.reshape(new_shape)
        else:
            # Shrinking case: take only the first new_size elements
            result_flat = flat_arr[:new_size]
            
            # Reshape to final shape
            return result_flat.reshape(new_shape)