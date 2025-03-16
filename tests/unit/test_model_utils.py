"""Tests for model utility functions."""

import torch
import pytest

from csm.models.model import _create_causal_mask, _index_causal_mask, sample_topk


def test_create_causal_mask():
    """Test the creation of causal attention masks."""
    seq_len = 4
    device = torch.device("cpu")
    
    mask = _create_causal_mask(seq_len, device)
    
    expected = torch.tensor([
        [True, False, False, False],
        [True, True, False, False],
        [True, True, True, False],
        [True, True, True, True]
    ], dtype=torch.bool)
    
    assert torch.all(mask == expected)
    assert mask.device == device
    assert mask.dtype == torch.bool


def test_index_causal_mask():
    """Test indexing into a causal mask with input positions."""
    seq_len = 4
    device = torch.device("cpu")
    
    mask = _create_causal_mask(seq_len, device)
    
    # Test with a batch of size 2, sequence length 2
    input_pos = torch.tensor([[0, 1], [2, 3]], device=device)
    result = _index_causal_mask(mask, input_pos)
    
    # Expected shape: (batch_size, seq_len, max_seq_len) = (2, 2, 4)
    assert result.shape == (2, 2, 4)
    
    # Check specific values
    # First batch, first position (0) should see only position 0
    assert torch.all(result[0, 0] == torch.tensor([True, False, False, False], device=device))
    # First batch, second position (1) should see positions 0 and 1
    assert torch.all(result[0, 1] == torch.tensor([True, True, False, False], device=device))
    # Second batch, first position (2) should see positions 0, 1, and 2
    assert torch.all(result[1, 0] == torch.tensor([True, True, True, False], device=device))
    # Second batch, second position (3) should see all positions
    assert torch.all(result[1, 1] == torch.tensor([True, True, True, True], device=device))