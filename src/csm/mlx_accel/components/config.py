"""
Configuration and constants for MLX acceleration.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Default generation parameters
DEFAULT_GENERATION_PARAMS = {
    "temperature": 1.0,
    "topk": 25,
}

@dataclass
class MLXConfig:
    """
    Configuration for MLX acceleration.
    
    Attributes:
        max_seq_len: Maximum sequence length
        rope_theta: RoPE theta parameter
        use_bfloat16: Whether to use bfloat16 precision
        epsilon: Small constant for numerical stability
        verbose: Whether to output verbose logs
        debug: Whether to output debug logs
        temperature: Sampling temperature
        topk: Top-k sampling parameter
        no_watermark: Whether to disable audio watermarking
    """
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    use_bfloat16: bool = False
    epsilon: float = 1e-5
    verbose: bool = False
    debug: bool = False
    temperature: Optional[float] = None
    topk: Optional[int] = None
    no_watermark: bool = False
    
    def __post_init__(self):
        # Initialize from environment variables if not set
        if self.verbose is False:
            self.verbose = os.environ.get("MLX_VERBOSE", "0") == "1"
        if self.debug is False:
            self.debug = os.environ.get("MLX_DEBUG", "0") == "1"

# Removed voice preset functionality