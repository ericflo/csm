#!/usr/bin/env python3
import sys

try:
    import mlx
    version = getattr(mlx, "__version__", "not available")
    print(f"MLX found! Version: {version}")
    print(f"MLX default device: {mlx.core.default_device()}")
except ImportError:
    print("MLX not found")