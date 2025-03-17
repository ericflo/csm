#!/usr/bin/env python3
import sys

try:
    import mlx
    version = getattr(mlx, "__version__", "not available")
    print(f"MLX found! Version: {version}")
    
    # Try different approaches to get the default device
    try:
        if hasattr(mlx, 'core') and hasattr(mlx.core, 'default_device'):
            print(f"MLX default device: {mlx.core.default_device()}")
        elif hasattr(mlx, 'default_device'):
            print(f"MLX default device: {mlx.default_device()}")
        else:
            print("MLX default device information not available")
            
        # Print MLX module structure for debugging
        print("\nMLX module attributes:")
        for attr in sorted(dir(mlx)):
            if not attr.startswith('__'):
                print(f"- {attr}")
    except Exception as e:
        print(f"Error getting MLX details: {e}")
except ImportError:
    print("MLX not found")