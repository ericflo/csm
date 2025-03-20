#!/bin/bash
set -e

# Build script for CCSM
mkdir -p build
cd build

# Configure with MLX support
cmake .. -DWITH_MLX=ON

# Build with all cores
cmake --build . -j$(sysctl -n hw.ncpu)

# Report success
echo "Build complete! Binaries are in the build directory."
echo "CPU version: ./build/ccsm-generate"
echo "MLX version: ./build/ccsm-generate-mlx"