#!/bin/bash
set -e

# Create build directory if it doesn't exist
mkdir -p build

# Move to build directory
cd build

# Configure the project with CMake
cmake ..

# Build the project
make -j$(nproc 2>/dev/null || echo 4)

echo "Build completed successfully!"