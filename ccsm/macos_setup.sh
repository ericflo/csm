#!/bin/bash
set -e

# Script to set up development environment for CCSM on macOS
echo "Setting up development environment for CCSM on macOS..."

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "Homebrew not found. Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "Homebrew found. Updating..."
    brew update
fi

# Install required dependencies
echo "Installing dependencies..."
brew install cmake
brew install python
brew install mlx

# Make sure MLX-C is properly set up
echo "Setting up MLX-C..."
if [ ! -d "../reference/mlx-c" ]; then
    echo "MLX-C not found. Cloning repository..."
    mkdir -p ../reference
    cd ../reference
    git clone https://github.com/ml-explore/mlx-c.git
    cd mlx-c
    
    # Build MLX-C
    echo "Building MLX-C..."
    mkdir -p build
    cd build
    cmake ..
    make -j$(sysctl -n hw.ncpu)
    cd ../../../ccsm
else
    echo "MLX-C found at ../reference/mlx-c"
    
    # Check if MLX-C is built
    if [ ! -d "../reference/mlx-c/build" ]; then
        echo "Building MLX-C..."
        cd ../reference/mlx-c
        mkdir -p build
        cd build
        cmake ..
        make -j$(sysctl -n hw.ncpu)
        cd ../../../ccsm
    else
        echo "MLX-C already built."
    fi
fi

# Make the build script executable
chmod +x build.sh

echo "Setup complete! You can now run ./build.sh to build CCSM."