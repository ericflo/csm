#!/bin/bash
set -e

# Build script for CCSM

# Command line argument handling
ENABLE_COVERAGE=0
ENABLE_MLX=1

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --coverage) ENABLE_COVERAGE=1 ;;
        --no-mlx) ENABLE_MLX=0 ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --coverage    Enable code coverage instrumentation"
            echo "  --no-mlx      Disable MLX support"
            echo "  --help        Show this help message"
            exit 0
            ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Create build directory
mkdir -p build
cd build

# Configure CMake options
CMAKE_OPTS=""

if [ $ENABLE_MLX -eq 1 ]; then
    CMAKE_OPTS="$CMAKE_OPTS -DWITH_MLX=ON"
else
    CMAKE_OPTS="$CMAKE_OPTS -DWITH_MLX=OFF"
fi

if [ $ENABLE_COVERAGE -eq 1 ]; then
    CMAKE_OPTS="$CMAKE_OPTS -DWITH_COVERAGE=ON"
    # Use debug build for coverage
    CMAKE_OPTS="$CMAKE_OPTS -DCMAKE_BUILD_TYPE=Debug"
else
    # Default to Release build
    CMAKE_OPTS="$CMAKE_OPTS -DCMAKE_BUILD_TYPE=Release"
fi

# Run CMake to configure the build
echo "Configuring with options: $CMAKE_OPTS"
cmake .. $CMAKE_OPTS

# Build with all cores
echo "Building with $(sysctl -n hw.ncpu) cores..."
cmake --build . -j$(sysctl -n hw.ncpu)

# Report success
echo "Build complete! Binaries are in the build directory."
echo "CPU version: ./build/ccsm-generate"

if [ $ENABLE_MLX -eq 1 ]; then
    echo "MLX version: ./build/ccsm-generate-mlx"
fi

if [ $ENABLE_COVERAGE -eq 1 ]; then
    echo ""
    echo "Code coverage support enabled. Run tests with:"
    echo "  cd build && ctest"
    echo ""
    echo "Generate coverage reports with:"
    echo "  cd build && make coverage"
    echo ""
    echo "Open coverage report with:"
    echo "  open build/coverage_unit/index.html"
fi