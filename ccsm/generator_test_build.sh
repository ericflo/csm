#!/bin/bash

# Build the generator test standalone

echo "Building generator_test standalone..."

# Ensure build directory exists
mkdir -p build

# Compile the test with correct includes and library paths
clang++ -std=c++17 -I. -Iinclude -o generator_test generator_test.cpp \
    -L./build/src -lccsm_core \
    -L./build/_deps/sentencepiece-build/src -lsentencepiece \
    -Wl,-rpath,./build/src \
    -Wl,-rpath,./build/_deps/sentencepiece-build/src

# Show the dependencies
if [ -f generator_test ]; then
    echo "Build successful!"
    echo "Dependencies:"
    otool -L generator_test
    
    # Run the test
    echo -e "\nRunning generator_test...\n"
    ./generator_test
else
    echo "Build failed, generator_test not created."
    echo "Make sure you've built the main library with './build.sh' first."
fi