#!/bin/bash

# Build the generator test
set -e

# Ensure we're in the ccsm directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Compile the generator test
g++ -o generator_test generator_test.cpp -I./include -L./lib -lccsm_core -std=c++17

echo "Build successful. Run ./generator_test to test the generator."