#!/bin/bash
set -e

# Script to build with coverage instrumentation and run tests

# Check if lcov is installed
if ! command -v lcov &> /dev/null; then
    echo "lcov is not installed. Please install it before running this script."
    echo "On macOS: brew install lcov"
    echo "On Ubuntu: apt-get install lcov"
    exit 1
fi

# Build with coverage enabled
echo "Building with coverage instrumentation..."
./build.sh --coverage

# Go to build directory
cd build

# Run the tests
echo "Running tests..."
ctest -V

# Generate coverage report
echo "Generating coverage report..."
make coverage

# Check if browser is available
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    echo "Opening coverage report in browser..."
    open coverage_unit/index.html
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    if command -v xdg-open &> /dev/null; then
        echo "Opening coverage report in browser..."
        xdg-open coverage_unit/index.html
    else
        echo "Coverage report generated at: $(pwd)/coverage_unit/index.html"
    fi
else
    echo "Coverage report generated at: $(pwd)/coverage_unit/index.html"
fi

# Print summary
echo "================ Test Coverage Summary ================"
echo "Check the HTML report for detailed information."
echo "To improve coverage, focus on these areas:"
echo "1. Implement missing tensor methods in tensor.cpp"
echo "2. Fix the ggml_tensor.cpp implementation"
echo "3. Add missing GGML_Context methods in ggml_tensor.cpp"
echo "4. Implement tokenizer.cpp with proper error handling"
echo "5. Add missing model methods in model.cpp"
echo "====================================================="

# Return to original directory
cd ..