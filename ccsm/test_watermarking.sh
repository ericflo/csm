#!/bin/bash
# Script to build and run watermarking tests

set -e  # Exit on any error

# Navigate to the ccsm directory
cd "$(dirname "$0")"

# Build the project if needed
if [ ! -d "build" ] || [ ! -f "build/Makefile" ]; then
    echo "Building ccsm project..."
    ./build.sh
fi

# Navigate to build directory
cd build

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Build and run basic watermarking tests
echo -e "${YELLOW}Building and running basic watermarking tests...${NC}"
make test_watermarking
make test_watermarking_edge_cases

# Run the tests with verbose output
echo -e "${YELLOW}Running basic watermarking tests...${NC}"
./tests/test_watermarking --gtest_color=yes

echo -e "${YELLOW}Running watermarking edge cases tests...${NC}"
./tests/test_watermarking_edge_cases --gtest_color=yes

# If coverage generation is requested
if [ "$1" == "--coverage" ]; then
    echo -e "${YELLOW}Generating coverage reports...${NC}"
    make coverage_watermarking
    make coverage_watermarking_edge_cases
    
    # Open coverage report
    echo -e "${GREEN}Opening coverage reports...${NC}"
    
    if [ "$(uname)" == "Darwin" ]; then
        # macOS
        open coverage_watermarking/index.html
        open coverage_watermarking_edge_cases/index.html
    elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
        # Linux
        if [ -x "$(command -v xdg-open)" ]; then
            xdg-open coverage_watermarking/index.html
            xdg-open coverage_watermarking_edge_cases/index.html
        else
            echo -e "${YELLOW}Coverage reports generated at:${NC}"
            echo "$(pwd)/coverage_watermarking/index.html"
            echo "$(pwd)/coverage_watermarking_edge_cases/index.html"
        fi
    else
        echo -e "${YELLOW}Coverage reports generated at:${NC}"
        echo "$(pwd)/coverage_watermarking/index.html"
        echo "$(pwd)/coverage_watermarking_edge_cases/index.html"
    fi
fi

echo -e "${GREEN}All watermarking tests completed!${NC}"