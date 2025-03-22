#!/bin/bash
# Script to build, run tests, and generate coverage reports for CCSM C++

set -e  # Exit on error

# Check if we should generate HTML coverage report
GENERATE_HTML=0
if [[ "$1" == "--html" ]]; then
    GENERATE_HTML=1
    shift
fi

# Determine which tests to run
TEST_FILTER=".*"
if [[ "$1" != "" ]]; then
    TEST_FILTER="$1"
fi

echo "🔨 Building with coverage instrumentation..."
if [ -d "build" ]; then
    # Clean build to avoid stale coverage data
    echo "  Cleaning existing build..."
    rm -rf build
fi

# Build with coverage support
./build.sh --coverage

echo "🧪 Running tests matching \"$TEST_FILTER\"..."
cd build
if [[ "$TEST_FILTER" == ".*" ]]; then
    # Run all tests
    ctest -V
else
    # Run filtered tests
    ctest -V -R "$TEST_FILTER"
fi

echo "📊 Generating coverage reports..."
make coverage

# Run coverage analysis script
echo "🔍 Analyzing coverage data..."
cd ..
if [[ $GENERATE_HTML -eq 1 ]]; then
    python3 scripts/analyze_coverage.py --html build/coverage.info
else
    python3 scripts/analyze_coverage.py build/coverage.info
fi

echo "✅ Done!"
echo ""
echo "Coverage reports are available at:"
echo "  build/coverage_unit/index.html (unit tests)"
echo "  build/coverage_simd/index.html (SIMD tests)"
echo "  build/coverage_tokenizer/index.html (tokenizer tests)"

if [[ $GENERATE_HTML -eq 1 ]]; then
    echo ""
    echo "Detailed analysis report is available at:"
    echo "  coverage_report/index.html"
fi