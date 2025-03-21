#!/bin/bash
set -e

# Check if a test name was provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <test_name> [--coverage]"
    echo "Examples:"
    echo "  $0 tensor             # Run tensor tests"
    echo "  $0 model --coverage   # Run model tests with coverage"
    echo ""
    echo "Available tests:"
    echo "  tensor        - Core tensor tests"
    echo "  ggml_tensor   - GGML backend tests"
    echo "  model         - Model interface tests"
    echo "  ggml_model    - GGML model implementation tests"
    echo "  tokenizer     - Tokenizer tests"
    echo "  generator     - Generator tests"
    echo "  watermarking  - Watermarking tests"
    echo "  simd          - SIMD optimization tests"
    echo "  version       - Version information tests"
    echo "  integration   - Integration tests"
    exit 1
fi

# Extract test name
TEST_NAME=$1
COVERAGE=0

# Check for coverage flag
if [ "$2" == "--coverage" ]; then
    COVERAGE=1
fi

# Build with or without coverage
if [ $COVERAGE -eq 1 ]; then
    echo "Building with coverage instrumentation..."
    ./build.sh --coverage
else
    echo "Building..."
    ./build.sh
fi

# Go to build directory
cd build

# Map test name to test executable
case $TEST_NAME in
    "tensor"|"test_tensor")
        TEST_TARGET="unit_tests/test_tensor"
        ;;
    "ggml_tensor"|"test_ggml_tensor")
        TEST_TARGET="unit_tests/test_ggml_tensor"
        ;;
    "model"|"test_model")
        TEST_TARGET="unit_tests/test_model"
        ;;
    "ggml_model"|"test_ggml_model")
        TEST_TARGET="unit_tests/test_ggml_model"
        ;;
    "tokenizer"|"test_tokenizer")
        TEST_TARGET="unit_tests/test_tokenizer"
        ;;
    "generator"|"test_generator")
        TEST_TARGET="unit_tests/test_generator"
        ;;
    "watermarking"|"test_watermarking")
        TEST_TARGET="unit_tests/test_watermarking"
        ;;
    "simd"|"test_simd")
        TEST_TARGET="unit_tests/test_simd"
        ;;
    "version"|"test_version")
        TEST_TARGET="unit_tests/test_version"
        ;;
    "integration"|"generation_workflow")
        TEST_TARGET="integration_tests/test_generation_workflow"
        ;;
    *)
        echo "Unknown test: $TEST_NAME"
        echo "Available tests: tensor, ggml_tensor, model, ggml_model, tokenizer, generator, watermarking, simd, version, integration"
        exit 1
        ;;
esac

# Run the test
echo "Running $TEST_NAME tests..."
if [ -f "$TEST_TARGET" ]; then
    ./$TEST_TARGET
else
    echo "Test executable not found: $TEST_TARGET"
    echo "Make sure the test exists and was built correctly."
    exit 1
fi

# Generate coverage if requested
if [ $COVERAGE -eq 1 ]; then
    echo "Generating coverage report..."
    lcov --capture --directory . --output-file coverage.info
    lcov --remove coverage.info '*/tests/*' '/usr/*' '*/external/*' '*/reference/*' --output-file coverage.info.cleaned
    genhtml coverage.info.cleaned --output-directory coverage_$TEST_NAME
    
    # Open coverage report
    if [[ "$OSTYPE" == "darwin"* ]]; then
        open coverage_$TEST_NAME/index.html
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v xdg-open &> /dev/null; then
            xdg-open coverage_$TEST_NAME/index.html
        else
            echo "Coverage report generated at: $(pwd)/coverage_$TEST_NAME/index.html"
        fi
    else
        echo "Coverage report generated at: $(pwd)/coverage_$TEST_NAME/index.html"
    fi
fi

# Return to original directory
cd ..