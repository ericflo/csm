# CCSM Test Suite

This directory contains the comprehensive test suite for the CCSM C++ codebase, built following the Test-Driven Development (TDD) approach.

## Overview

The test suite is designed to:

1. Define expected behavior for all CCSM components
2. Guide the implementation by specifying clear requirements
3. Maintain code quality and prevent regressions
4. Track code coverage to ensure thorough testing

## Test Organization

The test suite is organized into several categories:

### Unit Tests (`/tests/unit/`)

Unit tests for individual components:

- `test_tensor.cpp` - Tests for the core tensor system
- `test_ggml_tensor.cpp` - Tests for the GGML backend implementation
- `test_model.cpp` - Tests for model configuration and interface
- `test_ggml_model.cpp` - Tests for GGML-specific model implementation
- `test_tokenizer.cpp` - Tests for text and audio tokenization
- `test_generator.cpp` - Tests for text-to-speech generation
- `test_watermarking.cpp` - Tests for audio watermarking
- `test_version.cpp` - Tests for version information
- `test_simd.cpp` - Tests for SIMD optimizations

### Special Tests

- `/tests/tokenizer/test_tokenizer.cpp` - Standalone tokenizer test
- `/tests/simd/test_simd.cpp` - SIMD optimization tests

### Integration Tests (`/tests/integration/`)

- `test_generation_workflow.cpp` - End-to-end tests for the generation pipeline

## Running Tests

### Building and Running All Tests

```bash
# Basic build and test
./build.sh && cd build && ctest

# Build with coverage and generate report
./run_tests_with_coverage.sh
```

### Running Specific Tests

```bash
# Run a specific test
cd build && ./unit_tests/test_tensor

# Run tests with a specific filter
cd build && ctest -R tensor
```

## Code Coverage

Code coverage is tracked using LCOV and reported with the following metrics:

- **Line coverage**: Percentage of code lines executed by tests
- **Branch coverage**: Percentage of code branches (if/else) executed by tests
- **Function coverage**: Percentage of functions called by tests

To generate and view code coverage reports:

```bash
# Generate coverage report
cd build && make coverage

# View the report
open coverage_unit/index.html
```

## Test-Driven Development Workflow

This project follows a TDD approach:

1. **Write a failing test** that defines expected behavior
2. **Implement the minimum code** to make the test pass
3. **Refactor** the implementation while keeping tests passing
4. **Repeat** for new functionality

Many of the tests currently fail because the implementation is incomplete. This is by design - the tests serve as a specification for what the code should do.

## Next Steps

Refer to `/CCSM_TEST_PLAN.md` for the overall test coverage goals and current status.

Refer to `/NEXT_STEPS.md` for the implementation roadmap to make the tests pass.

## Contributing New Tests

When adding new tests:

1. Follow the existing patterns and naming conventions
2. Add appropriate test fixtures for setup and teardown
3. Include both normal operation and edge case tests
4. Update the test plan with new test coverage goals
5. Run all existing tests to verify nothing broke

Always commit test code that compiles, even if the tests fail due to incomplete implementation.