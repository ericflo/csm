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

- **Core Tensor System**
  - `test_tensor.cpp` - Basic tensor operations
  - `test_tensor_type_promotion.cpp` - Type promotion tests
  - `test_tensor_serialization.cpp` - Serialization tests
  - `test_tensor_quantized_operations.cpp` - Quantized operations

- **GGML Backend**
  - `test_ggml_tensor.cpp` - GGML tensor implementation
  - `test_ggml_model.cpp` - GGML model implementation
  - `test_ggml_advanced_quantization.cpp` - Advanced quantization
  - `test_ggml_kv_cache_quantization.cpp` - KV cache quantization
  - `test_ggml_model_quantization.cpp` - Model quantization
  - `test_ggml_kv_cache_pruning.cpp` - KV cache pruning

- **Model System**
  - `test_model.cpp` - Model interface tests
  - `test_model_factory.cpp` - Model factory tests
  - `test_unified_model.cpp` - Unified model interface

- **Generator**
  - `test_generator.cpp` - Core generator tests
  - `test_generator_basic.cpp` - Basic generation functionality
  - `test_generator_advanced.cpp` - Advanced generation features
  - `test_generator_configuration.cpp` - Configuration tests
  - `test_generator_memory_optimization.cpp` - Memory optimization
  - `test_generator_stress.cpp` - Stress tests and edge cases

- **Other Components**
  - `test_watermarking.cpp` - Audio watermarking
  - `test_version.cpp` - Version information
  - `test_thread_pool.cpp` - Thread pool implementation
  - `test_thread_pool_advanced.cpp` - Advanced threading
  - `test_thread_pool_stress.cpp` - Thread pool stress tests
  - `test_memory_optimization_edge_cases.cpp` - Memory optimization

- **MLX Acceleration**
  - `test_mlx_tensor.cpp` - MLX tensor operations
  - `test_mlx_model.cpp` - MLX model implementation
  - `test_mlx_transformer.cpp` - MLX transformer
  - `test_mlx_weight_converter.cpp` - Weight conversion
  - `test_mlx_optimizations.cpp` - Performance optimizations

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

## Implementation Status

Current implementation status for key components:

| Component | Status | Coverage | Description |
|-----------|--------|----------|-------------|
| Core Tensor System | 🟢 Complete | ~90% | Basic operations, type promotion, serialization, and quantized operations |
| GGML Backend | 🟡 In Progress | ~75% | Basic tensor operations, model implementation, advanced quantization |
| Model System | 🟡 In Progress | ~75% | Model interface, factory, unified model interface |
| Tokenizer | 🟢 Complete | ~90% | Text tokenization with SentencePiece |
| SIMD Optimizations | 🟢 Complete | ~85% | Vector operations, matrix multiplication, activation functions |
| MLX Acceleration | 🟡 In Progress | ~80% | Weight converter, tensor operations, transformer implementation |
| Generator | 🟢 Complete | ~75% | Comprehensive test suite with basic, advanced, and stress tests |
| Watermarking | 🟢 Complete | ~85% | Full implementation of watermarking with robust detection |
| Thread Pool | 🟢 Complete | ~90% | Task scheduling, work stealing, and stress tests |

## Generator Component

The Generator component now has a comprehensive test suite:

- `test_generator.cpp` - Core functionality tests
- `test_generator_basic.cpp` - Basic generation tests completing Phase 1 of the plan
- `test_generator_advanced.cpp` - Advanced feature tests implementing Phase 2 of the plan
- `test_generator_configuration.cpp` - Configuration tests
- `test_generator_memory_optimization.cpp` - Memory optimization tests completing Phase 3
- `test_generator_stress.cpp` - Stress tests with failure scenarios

Phase 1 testing (Basic Generation) has been completed with:
- Text-to-speech generation tests
- Sampling parameter tests
- Generation configuration tests
- Basic error handling tests

## Next Steps

For the next phase of development, focus will be on:

1. Completing GGML backend advanced features, particularly:
   - `ggml_graph_compute_with_ctx` function in `ggml_model.cpp`
   - Advanced quantization workflows

2. Model System improvements:
   - Forward pass tests
   - Attention mechanism tests
   - Transformer layer tests
   - KV cache tests

3. Performance optimization tests:
   - Tensor operation benchmarks
   - Memory allocation efficiency
   - Broadcasting performance

See DEVELOPMENT.md for the complete implementation roadmap.

## Contributing New Tests

When adding new tests:

1. Follow the existing patterns and naming conventions
2. Add appropriate test fixtures for setup and teardown
3. Include both normal operation and edge case tests
4. Update the test plan with new test coverage goals
5. Run all existing tests to verify nothing broke

Always commit test code that compiles, even if the tests fail due to incomplete implementation.