# CCSM Development Guide

This document provides a comprehensive guide for developing the CCSM C++ codebase, including implementation priorities, testing strategy, and development roadmap.

## Table of Contents

1. [Development Philosophy](#development-philosophy)
2. [Implementation Status](#implementation-status)
3. [Implementation Plan](#implementation-plan)
4. [Testing Strategy](#testing-strategy)
5. [Testing Infrastructure](#testing-infrastructure)
6. [Component Test Plans](#component-test-plans)
7. [Test Coverage Goals](#test-coverage-goals)
8. [Performance Optimization](#performance-optimization)
9. [Running Tests](#running-tests)

## Development Philosophy

CCSM follows a **Test-Driven Development (TDD)** approach:

1. Write tests that define expected behavior
2. Implement minimal code to make tests pass
3. Refactor for performance and clarity
4. Measure coverage and identify gaps
5. Repeat until all features are implemented

This approach ensures robust, well-tested code with high test coverage.

## Implementation Status

| Component | Status | Description |
|-----------|--------|-------------|
| Core Tensor System | 🟡 In Progress | Basic operations, broadcasting implemented |
| GGML Backend | 🟡 In Progress | Basic integration with pending fixes |
| Model System | 🟢 Basic Working | Model interface and factory implemented |
| Tokenizer | 🟢 Complete | Text tokenization with SentencePiece |
| SIMD Optimizations | 🟢 Complete | Matrix multiplication, normalization, activation functions, kernel fusion, mixed precision, edge case handling, and in-place operations implemented |
| MLX Acceleration | 🟠 Planning | Basic infrastructure only |
| Generator | 🟡 In Progress | Basic functionality and tests implemented |
| Watermarking | 🟡 In Progress | Initial interface implemented, needs full implementation |
| Thread Pool | 🟡 In Progress | Basic functionality implemented |
| CLI Arguments | 🟢 Complete | Argument parsing and validation |
| Utilities | 🟡 In Progress | Basic utilities implemented |

## Implementation Plan

### Priority Order

1. **Core Tensor System** - Foundation for all operations
2. **GGML Backend** - CPU-based tensor operations
3. **Model System** - Transformer model implementation
4. **Tokenizer** - Text and audio tokenization
5. **Generator** - Text-to-audio pipeline
6. **Watermarking** - Audio watermarking
7. **MLX Acceleration** - Apple Silicon optimizations
8. **Utility functions and CLI args** - Supporting infrastructure

### Implementation Process

For each component, follow this workflow:

1. Run existing tests to identify failing tests
2. Implement just enough code to make tests pass
3. Add more tests for additional functionality
4. Implement that functionality
5. Refactor for performance while keeping tests passing
6. Measure coverage and add tests for uncovered areas

## Testing Strategy

CCSM follows a comprehensive testing strategy with multiple test types:

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test interactions between components
3. **Performance Tests**: Test performance metrics and optimizations
4. **Edge Case Tests**: Test behavior under extreme conditions

### Test Design Principles

- Tests should define behavior, not implementation details
- Tests should be independent and reproducible
- Tests should cover both success and failure cases
- Tests should include performance benchmarks where appropriate

## Testing Infrastructure

- **Framework**: Google Test
- **Coverage Analysis**: LCOV/GCOV
- **Coverage Reports**: HTML reports via custom scripts
- **Test Runners**: CTest with custom scripts for coverage

## Component Test Plans

### 1. Core Tensor System (Target: 90%+)

#### Phase 1: Tensor Creation and Basic Operations ✅
- Basic tensor creation tests
- Shape manipulation tests
- Memory management tests
- Tensor conversion tests

#### Phase 2: Tensor Broadcasting and Advanced Operations ✅
- Broadcasting implementation tests ✅
- Tensor view/slice tests ✅
- Type conversion and promotion tests ✅
- Tensor serialization tests ✅
- Memory efficiency tests (views vs. copies) ✅

#### Phase 3: Performance Tests 🔴
- Basic operation benchmarks
- Memory allocation efficiency
- Broadcasting performance
- Comparison with other tensor libraries

### 2. SIMD Optimizations (Target: 85%+)

#### Phase 1: Basic SIMD Operations ✅
- Vector operation tests (add, multiply, etc.)
- Platform detection tests
- Basic activation function tests (ReLU, SiLU)

#### Phase 2: Advanced SIMD Operations ✅
- Matrix multiplication tests (basic) ✅
- Cache-aware matrix multiplication ✅
- Comprehensive activation function tests ✅
- Architecture-specific tests (AVX, AVX2, AVX-512, NEON) ✅
- Fallback implementation tests ✅
- RMS Normalization implementation ✅
- Layer Normalization implementation ✅
- Attention mechanism implementation ✅
- Kernel fusion implementations:
  - Fused RMS Norm + SiLU activation ✅
  - Fused Layer Norm + ReLU activation ✅

#### Phase 3: Performance and Stability ✅
- Performance comparison benchmarks ✅
- Numerical stability tests ✅
- Mixed precision tests ✅
- Edge case handling (denormals, NaN, Inf) ✅

### 3. Model System (Target: 90%+)

#### Phase 1: Model Interface ✅
- Model creation tests
- Configuration handling tests
- Parameter access tests
- Model state tests

#### Phase 2: Model Operations 🔴
- Forward pass tests
- Attention mechanism tests
- Transformer layer tests
- KV cache tests

### 4. GGML Backend (Target: 85%+)

#### Phase 1: Basic GGML Integration ✅
- GGML tensor creation tests
- GGML memory management tests
- GGML basic operation tests
- GGML computation graph tests

#### Phase 2: GGML Advanced Features 🔴
- GGML quantization tests
- GGML optimized operation tests
- Custom kernel tests
- GGML context management tests

### 5. Tokenizer (Target: 90%+)

#### Phase 1: Basic Tokenization ✅
- Text tokenization tests
- Token decoding tests
- Special token handling tests
- Error handling tests

#### Phase 2: Advanced Tokenization 🔴
- Vocabulary loading tests
- Token combination tests
- Unicode support tests
- Performance tests

### 6. Generator (Target: 90%+)

#### Phase 1: Basic Generation 🔴
- Text-to-audio generation tests
- Sampling parameter tests
- Generation configuration tests
- Basic error handling tests

### 7. Thread Pool (Target: 80%+)

#### Phase 1: Basic Threading 🟡
- Thread creation tests
- Task scheduling tests
- Basic load balancing tests
- Error handling tests

## Test Coverage Goals

| Component | Target Coverage | Current Coverage | Status |
|-----------|----------------|------------------|--------|
| Core Tensor System | 90%+ | ~75% | 🟢 Good |
| Model System | 90%+ | ~50% | 🟡 Partial |
| GGML Subsystem | 85%+ | ~40% | 🟡 Partial |
| SIMD Optimizations | 85%+ | ~60% | 🟢 Good |
| Tokenizer | 90%+ | ~80% | 🟢 Good |
| Generator | 90%+ | ~60% | 🟡 Partial |
| MLX Acceleration | 80%+ | <5% | 🔴 Not Started |
| Watermarking | 85%+ | <5% | 🔴 Not Started |
| Thread Pool | 80%+ | ~50% | 🟡 Partial |
| Command-line Arguments | 90%+ | ~70% | 🟢 Good |
| Utility Functions | 85%+ | ~30% | 🟡 Partial |

## Performance Optimization

Once basic functionality is working, focus on these performance optimizations:

### 1. Memory Management
- Optimize tensor allocation and reuse
- Minimize copies between operations
- Add memory-efficient KV cache implementation

### 2. SIMD Optimizations
- Implement SIMD-accelerated operations for different architectures
- Add platform-specific optimizations
- Focus on matrix multiplication and activation functions

### 3. Thread Pool Improvements
- Improve work distribution
- Add dynamic thread management
- Implement task prioritization

### 4. Model Optimizations
- Optimize attention implementations ✅
- Add kernel fusion where possible ✅
- Additional kernel fusion opportunities:
  - Matrix multiplication + activation functions ✅
  - Further attention fusion optimizations ✅
  - In-place operations to reduce memory traffic ✅
- Implement quantization-aware operations ✅
  - Q8_0 quantization (8-bit with zero bias) ✅
  - Q4_0 quantization (4-bit with zero bias) ✅
  - Q4_1 quantization (4-bit with non-zero bias) ✅

## Running Tests

### Basic Test Execution

```bash
# Build the project
./build.sh

# Run tests using CTest
cd build && ctest
```

### Running Tests with Coverage

```bash
# Build with coverage instrumentation
./build.sh --coverage

# Run all tests with coverage
./scripts/run_tests_with_coverage.sh

# Run specific tests with coverage
./scripts/run_tests_with_coverage.sh "TensorTest.*"

# Generate HTML report
./scripts/run_tests_with_coverage.sh --html
```

### Viewing Coverage Reports

- Basic coverage reports: `build/coverage_*/index.html`
- Detailed analysis report: `coverage_report/index.html`

## Current Implementation Issues

### Watermarking Tests ✅
- ✅ The `WatermarkResult` struct is referenced in tests but not defined in the header
- ✅ Solution: Define the struct in watermarking.h (implemented)

### GGML Functions
- Missing `ggml_graph_compute_with_ctx` function used in `ggml_model.cpp`
- Solution: Update the GGML backend to include this function

### Integration Tests
- Issues with the end-to-end workflow tests
- Solution: Focus on unit tests first, then fix integration tests

## Next Steps

1. ✅ Implement additional kernel fusion optimizations (matrix multiplication + activation, further attention optimizations)
2. ✅ Focus on Core Tensor System tests (type promotion, serialization)
3. ✅ Add quantization-aware operations with SIMD support
4. ✅ Implement additional SIMD optimizations (mixed precision, edge case handling, in-place operations)
5. ✅ Expand GGML backend tests to include quantization (KV cache quantization)
6. ✅ Begin implementing Generator tests
7. Create tests for Thread Pool implementation
8. Implement and test memory management optimizations
9. Complete watermarking implementation

## Release Criteria

Before releasing a stable version:

1. All tests must pass
2. Coverage should meet or exceed targets specified above
3. Performance meets or exceeds baseline metrics
4. All public APIs are properly documented
5. Command-line interface has appropriate help and error messages