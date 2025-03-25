# MLX Acceleration Tests

This directory contains tests for the MLX acceleration components of CCSM. These tests validate the performance optimizations and correctness of MLX-specific implementations.

## Test Components

### MLX Optimizations

The `test_mlx_optimizations.cpp` file tests the performance optimizations implemented for MLX acceleration:

- **Fused Attention**: Tests the efficient implementation of the attention mechanism, including multi-query attention support
- **Fast Rotary Position Embeddings**: Tests the optimized implementation of RoPE for efficient positional encoding
- **Memory-efficient Tensor Operations**: Tests in-place operations and memory reuse strategies
- **Tensor Pool**: Tests the tensor pool implementation for memory reuse
- **Batch Processing**: Tests the batch processor for fusing multiple operations
- **Configuration**: Tests environment variable-based configuration

### MLX Tensor Operations

The `test_mlx_tensor_ops.cpp` file tests the fundamental tensor operations with MLX:

- Basic tensor creation and manipulation
- Tensor type conversion
- Matrix operations
- Broadcasting
- Shape manipulation

### MLX Memory Management

The `test_mlx_memory_management.cpp` file tests memory optimization strategies:

- Memory usage configuration
- Automatic memory optimization based on system resources
- Explicit memory optimization strategies
- Memory-efficient tensor allocation and deallocation

## Running Tests

To run the MLX-specific tests:

```bash
# Run MLX optimizations tests
./test_mlx_optimizations

# Run tensor operations tests
./test_mlx_tensor_ops

# Run memory management tests
./test_mlx_memory_management
```

## Running with Coverage

To generate coverage information for the MLX components:

```bash
# Generate coverage for MLX optimizations
make coverage_mlx_optimizations

# Generate coverage for tensor operations
make coverage_mlx_tensor_ops

# Generate coverage for memory management
make coverage_mlx_memory_management
```

## Performance Measurement

Many of the tests include performance benchmarks that compare standard implementations with optimized ones. These benchmarks output:

- Average execution time for standard and optimized implementations
- Speedup ratios from the optimizations
- Memory usage statistics

The benchmarks are designed to be representative of real-world usage patterns and help identify opportunities for further optimization.

## Skipping Tests

If MLX is not available on your system, these tests will still compile but will skip the MLX-specific test cases, running only the compatibility tests.