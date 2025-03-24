#include <gtest/gtest.h>
#include <ccsm/mlx/mlx_optimizations.h>
#include <ccsm/utils.h>
#include <memory>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <cstdlib> // for setenv, unsetenv
#include <limits>

namespace ccsm {
namespace testing {

class MLXOptimizationsTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test MLXOptimizationConfig class and its methods
TEST_F(MLXOptimizationsTest, TestMLXOptimizationConfig) {
    // Test default configuration
    MLXOptimizationConfig config;
    EXPECT_EQ(config.compute_precision, MLXOptimizationConfig::ComputePrecision::BFLOAT16);
    EXPECT_EQ(config.memory_usage, MLXOptimizationConfig::MemoryUsage::BALANCED);
    EXPECT_EQ(config.num_compute_threads, 0);
    EXPECT_TRUE(config.use_autotune);
    EXPECT_TRUE(config.use_async_compute);
}

// Test loading configuration from environment variables
TEST_F(MLXOptimizationsTest, TestConfigFromEnv) {
    // Save original environment variables
    const char* orig_precision = getenv("MLX_COMPUTE_PRECISION");
    const char* orig_memory = getenv("MLX_MEMORY_USAGE");
    const char* orig_threads = getenv("MLX_NUM_THREADS");
    
    std::string saved_precision = orig_precision ? orig_precision : "";
    std::string saved_memory = orig_memory ? orig_memory : "";
    std::string saved_threads = orig_threads ? orig_threads : "";
    
    // Test different combinations of environment variables
    std::vector<std::tuple<std::string, std::string, std::string, MLXOptimizationConfig::ComputePrecision, MLXOptimizationConfig::MemoryUsage, int>> test_cases = {
        // precision, memory, threads, expected_precision, expected_memory, expected_threads
        {"float32", "minimal", "4", MLXOptimizationConfig::ComputePrecision::FLOAT32, MLXOptimizationConfig::MemoryUsage::MINIMAL, 4},
        {"FLOAT32", "MINIMAL", "4", MLXOptimizationConfig::ComputePrecision::FLOAT32, MLXOptimizationConfig::MemoryUsage::MINIMAL, 4},
        {"bfloat16", "balanced", "8", MLXOptimizationConfig::ComputePrecision::BFLOAT16, MLXOptimizationConfig::MemoryUsage::BALANCED, 8},
        {"BFLOAT16", "BALANCED", "8", MLXOptimizationConfig::ComputePrecision::BFLOAT16, MLXOptimizationConfig::MemoryUsage::BALANCED, 8},
        {"float16", "performance", "16", MLXOptimizationConfig::ComputePrecision::FLOAT16, MLXOptimizationConfig::MemoryUsage::PERFORMANCE, 16},
        {"FLOAT16", "PERFORMANCE", "16", MLXOptimizationConfig::ComputePrecision::FLOAT16, MLXOptimizationConfig::MemoryUsage::PERFORMANCE, 16},
        {"invalid", "invalid", "invalid", MLXOptimizationConfig::ComputePrecision::BFLOAT16, MLXOptimizationConfig::MemoryUsage::BALANCED, 0},
        {"", "", "", MLXOptimizationConfig::ComputePrecision::BFLOAT16, MLXOptimizationConfig::MemoryUsage::BALANCED, 0}
    };
    
    for (const auto& test_case : test_cases) {
        // Set environment variables
        if (!std::get<0>(test_case).empty()) {
            setenv("MLX_COMPUTE_PRECISION", std::get<0>(test_case).c_str(), 1);
        } else {
            unsetenv("MLX_COMPUTE_PRECISION");
        }
        
        if (!std::get<1>(test_case).empty()) {
            setenv("MLX_MEMORY_USAGE", std::get<1>(test_case).c_str(), 1);
        } else {
            unsetenv("MLX_MEMORY_USAGE");
        }
        
        if (!std::get<2>(test_case).empty()) {
            setenv("MLX_NUM_THREADS", std::get<2>(test_case).c_str(), 1);
        } else {
            unsetenv("MLX_NUM_THREADS");
        }
        
        // Load configuration from environment
        MLXOptimizationConfig config = MLXOptimizationConfig::from_env();
        
        // Check if configuration matches expected values
        EXPECT_EQ(config.compute_precision, std::get<3>(test_case));
        EXPECT_EQ(config.memory_usage, std::get<4>(test_case));
        EXPECT_EQ(config.num_compute_threads, std::get<5>(test_case));
    }
    
    // Restore original environment variables
    if (!saved_precision.empty()) {
        setenv("MLX_COMPUTE_PRECISION", saved_precision.c_str(), 1);
    } else {
        unsetenv("MLX_COMPUTE_PRECISION");
    }
    
    if (!saved_memory.empty()) {
        setenv("MLX_MEMORY_USAGE", saved_memory.c_str(), 1);
    } else {
        unsetenv("MLX_MEMORY_USAGE");
    }
    
    if (!saved_threads.empty()) {
        setenv("MLX_NUM_THREADS", saved_threads.c_str(), 1);
    } else {
        unsetenv("MLX_NUM_THREADS");
    }
}

// Test MLX availability and device configuration
TEST_F(MLXOptimizationsTest, TestMLXConfiguration) {
    // This test should pass regardless of MLX availability
    MLXOptimizationConfig config;
    
    // Should not throw an exception
    EXPECT_NO_THROW(configure_mlx_for_device(config));
}

// Create mock MLX tensors for testing
#ifdef CCSM_WITH_MLX
// Helper to create a mock MLX array
mlx_array create_mock_mlx_array(const std::vector<int>& shape, mlx_dtype dtype) {
    mlx_array array;
    mlx_array_zeros(const_cast<int*>(shape.data()), static_cast<int>(shape.size()), dtype, &array);
    return array;
}

// Helper to fill an MLX array with test data
void fill_mock_mlx_array(mlx_array array, float value) {
    if (!array.ctx) {
        return;
    }
    
    // Get array size
    size_t size = mlx_array_size(array);
    
    // Get dtype
    mlx_dtype dtype;
    mlx_array_dtype(array, &dtype);
    
    // Fill with data based on dtype
    if (dtype == MLX_FLOAT32) {
        float* data = (float*)mlx_array_data_float32(array);
        for (size_t i = 0; i < size; ++i) {
            data[i] = value;
        }
    } else if (dtype == MLX_FLOAT16 || dtype == MLX_BFLOAT16) {
        // Convert to F32, fill, then convert back
        mlx_array temp;
        mlx_array_astype(array, MLX_FLOAT32, &temp);
        
        float* data = (float*)mlx_array_data_float32(temp);
        for (size_t i = 0; i < size; ++i) {
            data[i] = value;
        }
        
        // Convert back and copy to original
        mlx_array_copy(temp, array);
        mlx_array_free(temp);
    }
}

// Helper to compare two MLX arrays
bool mlx_arrays_equal(mlx_array a, mlx_array b, float tolerance = 1e-5f) {
    if (!a.ctx || !b.ctx) {
        return false;
    }
    
    // Check dimensions
    uint32_t a_ndim, b_ndim;
    mlx_array_ndim(a, &a_ndim);
    mlx_array_ndim(b, &b_ndim);
    
    if (a_ndim != b_ndim) {
        return false;
    }
    
    // Check shapes
    const int* a_shape = mlx_array_shape(a);
    const int* b_shape = mlx_array_shape(b);
    
    for (uint32_t i = 0; i < a_ndim; ++i) {
        if (a_shape[i] != b_shape[i]) {
            return false;
        }
    }
    
    // Get data types
    mlx_dtype a_dtype, b_dtype;
    mlx_array_dtype(a, &a_dtype);
    mlx_array_dtype(b, &b_dtype);
    
    // Convert arrays to F32 for comparison
    mlx_array a_f32 = a, b_f32 = b;
    bool converted_a = false, converted_b = false;
    
    if (a_dtype != MLX_FLOAT32) {
        mlx_array_astype(a, MLX_FLOAT32, &a_f32);
        converted_a = true;
    }
    
    if (b_dtype != MLX_FLOAT32) {
        mlx_array_astype(b, MLX_FLOAT32, &b_f32);
        converted_b = true;
    }
    
    // Compare data
    size_t size = mlx_array_size(a_f32);
    const float* a_data = mlx_array_data_float32(a_f32);
    const float* b_data = mlx_array_data_float32(b_f32);
    
    bool equal = true;
    for (size_t i = 0; i < size; ++i) {
        if (std::abs(a_data[i] - b_data[i]) > tolerance) {
            equal = false;
            break;
        }
    }
    
    // Clean up converted arrays
    if (converted_a) {
        mlx_array_free(a_f32);
    }
    
    if (converted_b) {
        mlx_array_free(b_f32);
    }
    
    return equal;
}
#endif

// Test precision conversion functions
TEST_F(MLXOptimizationsTest, TestPrecisionConversion) {
#ifdef CCSM_WITH_MLX
    // Create a stream
    mlx_stream stream = mlx_default_cpu_stream_new();
    
    // Create test arrays with different dtypes
    std::vector<int> shape = {2, 3};
    mlx_array array_f32 = create_mock_mlx_array(shape, MLX_FLOAT32);
    fill_mock_mlx_array(array_f32, 1.5f);
    
    // Test conversion to different precisions
    mlx_array array_bf16 = convert_precision(array_f32, MLXOptimizationConfig::ComputePrecision::BFLOAT16, stream);
    mlx_array array_f16 = convert_precision(array_f32, MLXOptimizationConfig::ComputePrecision::FLOAT16, stream);
    
    // Check dtypes
    mlx_dtype dtype_bf16, dtype_f16;
    mlx_array_dtype(array_bf16, &dtype_bf16);
    mlx_array_dtype(array_f16, &dtype_f16);
    
    EXPECT_EQ(dtype_bf16, MLX_BFLOAT16);
    EXPECT_EQ(dtype_f16, MLX_FLOAT16);
    
    // Test converting back to F32
    mlx_array array_f32_2 = convert_precision(array_bf16, MLXOptimizationConfig::ComputePrecision::FLOAT32, stream);
    
    mlx_dtype dtype_f32_2;
    mlx_array_dtype(array_f32_2, &dtype_f32_2);
    EXPECT_EQ(dtype_f32_2, MLX_FLOAT32);
    
    // Check that values are approximately equal (allowing for precision loss)
    EXPECT_TRUE(mlx_arrays_equal(array_f32, array_f32_2, 0.1f));
    
    // Clean up
    mlx_array_free(array_f32);
    mlx_array_free(array_bf16);
    mlx_array_free(array_f16);
    mlx_array_free(array_f32_2);
#else
    GTEST_SKIP() << "MLX not available, skipping test";
#endif
}

// Test memory optimization functions
TEST_F(MLXOptimizationsTest, TestMemoryOptimization) {
#ifdef CCSM_WITH_MLX
    // Create some test arrays
    std::vector<int> shape1 = {10, 20};
    std::vector<int> shape2 = {30, 40};
    std::vector<int> shape3 = {50, 60};
    
    mlx_array array1 = create_mock_mlx_array(shape1, MLX_FLOAT32);
    mlx_array array2 = create_mock_mlx_array(shape2, MLX_FLOAT32);
    mlx_array array3 = create_mock_mlx_array(shape3, MLX_FLOAT32);
    
    // Fill with test data
    fill_mock_mlx_array(array1, 1.0f);
    fill_mock_mlx_array(array2, 2.0f);
    fill_mock_mlx_array(array3, 3.0f);
    
    // Create a vector of arrays to optimize
    std::vector<mlx_array> arrays = {array1, array2, array3};
    
    // Test optimization at different levels
    for (auto level : {MLXOptimizationConfig::MemoryUsage::MINIMAL, 
                       MLXOptimizationConfig::MemoryUsage::BALANCED, 
                       MLXOptimizationConfig::MemoryUsage::PERFORMANCE}) {
        // Create copies of arrays to test
        std::vector<mlx_array> test_arrays;
        for (const auto& arr : arrays) {
            mlx_array copy;
            mlx_array_copy(arr, copy);
            test_arrays.push_back(copy);
        }
        
        // Optimize memory usage
        optimize_memory_usage(test_arrays, level);
        
        // Verify arrays are still valid and contain correct data
        EXPECT_EQ(test_arrays.size(), arrays.size());
        for (size_t i = 0; i < test_arrays.size(); ++i) {
            EXPECT_TRUE(test_arrays[i].ctx != nullptr);
            
            // In MINIMAL mode, arrays should be converted to BF16
            if (level == MLXOptimizationConfig::MemoryUsage::MINIMAL) {
                mlx_dtype dtype;
                mlx_array_dtype(test_arrays[i], &dtype);
                EXPECT_EQ(dtype, MLX_BFLOAT16);
            }
            
            // Check that data values are approximately preserved
            EXPECT_TRUE(mlx_arrays_equal(test_arrays[i], arrays[i], 0.1f));
        }
        
        // Clean up test arrays
        for (auto& arr : test_arrays) {
            mlx_array_free(arr);
        }
    }
    
    // Clean up original arrays
    for (auto& arr : arrays) {
        mlx_array_free(arr);
    }
#else
    GTEST_SKIP() << "MLX not available, skipping test";
#endif
}

// Test fast matrix multiplication
TEST_F(MLXOptimizationsTest, TestFastMatmul) {
#ifdef CCSM_WITH_MLX
    // Create a stream
    mlx_stream stream = mlx_default_cpu_stream_new();
    
    // Create test matrices
    std::vector<int> shape_a = {2, 3};
    std::vector<int> shape_b = {3, 4};
    
    mlx_array a = create_mock_mlx_array(shape_a, MLX_FLOAT32);
    mlx_array b = create_mock_mlx_array(shape_b, MLX_FLOAT32);
    
    // Fill with test data
    float* a_data = (float*)mlx_array_data_float32(a);
    float* b_data = (float*)mlx_array_data_float32(b);
    
    for (int i = 0; i < 2 * 3; ++i) {
        a_data[i] = i + 1.0f;
    }
    
    for (int i = 0; i < 3 * 4; ++i) {
        b_data[i] = i + 1.0f;
    }
    
    // Test optimized matrix multiplication
    mlx_array result = mlx_fast::matmul_optimized(a, b, false, stream);
    
    // Verify result dimensions
    uint32_t result_ndim;
    mlx_array_ndim(result, &result_ndim);
    EXPECT_EQ(result_ndim, 2);
    
    const int* result_shape = mlx_array_shape(result);
    EXPECT_EQ(result_shape[0], 2);
    EXPECT_EQ(result_shape[1], 4);
    
    // Test with transposed second matrix
    std::vector<int> shape_b_t = {4, 3};
    mlx_array b_t = create_mock_mlx_array(shape_b_t, MLX_FLOAT32);
    
    // Fill with test data (equivalent to transpose of b)
    float* b_t_data = (float*)mlx_array_data_float32(b_t);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 3; ++j) {
            b_t_data[i * 3 + j] = b_data[j * 4 + i];
        }
    }
    
    // Perform multiplication with transpose flag
    mlx_array result_t = mlx_fast::matmul_optimized(a, b_t, true, stream);
    
    // Verify results are the same
    EXPECT_TRUE(mlx_arrays_equal(result, result_t));
    
    // Clean up
    mlx_array_free(a);
    mlx_array_free(b);
    mlx_array_free(b_t);
    mlx_array_free(result);
    mlx_array_free(result_t);
#else
    GTEST_SKIP() << "MLX not available, skipping test";
#endif
}

// Test fused layer norm + linear operation
TEST_F(MLXOptimizationsTest, TestFusedLayerNormLinear) {
#ifdef CCSM_WITH_MLX
    // Create a stream
    mlx_stream stream = mlx_default_cpu_stream_new();
    
    // Create test tensors
    std::vector<int> x_shape = {2, 3, 4}; // [batch, seq_len, hidden_dim]
    std::vector<int> norm_weight_shape = {4}; // [hidden_dim]
    std::vector<int> norm_bias_shape = {4}; // [hidden_dim]
    std::vector<int> linear_weight_shape = {8, 4}; // [out_dim, hidden_dim]
    
    mlx_array x = create_mock_mlx_array(x_shape, MLX_FLOAT32);
    mlx_array norm_weight = create_mock_mlx_array(norm_weight_shape, MLX_FLOAT32);
    mlx_array norm_bias = create_mock_mlx_array(norm_bias_shape, MLX_FLOAT32);
    mlx_array linear_weight = create_mock_mlx_array(linear_weight_shape, MLX_FLOAT32);
    
    // Fill with test data
    fill_mock_mlx_array(x, 1.0f);
    fill_mock_mlx_array(norm_weight, 1.0f);
    fill_mock_mlx_array(norm_bias, 0.5f);
    fill_mock_mlx_array(linear_weight, 1.0f);
    
    // Perform fused layer norm + linear operation
    float eps = 1e-5f;
    mlx_array result = mlx_fast::fused_layer_norm_linear(
        x, norm_weight, norm_bias, linear_weight, eps, stream);
    
    // Verify result dimensions
    uint32_t result_ndim;
    mlx_array_ndim(result, &result_ndim);
    EXPECT_EQ(result_ndim, 3);
    
    const int* result_shape = mlx_array_shape(result);
    EXPECT_EQ(result_shape[0], 2); // batch
    EXPECT_EQ(result_shape[1], 3); // seq_len
    EXPECT_EQ(result_shape[2], 8); // out_dim
    
    // Clean up
    mlx_array_free(x);
    mlx_array_free(norm_weight);
    mlx_array_free(norm_bias);
    mlx_array_free(linear_weight);
    mlx_array_free(result);
#else
    GTEST_SKIP() << "MLX not available, skipping test";
#endif
}

// Test batch processor
TEST_F(MLXOptimizationsTest, TestBatchProcessor) {
#ifdef CCSM_WITH_MLX
    // Create a stream
    mlx_stream stream = mlx_default_cpu_stream_new();
    
    // Create a batch processor
    MLXBatchProcessor processor;
    
    // Create test data
    std::vector<int> shape = {2, 3};
    mlx_array a = create_mock_mlx_array(shape, MLX_FLOAT32);
    mlx_array b = create_mock_mlx_array(shape, MLX_FLOAT32);
    
    fill_mock_mlx_array(a, 1.0f);
    fill_mock_mlx_array(b, 2.0f);
    
    // Add operations to the batch
    processor.add_operation([a, b]() {
        mlx_array result;
        mlx_add(a, b, &result);
        return result;
    });
    
    processor.add_operation([a, b]() {
        mlx_array result;
        mlx_multiply(a, b, &result);
        return result;
    });
    
    processor.add_operation([a]() {
        mlx_array result;
        mlx_square(a, &result);
        return result;
    });
    
    // Execute the batch
    std::vector<mlx_array> results = processor.execute(stream);
    
    // Verify results
    EXPECT_EQ(results.size(), 3);
    
    // Check first result (add)
    const float* add_data = mlx_array_data_float32(results[0]);
    for (int i = 0; i < 2 * 3; ++i) {
        EXPECT_FLOAT_EQ(add_data[i], 3.0f); // 1.0 + 2.0
    }
    
    // Check second result (multiply)
    const float* mul_data = mlx_array_data_float32(results[1]);
    for (int i = 0; i < 2 * 3; ++i) {
        EXPECT_FLOAT_EQ(mul_data[i], 2.0f); // 1.0 * 2.0
    }
    
    // Check third result (square)
    const float* square_data = mlx_array_data_float32(results[2]);
    for (int i = 0; i < 2 * 3; ++i) {
        EXPECT_FLOAT_EQ(square_data[i], 1.0f); // 1.0^2
    }
    
    // Test clearing the processor
    processor.clear();
    
    // Add a new operation
    processor.add_operation([a]() {
        mlx_array result;
        mlx_exp(a, &result);
        return result;
    });
    
    // Execute again
    std::vector<mlx_array> results2 = processor.execute(stream);
    
    // Verify new results
    EXPECT_EQ(results2.size(), 1);
    
    // Clean up
    mlx_array_free(a);
    mlx_array_free(b);
    
    for (auto& result : results) {
        mlx_array_free(result);
    }
    
    for (auto& result : results2) {
        mlx_array_free(result);
    }
#else
    GTEST_SKIP() << "MLX not available, skipping test";
#endif
}

// Test memory-efficient tensor operations
TEST_F(MLXOptimizationsTest, TestMemoryEfficientOperations) {
#ifdef CCSM_WITH_MLX
    // Create a stream
    mlx_stream stream = mlx_default_cpu_stream_new();
    
    // Create test tensors
    std::vector<int> shape = {2, 3};
    mlx_array a = create_mock_mlx_array(shape, MLX_FLOAT32);
    mlx_array b = create_mock_mlx_array(shape, MLX_FLOAT32);
    
    // Fill with test data
    float* a_data = (float*)mlx_array_data_float32(a);
    float* b_data = (float*)mlx_array_data_float32(b);
    
    for (int i = 0; i < 2 * 3; ++i) {
        a_data[i] = i + 1.0f;
        b_data[i] = 0.5f;
    }
    
    // Test in-place addition
    mlx_array a_copy;
    mlx_array_copy(a, a_copy);
    
    mlx_array result_add = mlx_memory::add_inplace(a_copy, b, stream);
    
    // Verify result (should be a + b, and a_copy should be updated in place)
    EXPECT_EQ(result_add.ctx, a_copy.ctx);
    
    const float* result_add_data = mlx_array_data_float32(result_add);
    for (int i = 0; i < 2 * 3; ++i) {
        EXPECT_FLOAT_EQ(result_add_data[i], (i + 1.0f) + 0.5f);
    }
    
    // Test in-place multiplication
    mlx_array a_copy2;
    mlx_array_copy(a, a_copy2);
    
    mlx_array result_mul = mlx_memory::multiply_inplace(a_copy2, b, stream);
    
    // Verify result (should be a * b, and a_copy2 should be updated in place)
    EXPECT_EQ(result_mul.ctx, a_copy2.ctx);
    
    const float* result_mul_data = mlx_array_data_float32(result_mul);
    for (int i = 0; i < 2 * 3; ++i) {
        EXPECT_FLOAT_EQ(result_mul_data[i], (i + 1.0f) * 0.5f);
    }
    
    // Clean up
    mlx_array_free(a);
    mlx_array_free(b);
    mlx_array_free(a_copy);
    mlx_array_free(a_copy2);
#else
    GTEST_SKIP() << "MLX not available, skipping test";
#endif
}

// Test tensor pool
TEST_F(MLXOptimizationsTest, TestTensorPool) {
#ifdef CCSM_WITH_MLX
    // Create a tensor pool with a small max size to test recycling
    mlx_memory::TensorPool pool(3);
    
    // Create different shapes to test
    std::vector<int> shape1 = {2, 3};
    std::vector<int> shape2 = {4, 5};
    std::vector<int> shape3 = {6, 7};
    std::vector<int> shape4 = {8, 9};
    
    // Get tensors from the pool
    mlx_array t1 = pool.get(shape1, MLX_FLOAT32);
    mlx_array t2 = pool.get(shape2, MLX_FLOAT32);
    mlx_array t3 = pool.get(shape3, MLX_FLOAT32);
    
    // Verify shapes
    uint32_t ndim1, ndim2, ndim3;
    mlx_array_ndim(t1, &ndim1);
    mlx_array_ndim(t2, &ndim2);
    mlx_array_ndim(t3, &ndim3);
    
    EXPECT_EQ(ndim1, 2);
    EXPECT_EQ(ndim2, 2);
    EXPECT_EQ(ndim3, 2);
    
    const int* t1_shape = mlx_array_shape(t1);
    const int* t2_shape = mlx_array_shape(t2);
    const int* t3_shape = mlx_array_shape(t3);
    
    EXPECT_EQ(t1_shape[0], 2);
    EXPECT_EQ(t1_shape[1], 3);
    EXPECT_EQ(t2_shape[0], 4);
    EXPECT_EQ(t2_shape[1], 5);
    EXPECT_EQ(t3_shape[0], 6);
    EXPECT_EQ(t3_shape[1], 7);
    
    // Recycle tensors
    pool.recycle(t1);
    pool.recycle(t2);
    pool.recycle(t3);
    
    // Get a tensor with a shape that matches one in the pool
    mlx_array t1_reused = pool.get(shape1, MLX_FLOAT32);
    
    // Should have reused the existing tensor
    uint32_t ndim_reused;
    mlx_array_ndim(t1_reused, &ndim_reused);
    EXPECT_EQ(ndim_reused, 2);
    
    const int* t1_reused_shape = mlx_array_shape(t1_reused);
    EXPECT_EQ(t1_reused_shape[0], 2);
    EXPECT_EQ(t1_reused_shape[1], 3);
    
    // Get another tensor with a new shape
    mlx_array t4 = pool.get(shape4, MLX_FLOAT32);
    
    // Since pool max size is 3, getting a 4th tensor should have removed the oldest one
    
    // Get tensors with shapes matching those in the pool
    mlx_array t2_reused = pool.get(shape2, MLX_FLOAT32);
    mlx_array t3_reused = pool.get(shape3, MLX_FLOAT32);
    
    // Should have reused existing tensors
    uint32_t ndim2_reused, ndim3_reused;
    mlx_array_ndim(t2_reused, &ndim2_reused);
    mlx_array_ndim(t3_reused, &ndim3_reused);
    
    EXPECT_EQ(ndim2_reused, 2);
    EXPECT_EQ(ndim3_reused, 2);
    
    const int* t2_reused_shape = mlx_array_shape(t2_reused);
    const int* t3_reused_shape = mlx_array_shape(t3_reused);
    
    EXPECT_EQ(t2_reused_shape[0], 4);
    EXPECT_EQ(t2_reused_shape[1], 5);
    EXPECT_EQ(t3_reused_shape[0], 6);
    EXPECT_EQ(t3_reused_shape[1], 7);
    
    // Testing shape matching - get a tensor with the same dimensions but different values
    std::vector<int> shape5 = {6, 7}; // Same shape as shape3
    mlx_array t5 = pool.get(shape5, MLX_FLOAT32);
    
    // Should have a new tensor since all matching ones are already in use
    EXPECT_NE(t5.ctx, t3_reused.ctx);
    
    // Clean up
    mlx_array_free(t1_reused);
    mlx_array_free(t2_reused);
    mlx_array_free(t3_reused);
    mlx_array_free(t4);
    mlx_array_free(t5);
    
    // Test clear function
    mlx_array t6 = pool.get(shape1, MLX_FLOAT32);
    pool.recycle(t6);
    
    // Clear the pool
    pool.clear();
    
    // Get a tensor again - should be a new one
    mlx_array t7 = pool.get(shape1, MLX_FLOAT32);
    
    // Clean up
    mlx_array_free(t7);
#else
    GTEST_SKIP() << "MLX not available, skipping test";
#endif
}

// Test fused attention
TEST_F(MLXOptimizationsTest, TestFusedAttention) {
#ifdef CCSM_WITH_MLX
    // Create a stream
    mlx_stream stream = mlx_default_cpu_stream_new();
    
    // Create test tensors for attention
    std::vector<int> x_shape = {2, 3, 32}; // [batch, seq_len, d_model]
    std::vector<int> wq_shape = {32, 32}; // [d_model, d_model]
    std::vector<int> wk_shape = {32, 32}; // [d_model, d_model]
    std::vector<int> wv_shape = {32, 32}; // [d_model, d_model]
    std::vector<int> wo_shape = {32, 32}; // [d_model, d_model]
    
    mlx_array x = create_mock_mlx_array(x_shape, MLX_FLOAT32);
    mlx_array wq = create_mock_mlx_array(wq_shape, MLX_FLOAT32);
    mlx_array wk = create_mock_mlx_array(wk_shape, MLX_FLOAT32);
    mlx_array wv = create_mock_mlx_array(wv_shape, MLX_FLOAT32);
    mlx_array wo = create_mock_mlx_array(wo_shape, MLX_FLOAT32);
    
    // Fill with test data
    fill_mock_mlx_array(x, 1.0f);
    fill_mock_mlx_array(wq, 0.1f);
    fill_mock_mlx_array(wk, 0.1f);
    fill_mock_mlx_array(wv, 0.1f);
    fill_mock_mlx_array(wo, 0.1f);
    
    // Create positions for RoPE
    std::vector<int> positions = {0, 1, 2};
    
    // Test fused attention
    int n_heads = 4;
    int n_kv_heads = 4;
    float rope_theta = 10000.0f;
    
    mlx_array result = mlx_fast::fused_attention(
        x, wq, wk, wv, wo, positions, rope_theta, n_heads, n_kv_heads, stream);
    
    // Since this is a stub, it will return an empty array, but we can at least make sure it doesn't crash
    
    // Clean up
    mlx_array_free(x);
    mlx_array_free(wq);
    mlx_array_free(wk);
    mlx_array_free(wv);
    mlx_array_free(wo);
    if (result.ctx) {
        mlx_array_free(result);
    }
#else
    GTEST_SKIP() << "MLX not available, skipping test";
#endif
}

// Test fast RoPE implementation
TEST_F(MLXOptimizationsTest, TestFastRoPE) {
#ifdef CCSM_WITH_MLX
    // Create a stream
    mlx_stream stream = mlx_default_cpu_stream_new();
    
    // Create test tensor
    std::vector<int> x_shape = {2, 3, 4, 8}; // [batch, seq_len, heads, head_dim]
    mlx_array x = create_mock_mlx_array(x_shape, MLX_FLOAT32);
    
    // Fill with test data
    fill_mock_mlx_array(x, 1.0f);
    
    // Create positions
    std::vector<int> positions = {0, 1, 2};
    
    // Test fast RoPE
    float theta = 10000.0f;
    
    mlx_array result = mlx_fast::fast_rope(x, positions, theta, stream);
    
    // Since this is a stub, it will return an empty array, but we can at least make sure it doesn't crash
    
    // Clean up
    mlx_array_free(x);
    if (result.ctx) {
        mlx_array_free(result);
    }
#else
    GTEST_SKIP() << "MLX not available, skipping test";
#endif
}

} // namespace testing
} // namespace ccsm