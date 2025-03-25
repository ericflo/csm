#include <gtest/gtest.h>
#include <ccsm/mlx/mlx_optimizations.h>
#include <ccsm/utils.h>
#include <chrono>
#include <vector>
#include <random>
#include <algorithm>

namespace ccsm {

// Helper to measure execution time
template<typename Func>
double measure_time_ms(Func func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

#ifdef CCSM_WITH_MLX
// Test MLXOptimizationConfig loading from environment
TEST(MLXOptimizationsTest, ConfigFromEnv) {
    // Set environment variables
    setenv("MLX_COMPUTE_PRECISION", "float16", 1);
    setenv("MLX_MEMORY_USAGE", "minimal", 1);
    setenv("MLX_NUM_THREADS", "4", 1);
    
    // Get config
    MLXOptimizationConfig config = MLXOptimizationConfig::from_env();
    
    // Check values
    EXPECT_EQ(config.compute_precision, MLXOptimizationConfig::ComputePrecision::FLOAT16);
    EXPECT_EQ(config.memory_usage, MLXOptimizationConfig::MemoryUsage::MINIMAL);
    EXPECT_EQ(config.num_compute_threads, 4);
    
    // Reset environment
    unsetenv("MLX_COMPUTE_PRECISION");
    unsetenv("MLX_MEMORY_USAGE");
    unsetenv("MLX_NUM_THREADS");
}

// Test MLX precision conversion
TEST(MLXOptimizationsTest, PrecisionConversion) {
    mlx_device device;
    mlx_default_device(&device);
    
    mlx_stream stream = mlx_default_cpu_stream_new();
    
    // Create a test array in FLOAT32
    std::vector<int> shape = {2, 3};
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    
    mlx_array array;
    mlx_array_from_data(data.data(), shape.data(), shape.size(), MLX_FLOAT32, &array);
    
    // Convert to BFLOAT16
    mlx_array bf16_array = convert_precision(array, MLXOptimizationConfig::ComputePrecision::BFLOAT16, stream);
    
    // Check dtype
    mlx_dtype dtype;
    mlx_array_dtype(bf16_array, &dtype);
    EXPECT_EQ(dtype, MLX_BFLOAT16);
    
    // Convert back to FLOAT32 for verification
    mlx_array back_to_f32;
    mlx_array_astype(bf16_array, MLX_FLOAT32, &back_to_f32);
    
    // Verify data is approximately the same (allowing for precision loss)
    float* result_data = (float*)mlx_array_data_float32(back_to_f32);
    for (int i = 0; i < 6; i++) {
        EXPECT_NEAR(result_data[i], data[i], 0.001f);
    }
    
    // Free arrays
    mlx_array_free(array);
    mlx_array_free(bf16_array);
    mlx_array_free(back_to_f32);
    mlx_stream_free(stream);
}

// Test memory optimization
TEST(MLXOptimizationsTest, MemoryOptimization) {
    mlx_stream stream = mlx_default_cpu_stream_new();
    
    // Create test arrays
    std::vector<int> shape = {10, 20};
    
    mlx_array array1, array2, array3;
    mlx_array_zeros(shape.data(), shape.size(), MLX_FLOAT32, &array1);
    mlx_array_zeros(shape.data(), shape.size(), MLX_FLOAT32, &array2);
    mlx_array_zeros(shape.data(), shape.size(), MLX_FLOAT32, &array3);
    
    // Test minimal memory usage
    std::vector<mlx_array> arrays = {array1, array2, array3};
    optimize_memory_usage(arrays, MLXOptimizationConfig::MemoryUsage::MINIMAL);
    
    // Verify arrays are still valid
    for (const auto& array : arrays) {
        uint32_t ndim;
        mlx_array_ndim(array, &ndim);
        EXPECT_EQ(ndim, 2);
        
        const int* array_shape = mlx_array_shape(array);
        EXPECT_EQ(array_shape[0], 10);
        EXPECT_EQ(array_shape[1], 20);
        
        mlx_dtype dtype;
        mlx_array_dtype(array, &dtype);
        EXPECT_EQ(dtype, MLX_BFLOAT16); // Should be converted to BF16 in minimal mode
    }
    
    // Free arrays
    for (auto& array : arrays) {
        mlx_array_free(array);
    }
    
    mlx_stream_free(stream);
}

// Test optimized matrix multiplication
TEST(MLXOptimizationsTest, MatmulOptimized) {
    mlx_stream stream = mlx_default_cpu_stream_new();
    
    // Create test matrices
    std::vector<int> shape_a = {2, 3};
    std::vector<int> shape_b = {3, 4};
    
    std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> data_b = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    
    mlx_array a, b;
    mlx_array_from_data(data_a.data(), shape_a.data(), shape_a.size(), MLX_FLOAT32, &a);
    mlx_array_from_data(data_b.data(), shape_b.data(), shape_b.size(), MLX_FLOAT32, &b);
    
    // Standard matmul
    mlx_array standard_result;
    mlx_matmul(a, b, &standard_result);
    
    // Optimized matmul
    mlx_array optimized_result = mlx_fast::matmul_optimized(a, b, false, stream);
    
    // Verify results are the same
    float* standard_data = (float*)mlx_array_data_float32(standard_result);
    float* optimized_data = (float*)mlx_array_data_float32(optimized_result);
    
    uint32_t result_size;
    mlx_array_size(standard_result, &result_size);
    
    for (uint32_t i = 0; i < result_size; i++) {
        EXPECT_NEAR(standard_data[i], optimized_data[i], 0.001f);
    }
    
    // Performance benchmark
    int num_runs = 10;
    
    double standard_time = measure_time_ms([&]() {
        for (int i = 0; i < num_runs; i++) {
            mlx_array result;
            mlx_matmul(a, b, &result);
            mlx_array_free(result);
        }
    });
    
    double optimized_time = measure_time_ms([&]() {
        for (int i = 0; i < num_runs; i++) {
            mlx_array result = mlx_fast::matmul_optimized(a, b, false, stream);
            mlx_array_free(result);
        }
    });
    
    CCSM_INFO("MatMul Benchmark:");
    CCSM_INFO("  Standard Matmul: ", standard_time / num_runs, " ms (avg of ", num_runs, " runs)");
    CCSM_INFO("  Optimized Matmul: ", optimized_time / num_runs, " ms (avg of ", num_runs, " runs)");
    CCSM_INFO("  Speedup: ", standard_time / optimized_time, "x");
    
    // Free arrays
    mlx_array_free(a);
    mlx_array_free(b);
    mlx_array_free(standard_result);
    mlx_array_free(optimized_result);
    
    mlx_stream_free(stream);
}

// Test fused layer norm + linear
TEST(MLXOptimizationsTest, FusedLayerNormLinear) {
    mlx_stream stream = mlx_default_cpu_stream_new();
    
    // Create test input
    std::vector<int> input_shape = {2, 5, 4}; // [batch, seq_len, hidden_size]
    std::vector<float> input_data(2 * 5 * 4);
    std::generate(input_data.begin(), input_data.end(), 
                 [&]() { return static_cast<float>(rand()) / RAND_MAX; });
    
    mlx_array input;
    mlx_array_from_data(input_data.data(), input_shape.data(), input_shape.size(), MLX_FLOAT32, &input);
    
    // Create layer norm weights and bias
    std::vector<int> norm_weight_shape = {4}; // [hidden_size]
    std::vector<float> norm_weight_data = {0.5f, 0.5f, 0.5f, 0.5f};
    std::vector<float> norm_bias_data = {0.1f, 0.1f, 0.1f, 0.1f};
    
    mlx_array norm_weight, norm_bias;
    mlx_array_from_data(norm_weight_data.data(), norm_weight_shape.data(), 
                        norm_weight_shape.size(), MLX_FLOAT32, &norm_weight);
    mlx_array_from_data(norm_bias_data.data(), norm_weight_shape.data(), 
                        norm_weight_shape.size(), MLX_FLOAT32, &norm_bias);
    
    // Create linear weights
    std::vector<int> linear_weight_shape = {6, 4}; // [out_features, in_features]
    std::vector<float> linear_weight_data(6 * 4);
    std::generate(linear_weight_data.begin(), linear_weight_data.end(), 
                 [&]() { return static_cast<float>(rand()) / RAND_MAX; });
    
    mlx_array linear_weight;
    mlx_array_from_data(linear_weight_data.data(), linear_weight_shape.data(), 
                        linear_weight_shape.size(), MLX_FLOAT32, &linear_weight);
    
    // Standard approach: layer norm followed by linear
    mlx_array normalized;
    mlx_layer_norm(input, norm_weight, norm_bias, 1e-5f, &normalized);
    
    // For the linear part, reshape to 2D
    std::vector<int> flat_shape = {2 * 5, 4};
    mlx_array normalized_flat;
    mlx_array_reshape(normalized, flat_shape.data(), flat_shape.size(), &normalized_flat);
    
    mlx_array standard_result_flat;
    mlx_matmul(normalized_flat, linear_weight, &standard_result_flat);
    
    std::vector<int> result_shape = {2, 5, 6};
    mlx_array standard_result;
    mlx_array_reshape(standard_result_flat, result_shape.data(), result_shape.size(), &standard_result);
    
    // Fused approach
    mlx_array fused_result = mlx_fast::fused_layer_norm_linear(
        input, norm_weight, norm_bias, linear_weight, 1e-5f, stream);
    
    // Verify results are approximately the same
    float* standard_data = (float*)mlx_array_data_float32(standard_result);
    float* fused_data = (float*)mlx_array_data_float32(fused_result);
    
    uint32_t result_size;
    mlx_array_size(standard_result, &result_size);
    
    for (uint32_t i = 0; i < result_size; i++) {
        EXPECT_NEAR(standard_data[i], fused_data[i], 0.001f);
    }
    
    // Performance benchmark
    int num_runs = 10;
    
    double standard_time = measure_time_ms([&]() {
        for (int i = 0; i < num_runs; i++) {
            mlx_array norm_result;
            mlx_layer_norm(input, norm_weight, norm_bias, 1e-5f, &norm_result);
            
            mlx_array norm_flat;
            mlx_array_reshape(norm_result, flat_shape.data(), flat_shape.size(), &norm_flat);
            
            mlx_array linear_result;
            mlx_matmul(norm_flat, linear_weight, &linear_result);
            
            mlx_array final_result;
            mlx_array_reshape(linear_result, result_shape.data(), result_shape.size(), &final_result);
            
            mlx_array_free(norm_result);
            mlx_array_free(norm_flat);
            mlx_array_free(linear_result);
            mlx_array_free(final_result);
        }
    });
    
    double fused_time = measure_time_ms([&]() {
        for (int i = 0; i < num_runs; i++) {
            mlx_array result = mlx_fast::fused_layer_norm_linear(
                input, norm_weight, norm_bias, linear_weight, 1e-5f, stream);
            mlx_array_free(result);
        }
    });
    
    CCSM_INFO("Layer Norm + Linear Benchmark:");
    CCSM_INFO("  Standard approach: ", standard_time / num_runs, " ms (avg of ", num_runs, " runs)");
    CCSM_INFO("  Fused approach: ", fused_time / num_runs, " ms (avg of ", num_runs, " runs)");
    CCSM_INFO("  Speedup: ", standard_time / fused_time, "x");
    
    // Free arrays
    mlx_array_free(input);
    mlx_array_free(norm_weight);
    mlx_array_free(norm_bias);
    mlx_array_free(linear_weight);
    mlx_array_free(normalized);
    mlx_array_free(normalized_flat);
    mlx_array_free(standard_result_flat);
    mlx_array_free(standard_result);
    mlx_array_free(fused_result);
    
    mlx_stream_free(stream);
}

// Test fast rotary position embeddings
TEST(MLXOptimizationsTest, FastRope) {
    mlx_stream stream = mlx_default_cpu_stream_new();
    
    // Create test input: [batch_size, seq_len, n_heads, head_dim]
    int batch_size = 2;
    int seq_len = 5;
    int n_heads = 4;
    int head_dim = 16; // Must be even for RoPE
    
    std::vector<int> input_shape = {batch_size, seq_len, n_heads, head_dim};
    std::vector<float> input_data(batch_size * seq_len * n_heads * head_dim);
    std::generate(input_data.begin(), input_data.end(), 
                 [&]() { return static_cast<float>(rand()) / RAND_MAX - 0.5f; });
    
    mlx_array input;
    mlx_array_from_data(input_data.data(), input_shape.data(), input_shape.size(), MLX_FLOAT32, &input);
    
    // Create position indices
    std::vector<int> positions = {0, 1, 2, 3, 4}; // Simple case: positions match sequence indices
    float rope_theta = 10000.0f;
    
    // Apply fast RoPE
    mlx_array result = mlx_fast::fast_rope(input, positions, rope_theta, stream);
    
    // Verify result has the same shape
    uint32_t result_ndim;
    mlx_array_ndim(result, &result_ndim);
    EXPECT_EQ(result_ndim, 4);
    
    const int* result_shape = mlx_array_shape(result);
    EXPECT_EQ(result_shape[0], batch_size);
    EXPECT_EQ(result_shape[1], seq_len);
    EXPECT_EQ(result_shape[2], n_heads);
    EXPECT_EQ(result_shape[3], head_dim);
    
    // Check if the output is different from the input
    // (If RoPE is applied, the output should not be identical to the input)
    float* input_data_ptr = (float*)mlx_array_data_float32(input);
    float* result_data_ptr = (float*)mlx_array_data_float32(result);
    
    bool has_difference = false;
    for (int i = 0; i < batch_size * seq_len * n_heads * head_dim; i++) {
        if (std::abs(input_data_ptr[i] - result_data_ptr[i]) > 1e-5f) {
            has_difference = true;
            break;
        }
    }
    
    EXPECT_TRUE(has_difference) << "RoPE doesn't seem to have changed the input tensor";
    
    // Benchmark RoPE implementation
    int num_runs = 10;
    
    double rope_time = measure_time_ms([&]() {
        for (int i = 0; i < num_runs; i++) {
            mlx_array result = mlx_fast::fast_rope(input, positions, rope_theta, stream);
            mlx_array_free(result);
        }
    });
    
    CCSM_INFO("Fast RoPE Benchmark:");
    CCSM_INFO("  Average time: ", rope_time / num_runs, " ms (avg of ", num_runs, " runs)");
    
    // Free arrays
    mlx_array_free(input);
    mlx_array_free(result);
    
    mlx_stream_free(stream);
}

// Test fused attention
TEST(MLXOptimizationsTest, FusedAttention) {
    mlx_stream stream = mlx_default_cpu_stream_new();
    
    // Create test input: [batch_size, seq_len, hidden_size]
    int batch_size = 2;
    int seq_len = 8;
    int hidden_size = 64;
    int n_heads = 4;
    int n_kv_heads = 4; // Regular attention
    int head_dim = hidden_size / n_heads;
    
    std::vector<int> input_shape = {batch_size, seq_len, hidden_size};
    std::vector<float> input_data(batch_size * seq_len * hidden_size);
    std::generate(input_data.begin(), input_data.end(), 
                 [&]() { return static_cast<float>(rand()) / RAND_MAX * 0.1f; });
    
    mlx_array input;
    mlx_array_from_data(input_data.data(), input_shape.data(), input_shape.size(), MLX_FLOAT32, &input);
    
    // Create projection weights
    std::vector<int> proj_shape = {hidden_size, hidden_size}; // [hidden_size, hidden_size]
    
    std::vector<float> wq_data(hidden_size * hidden_size);
    std::vector<float> wk_data(hidden_size * hidden_size);
    std::vector<float> wv_data(hidden_size * hidden_size);
    std::vector<float> wo_data(hidden_size * hidden_size);
    
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::normal_distribution<float> dist(0.0f, 0.02f);
    
    std::generate(wq_data.begin(), wq_data.end(), [&]() { return dist(gen); });
    std::generate(wk_data.begin(), wk_data.end(), [&]() { return dist(gen); });
    std::generate(wv_data.begin(), wv_data.end(), [&]() { return dist(gen); });
    std::generate(wo_data.begin(), wo_data.end(), [&]() { return dist(gen); });
    
    mlx_array wq, wk, wv, wo;
    mlx_array_from_data(wq_data.data(), proj_shape.data(), proj_shape.size(), MLX_FLOAT32, &wq);
    mlx_array_from_data(wk_data.data(), proj_shape.data(), proj_shape.size(), MLX_FLOAT32, &wk);
    mlx_array_from_data(wv_data.data(), proj_shape.data(), proj_shape.size(), MLX_FLOAT32, &wv);
    mlx_array_from_data(wo_data.data(), proj_shape.data(), proj_shape.size(), MLX_FLOAT32, &wo);
    
    // Create position indices
    std::vector<int> positions;
    for (int i = 0; i < seq_len; i++) {
        positions.push_back(i);
    }
    float rope_theta = 10000.0f;
    
    // Apply fused attention
    mlx_array result = mlx_fast::fused_attention(
        input, wq, wk, wv, wo, positions, rope_theta, n_heads, n_kv_heads, stream);
    
    // Verify result has the same shape as input
    uint32_t result_ndim;
    mlx_array_ndim(result, &result_ndim);
    EXPECT_EQ(result_ndim, 3);
    
    const int* result_shape = mlx_array_shape(result);
    EXPECT_EQ(result_shape[0], batch_size);
    EXPECT_EQ(result_shape[1], seq_len);
    EXPECT_EQ(result_shape[2], hidden_size);
    
    // Benchmark standard vs. fused approach
    // For a full implementation we would need to measure standard attention too,
    // but for simplicity we'll just measure the fused version
    int num_runs = 5;
    
    double fused_time = measure_time_ms([&]() {
        for (int i = 0; i < num_runs; i++) {
            mlx_array result = mlx_fast::fused_attention(
                input, wq, wk, wv, wo, positions, rope_theta, n_heads, n_kv_heads, stream);
            mlx_array_free(result);
        }
    });
    
    CCSM_INFO("Fused Attention Benchmark:");
    CCSM_INFO("  Average time: ", fused_time / num_runs, " ms (avg of ", num_runs, " runs)");
    
    // Test with grouped-query attention (MQA, n_kv_heads < n_heads)
    n_kv_heads = 2; // Half the number of heads
    
    mlx_array mqa_result = mlx_fast::fused_attention(
        input, wq, wk, wv, wo, positions, rope_theta, n_heads, n_kv_heads, stream);
    
    // Verify MQA result has the same shape as input
    uint32_t mqa_ndim;
    mlx_array_ndim(mqa_result, &mqa_ndim);
    EXPECT_EQ(mqa_ndim, 3);
    
    const int* mqa_shape = mlx_array_shape(mqa_result);
    EXPECT_EQ(mqa_shape[0], batch_size);
    EXPECT_EQ(mqa_shape[1], seq_len);
    EXPECT_EQ(mqa_shape[2], hidden_size);
    
    // Free arrays
    mlx_array_free(input);
    mlx_array_free(wq);
    mlx_array_free(wk);
    mlx_array_free(wv);
    mlx_array_free(wo);
    mlx_array_free(result);
    mlx_array_free(mqa_result);
    
    mlx_stream_free(stream);
}

// Test MLX batch processor
TEST(MLXOptimizationsTest, BatchProcessor) {
    mlx_stream stream = mlx_default_cpu_stream_new();
    
    // Create test arrays
    std::vector<int> shape_a = {10, 20};
    std::vector<int> shape_b = {20, 30};
    std::vector<int> shape_c = {10, 30};
    
    mlx_array a, b, c;
    mlx_array_zeros(shape_a.data(), shape_a.size(), MLX_FLOAT32, &a);
    mlx_array_zeros(shape_b.data(), shape_b.size(), MLX_FLOAT32, &b);
    mlx_array_zeros(shape_c.data(), shape_c.size(), MLX_FLOAT32, &c);
    
    // Create batch processor
    MLXBatchProcessor batch_processor;
    
    // Add operations
    batch_processor.add_operation([&]() {
        mlx_array result;
        mlx_matmul(a, b, &result);
        return result;
    });
    
    batch_processor.add_operation([&]() {
        mlx_array result;
        mlx_add(c, c, &result);
        return result;
    });
    
    // Execute batch
    auto results = batch_processor.execute(stream);
    
    // Verify we got two results
    EXPECT_EQ(results.size(), 2);
    
    // Verify the shapes
    uint32_t result1_ndim, result2_ndim;
    mlx_array_ndim(results[0], &result1_ndim);
    mlx_array_ndim(results[1], &result2_ndim);
    
    EXPECT_EQ(result1_ndim, 2);
    EXPECT_EQ(result2_ndim, 2);
    
    const int* result1_shape = mlx_array_shape(results[0]);
    const int* result2_shape = mlx_array_shape(results[1]);
    
    EXPECT_EQ(result1_shape[0], 10);
    EXPECT_EQ(result1_shape[1], 30);
    
    EXPECT_EQ(result2_shape[0], 10);
    EXPECT_EQ(result2_shape[1], 30);
    
    // Benchmark batch vs. sequential
    int num_runs = 10;
    
    double sequential_time = measure_time_ms([&]() {
        for (int i = 0; i < num_runs; i++) {
            mlx_array result1;
            mlx_matmul(a, b, &result1);
            
            mlx_array result2;
            mlx_add(c, c, &result2);
            
            mlx_array_free(result1);
            mlx_array_free(result2);
        }
    });
    
    double batch_time = measure_time_ms([&]() {
        for (int i = 0; i < num_runs; i++) {
            MLXBatchProcessor processor;
            
            processor.add_operation([&]() {
                mlx_array result;
                mlx_matmul(a, b, &result);
                return result;
            });
            
            processor.add_operation([&]() {
                mlx_array result;
                mlx_add(c, c, &result);
                return result;
            });
            
            auto batch_results = processor.execute(stream);
            
            for (auto& result : batch_results) {
                mlx_array_free(result);
            }
        }
    });
    
    CCSM_INFO("Batch Processing Benchmark:");
    CCSM_INFO("  Sequential time: ", sequential_time / num_runs, " ms (avg of ", num_runs, " runs)");
    CCSM_INFO("  Batch time: ", batch_time / num_runs, " ms (avg of ", num_runs, " runs)");
    CCSM_INFO("  Speedup: ", sequential_time / batch_time, "x");
    
    // Free arrays
    mlx_array_free(a);
    mlx_array_free(b);
    mlx_array_free(c);
    
    for (auto& result : results) {
        mlx_array_free(result);
    }
    
    mlx_stream_free(stream);
}

// Test in-place operations
TEST(MLXOptimizationsTest, InplaceOperations) {
    mlx_stream stream = mlx_default_cpu_stream_new();
    
    // Create test arrays
    std::vector<int> shape = {10, 20};
    
    mlx_array a, b;
    mlx_array_zeros(shape.data(), shape.size(), MLX_FLOAT32, &a);
    mlx_array_ones(shape.data(), shape.size(), MLX_FLOAT32, &b);
    
    // Make a copy of a for verification
    mlx_array a_copy;
    mlx_array_copy(a, &a_copy);
    
    // In-place addition
    mlx_memory::add_inplace(a, b, stream);
    
    // Verify a has been modified to equal a+b
    float* a_data = (float*)mlx_array_data_float32(a);
    float* a_copy_data = (float*)mlx_array_data_float32(a_copy);
    float* b_data = (float*)mlx_array_data_float32(b);
    
    uint32_t size;
    mlx_array_size(a, &size);
    
    for (uint32_t i = 0; i < size; i++) {
        EXPECT_NEAR(a_data[i], a_copy_data[i] + b_data[i], 0.001f);
    }
    
    // Make a copy of a for verification
    mlx_array a_copy2;
    mlx_array_copy(a, &a_copy2);
    
    // In-place multiplication
    mlx_memory::multiply_inplace(a, b, stream);
    
    // Verify a has been modified to equal a*b
    for (uint32_t i = 0; i < size; i++) {
        EXPECT_NEAR(a_data[i], a_copy2_data[i] * b_data[i], 0.001f);
    }
    
    // Benchmark standard vs. in-place operations
    int num_runs = 10;
    
    // Create new arrays for benchmark
    mlx_array x, y, x_copy;
    mlx_array_zeros(shape.data(), shape.size(), MLX_FLOAT32, &x);
    mlx_array_ones(shape.data(), shape.size(), MLX_FLOAT32, &y);
    mlx_array_copy(x, &x_copy);
    
    double standard_add_time = measure_time_ms([&]() {
        for (int i = 0; i < num_runs; i++) {
            mlx_array result;
            mlx_add(x, y, &result);
            
            // Free original and replace
            mlx_array_free(x);
            x = result;
        }
    });
    
    // Reset x
    mlx_array_free(x);
    mlx_array_copy(x_copy, &x);
    
    double inplace_add_time = measure_time_ms([&]() {
        for (int i = 0; i < num_runs; i++) {
            mlx_memory::add_inplace(x, y, stream);
        }
    });
    
    CCSM_INFO("In-place Addition Benchmark:");
    CCSM_INFO("  Standard add: ", standard_add_time / num_runs, " ms (avg of ", num_runs, " runs)");
    CCSM_INFO("  In-place add: ", inplace_add_time / num_runs, " ms (avg of ", num_runs, " runs)");
    CCSM_INFO("  Speedup: ", standard_add_time / inplace_add_time, "x");
    
    // Free arrays
    mlx_array_free(a);
    mlx_array_free(b);
    mlx_array_free(a_copy);
    mlx_array_free(a_copy2);
    mlx_array_free(x);
    mlx_array_free(y);
    mlx_array_free(x_copy);
    
    mlx_stream_free(stream);
}

// Test tensor pool
TEST(MLXOptimizationsTest, TensorPool) {
    // Create tensor pool
    mlx_memory::TensorPool pool(10);
    
    // Create some tensors with different shapes and dtypes
    std::vector<int> shape1 = {10, 20};
    std::vector<int> shape2 = {30, 40};
    
    // Get tensors from pool
    mlx_array tensor1 = pool.get(shape1, MLX_FLOAT32);
    mlx_array tensor2 = pool.get(shape2, MLX_BFLOAT16);
    
    // Verify tensor shapes and dtypes
    uint32_t ndim1, ndim2;
    mlx_array_ndim(tensor1, &ndim1);
    mlx_array_ndim(tensor2, &ndim2);
    
    EXPECT_EQ(ndim1, 2);
    EXPECT_EQ(ndim2, 2);
    
    const int* shape1_result = mlx_array_shape(tensor1);
    const int* shape2_result = mlx_array_shape(tensor2);
    
    EXPECT_EQ(shape1_result[0], 10);
    EXPECT_EQ(shape1_result[1], 20);
    EXPECT_EQ(shape2_result[0], 30);
    EXPECT_EQ(shape2_result[1], 40);
    
    mlx_dtype dtype1, dtype2;
    mlx_array_dtype(tensor1, &dtype1);
    mlx_array_dtype(tensor2, &dtype2);
    
    EXPECT_EQ(dtype1, MLX_FLOAT32);
    EXPECT_EQ(dtype2, MLX_BFLOAT16);
    
    // Recycle tensors
    pool.recycle(tensor1);
    pool.recycle(tensor2);
    
    // Get tensor with same shape and dtype as tensor1
    mlx_array tensor3 = pool.get(shape1, MLX_FLOAT32);
    
    // Verify it's reusing the recycled tensor
    uint32_t ndim3;
    mlx_array_ndim(tensor3, &ndim3);
    
    EXPECT_EQ(ndim3, 2);
    
    const int* shape3_result = mlx_array_shape(tensor3);
    EXPECT_EQ(shape3_result[0], 10);
    EXPECT_EQ(shape3_result[1], 20);
    
    mlx_dtype dtype3;
    mlx_array_dtype(tensor3, &dtype3);
    EXPECT_EQ(dtype3, MLX_FLOAT32);
    
    // Benchmark tensor allocation with and without pool
    int num_runs = 100;
    
    double standard_alloc_time = measure_time_ms([&]() {
        for (int i = 0; i < num_runs; i++) {
            mlx_array tensor;
            mlx_array_zeros(shape1.data(), shape1.size(), MLX_FLOAT32, &tensor);
            mlx_array_free(tensor);
        }
    });
    
    double pool_alloc_time = measure_time_ms([&]() {
        mlx_memory::TensorPool benchmark_pool(100);
        
        for (int i = 0; i < num_runs; i++) {
            mlx_array tensor = benchmark_pool.get(shape1, MLX_FLOAT32);
            benchmark_pool.recycle(tensor);
        }
    });
    
    CCSM_INFO("Tensor Pool Benchmark:");
    CCSM_INFO("  Standard allocation: ", standard_alloc_time / num_runs, " ms (avg of ", num_runs, " runs)");
    CCSM_INFO("  Pool allocation: ", pool_alloc_time / num_runs, " ms (avg of ", num_runs, " runs)");
    CCSM_INFO("  Speedup: ", standard_alloc_time / pool_alloc_time, "x");
    
    // Cleanup
    mlx_array_free(tensor3);
    pool.clear();
}

#endif // CCSM_WITH_MLX

// Test that runs even without MLX
TEST(MLXOptimizationsTest, BasicTest) {
    // Simple test that always passes
    EXPECT_TRUE(true);
}

} // namespace ccsm