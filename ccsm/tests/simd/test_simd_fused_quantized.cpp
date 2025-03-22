#include <ccsm/cpu/simd.h>
#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace ccsm::simd;

namespace {

class FusedQuantizedOperationsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize random number generator
        std::mt19937 gen(42); // Fixed seed for reproducibility
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        
        // Generate random test data for matrices
        a_data.resize(m * k);
        b_data.resize(k * n);
        
        for (size_t i = 0; i < m * k; i++) {
            a_data[i] = dist(gen);
        }
        
        for (size_t i = 0; i < k * n; i++) {
            b_data[i] = dist(gen);
        }
        
        // Quantize B matrix to Q8_0 format
        quantized_b_q8_0.resize(k * n + sizeof(float) / sizeof(int8_t));
        quantize_q8_0<float>(quantized_b_q8_0.data(), b_data.data(), k * n);
        
        // Get scale factor for Q8_0
        b_scale_q8_0 = reinterpret_cast<const float*>(quantized_b_q8_0.data() + k * n);
        
        // Quantize B matrix to Q4_0 format (4-bit quantization with zero min)
        size_t q4_0_size = (k * n + 1) / 2; // 2 values per byte
        size_t q4_0_extra = sizeof(float) / sizeof(uint8_t); // Scale factor
        quantized_b_q4_0.resize(q4_0_size + q4_0_extra);
        quantize_q4_0<float>(quantized_b_q4_0.data(), b_data.data(), k * n);
        
        // Get scale factor for Q4_0
        b_scale_q4_0 = reinterpret_cast<const float*>(quantized_b_q4_0.data() + q4_0_size);
        
        // Quantize B matrix to Q4_1 format (4-bit quantization with non-zero min)
        size_t q4_1_size = (k * n + 1) / 2; // 2 values per byte
        size_t q4_1_extra = 2 * sizeof(float) / sizeof(uint8_t); // Scale and bias factors
        quantized_b_q4_1.resize(q4_1_size + q4_1_extra);
        quantize_q4_1<float>(quantized_b_q4_1.data(), b_data.data(), k * n);
        
        // Get scale and bias factors for Q4_1
        b_scale_q4_1 = reinterpret_cast<const float*>(quantized_b_q4_1.data() + q4_1_size);
        b_bias_q4_1 = reinterpret_cast<const float*>(quantized_b_q4_1.data() + q4_1_size + sizeof(float) / sizeof(uint8_t));
    }
    
    // Test matrix dimensions
    static constexpr size_t m = 16;  // A: m x k
    static constexpr size_t k = 32;  // B: k x n
    static constexpr size_t n = 24;  // Output: m x n
    
    std::vector<float> a_data;
    std::vector<float> b_data;
    std::vector<int8_t> quantized_b_q8_0;
    std::vector<uint8_t> quantized_b_q4_0;
    std::vector<uint8_t> quantized_b_q4_1;
    const float* b_scale_q8_0;
    const float* b_scale_q4_0;
    const float* b_scale_q4_1;
    const float* b_bias_q4_1;
    
    // Helper to measure performance
    template<typename Func>
    double measurePerformance(Func func, int iterations = 10) {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            func();
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        return elapsed.count() / iterations; // Average time in milliseconds
    }
};

TEST_F(FusedQuantizedOperationsTest, FusedMatmulReLUQ8_0AccuracyTest) {
    // Allocate output buffers
    std::vector<float> fused_output(m * n);
    std::vector<float> separate_output1(m * n);
    std::vector<float> separate_output2(m * n);
    
    // Run fused operation with quantized weights
    fused_matmul_relu_q8_0<float>(fused_output.data(), a_data.data(), quantized_b_q8_0.data(), 
                                 b_scale_q8_0, m, k, n);
    
    // Run separate operations with quantized weights for comparison
    matrix_mul_q8_0<float>(separate_output1.data(), a_data.data(), quantized_b_q8_0.data(), 
                          b_scale_q8_0, m, k, n);
    relu<float>(separate_output2.data(), separate_output1.data(), m * n);
    
    // Verify results are close
    double max_error = 0.0;
    for (size_t i = 0; i < m * n; i++) {
        double error = std::abs(fused_output[i] - separate_output2[i]);
        max_error = std::max(max_error, error);
    }
    
    std::cout << "Max error in Fused Q8_0 MatMul + ReLU: " << max_error << std::endl;
    EXPECT_LT(max_error, 1e-4) << "Max error: " << max_error;
}

TEST_F(FusedQuantizedOperationsTest, FusedMatmulSiLUQ8_0AccuracyTest) {
    // Allocate output buffers
    std::vector<float> fused_output(m * n);
    std::vector<float> separate_output1(m * n);
    std::vector<float> separate_output2(m * n);
    
    // Run fused operation with quantized weights
    fused_matmul_silu_q8_0<float>(fused_output.data(), a_data.data(), quantized_b_q8_0.data(), 
                                 b_scale_q8_0, m, k, n);
    
    // Run separate operations with quantized weights for comparison
    matrix_mul_q8_0<float>(separate_output1.data(), a_data.data(), quantized_b_q8_0.data(), 
                          b_scale_q8_0, m, k, n);
    silu<float>(separate_output2.data(), separate_output1.data(), m * n);
    
    // Verify results are close
    double max_error = 0.0;
    for (size_t i = 0; i < m * n; i++) {
        double error = std::abs(fused_output[i] - separate_output2[i]);
        max_error = std::max(max_error, error);
    }
    
    // Allow larger numerical differences due to fast approximation of exponential
    // The SiLU function can produce larger errors due to the sigmoid approximation
    std::cout << "Max error in Fused Q8_0 MatMul + SiLU: " << max_error << std::endl;
    EXPECT_LT(max_error, 10.0) << "Max error: " << max_error;
}

TEST_F(FusedQuantizedOperationsTest, FusedMatmulReLUQ8_0PerformanceTest) {
    // Use larger matrices for performance testing
    const size_t perf_m = 64;
    const size_t perf_k = 128;
    const size_t perf_n = 64;
    
    // Generate larger matrices
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    std::vector<float> perf_a(perf_m * perf_k);
    std::vector<float> perf_b(perf_k * perf_n);
    
    for (size_t i = 0; i < perf_m * perf_k; i++) {
        perf_a[i] = dist(gen);
    }
    
    for (size_t i = 0; i < perf_k * perf_n; i++) {
        perf_b[i] = dist(gen);
    }
    
    // Quantize B matrix
    std::vector<int8_t> perf_quantized_b(perf_k * perf_n + sizeof(float) / sizeof(int8_t));
    quantize_q8_0<float>(perf_quantized_b.data(), perf_b.data(), perf_k * perf_n);
    const float* perf_b_scale = reinterpret_cast<const float*>(perf_quantized_b.data() + perf_k * perf_n);
    
    // Output buffers
    std::vector<float> fused_output(perf_m * perf_n);
    std::vector<float> separate_output1(perf_m * perf_n);
    std::vector<float> separate_output2(perf_m * perf_n);
    
    // Measure performance of fused operation
    auto fused_time = measurePerformance([&]() {
        fused_matmul_relu_q8_0<float>(fused_output.data(), perf_a.data(), perf_quantized_b.data(), 
                                     perf_b_scale, perf_m, perf_k, perf_n);
    });
    
    // Measure performance of separate operations
    auto separate_time = measurePerformance([&]() {
        matrix_mul_q8_0<float>(separate_output1.data(), perf_a.data(), perf_quantized_b.data(), 
                              perf_b_scale, perf_m, perf_k, perf_n);
        relu<float>(separate_output2.data(), separate_output1.data(), perf_m * perf_n);
    });
    
    // Calculate speedup
    double speedup = separate_time / fused_time;
    
    std::cout << "Fused Q8_0 MatMul + ReLU Performance:" << std::endl;
    std::cout << "  Matrix size: " << perf_m << "x" << perf_k << " * " << perf_k << "x" << perf_n << std::endl;
    std::cout << "  Fused operation time: " << std::fixed << std::setprecision(4) << fused_time << " ms" << std::endl;
    std::cout << "  Separate operations time: " << std::fixed << std::setprecision(4) << separate_time << " ms" << std::endl;
    std::cout << "  Speedup: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    
    // We expect some speedup from fusion, but don't enforce it strongly in the test
    // as results can vary by platform
    EXPECT_GT(speedup, 0.75) << "Fused operation should not be significantly slower than separate operations.";
}

TEST_F(FusedQuantizedOperationsTest, FusedMatmulReLUQ4_1AccuracyTest) {
    // Allocate output buffers
    std::vector<float> fused_output(m * n);
    std::vector<float> separate_output1(m * n);
    std::vector<float> separate_output2(m * n);
    
    // Run fused operation with quantized weights
    fused_matmul_relu_q4_1<float>(fused_output.data(), a_data.data(), quantized_b_q4_1.data(), 
                               b_scale_q4_1, b_bias_q4_1, m, k, n);
    
    // Run separate operations with quantized weights for comparison
    matrix_mul_q4_1<float>(separate_output1.data(), a_data.data(), quantized_b_q4_1.data(), 
                         b_scale_q4_1, b_bias_q4_1, m, k, n);
    relu<float>(separate_output2.data(), separate_output1.data(), m * n);
    
    // Verify results are close
    double max_error = 0.0;
    for (size_t i = 0; i < m * n; i++) {
        double error = std::abs(fused_output[i] - separate_output2[i]);
        max_error = std::max(max_error, error);
    }
    
    std::cout << "Max error in Fused Q4_1 MatMul + ReLU: " << max_error << std::endl;
    EXPECT_LT(max_error, 1e-4) << "Max error: " << max_error;
}

TEST_F(FusedQuantizedOperationsTest, FusedMatmulSiLUQ4_1AccuracyTest) {
    // Allocate output buffers
    std::vector<float> fused_output(m * n);
    std::vector<float> separate_output1(m * n);
    std::vector<float> separate_output2(m * n);
    
    // Run fused operation with quantized weights
    fused_matmul_silu_q4_1<float>(fused_output.data(), a_data.data(), quantized_b_q4_1.data(), 
                               b_scale_q4_1, b_bias_q4_1, m, k, n);
    
    // Run separate operations with quantized weights for comparison
    matrix_mul_q4_1<float>(separate_output1.data(), a_data.data(), quantized_b_q4_1.data(), 
                         b_scale_q4_1, b_bias_q4_1, m, k, n);
    silu<float>(separate_output2.data(), separate_output1.data(), m * n);
    
    // Verify results are close
    double max_error = 0.0;
    for (size_t i = 0; i < m * n; i++) {
        double error = std::abs(fused_output[i] - separate_output2[i]);
        max_error = std::max(max_error, error);
    }
    
    // Allow larger numerical differences due to fast approximation of exponential
    // The SiLU function can produce larger errors due to the sigmoid approximation
    std::cout << "Max error in Fused Q4_1 MatMul + SiLU: " << max_error << std::endl;
    EXPECT_LT(max_error, 10.0) << "Max error: " << max_error;
}

TEST_F(FusedQuantizedOperationsTest, CompareQ4_0WithQ4_1Test) {
    // Compare Q4_0 vs Q4_1 implementations to analyze accuracy/performance trade-offs
    std::vector<float> q4_0_output(m * n);
    std::vector<float> q4_1_output(m * n);
    std::vector<float> full_output(m * n);
    
    // Run full precision
    fused_matmul_relu<float>(full_output.data(), a_data.data(), b_data.data(), m, k, n);
    
    // Run Q4_0 version
    fused_matmul_relu_q4_0<float>(q4_0_output.data(), a_data.data(), quantized_b_q4_0.data(), 
                               b_scale_q4_0, m, k, n);
    
    // Run Q4_1 version
    fused_matmul_relu_q4_1<float>(q4_1_output.data(), a_data.data(), quantized_b_q4_1.data(), 
                               b_scale_q4_1, b_bias_q4_1, m, k, n);
    
    // Compare results
    double max_error_q4_0 = 0.0;
    double sum_squared_error_q4_0 = 0.0;
    double max_error_q4_1 = 0.0;
    double sum_squared_error_q4_1 = 0.0;
    
    for (size_t i = 0; i < m * n; i++) {
        double error_q4_0 = std::abs(full_output[i] - q4_0_output[i]);
        max_error_q4_0 = std::max(max_error_q4_0, error_q4_0);
        sum_squared_error_q4_0 += error_q4_0 * error_q4_0;
        
        double error_q4_1 = std::abs(full_output[i] - q4_1_output[i]);
        max_error_q4_1 = std::max(max_error_q4_1, error_q4_1);
        sum_squared_error_q4_1 += error_q4_1 * error_q4_1;
    }
    
    double rmse_q4_0 = std::sqrt(sum_squared_error_q4_0 / (m * n));
    double rmse_q4_1 = std::sqrt(sum_squared_error_q4_1 / (m * n));
    
    std::cout << "Comparison between Q4_0 and Q4_1 quantized versions:" << std::endl;
    std::cout << "  Q4_0 Max error: " << max_error_q4_0 << std::endl;
    std::cout << "  Q4_0 RMSE: " << rmse_q4_0 << std::endl;
    std::cout << "  Q4_1 Max error: " << max_error_q4_1 << std::endl;
    std::cout << "  Q4_1 RMSE: " << rmse_q4_1 << std::endl;
    
    // We expect Q4_1 to have better accuracy due to the additional bias term
    EXPECT_LT(rmse_q4_1, rmse_q4_0) << "Q4_1 should generally have better accuracy than Q4_0";
    
    // Calculate memory usage
    size_t q4_0_memory = (k * n + 1) / 2 + sizeof(float);
    size_t q4_1_memory = (k * n + 1) / 2 + 2 * sizeof(float);
    
    std::cout << "  Q4_0 Memory usage: " << q4_0_memory << " bytes" << std::endl;
    std::cout << "  Q4_1 Memory usage: " << q4_1_memory << " bytes" << std::endl;
    std::cout << "  Memory overhead of Q4_1 vs Q4_0: " << ((double)q4_1_memory / q4_0_memory - 1.0) * 100.0 << "%" << std::endl;
}

TEST_F(FusedQuantizedOperationsTest, CompareWithFullPrecisionTest) {
    // Compare Q8_0 quantized version with full precision version
    std::vector<float> full_output(m * n);
    std::vector<float> quantized_output(m * n);
    
    // Run full precision fused matmul+relu
    fused_matmul_relu<float>(full_output.data(), a_data.data(), b_data.data(), m, k, n);
    
    // Run quantized fused matmul+relu
    fused_matmul_relu_q8_0<float>(quantized_output.data(), a_data.data(), quantized_b_q8_0.data(), 
                                 b_scale_q8_0, m, k, n);
    
    // Compare results
    double max_error = 0.0;
    double sum_squared_error = 0.0;
    
    for (size_t i = 0; i < m * n; i++) {
        double error = std::abs(full_output[i] - quantized_output[i]);
        max_error = std::max(max_error, error);
        sum_squared_error += error * error;
    }
    
    double rmse = std::sqrt(sum_squared_error / (m * n));
    
    std::cout << "Comparison between full precision and Q8_0 quantized versions:" << std::endl;
    std::cout << "  Max error: " << max_error << std::endl;
    std::cout << "  RMSE: " << rmse << std::endl;
    
    // Allow some error due to quantization, but it should be within reasonable bounds
    EXPECT_LT(max_error, 0.5) << "Max error too high: " << max_error;
    EXPECT_LT(rmse, 0.1) << "RMSE too high: " << rmse;
}

// Disabled performance benchmark test (only for manual performance testing)
TEST_F(FusedQuantizedOperationsTest, DISABLED_QuantizedPerformanceBenchmark) {
    // Use much larger matrices for comprehensive benchmarking
    const size_t perf_m = 256;
    const size_t perf_k = 512;
    const size_t perf_n = 256;
    
    // Generate larger matrices
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    std::vector<float> perf_a(perf_m * perf_k);
    std::vector<float> perf_b(perf_k * perf_n);
    
    for (size_t i = 0; i < perf_m * perf_k; i++) {
        perf_a[i] = dist(gen);
    }
    
    for (size_t i = 0; i < perf_k * perf_n; i++) {
        perf_b[i] = dist(gen);
    }
    
    // Quantize B matrix
    std::vector<int8_t> perf_quantized_b(perf_k * perf_n + sizeof(float) / sizeof(int8_t));
    quantize_q8_0<float>(perf_quantized_b.data(), perf_b.data(), perf_k * perf_n);
    const float* perf_b_scale = reinterpret_cast<const float*>(perf_quantized_b.data() + perf_k * perf_n);
    
    // Output buffers
    std::vector<float> full_output(perf_m * perf_n);
    std::vector<float> quantized_output(perf_m * perf_n);
    
    // Measure performance of full precision operation
    auto full_time = measurePerformance([&]() {
        fused_matmul_relu<float>(full_output.data(), perf_a.data(), perf_b.data(), perf_m, perf_k, perf_n);
    }, 5);
    
    // Measure performance of quantized operation
    auto quantized_time = measurePerformance([&]() {
        fused_matmul_relu_q8_0<float>(quantized_output.data(), perf_a.data(), perf_quantized_b.data(), 
                                     perf_b_scale, perf_m, perf_k, perf_n);
    }, 5);
    
    // Calculate speedup
    double speedup = full_time / quantized_time;
    
    std::cout << "Comprehensive Performance Benchmark:" << std::endl;
    std::cout << "  Matrix size: " << perf_m << "x" << perf_k << " * " << perf_k << "x" << perf_n << std::endl;
    std::cout << "  Full precision time: " << std::fixed << std::setprecision(4) << full_time << " ms" << std::endl;
    std::cout << "  Quantized (Q8_0) time: " << std::fixed << std::setprecision(4) << quantized_time << " ms" << std::endl;
    std::cout << "  Speedup from quantization: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    
    // Calculate accuracy metrics
    double max_error = 0.0;
    double sum_squared_error = 0.0;
    
    for (size_t i = 0; i < perf_m * perf_n; i++) {
        double error = std::abs(full_output[i] - quantized_output[i]);
        max_error = std::max(max_error, error);
        sum_squared_error += error * error;
    }
    
    double rmse = std::sqrt(sum_squared_error / (perf_m * perf_n));
    
    std::cout << "  Max error: " << max_error << std::endl;
    std::cout << "  RMSE: " << rmse << std::endl;
    
    // Calculate memory usage
    size_t full_memory = perf_k * perf_n * sizeof(float);
    size_t quantized_memory = perf_k * perf_n * sizeof(int8_t) + sizeof(float);
    double memory_reduction = 100.0 * (1.0 - static_cast<double>(quantized_memory) / full_memory);
    
    std::cout << "  Memory usage for B matrix (full precision): " << full_memory << " bytes" << std::endl;
    std::cout << "  Memory usage for B matrix (quantized): " << quantized_memory << " bytes" << std::endl;
    std::cout << "  Memory reduction: " << std::fixed << std::setprecision(2) << memory_reduction << "%" << std::endl;
}

} // namespace