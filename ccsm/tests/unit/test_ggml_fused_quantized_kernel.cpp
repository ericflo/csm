#include <gtest/gtest.h>
#include <ccsm/cpu/ggml_model.h>
#include <ccsm/cpu/ggml_tensor.h>
#include <ccsm/cpu/simd.h>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <iostream>
#include <iomanip>

using namespace ccsm;

// Test fixture for GGML fused quantized kernel operations
class GGMLFusedQuantizedKernelTest : public ::testing::Test {
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
        
        // Create a GGMLContext for tensor operations
        ggml_ctx = std::make_shared<GGMLContext>();
        
        // Create input tensors
        tensor_a = ggml_ctx->create_tensor({m, k}, DataType::F32);
        tensor_b = ggml_ctx->create_tensor({k, n}, DataType::F32);
        
        // Copy data to tensors
        std::memcpy(tensor_a.data(), a_data.data(), m * k * sizeof(float));
        std::memcpy(tensor_b.data(), b_data.data(), k * n * sizeof(float));
        
        // Create quantized versions of tensor_b
        tensor_b_q8_0 = ggml_ctx->cast(tensor_b, DataType::Q8_0);
        tensor_b_q4_0 = ggml_ctx->cast(tensor_b, DataType::Q4_0);
        tensor_b_q4_1 = ggml_ctx->cast(tensor_b, DataType::Q4_1);
    }
    
    // Helper to measure performance
    template<typename Func>
    double measurePerformance(Func func, int iterations = 5) {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            func();
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        return elapsed.count() / iterations; // Average time in milliseconds
    }
    
    // Helper to calculate error metrics
    void calculateErrorMetrics(const std::vector<float>& reference, const std::vector<float>& test, 
                              double& max_error, double& rmse) {
        if (reference.size() != test.size()) {
            throw std::runtime_error("Size mismatch in error calculation");
        }
        
        double sum_squared_error = 0.0;
        max_error = 0.0;
        
        for (size_t i = 0; i < reference.size(); i++) {
            double error = std::abs(reference[i] - test[i]);
            max_error = std::max(max_error, error);
            sum_squared_error += error * error;
        }
        
        rmse = std::sqrt(sum_squared_error / reference.size());
    }
    
    // Test matrix dimensions
    static constexpr size_t m = 64;  // A: m x k
    static constexpr size_t k = 128; // B: k x n
    static constexpr size_t n = 64;  // Output: m x n
    
    // Test data
    std::vector<float> a_data;
    std::vector<float> b_data;
    
    // Tensors
    Tensor tensor_a;
    Tensor tensor_b;
    Tensor tensor_b_q8_0;
    Tensor tensor_b_q4_0;
    Tensor tensor_b_q4_1;
    
    // Context
    std::shared_ptr<GGMLContext> ggml_ctx;
};

// Test standard matrix multiplication with different quantized formats
TEST_F(GGMLFusedQuantizedKernelTest, StandardMatMul) {
    // Perform standard matrix multiplication with different data types
    Tensor result_f32 = ggml_ctx->matmul(tensor_a, tensor_b);
    Tensor result_q8_0 = ggml_ctx->matmul(tensor_a, tensor_b_q8_0);
    Tensor result_q4_0 = ggml_ctx->matmul(tensor_a, tensor_b_q4_0);
    Tensor result_q4_1 = ggml_ctx->matmul(tensor_a, tensor_b_q4_1);
    
    // Verify dimensions of results
    EXPECT_EQ(result_f32.shape(0), m);
    EXPECT_EQ(result_f32.shape(1), n);
    EXPECT_EQ(result_q8_0.shape(0), m);
    EXPECT_EQ(result_q8_0.shape(1), n);
    EXPECT_EQ(result_q4_0.shape(0), m);
    EXPECT_EQ(result_q4_0.shape(1), n);
    EXPECT_EQ(result_q4_1.shape(0), m);
    EXPECT_EQ(result_q4_1.shape(1), n);
    
    // Extract data for comparison
    std::vector<float> f32_data(m * n);
    std::vector<float> q8_0_data(m * n);
    std::vector<float> q4_0_data(m * n);
    std::vector<float> q4_1_data(m * n);
    
    std::memcpy(f32_data.data(), result_f32.data(), m * n * sizeof(float));
    std::memcpy(q8_0_data.data(), result_q8_0.data(), m * n * sizeof(float));
    std::memcpy(q4_0_data.data(), result_q4_0.data(), m * n * sizeof(float));
    std::memcpy(q4_1_data.data(), result_q4_1.data(), m * n * sizeof(float));
    
    // Calculate error metrics
    double q8_0_max_error, q8_0_rmse;
    double q4_0_max_error, q4_0_rmse;
    double q4_1_max_error, q4_1_rmse;
    
    calculateErrorMetrics(f32_data, q8_0_data, q8_0_max_error, q8_0_rmse);
    calculateErrorMetrics(f32_data, q4_0_data, q4_0_max_error, q4_0_rmse);
    calculateErrorMetrics(f32_data, q4_1_data, q4_1_max_error, q4_1_rmse);
    
    // Print error metrics
    std::cout << "Standard Matrix Multiplication Error Metrics:" << std::endl;
    std::cout << "  Q8_0 - Max Error: " << q8_0_max_error << ", RMSE: " << q8_0_rmse << std::endl;
    std::cout << "  Q4_0 - Max Error: " << q4_0_max_error << ", RMSE: " << q4_0_rmse << std::endl;
    std::cout << "  Q4_1 - Max Error: " << q4_1_max_error << ", RMSE: " << q4_1_rmse << std::endl;
    
    // Verify errors are within acceptable limits
    EXPECT_LT(q8_0_max_error, 0.1);
    EXPECT_LT(q8_0_rmse, 0.02);
    
    // Q4 formats have higher error due to lower precision
    EXPECT_LT(q4_0_max_error, 0.5);
    EXPECT_LT(q4_0_rmse, 0.1);
    
    // Q4_1 should be more accurate than Q4_0 due to bias
    EXPECT_LT(q4_1_max_error, 0.5);
    EXPECT_LT(q4_1_rmse, 0.1);
    EXPECT_LT(q4_1_rmse, q4_0_rmse);
}

// Test fused matrix multiplication with ReLU and different quantized formats
TEST_F(GGMLFusedQuantizedKernelTest, FusedMatMulReLU) {
    // Perform standard operations (separate matmul and ReLU)
    Tensor std_result_f32 = ggml_ctx->matmul(tensor_a, tensor_b);
    Tensor std_relu_f32 = ggml_ctx->relu(std_result_f32);
    
    Tensor std_result_q8_0 = ggml_ctx->matmul(tensor_a, tensor_b_q8_0);
    Tensor std_relu_q8_0 = ggml_ctx->relu(std_result_q8_0);
    
    Tensor std_result_q4_0 = ggml_ctx->matmul(tensor_a, tensor_b_q4_0);
    Tensor std_relu_q4_0 = ggml_ctx->relu(std_result_q4_0);
    
    Tensor std_result_q4_1 = ggml_ctx->matmul(tensor_a, tensor_b_q4_1);
    Tensor std_relu_q4_1 = ggml_ctx->relu(std_result_q4_1);
    
    // Prepare output buffers for fused operations
    float* fused_output_f32 = new float[m * n];
    float* fused_output_q8_0 = new float[m * n];
    float* fused_output_q4_0 = new float[m * n];
    float* fused_output_q4_1 = new float[m * n];
    
    // Use SIMD's fused operations
    simd::fused_matmul_relu<float>(fused_output_f32, a_data.data(), b_data.data(), m, k, n);
    
    // Get raw data from quantized tensors
    const int8_t* b_q8_0_data = static_cast<const int8_t*>(tensor_b_q8_0.data());
    const float* b_q8_0_scale = reinterpret_cast<const float*>(b_q8_0_data + k * n);
    
    const uint8_t* b_q4_0_data = static_cast<const uint8_t*>(tensor_b_q4_0.data());
    const float* b_q4_0_scale = reinterpret_cast<const float*>(b_q4_0_data + (k * n + 1) / 2);
    
    const uint8_t* b_q4_1_data = static_cast<const uint8_t*>(tensor_b_q4_1.data());
    const float* b_q4_1_scale = reinterpret_cast<const float*>(b_q4_1_data + (k * n + 1) / 2);
    const float* b_q4_1_bias = reinterpret_cast<const float*>(b_q4_1_scale + 1);
    
    // Use fused quantized operations
    simd::fused_matmul_relu_q8_0<float>(fused_output_q8_0, a_data.data(), b_q8_0_data, b_q8_0_scale, m, k, n);
    simd::fused_matmul_relu_q4_0<float>(fused_output_q4_0, a_data.data(), b_q4_0_data, b_q4_0_scale, m, k, n);
    simd::fused_matmul_relu_q4_1<float>(fused_output_q4_1, a_data.data(), b_q4_1_data, b_q4_1_scale, b_q4_1_bias, m, k, n);
    
    // Extract standard results for comparison
    std::vector<float> std_f32_data(m * n);
    std::vector<float> std_q8_0_data(m * n);
    std::vector<float> std_q4_0_data(m * n);
    std::vector<float> std_q4_1_data(m * n);
    
    std::memcpy(std_f32_data.data(), std_relu_f32.data(), m * n * sizeof(float));
    std::memcpy(std_q8_0_data.data(), std_relu_q8_0.data(), m * n * sizeof(float));
    std::memcpy(std_q4_0_data.data(), std_relu_q4_0.data(), m * n * sizeof(float));
    std::memcpy(std_q4_1_data.data(), std_relu_q4_1.data(), m * n * sizeof(float));
    
    // Create vectors for fused results
    std::vector<float> fused_f32_data(fused_output_f32, fused_output_f32 + m * n);
    std::vector<float> fused_q8_0_data(fused_output_q8_0, fused_output_q8_0 + m * n);
    std::vector<float> fused_q4_0_data(fused_output_q4_0, fused_output_q4_0 + m * n);
    std::vector<float> fused_q4_1_data(fused_output_q4_1, fused_output_q4_1 + m * n);
    
    // Calculate error metrics - comparing fused operations vs. separate operations
    double f32_max_error, f32_rmse;
    double q8_0_max_error, q8_0_rmse;
    double q4_0_max_error, q4_0_rmse;
    double q4_1_max_error, q4_1_rmse;
    
    calculateErrorMetrics(std_f32_data, fused_f32_data, f32_max_error, f32_rmse);
    calculateErrorMetrics(std_q8_0_data, fused_q8_0_data, q8_0_max_error, q8_0_rmse);
    calculateErrorMetrics(std_q4_0_data, fused_q4_0_data, q4_0_max_error, q4_0_rmse);
    calculateErrorMetrics(std_q4_1_data, fused_q4_1_data, q4_1_max_error, q4_1_rmse);
    
    // Print error metrics
    std::cout << "Fused MatMul+ReLU vs. Separate Operations Error Metrics:" << std::endl;
    std::cout << "  F32 - Max Error: " << f32_max_error << ", RMSE: " << f32_rmse << std::endl;
    std::cout << "  Q8_0 - Max Error: " << q8_0_max_error << ", RMSE: " << q8_0_rmse << std::endl;
    std::cout << "  Q4_0 - Max Error: " << q4_0_max_error << ", RMSE: " << q4_0_rmse << std::endl;
    std::cout << "  Q4_1 - Max Error: " << q4_1_max_error << ", RMSE: " << q4_1_rmse << std::endl;
    
    // Verify errors are within acceptable limits
    EXPECT_LT(f32_max_error, 1e-4);
    EXPECT_LT(f32_rmse, 1e-5);
    
    EXPECT_LT(q8_0_max_error, 1e-4);
    EXPECT_LT(q8_0_rmse, 1e-5);
    
    EXPECT_LT(q4_0_max_error, 1e-4);
    EXPECT_LT(q4_0_rmse, 1e-5);
    
    EXPECT_LT(q4_1_max_error, 1e-4);
    EXPECT_LT(q4_1_rmse, 1e-5);
    
    // Verify ReLU was correctly applied (no negative values)
    for (size_t i = 0; i < m * n; i++) {
        EXPECT_GE(fused_f32_data[i], 0.0f);
        EXPECT_GE(fused_q8_0_data[i], 0.0f);
        EXPECT_GE(fused_q4_0_data[i], 0.0f);
        EXPECT_GE(fused_q4_1_data[i], 0.0f);
    }
    
    // Clean up
    delete[] fused_output_f32;
    delete[] fused_output_q8_0;
    delete[] fused_output_q4_0;
    delete[] fused_output_q4_1;
}

// Test fused matrix multiplication with SiLU and different quantized formats
TEST_F(GGMLFusedQuantizedKernelTest, FusedMatMulSiLU) {
    // Perform standard operations (separate matmul and SiLU)
    Tensor std_result_f32 = ggml_ctx->matmul(tensor_a, tensor_b);
    Tensor std_silu_f32 = ggml_ctx->silu(std_result_f32);
    
    Tensor std_result_q8_0 = ggml_ctx->matmul(tensor_a, tensor_b_q8_0);
    Tensor std_silu_q8_0 = ggml_ctx->silu(std_result_q8_0);
    
    Tensor std_result_q4_0 = ggml_ctx->matmul(tensor_a, tensor_b_q4_0);
    Tensor std_silu_q4_0 = ggml_ctx->silu(std_result_q4_0);
    
    Tensor std_result_q4_1 = ggml_ctx->matmul(tensor_a, tensor_b_q4_1);
    Tensor std_silu_q4_1 = ggml_ctx->silu(std_result_q4_1);
    
    // Prepare output buffers for fused operations
    float* fused_output_f32 = new float[m * n];
    float* fused_output_q8_0 = new float[m * n];
    float* fused_output_q4_0 = new float[m * n];
    float* fused_output_q4_1 = new float[m * n];
    
    // Use SIMD's fused operations
    simd::fused_matmul_silu<float>(fused_output_f32, a_data.data(), b_data.data(), m, k, n);
    
    // Get raw data from quantized tensors
    const int8_t* b_q8_0_data = static_cast<const int8_t*>(tensor_b_q8_0.data());
    const float* b_q8_0_scale = reinterpret_cast<const float*>(b_q8_0_data + k * n);
    
    const uint8_t* b_q4_0_data = static_cast<const uint8_t*>(tensor_b_q4_0.data());
    const float* b_q4_0_scale = reinterpret_cast<const float*>(b_q4_0_data + (k * n + 1) / 2);
    
    const uint8_t* b_q4_1_data = static_cast<const uint8_t*>(tensor_b_q4_1.data());
    const float* b_q4_1_scale = reinterpret_cast<const float*>(b_q4_1_data + (k * n + 1) / 2);
    const float* b_q4_1_bias = reinterpret_cast<const float*>(b_q4_1_scale + 1);
    
    // Use fused quantized operations
    simd::fused_matmul_silu_q8_0<float>(fused_output_q8_0, a_data.data(), b_q8_0_data, b_q8_0_scale, m, k, n);
    simd::fused_matmul_silu_q4_0<float>(fused_output_q4_0, a_data.data(), b_q4_0_data, b_q4_0_scale, m, k, n);
    simd::fused_matmul_silu_q4_1<float>(fused_output_q4_1, a_data.data(), b_q4_1_data, b_q4_1_scale, b_q4_1_bias, m, k, n);
    
    // Extract standard results for comparison
    std::vector<float> std_f32_data(m * n);
    std::vector<float> std_q8_0_data(m * n);
    std::vector<float> std_q4_0_data(m * n);
    std::vector<float> std_q4_1_data(m * n);
    
    std::memcpy(std_f32_data.data(), std_silu_f32.data(), m * n * sizeof(float));
    std::memcpy(std_q8_0_data.data(), std_silu_q8_0.data(), m * n * sizeof(float));
    std::memcpy(std_q4_0_data.data(), std_silu_q4_0.data(), m * n * sizeof(float));
    std::memcpy(std_q4_1_data.data(), std_silu_q4_1.data(), m * n * sizeof(float));
    
    // Create vectors for fused results
    std::vector<float> fused_f32_data(fused_output_f32, fused_output_f32 + m * n);
    std::vector<float> fused_q8_0_data(fused_output_q8_0, fused_output_q8_0 + m * n);
    std::vector<float> fused_q4_0_data(fused_output_q4_0, fused_output_q4_0 + m * n);
    std::vector<float> fused_q4_1_data(fused_output_q4_1, fused_output_q4_1 + m * n);
    
    // Calculate error metrics - comparing fused operations vs. separate operations
    double f32_max_error, f32_rmse;
    double q8_0_max_error, q8_0_rmse;
    double q4_0_max_error, q4_0_rmse;
    double q4_1_max_error, q4_1_rmse;
    
    calculateErrorMetrics(std_f32_data, fused_f32_data, f32_max_error, f32_rmse);
    calculateErrorMetrics(std_q8_0_data, fused_q8_0_data, q8_0_max_error, q8_0_rmse);
    calculateErrorMetrics(std_q4_0_data, fused_q4_0_data, q4_0_max_error, q4_0_rmse);
    calculateErrorMetrics(std_q4_1_data, fused_q4_1_data, q4_1_max_error, q4_1_rmse);
    
    // Print error metrics
    std::cout << "Fused MatMul+SiLU vs. Separate Operations Error Metrics:" << std::endl;
    std::cout << "  F32 - Max Error: " << f32_max_error << ", RMSE: " << f32_rmse << std::endl;
    std::cout << "  Q8_0 - Max Error: " << q8_0_max_error << ", RMSE: " << q8_0_rmse << std::endl;
    std::cout << "  Q4_0 - Max Error: " << q4_0_max_error << ", RMSE: " << q4_0_rmse << std::endl;
    std::cout << "  Q4_1 - Max Error: " << q4_1_max_error << ", RMSE: " << q4_1_rmse << std::endl;
    
    // Verify errors are within acceptable limits - SiLU allows slightly larger errors due to exponential approximation
    EXPECT_LT(f32_max_error, 0.1);
    EXPECT_LT(f32_rmse, 0.01);
    
    EXPECT_LT(q8_0_max_error, 0.1);
    EXPECT_LT(q8_0_rmse, 0.01);
    
    EXPECT_LT(q4_0_max_error, 0.2);
    EXPECT_LT(q4_0_rmse, 0.05);
    
    EXPECT_LT(q4_1_max_error, 0.2);
    EXPECT_LT(q4_1_rmse, 0.05);
    
    // Verify SiLU properties
    // SiLU(x) = x * sigmoid(x), so it should be positive for positive x and negative for negative x, but bounded
    for (size_t i = 0; i < m * n; i++) {
        // SiLU can produce both positive and negative values
        // It approaches the identity function for large positive inputs and zero for large negative inputs
        if (std_f32_data[i] > 0.5) {
            // For positive values, fused result should be close to the tensor result
            EXPECT_NEAR(fused_f32_data[i], std_f32_data[i], 0.1);
        }
    }
    
    // Clean up
    delete[] fused_output_f32;
    delete[] fused_output_q8_0;
    delete[] fused_output_q4_0;
    delete[] fused_output_q4_1;
}

// Test performance of fused operations
TEST_F(GGMLFusedQuantizedKernelTest, FusedOperationPerformance) {
    // Prepare output buffers
    float* separate_matmul_output = new float[m * n];
    float* separate_activation_output = new float[m * n];
    float* fused_output = new float[m * n];
    
    // Get raw data from quantized tensors
    const int8_t* b_q8_0_data = static_cast<const int8_t*>(tensor_b_q8_0.data());
    const float* b_q8_0_scale = reinterpret_cast<const float*>(b_q8_0_data + k * n);
    
    const uint8_t* b_q4_0_data = static_cast<const uint8_t*>(tensor_b_q4_0.data());
    const float* b_q4_0_scale = reinterpret_cast<const float*>(b_q4_0_data + (k * n + 1) / 2);
    
    const uint8_t* b_q4_1_data = static_cast<const uint8_t*>(tensor_b_q4_1.data());
    const float* b_q4_1_scale = reinterpret_cast<const float*>(b_q4_1_data + (k * n + 1) / 2);
    const float* b_q4_1_bias = reinterpret_cast<const float*>(b_q4_1_scale + 1);
    
    // Measure performance of separate operations (F32)
    double separate_f32_matmul_relu_time = measurePerformance([&]() {
        simd::matrix_mul<float>(separate_matmul_output, a_data.data(), b_data.data(), m, k, n);
        simd::relu<float>(separate_activation_output, separate_matmul_output, m * n);
    });
    
    // Measure performance of fused operations (F32)
    double fused_f32_matmul_relu_time = measurePerformance([&]() {
        simd::fused_matmul_relu<float>(fused_output, a_data.data(), b_data.data(), m, k, n);
    });
    
    // Measure performance of separate operations (Q8_0)
    double separate_q8_0_matmul_relu_time = measurePerformance([&]() {
        simd::matrix_mul_q8_0<float>(separate_matmul_output, a_data.data(), b_q8_0_data, b_q8_0_scale, m, k, n);
        simd::relu<float>(separate_activation_output, separate_matmul_output, m * n);
    });
    
    // Measure performance of fused operations (Q8_0)
    double fused_q8_0_matmul_relu_time = measurePerformance([&]() {
        simd::fused_matmul_relu_q8_0<float>(fused_output, a_data.data(), b_q8_0_data, b_q8_0_scale, m, k, n);
    });
    
    // Measure performance of separate operations (Q4_0)
    double separate_q4_0_matmul_relu_time = measurePerformance([&]() {
        simd::matrix_mul_q4_0<float>(separate_matmul_output, a_data.data(), b_q4_0_data, b_q4_0_scale, m, k, n);
        simd::relu<float>(separate_activation_output, separate_matmul_output, m * n);
    });
    
    // Measure performance of fused operations (Q4_0)
    double fused_q4_0_matmul_relu_time = measurePerformance([&]() {
        simd::fused_matmul_relu_q4_0<float>(fused_output, a_data.data(), b_q4_0_data, b_q4_0_scale, m, k, n);
    });
    
    // Measure performance of separate operations (Q4_1)
    double separate_q4_1_matmul_relu_time = measurePerformance([&]() {
        simd::matrix_mul_q4_1<float>(separate_matmul_output, a_data.data(), b_q4_1_data, b_q4_1_scale, b_q4_1_bias, m, k, n);
        simd::relu<float>(separate_activation_output, separate_matmul_output, m * n);
    });
    
    // Measure performance of fused operations (Q4_1)
    double fused_q4_1_matmul_relu_time = measurePerformance([&]() {
        simd::fused_matmul_relu_q4_1<float>(fused_output, a_data.data(), b_q4_1_data, b_q4_1_scale, b_q4_1_bias, m, k, n);
    });
    
    // Calculate speedups
    double f32_speedup = separate_f32_matmul_relu_time / fused_f32_matmul_relu_time;
    double q8_0_speedup = separate_q8_0_matmul_relu_time / fused_q8_0_matmul_relu_time;
    double q4_0_speedup = separate_q4_0_matmul_relu_time / fused_q4_0_matmul_relu_time;
    double q4_1_speedup = separate_q4_1_matmul_relu_time / fused_q4_1_matmul_relu_time;
    
    // Calculate quantization speedups
    double q8_0_vs_f32_speedup = separate_f32_matmul_relu_time / fused_q8_0_matmul_relu_time;
    double q4_0_vs_f32_speedup = separate_f32_matmul_relu_time / fused_q4_0_matmul_relu_time;
    double q4_1_vs_f32_speedup = separate_f32_matmul_relu_time / fused_q4_1_matmul_relu_time;
    
    // Print performance results
    std::cout << "\nPerformance Results (Matrix Size: " << m << "x" << k << " * " << k << "x" << n << ")" << std::endl;
    std::cout << std::setw(20) << "Operation" << std::setw(15) << "Type" << std::setw(15) << "Time (ms)" << std::setw(15) << "Speedup" << std::endl;
    std::cout << std::setfill('-') << std::setw(65) << "-" << std::setfill(' ') << std::endl;
    
    std::cout << std::setw(20) << "Separate MatMul+ReLU" << std::setw(15) << "F32" << std::setw(15) << std::fixed << std::setprecision(3) << separate_f32_matmul_relu_time << std::setw(15) << "1.00x" << std::endl;
    std::cout << std::setw(20) << "Fused MatMul+ReLU" << std::setw(15) << "F32" << std::setw(15) << std::fixed << std::setprecision(3) << fused_f32_matmul_relu_time << std::setw(15) << std::fixed << std::setprecision(2) << f32_speedup << "x" << std::endl;
    
    std::cout << std::setw(20) << "Separate MatMul+ReLU" << std::setw(15) << "Q8_0" << std::setw(15) << std::fixed << std::setprecision(3) << separate_q8_0_matmul_relu_time << std::setw(15) << std::fixed << std::setprecision(2) << separate_f32_matmul_relu_time / separate_q8_0_matmul_relu_time << "x" << std::endl;
    std::cout << std::setw(20) << "Fused MatMul+ReLU" << std::setw(15) << "Q8_0" << std::setw(15) << std::fixed << std::setprecision(3) << fused_q8_0_matmul_relu_time << std::setw(15) << std::fixed << std::setprecision(2) << q8_0_vs_f32_speedup << "x" << std::endl;
    
    std::cout << std::setw(20) << "Separate MatMul+ReLU" << std::setw(15) << "Q4_0" << std::setw(15) << std::fixed << std::setprecision(3) << separate_q4_0_matmul_relu_time << std::setw(15) << std::fixed << std::setprecision(2) << separate_f32_matmul_relu_time / separate_q4_0_matmul_relu_time << "x" << std::endl;
    std::cout << std::setw(20) << "Fused MatMul+ReLU" << std::setw(15) << "Q4_0" << std::setw(15) << std::fixed << std::setprecision(3) << fused_q4_0_matmul_relu_time << std::setw(15) << std::fixed << std::setprecision(2) << q4_0_vs_f32_speedup << "x" << std::endl;
    
    std::cout << std::setw(20) << "Separate MatMul+ReLU" << std::setw(15) << "Q4_1" << std::setw(15) << std::fixed << std::setprecision(3) << separate_q4_1_matmul_relu_time << std::setw(15) << std::fixed << std::setprecision(2) << separate_f32_matmul_relu_time / separate_q4_1_matmul_relu_time << "x" << std::endl;
    std::cout << std::setw(20) << "Fused MatMul+ReLU" << std::setw(15) << "Q4_1" << std::setw(15) << std::fixed << std::setprecision(3) << fused_q4_1_matmul_relu_time << std::setw(15) << std::fixed << std::setprecision(2) << q4_1_vs_f32_speedup << "x" << std::endl;
    
    // Fusion speedup within each type
    std::cout << "\nFusion Speedups:" << std::endl;
    std::cout << "  F32: " << std::fixed << std::setprecision(2) << f32_speedup << "x" << std::endl;
    std::cout << "  Q8_0: " << std::fixed << std::setprecision(2) << q8_0_speedup << "x" << std::endl;
    std::cout << "  Q4_0: " << std::fixed << std::setprecision(2) << q4_0_speedup << "x" << std::endl;
    std::cout << "  Q4_1: " << std::fixed << std::setprecision(2) << q4_1_speedup << "x" << std::endl;
    
    // Verify we get at least some speedup from fusion within each type
    EXPECT_GE(f32_speedup, 0.95);  // Might not get major speedup in all cases, but should not be much slower
    EXPECT_GE(q8_0_speedup, 0.95);
    EXPECT_GE(q4_0_speedup, 0.95);
    EXPECT_GE(q4_1_speedup, 0.95);
    
    // Verify quantized versions are faster than full precision
    EXPECT_GE(q8_0_vs_f32_speedup, 1.0);
    EXPECT_GE(q4_0_vs_f32_speedup, 1.0);
    EXPECT_GE(q4_1_vs_f32_speedup, 1.0);
    
    // Clean up
    delete[] separate_matmul_output;
    delete[] separate_activation_output;
    delete[] fused_output;
}