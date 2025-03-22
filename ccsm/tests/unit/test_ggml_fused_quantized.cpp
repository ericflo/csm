#include <ccsm/cpu/ggml_model.h>
#include <ccsm/cpu/ggml_tensor.h>
#include <ccsm/cpu/simd.h>
#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>

namespace ccsm {
namespace {

// Test fixture for GGML fused quantized operations
class GGMLFusedQuantizedTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a GGML context
        ggml_ctx = std::make_shared<GGMLContext>();
        
        // Initialize random number generator
        std::mt19937 gen(42); // Fixed seed for reproducibility
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        
        // Generate random test data
        data_a.resize(m * k);
        data_b.resize(k * n);
        
        for (size_t i = 0; i < m * k; i++) {
            data_a[i] = dist(gen);
        }
        
        for (size_t i = 0; i < k * n; i++) {
            data_b[i] = dist(gen);
        }
        
        // Create tensors with the test data
        tensor_a = ggml_ctx->create_tensor({m, k}, DataType::F32);
        tensor_b = ggml_ctx->create_tensor({k, n}, DataType::F32);
        
        std::memcpy(tensor_a.data(), data_a.data(), m * k * sizeof(float));
        std::memcpy(tensor_b.data(), data_b.data(), k * n * sizeof(float));
        
        // Quantize tensor_b to different formats
        tensor_b_q8_0 = ggml_ctx->cast(tensor_b, DataType::Q8_0);
        tensor_b_q4_0 = ggml_ctx->cast(tensor_b, DataType::Q4_0);
        tensor_b_q4_1 = ggml_ctx->cast(tensor_b, DataType::Q4_1);
    }
    
    // Test data dimensions
    static constexpr size_t m = 16;
    static constexpr size_t k = 32;
    static constexpr size_t n = 24;
    
    // Test data
    std::vector<float> data_a;
    std::vector<float> data_b;
    
    // Tensors
    Tensor tensor_a;
    Tensor tensor_b;
    Tensor tensor_b_q8_0;
    Tensor tensor_b_q4_0;
    Tensor tensor_b_q4_1;
    
    // GGML context
    std::shared_ptr<GGMLContext> ggml_ctx;
};

// Test matrix multiplication with quantized weights
TEST_F(GGMLFusedQuantizedTest, MatMulQuantized) {
    // Regular matrix multiplication (F32 * F32)
    Tensor result_f32 = ggml_ctx->matmul(tensor_a, tensor_b);
    
    // Matrix multiplication with Q8_0 weights (F32 * Q8_0)
    Tensor result_q8_0 = ggml_ctx->matmul(tensor_a, tensor_b_q8_0);
    
    // Matrix multiplication with Q4_0 weights (F32 * Q4_0)
    Tensor result_q4_0 = ggml_ctx->matmul(tensor_a, tensor_b_q4_0);
    
    // Matrix multiplication with Q4_1 weights (F32 * Q4_1)
    Tensor result_q4_1 = ggml_ctx->matmul(tensor_a, tensor_b_q4_1);
    
    // Verify dimensions
    EXPECT_EQ(result_f32.shape(0), m);
    EXPECT_EQ(result_f32.shape(1), n);
    
    EXPECT_EQ(result_q8_0.shape(0), m);
    EXPECT_EQ(result_q8_0.shape(1), n);
    
    EXPECT_EQ(result_q4_0.shape(0), m);
    EXPECT_EQ(result_q4_0.shape(1), n);
    
    EXPECT_EQ(result_q4_1.shape(0), m);
    EXPECT_EQ(result_q4_1.shape(1), n);
    
    // Convert results to vectors for comparison
    std::vector<float> f32_result(m * n);
    std::vector<float> q8_0_result(m * n);
    std::vector<float> q4_0_result(m * n);
    std::vector<float> q4_1_result(m * n);
    
    std::memcpy(f32_result.data(), result_f32.data(), m * n * sizeof(float));
    std::memcpy(q8_0_result.data(), result_q8_0.data(), m * n * sizeof(float));
    std::memcpy(q4_0_result.data(), result_q4_0.data(), m * n * sizeof(float));
    std::memcpy(q4_1_result.data(), result_q4_1.data(), m * n * sizeof(float));
    
    // Calculate error metrics for Q8_0
    double q8_0_max_error = 0.0;
    double q8_0_sum_squared_error = 0.0;
    
    for (size_t i = 0; i < m * n; i++) {
        double error = std::abs(f32_result[i] - q8_0_result[i]);
        q8_0_max_error = std::max(q8_0_max_error, error);
        q8_0_sum_squared_error += error * error;
    }
    
    double q8_0_rmse = std::sqrt(q8_0_sum_squared_error / (m * n));
    
    // Calculate error metrics for Q4_0
    double q4_0_max_error = 0.0;
    double q4_0_sum_squared_error = 0.0;
    
    for (size_t i = 0; i < m * n; i++) {
        double error = std::abs(f32_result[i] - q4_0_result[i]);
        q4_0_max_error = std::max(q4_0_max_error, error);
        q4_0_sum_squared_error += error * error;
    }
    
    double q4_0_rmse = std::sqrt(q4_0_sum_squared_error / (m * n));
    
    // Calculate error metrics for Q4_1
    double q4_1_max_error = 0.0;
    double q4_1_sum_squared_error = 0.0;
    
    for (size_t i = 0; i < m * n; i++) {
        double error = std::abs(f32_result[i] - q4_1_result[i]);
        q4_1_max_error = std::max(q4_1_max_error, error);
        q4_1_sum_squared_error += error * error;
    }
    
    double q4_1_rmse = std::sqrt(q4_1_sum_squared_error / (m * n));
    
    // Print error metrics
    std::cout << "Q8_0 Matrix Multiplication - Max Error: " << q8_0_max_error << ", RMSE: " << q8_0_rmse << std::endl;
    std::cout << "Q4_0 Matrix Multiplication - Max Error: " << q4_0_max_error << ", RMSE: " << q4_0_rmse << std::endl;
    std::cout << "Q4_1 Matrix Multiplication - Max Error: " << q4_1_max_error << ", RMSE: " << q4_1_rmse << std::endl;
    
    // Verify errors are within acceptable limits
    EXPECT_LT(q8_0_max_error, 1.0);  // Max absolute error < 1.0
    EXPECT_LT(q8_0_rmse, 0.2);       // RMSE < 0.2
    
    EXPECT_LT(q4_0_max_error, 1.5);  // Max absolute error < 1.5
    EXPECT_LT(q4_0_rmse, 0.3);       // RMSE < 0.3
    
    EXPECT_LT(q4_1_max_error, 1.0);  // Max absolute error < 1.0
    EXPECT_LT(q4_1_rmse, 0.25);      // RMSE < 0.25
}

// Test fused quantized matrix multiplication with ReLU activation
TEST_F(GGMLFusedQuantizedTest, MatMulReLUQuantized) {
    // Regular matrix multiplication with ReLU (F32 * F32)
    Tensor regular_matmul = ggml_ctx->matmul(tensor_a, tensor_b);
    Tensor result_f32 = ggml_ctx->relu(regular_matmul);
    
    // Matrix multiplication with Q8_0 weights followed by ReLU
    Tensor q8_0_matmul = ggml_ctx->matmul(tensor_a, tensor_b_q8_0);
    Tensor result_q8_0 = ggml_ctx->relu(q8_0_matmul);
    
    // Matrix multiplication with Q4_0 weights followed by ReLU
    Tensor q4_0_matmul = ggml_ctx->matmul(tensor_a, tensor_b_q4_0);
    Tensor result_q4_0 = ggml_ctx->relu(q4_0_matmul);
    
    // Matrix multiplication with Q4_1 weights followed by ReLU
    Tensor q4_1_matmul = ggml_ctx->matmul(tensor_a, tensor_b_q4_1);
    Tensor result_q4_1 = ggml_ctx->relu(q4_1_matmul);
    
    // Convert results to vectors for comparison
    std::vector<float> f32_result(m * n);
    std::vector<float> q8_0_result(m * n);
    std::vector<float> q4_0_result(m * n);
    std::vector<float> q4_1_result(m * n);
    
    std::memcpy(f32_result.data(), result_f32.data(), m * n * sizeof(float));
    std::memcpy(q8_0_result.data(), result_q8_0.data(), m * n * sizeof(float));
    std::memcpy(q4_0_result.data(), result_q4_0.data(), m * n * sizeof(float));
    std::memcpy(q4_1_result.data(), result_q4_1.data(), m * n * sizeof(float));
    
    // Verify ReLU function is applied (no negative values)
    for (size_t i = 0; i < m * n; i++) {
        EXPECT_GE(f32_result[i], 0.0f);
        EXPECT_GE(q8_0_result[i], 0.0f);
        EXPECT_GE(q4_0_result[i], 0.0f);
        EXPECT_GE(q4_1_result[i], 0.0f);
    }
    
    // Calculate error metrics for Q8_0 with ReLU
    double q8_0_max_error = 0.0;
    double q8_0_sum_squared_error = 0.0;
    
    for (size_t i = 0; i < m * n; i++) {
        double error = std::abs(f32_result[i] - q8_0_result[i]);
        q8_0_max_error = std::max(q8_0_max_error, error);
        q8_0_sum_squared_error += error * error;
    }
    
    double q8_0_rmse = std::sqrt(q8_0_sum_squared_error / (m * n));
    
    // Calculate error metrics for Q4_0 with ReLU
    double q4_0_max_error = 0.0;
    double q4_0_sum_squared_error = 0.0;
    
    for (size_t i = 0; i < m * n; i++) {
        double error = std::abs(f32_result[i] - q4_0_result[i]);
        q4_0_max_error = std::max(q4_0_max_error, error);
        q4_0_sum_squared_error += error * error;
    }
    
    double q4_0_rmse = std::sqrt(q4_0_sum_squared_error / (m * n));
    
    // Calculate error metrics for Q4_1 with ReLU
    double q4_1_max_error = 0.0;
    double q4_1_sum_squared_error = 0.0;
    
    for (size_t i = 0; i < m * n; i++) {
        double error = std::abs(f32_result[i] - q4_1_result[i]);
        q4_1_max_error = std::max(q4_1_max_error, error);
        q4_1_sum_squared_error += error * error;
    }
    
    double q4_1_rmse = std::sqrt(q4_1_sum_squared_error / (m * n));
    
    // Print error metrics
    std::cout << "Q8_0 Matrix Multiplication with ReLU - Max Error: " << q8_0_max_error << ", RMSE: " << q8_0_rmse << std::endl;
    std::cout << "Q4_0 Matrix Multiplication with ReLU - Max Error: " << q4_0_max_error << ", RMSE: " << q4_0_rmse << std::endl;
    std::cout << "Q4_1 Matrix Multiplication with ReLU - Max Error: " << q4_1_max_error << ", RMSE: " << q4_1_rmse << std::endl;
    
    // Verify errors are within acceptable limits
    EXPECT_LT(q8_0_max_error, 1.0);
    EXPECT_LT(q8_0_rmse, 0.2);
    
    EXPECT_LT(q4_0_max_error, 1.5);
    EXPECT_LT(q4_0_rmse, 0.3);
    
    EXPECT_LT(q4_1_max_error, 1.0);
    EXPECT_LT(q4_1_rmse, 0.25);
}

// Test fused quantized matrix multiplication with SiLU activation
TEST_F(GGMLFusedQuantizedTest, MatMulSiLUQuantized) {
    // Regular matrix multiplication with SiLU (F32 * F32)
    Tensor regular_matmul = ggml_ctx->matmul(tensor_a, tensor_b);
    Tensor result_f32 = ggml_ctx->silu(regular_matmul);
    
    // Matrix multiplication with Q8_0 weights followed by SiLU
    Tensor q8_0_matmul = ggml_ctx->matmul(tensor_a, tensor_b_q8_0);
    Tensor result_q8_0 = ggml_ctx->silu(q8_0_matmul);
    
    // Matrix multiplication with Q4_0 weights followed by SiLU
    Tensor q4_0_matmul = ggml_ctx->matmul(tensor_a, tensor_b_q4_0);
    Tensor result_q4_0 = ggml_ctx->silu(q4_0_matmul);
    
    // Matrix multiplication with Q4_1 weights followed by SiLU
    Tensor q4_1_matmul = ggml_ctx->matmul(tensor_a, tensor_b_q4_1);
    Tensor result_q4_1 = ggml_ctx->silu(q4_1_matmul);
    
    // Convert results to vectors for comparison
    std::vector<float> f32_result(m * n);
    std::vector<float> q8_0_result(m * n);
    std::vector<float> q4_0_result(m * n);
    std::vector<float> q4_1_result(m * n);
    
    std::memcpy(f32_result.data(), result_f32.data(), m * n * sizeof(float));
    std::memcpy(q8_0_result.data(), result_q8_0.data(), m * n * sizeof(float));
    std::memcpy(q4_0_result.data(), result_q4_0.data(), m * n * sizeof(float));
    std::memcpy(q4_1_result.data(), result_q4_1.data(), m * n * sizeof(float));
    
    // Calculate error metrics for Q8_0 with SiLU
    double q8_0_max_error = 0.0;
    double q8_0_sum_squared_error = 0.0;
    
    for (size_t i = 0; i < m * n; i++) {
        double error = std::abs(f32_result[i] - q8_0_result[i]);
        q8_0_max_error = std::max(q8_0_max_error, error);
        q8_0_sum_squared_error += error * error;
    }
    
    double q8_0_rmse = std::sqrt(q8_0_sum_squared_error / (m * n));
    
    // Calculate error metrics for Q4_0 with SiLU
    double q4_0_max_error = 0.0;
    double q4_0_sum_squared_error = 0.0;
    
    for (size_t i = 0; i < m * n; i++) {
        double error = std::abs(f32_result[i] - q4_0_result[i]);
        q4_0_max_error = std::max(q4_0_max_error, error);
        q4_0_sum_squared_error += error * error;
    }
    
    double q4_0_rmse = std::sqrt(q4_0_sum_squared_error / (m * n));
    
    // Calculate error metrics for Q4_1 with SiLU
    double q4_1_max_error = 0.0;
    double q4_1_sum_squared_error = 0.0;
    
    for (size_t i = 0; i < m * n; i++) {
        double error = std::abs(f32_result[i] - q4_1_result[i]);
        q4_1_max_error = std::max(q4_1_max_error, error);
        q4_1_sum_squared_error += error * error;
    }
    
    double q4_1_rmse = std::sqrt(q4_1_sum_squared_error / (m * n));
    
    // Print error metrics
    std::cout << "Q8_0 Matrix Multiplication with SiLU - Max Error: " << q8_0_max_error << ", RMSE: " << q8_0_rmse << std::endl;
    std::cout << "Q4_0 Matrix Multiplication with SiLU - Max Error: " << q4_0_max_error << ", RMSE: " << q4_0_rmse << std::endl;
    std::cout << "Q4_1 Matrix Multiplication with SiLU - Max Error: " << q4_1_max_error << ", RMSE: " << q4_1_rmse << std::endl;
    
    // Verify errors are within acceptable limits
    EXPECT_LT(q8_0_max_error, 1.0);
    EXPECT_LT(q8_0_rmse, 0.2);
    
    EXPECT_LT(q4_0_max_error, 1.5);
    EXPECT_LT(q4_0_rmse, 0.3);
    
    EXPECT_LT(q4_1_max_error, 1.0);
    EXPECT_LT(q4_1_rmse, 0.25);
}

// Performance benchmark for quantized matrix multiplication (disabled by default)
TEST_F(GGMLFusedQuantizedTest, DISABLED_QuantizedMatMulPerformance) {
    // Create larger matrices for performance testing
    const size_t large_m = 128;
    const size_t large_k = 512;
    const size_t large_n = 128;
    
    // Create data for matrices
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    std::vector<float> large_a(large_m * large_k);
    std::vector<float> large_b(large_k * large_n);
    
    for (size_t i = 0; i < large_m * large_k; i++) {
        large_a[i] = dist(gen);
    }
    
    for (size_t i = 0; i < large_k * large_n; i++) {
        large_b[i] = dist(gen);
    }
    
    // Create tensors
    Tensor large_tensor_a = ggml_ctx->create_tensor({large_m, large_k}, DataType::F32);
    Tensor large_tensor_b = ggml_ctx->create_tensor({large_k, large_n}, DataType::F32);
    
    std::memcpy(large_tensor_a.data(), large_a.data(), large_m * large_k * sizeof(float));
    std::memcpy(large_tensor_b.data(), large_b.data(), large_k * large_n * sizeof(float));
    
    // Quantize tensor_b to different formats
    Tensor large_tensor_b_q8_0 = ggml_ctx->cast(large_tensor_b, DataType::Q8_0);
    Tensor large_tensor_b_q4_0 = ggml_ctx->cast(large_tensor_b, DataType::Q4_0);
    Tensor large_tensor_b_q4_1 = ggml_ctx->cast(large_tensor_b, DataType::Q4_1);
    
    // F32 MatMul + ReLU
    auto start_f32 = std::chrono::high_resolution_clock::now();
    Tensor matmul_f32 = ggml_ctx->matmul(large_tensor_a, large_tensor_b);
    Tensor result_f32 = ggml_ctx->relu(matmul_f32);
    auto end_f32 = std::chrono::high_resolution_clock::now();
    auto duration_f32 = std::chrono::duration_cast<std::chrono::milliseconds>(end_f32 - start_f32);
    
    // Q8_0 MatMul + ReLU
    auto start_q8 = std::chrono::high_resolution_clock::now();
    Tensor matmul_q8 = ggml_ctx->matmul(large_tensor_a, large_tensor_b_q8_0);
    Tensor result_q8 = ggml_ctx->relu(matmul_q8);
    auto end_q8 = std::chrono::high_resolution_clock::now();
    auto duration_q8 = std::chrono::duration_cast<std::chrono::milliseconds>(end_q8 - start_q8);
    
    // Q4_0 MatMul + ReLU
    auto start_q4_0 = std::chrono::high_resolution_clock::now();
    Tensor matmul_q4_0 = ggml_ctx->matmul(large_tensor_a, large_tensor_b_q4_0);
    Tensor result_q4_0 = ggml_ctx->relu(matmul_q4_0);
    auto end_q4_0 = std::chrono::high_resolution_clock::now();
    auto duration_q4_0 = std::chrono::duration_cast<std::chrono::milliseconds>(end_q4_0 - start_q4_0);
    
    // Q4_1 MatMul + ReLU
    auto start_q4_1 = std::chrono::high_resolution_clock::now();
    Tensor matmul_q4_1 = ggml_ctx->matmul(large_tensor_a, large_tensor_b_q4_1);
    Tensor result_q4_1 = ggml_ctx->relu(matmul_q4_1);
    auto end_q4_1 = std::chrono::high_resolution_clock::now();
    auto duration_q4_1 = std::chrono::duration_cast<std::chrono::milliseconds>(end_q4_1 - start_q4_1);
    
    // Print performance results
    std::cout << "Performance Benchmark for matrix multiplication with ReLU:" << std::endl;
    std::cout << "Matrix dimensions: " << large_m << "x" << large_k << " * " << large_k << "x" << large_n << std::endl;
    std::cout << "F32 MatMul + ReLU: " << duration_f32.count() << " ms" << std::endl;
    std::cout << "Q8_0 MatMul + ReLU: " << duration_q8.count() << " ms ";
    std::cout << "(Speedup: " << static_cast<float>(duration_f32.count()) / duration_q8.count() << "x)" << std::endl;
    std::cout << "Q4_0 MatMul + ReLU: " << duration_q4_0.count() << " ms ";
    std::cout << "(Speedup: " << static_cast<float>(duration_f32.count()) / duration_q4_0.count() << "x)" << std::endl;
    std::cout << "Q4_1 MatMul + ReLU: " << duration_q4_1.count() << " ms ";
    std::cout << "(Speedup: " << static_cast<float>(duration_f32.count()) / duration_q4_1.count() << "x)" << std::endl;
    
    // Calculate and print error metrics
    std::vector<float> f32_data(large_m * large_n);
    std::vector<float> q8_data(large_m * large_n);
    std::vector<float> q4_0_data(large_m * large_n);
    std::vector<float> q4_1_data(large_m * large_n);
    
    std::memcpy(f32_data.data(), result_f32.data(), large_m * large_n * sizeof(float));
    std::memcpy(q8_data.data(), result_q8.data(), large_m * large_n * sizeof(float));
    std::memcpy(q4_0_data.data(), result_q4_0.data(), large_m * large_n * sizeof(float));
    std::memcpy(q4_1_data.data(), result_q4_1.data(), large_m * large_n * sizeof(float));
    
    // Calculate error metrics for Q8_0
    double q8_max_error = 0.0;
    double q8_sum_squared_error = 0.0;
    
    for (size_t i = 0; i < large_m * large_n; i++) {
        double error = std::abs(f32_data[i] - q8_data[i]);
        q8_max_error = std::max(q8_max_error, error);
        q8_sum_squared_error += error * error;
    }
    
    double q8_rmse = std::sqrt(q8_sum_squared_error / (large_m * large_n));
    
    // Calculate error metrics for Q4_0
    double q4_0_max_error = 0.0;
    double q4_0_sum_squared_error = 0.0;
    
    for (size_t i = 0; i < large_m * large_n; i++) {
        double error = std::abs(f32_data[i] - q4_0_data[i]);
        q4_0_max_error = std::max(q4_0_max_error, error);
        q4_0_sum_squared_error += error * error;
    }
    
    double q4_0_rmse = std::sqrt(q4_0_sum_squared_error / (large_m * large_n));
    
    // Calculate error metrics for Q4_1
    double q4_1_max_error = 0.0;
    double q4_1_sum_squared_error = 0.0;
    
    for (size_t i = 0; i < large_m * large_n; i++) {
        double error = std::abs(f32_data[i] - q4_1_data[i]);
        q4_1_max_error = std::max(q4_1_max_error, error);
        q4_1_sum_squared_error += error * error;
    }
    
    double q4_1_rmse = std::sqrt(q4_1_sum_squared_error / (large_m * large_n));
    
    // Print error metrics
    std::cout << "Q8_0 Error Metrics - Max Error: " << q8_max_error << ", RMSE: " << q8_rmse << std::endl;
    std::cout << "Q4_0 Error Metrics - Max Error: " << q4_0_max_error << ", RMSE: " << q4_0_rmse << std::endl;
    std::cout << "Q4_1 Error Metrics - Max Error: " << q4_1_max_error << ", RMSE: " << q4_1_rmse << std::endl;
}

// Test quantization compatibility with other tensor operations
TEST_F(GGMLFusedQuantizedTest, QuantizationCompatibility) {
    // Test element-wise operations with quantized tensors
    
    // Create a vector tensor
    std::vector<float> vec_data(n);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (size_t i = 0; i < n; i++) {
        vec_data[i] = dist(gen);
    }
    
    Tensor vec_tensor = ggml_ctx->create_tensor({n}, DataType::F32);
    std::memcpy(vec_tensor.data(), vec_data.data(), n * sizeof(float));
    
    // Quantize vector
    Tensor vec_q8_0 = ggml_ctx->cast(vec_tensor, DataType::Q8_0);
    
    // Perform matrix multiplication first
    Tensor matmul_result = ggml_ctx->matmul(tensor_a, tensor_b);
    Tensor q8_matmul_result = ggml_ctx->matmul(tensor_a, tensor_b_q8_0);
    
    // Extract first row from results
    Tensor matmul_row = matmul_result.slice(0, 0, 1);
    Tensor q8_matmul_row = q8_matmul_result.slice(0, 0, 1);
    
    // Reshape to 1D to apply element-wise operations
    std::vector<size_t> vec_shape = {n};
    Tensor reshaped_row = matmul_row.reshape(vec_shape);
    Tensor reshaped_q8_row = q8_matmul_row.reshape(vec_shape);
    
    // Test element-wise operations with the quantized results
    
    // 1. Add
    Tensor add_result = ggml_ctx->add(reshaped_row, vec_tensor);
    Tensor add_q8_result = ggml_ctx->add(reshaped_q8_row, vec_tensor);
    
    // 2. Multiply
    Tensor mul_result = ggml_ctx->multiply(reshaped_row, vec_tensor);
    Tensor mul_q8_result = ggml_ctx->multiply(reshaped_q8_row, vec_tensor);
    
    // 3. SoftMax
    Tensor softmax_result = ggml_ctx->softmax(reshaped_row, 0);
    Tensor softmax_q8_result = ggml_ctx->softmax(reshaped_q8_row, 0);
    
    // Convert to vectors for comparison
    std::vector<float> add_data(n);
    std::vector<float> add_q8_data(n);
    std::vector<float> mul_data(n);
    std::vector<float> mul_q8_data(n);
    std::vector<float> softmax_data(n);
    std::vector<float> softmax_q8_data(n);
    
    std::memcpy(add_data.data(), add_result.data(), n * sizeof(float));
    std::memcpy(add_q8_data.data(), add_q8_result.data(), n * sizeof(float));
    std::memcpy(mul_data.data(), mul_result.data(), n * sizeof(float));
    std::memcpy(mul_q8_data.data(), mul_q8_result.data(), n * sizeof(float));
    std::memcpy(softmax_data.data(), softmax_result.data(), n * sizeof(float));
    std::memcpy(softmax_q8_data.data(), softmax_q8_result.data(), n * sizeof(float));
    
    // Calculate error metrics for add
    double add_max_error = 0.0;
    double add_sum_squared_error = 0.0;
    
    for (size_t i = 0; i < n; i++) {
        double error = std::abs(add_data[i] - add_q8_data[i]);
        add_max_error = std::max(add_max_error, error);
        add_sum_squared_error += error * error;
    }
    
    double add_rmse = std::sqrt(add_sum_squared_error / n);
    
    // Calculate error metrics for multiply
    double mul_max_error = 0.0;
    double mul_sum_squared_error = 0.0;
    
    for (size_t i = 0; i < n; i++) {
        double error = std::abs(mul_data[i] - mul_q8_data[i]);
        mul_max_error = std::max(mul_max_error, error);
        mul_sum_squared_error += error * error;
    }
    
    double mul_rmse = std::sqrt(mul_sum_squared_error / n);
    
    // Calculate error metrics for softmax
    double softmax_max_error = 0.0;
    double softmax_sum_squared_error = 0.0;
    
    for (size_t i = 0; i < n; i++) {
        double error = std::abs(softmax_data[i] - softmax_q8_data[i]);
        softmax_max_error = std::max(softmax_max_error, error);
        softmax_sum_squared_error += error * error;
    }
    
    double softmax_rmse = std::sqrt(softmax_sum_squared_error / n);
    
    // Print error metrics
    std::cout << "Add with Q8_0 - Max Error: " << add_max_error << ", RMSE: " << add_rmse << std::endl;
    std::cout << "Multiply with Q8_0 - Max Error: " << mul_max_error << ", RMSE: " << mul_rmse << std::endl;
    std::cout << "SoftMax with Q8_0 - Max Error: " << softmax_max_error << ", RMSE: " << softmax_rmse << std::endl;
    
    // Verify errors are within acceptable limits
    EXPECT_LT(add_max_error, 1.0);
    EXPECT_LT(add_rmse, 0.2);
    
    EXPECT_LT(mul_max_error, 1.0);
    EXPECT_LT(mul_rmse, 0.2);
    
    EXPECT_LT(softmax_max_error, 0.1);
    EXPECT_LT(softmax_rmse, 0.01);
    
    // Verify softmax properties
    float softmax_sum = 0.0f;
    float softmax_q8_sum = 0.0f;
    
    for (size_t i = 0; i < n; i++) {
        softmax_sum += softmax_data[i];
        softmax_q8_sum += softmax_q8_data[i];
        EXPECT_GE(softmax_data[i], 0.0f);
        EXPECT_GE(softmax_q8_data[i], 0.0f);
        EXPECT_LE(softmax_data[i], 1.0f);
        EXPECT_LE(softmax_q8_data[i], 1.0f);
    }
    
    EXPECT_NEAR(softmax_sum, 1.0f, 0.01f);
    EXPECT_NEAR(softmax_q8_sum, 1.0f, 0.01f);
}

} // namespace
} // namespace ccsm