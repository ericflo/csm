#include <ccsm/cpu/ggml_model.h>
#include <ccsm/cpu/ggml_tensor.h>
#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <cmath>

namespace ccsm {
namespace {

class GGMLQuantizationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a GGML context and tensor factory
        ggml_ctx = std::make_shared<GGMLContext>();
        
        // Initialize random number generator
        std::mt19937 gen(42); // Fixed seed for reproducibility
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        
        // Generate random test data
        data.resize(test_size);
        for (size_t i = 0; i < test_size; i++) {
            data[i] = dist(gen);
        }
        
        // Create a tensor with the test data
        std::vector<size_t> shape = {test_size};
        original_tensor = ggml_ctx->create_tensor(shape, DataType::F32);
        std::memcpy(original_tensor.data(), data.data(), test_size * sizeof(float));
    }
    
    // Test data
    static constexpr size_t test_size = 1024;
    std::vector<float> data;
    
    // Tensor objects
    Tensor original_tensor;
    
    // GGML context
    std::shared_ptr<GGMLContext> ggml_ctx;
};

// Test Q8_0 quantization/dequantization in GGML
TEST_F(GGMLQuantizationTest, TestQ8_0Quantization) {
    // Create a Q8_0 tensor by casting from F32
    Tensor q8_tensor = ggml_ctx->cast(original_tensor, DataType::Q8_0);
    
    // Verify the dtype
    EXPECT_EQ(q8_tensor.dtype(), DataType::Q8_0);
    
    // Cast back to F32 for comparison
    Tensor dequantized = ggml_ctx->cast(q8_tensor, DataType::F32);
    
    // Get the data from the dequantized tensor
    std::vector<float> result_data(test_size);
    std::memcpy(result_data.data(), dequantized.data(), test_size * sizeof(float));
    
    // Verify the roundtrip: not exact due to quantization, but should be close
    double max_abs_error = 0.0;
    double sum_squared_error = 0.0;
    
    for (size_t i = 0; i < test_size; i++) {
        double error = std::abs(data[i] - result_data[i]);
        max_abs_error = std::max(max_abs_error, error);
        sum_squared_error += error * error;
    }
    
    double rmse = std::sqrt(sum_squared_error / test_size);
    
    // Check that errors are within acceptable limits
    // For 8-bit quantization, we expect some error but not too much
    EXPECT_LT(max_abs_error, 0.1); // Max absolute error < 0.1
    EXPECT_LT(rmse, 0.02);         // Root mean square error < 0.02
}

// Test Q4_0 quantization/dequantization in GGML
TEST_F(GGMLQuantizationTest, TestQ4_0Quantization) {
    // Create a Q4_0 tensor by casting from F32
    Tensor q4_tensor = ggml_ctx->cast(original_tensor, DataType::Q4_0);
    
    // Verify the dtype
    EXPECT_EQ(q4_tensor.dtype(), DataType::Q4_0);
    
    // Cast back to F32 for comparison
    Tensor dequantized = ggml_ctx->cast(q4_tensor, DataType::F32);
    
    // Get the data from the dequantized tensor
    std::vector<float> result_data(test_size);
    std::memcpy(result_data.data(), dequantized.data(), test_size * sizeof(float));
    
    // Verify the roundtrip: not exact due to quantization, but should be close
    double max_abs_error = 0.0;
    double sum_squared_error = 0.0;
    
    for (size_t i = 0; i < test_size; i++) {
        double error = std::abs(data[i] - result_data[i]);
        max_abs_error = std::max(max_abs_error, error);
        sum_squared_error += error * error;
    }
    
    double rmse = std::sqrt(sum_squared_error / test_size);
    
    // Check that errors are within acceptable limits
    // For 4-bit quantization, we expect more error than with 8-bit
    EXPECT_LT(max_abs_error, 0.3); // Max absolute error < 0.3
    EXPECT_LT(rmse, 0.1);         // Root mean square error < 0.1
}

// Test Q4_1 quantization/dequantization in GGML
TEST_F(GGMLQuantizationTest, TestQ4_1Quantization) {
    // Create a Q4_1 tensor by casting from F32
    Tensor q4_tensor = ggml_ctx->cast(original_tensor, DataType::Q4_1);
    
    // Verify the dtype
    EXPECT_EQ(q4_tensor.dtype(), DataType::Q4_1);
    
    // Cast back to F32 for comparison
    Tensor dequantized = ggml_ctx->cast(q4_tensor, DataType::F32);
    
    // Get the data from the dequantized tensor
    std::vector<float> result_data(test_size);
    std::memcpy(result_data.data(), dequantized.data(), test_size * sizeof(float));
    
    // Verify the roundtrip: not exact due to quantization, but should be close
    double max_abs_error = 0.0;
    double sum_squared_error = 0.0;
    
    for (size_t i = 0; i < test_size; i++) {
        double error = std::abs(data[i] - result_data[i]);
        max_abs_error = std::max(max_abs_error, error);
        sum_squared_error += error * error;
    }
    
    double rmse = std::sqrt(sum_squared_error / test_size);
    
    // Check that errors are within acceptable limits
    // For 4-bit quantization with bias, we expect better results than Q4_0
    EXPECT_LT(max_abs_error, 0.2); // Max absolute error < 0.2
    EXPECT_LT(rmse, 0.05);         // Root mean square error < 0.05
}

// Test GGML matrix multiplication with quantized weights
TEST_F(GGMLQuantizationTest, TestMatMulQ8_0) {
    // Create two random matrices
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    // Matrix dimensions
    size_t m = 16;
    size_t k = 32;
    size_t n = 24;
    
    // Create data for matrices
    std::vector<float> a_data(m * k);
    std::vector<float> b_data(k * n);
    
    // Fill with random data
    for (size_t i = 0; i < m * k; i++) {
        a_data[i] = dist(gen);
    }
    for (size_t i = 0; i < k * n; i++) {
        b_data[i] = dist(gen);
    }
    
    // Create GGML tensors
    Tensor a_tensor = ggml_ctx->create_tensor({m, k}, DataType::F32);
    Tensor b_tensor = ggml_ctx->create_tensor({k, n}, DataType::F32);
    
    // Copy data to tensors
    std::memcpy(a_tensor.data(), a_data.data(), m * k * sizeof(float));
    std::memcpy(b_tensor.data(), b_data.data(), k * n * sizeof(float));
    
    // Compute reference result with standard matmul
    Tensor ref_result = ggml_ctx->matmul(a_tensor, b_tensor);
    
    // Quantize the second matrix to Q8_0
    Tensor b_q8 = ggml_ctx->cast(b_tensor, DataType::Q8_0);
    
    // Matmul with quantized weights
    Tensor q_result = ggml_ctx->matmul(a_tensor, b_q8);
    
    // Extract results for comparison
    std::vector<float> ref_data(m * n);
    std::vector<float> q_data(m * n);
    
    std::memcpy(ref_data.data(), ref_result.data(), m * n * sizeof(float));
    std::memcpy(q_data.data(), q_result.data(), m * n * sizeof(float));
    
    // Calculate error metrics
    double max_abs_error = 0.0;
    double sum_squared_error = 0.0;
    
    for (size_t i = 0; i < m * n; i++) {
        double error = std::abs(ref_data[i] - q_data[i]);
        max_abs_error = std::max(max_abs_error, error);
        sum_squared_error += error * error;
    }
    
    double rmse = std::sqrt(sum_squared_error / (m * n));
    
    // Check that errors are within acceptable limits
    EXPECT_LT(max_abs_error, 1.0); // Max absolute error < 1.0
    EXPECT_LT(rmse, 0.2);          // Root mean square error < 0.2
}

// Test GGML matrix multiplication with Q4_0 quantized weights
TEST_F(GGMLQuantizationTest, TestMatMulQ4_0) {
    // Create two random matrices
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    // Matrix dimensions
    size_t m = 16;
    size_t k = 32;
    size_t n = 24;
    
    // Create data for matrices
    std::vector<float> a_data(m * k);
    std::vector<float> b_data(k * n);
    
    // Fill with random data
    for (size_t i = 0; i < m * k; i++) {
        a_data[i] = dist(gen);
    }
    for (size_t i = 0; i < k * n; i++) {
        b_data[i] = dist(gen);
    }
    
    // Create GGML tensors
    Tensor a_tensor = ggml_ctx->create_tensor({m, k}, DataType::F32);
    Tensor b_tensor = ggml_ctx->create_tensor({k, n}, DataType::F32);
    
    // Copy data to tensors
    std::memcpy(a_tensor.data(), a_data.data(), m * k * sizeof(float));
    std::memcpy(b_tensor.data(), b_data.data(), k * n * sizeof(float));
    
    // Compute reference result with standard matmul
    Tensor ref_result = ggml_ctx->matmul(a_tensor, b_tensor);
    
    // Quantize the second matrix to Q4_0
    Tensor b_q4 = ggml_ctx->cast(b_tensor, DataType::Q4_0);
    
    // Matmul with quantized weights
    Tensor q_result = ggml_ctx->matmul(a_tensor, b_q4);
    
    // Extract results for comparison
    std::vector<float> ref_data(m * n);
    std::vector<float> q_data(m * n);
    
    std::memcpy(ref_data.data(), ref_result.data(), m * n * sizeof(float));
    std::memcpy(q_data.data(), q_result.data(), m * n * sizeof(float));
    
    // Calculate error metrics
    double max_abs_error = 0.0;
    double sum_squared_error = 0.0;
    
    for (size_t i = 0; i < m * n; i++) {
        double error = std::abs(ref_data[i] - q_data[i]);
        max_abs_error = std::max(max_abs_error, error);
        sum_squared_error += error * error;
    }
    
    double rmse = std::sqrt(sum_squared_error / (m * n));
    
    // Check that errors are within acceptable limits
    // Q4_0 has higher error than Q8_0
    EXPECT_LT(max_abs_error, 1.5); // Max absolute error < 1.5
    EXPECT_LT(rmse, 0.3);          // Root mean square error < 0.3
}

} // namespace
} // namespace ccsm