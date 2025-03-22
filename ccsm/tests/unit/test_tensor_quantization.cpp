#include <ccsm/tensor.h>
#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <cmath>

namespace ccsm {
namespace {

class TensorQuantizationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a tensor factory and context
        ctx = ContextFactory::create();
        
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
        original_tensor = TensorFactory::from_data(data.data(), shape, DataType::F32);
    }
    
    // Test data
    static constexpr size_t test_size = 1024;
    std::vector<float> data;
    
    // Tensor objects
    Tensor original_tensor;
    
    // Context for operations
    std::shared_ptr<Context> ctx;
};

// Test Q8_0 quantization/dequantization via type casting
TEST_F(TensorQuantizationTest, TestQ8_0Quantization) {
    // Create a Q8_0 tensor by casting from F32
    Tensor q8_tensor = ctx->cast(original_tensor, DataType::Q8_0);
    
    // Verify the dtype
    EXPECT_EQ(q8_tensor.dtype(), DataType::Q8_0);
    
    // Cast back to F32 for comparison
    Tensor dequantized = ctx->cast(q8_tensor, DataType::F32);
    
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

// Test Q4_0 quantization/dequantization via type casting
TEST_F(TensorQuantizationTest, TestQ4_0Quantization) {
    // Create a Q4_0 tensor by casting from F32
    Tensor q4_tensor = ctx->cast(original_tensor, DataType::Q4_0);
    
    // Verify the dtype
    EXPECT_EQ(q4_tensor.dtype(), DataType::Q4_0);
    
    // Cast back to F32 for comparison
    Tensor dequantized = ctx->cast(q4_tensor, DataType::F32);
    
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

// Test Q4_1 quantization/dequantization via type casting
TEST_F(TensorQuantizationTest, TestQ4_1Quantization) {
    // Create a Q4_1 tensor by casting from F32
    Tensor q4_tensor = ctx->cast(original_tensor, DataType::Q4_1);
    
    // Verify the dtype
    EXPECT_EQ(q4_tensor.dtype(), DataType::Q4_1);
    
    // Cast back to F32 for comparison
    Tensor dequantized = ctx->cast(q4_tensor, DataType::F32);
    
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

// Test quantized matrix multiplication
TEST_F(TensorQuantizationTest, TestMatMulQ8_0) {
    // Create two random matrices
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    // Matrix dimensions
    size_t m = 16;
    size_t k = 32;
    size_t n = 24;
    
    // Create matrices
    std::vector<float> a_data(m * k);
    std::vector<float> b_data(k * n);
    
    // Fill with random data
    for (size_t i = 0; i < m * k; i++) {
        a_data[i] = dist(gen);
    }
    for (size_t i = 0; i < k * n; i++) {
        b_data[i] = dist(gen);
    }
    
    // Create tensors
    Tensor a_tensor = TensorFactory::from_data(a_data.data(), {m, k}, DataType::F32);
    Tensor b_tensor = TensorFactory::from_data(b_data.data(), {k, n}, DataType::F32);
    
    // Compute reference result with standard matmul
    Tensor ref_result = ctx->matmul(a_tensor, b_tensor);
    
    // Quantize the second matrix
    Tensor b_q8 = ctx->cast(b_tensor, DataType::Q8_0);
    
    // Matmul with quantized weights should use the appropriate implementation
    Tensor q_result = ctx->matmul(a_tensor, b_q8);
    
    // Cast back to F32 if needed
    if (q_result.dtype() != DataType::F32) {
        q_result = ctx->cast(q_result, DataType::F32);
    }
    
    // Compare results
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

} // namespace
} // namespace ccsm