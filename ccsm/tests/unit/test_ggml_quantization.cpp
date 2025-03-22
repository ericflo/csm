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

// Test quantization edge cases
TEST_F(GGMLQuantizationTest, EdgeCases) {
    // Test quantization with all zeros
    std::vector<float> zeros(test_size, 0.0f);
    
    // Create a tensor with zeros
    Tensor zeros_tensor = ggml_ctx->create_tensor({test_size}, DataType::F32);
    std::memcpy(zeros_tensor.data(), zeros.data(), test_size * sizeof(float));
    
    // Quantize to different formats
    Tensor zeros_q8 = ggml_ctx->cast(zeros_tensor, DataType::Q8_0);
    Tensor zeros_q4_0 = ggml_ctx->cast(zeros_tensor, DataType::Q4_0);
    Tensor zeros_q4_1 = ggml_ctx->cast(zeros_tensor, DataType::Q4_1);
    
    // Cast back to F32 for verification
    Tensor dequant_q8 = ggml_ctx->cast(zeros_q8, DataType::F32);
    Tensor dequant_q4_0 = ggml_ctx->cast(zeros_q4_0, DataType::F32);
    Tensor dequant_q4_1 = ggml_ctx->cast(zeros_q4_1, DataType::F32);
    
    // Verify all values are still close to zero
    const float* data_q8 = static_cast<const float*>(dequant_q8.data());
    const float* data_q4_0 = static_cast<const float*>(dequant_q4_0.data());
    const float* data_q4_1 = static_cast<const float*>(dequant_q4_1.data());
    
    for (size_t i = 0; i < test_size; i++) {
        EXPECT_NEAR(data_q8[i], 0.0f, 0.01f);
        EXPECT_NEAR(data_q4_0[i], 0.0f, 0.01f);
        EXPECT_NEAR(data_q4_1[i], 0.0f, 0.01f);
    }
    
    // Test with extreme values
    std::vector<float> extremes(test_size);
    for (size_t i = 0; i < test_size; i++) {
        extremes[i] = (i % 2 == 0) ? 1e6f : -1e6f;
    }
    
    // Create a tensor with extreme values
    Tensor extremes_tensor = ggml_ctx->create_tensor({test_size}, DataType::F32);
    std::memcpy(extremes_tensor.data(), extremes.data(), test_size * sizeof(float));
    
    // Quantize to different formats
    Tensor extremes_q8 = ggml_ctx->cast(extremes_tensor, DataType::Q8_0);
    Tensor extremes_q4_0 = ggml_ctx->cast(extremes_tensor, DataType::Q4_0);
    Tensor extremes_q4_1 = ggml_ctx->cast(extremes_tensor, DataType::Q4_1);
    
    // Cast back to F32 for verification - we mainly want to ensure no crashes
    Tensor dequant_extremes_q8 = ggml_ctx->cast(extremes_q8, DataType::F32);
    Tensor dequant_extremes_q4_0 = ggml_ctx->cast(extremes_q4_0, DataType::F32);
    Tensor dequant_extremes_q4_1 = ggml_ctx->cast(extremes_q4_1, DataType::F32);
    
    // Test with a mix of large and small values
    std::vector<float> mixed(test_size);
    for (size_t i = 0; i < test_size; i++) {
        mixed[i] = (i % 4 == 0) ? 1e6f : 
                   (i % 4 == 1) ? 1e-6f : 
                   (i % 4 == 2) ? -1e6f : -1e-6f;
    }
    
    // Create a tensor with mixed values
    Tensor mixed_tensor = ggml_ctx->create_tensor({test_size}, DataType::F32);
    std::memcpy(mixed_tensor.data(), mixed.data(), test_size * sizeof(float));
    
    // Quantize to different formats
    Tensor mixed_q8 = ggml_ctx->cast(mixed_tensor, DataType::Q8_0);
    Tensor mixed_q4_0 = ggml_ctx->cast(mixed_tensor, DataType::Q4_0);
    Tensor mixed_q4_1 = ggml_ctx->cast(mixed_tensor, DataType::Q4_1);
    
    // Cast back to F32 - we mainly want to ensure no crashes
    Tensor dequant_mixed_q8 = ggml_ctx->cast(mixed_q8, DataType::F32);
    Tensor dequant_mixed_q4_0 = ggml_ctx->cast(mixed_q4_0, DataType::F32);
    Tensor dequant_mixed_q4_1 = ggml_ctx->cast(mixed_q4_1, DataType::F32);
}

// Benchmark comparing quantized vs. non-quantized operations
TEST_F(GGMLQuantizationTest, DISABLED_QuantizationBenchmark) {
    // Create larger matrices for benchmarking
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    // Matrix dimensions
    size_t m = 128;
    size_t k = 256;
    size_t n = 128;
    
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
    
    // Run F32 matmul and measure time
    auto start_f32 = std::chrono::high_resolution_clock::now();
    Tensor result_f32 = ggml_ctx->matmul(a_tensor, b_tensor);
    auto end_f32 = std::chrono::high_resolution_clock::now();
    auto duration_f32 = std::chrono::duration_cast<std::chrono::milliseconds>(end_f32 - start_f32);
    
    // Quantize b to Q8_0
    Tensor b_q8 = ggml_ctx->cast(b_tensor, DataType::Q8_0);
    
    // Run Q8_0 matmul and measure time
    auto start_q8 = std::chrono::high_resolution_clock::now();
    Tensor result_q8 = ggml_ctx->matmul(a_tensor, b_q8);
    auto end_q8 = std::chrono::high_resolution_clock::now();
    auto duration_q8 = std::chrono::duration_cast<std::chrono::milliseconds>(end_q8 - start_q8);
    
    // Quantize b to Q4_0
    Tensor b_q4 = ggml_ctx->cast(b_tensor, DataType::Q4_0);
    
    // Run Q4_0 matmul and measure time
    auto start_q4 = std::chrono::high_resolution_clock::now();
    Tensor result_q4 = ggml_ctx->matmul(a_tensor, b_q4);
    auto end_q4 = std::chrono::high_resolution_clock::now();
    auto duration_q4 = std::chrono::duration_cast<std::chrono::milliseconds>(end_q4 - start_q4);
    
    // Compare dimensions
    EXPECT_EQ(result_f32.shape(0), m);
    EXPECT_EQ(result_f32.shape(1), n);
    EXPECT_EQ(result_q8.shape(0), m);
    EXPECT_EQ(result_q8.shape(1), n);
    EXPECT_EQ(result_q4.shape(0), m);
    EXPECT_EQ(result_q4.shape(1), n);
    
    // Extract data for error analysis
    std::vector<float> f32_data(m * n);
    std::vector<float> q8_data(m * n);
    std::vector<float> q4_data(m * n);
    
    std::memcpy(f32_data.data(), result_f32.data(), m * n * sizeof(float));
    std::memcpy(q8_data.data(), result_q8.data(), m * n * sizeof(float));
    std::memcpy(q4_data.data(), result_q4.data(), m * n * sizeof(float));
    
    // Calculate errors
    double q8_max_error = 0.0;
    double q8_sum_squared_error = 0.0;
    double q4_max_error = 0.0;
    double q4_sum_squared_error = 0.0;
    
    for (size_t i = 0; i < m * n; i++) {
        double q8_error = std::abs(f32_data[i] - q8_data[i]);
        double q4_error = std::abs(f32_data[i] - q4_data[i]);
        
        q8_max_error = std::max(q8_max_error, q8_error);
        q8_sum_squared_error += q8_error * q8_error;
        
        q4_max_error = std::max(q4_max_error, q4_error);
        q4_sum_squared_error += q4_error * q4_error;
    }
    
    double q8_rmse = std::sqrt(q8_sum_squared_error / (m * n));
    double q4_rmse = std::sqrt(q4_sum_squared_error / (m * n));
    
    // Print results
    std::cout << "F32 MatMul Time: " << duration_f32.count() << " ms" << std::endl;
    std::cout << "Q8_0 MatMul Time: " << duration_q8.count() << " ms (speedup: " 
              << static_cast<float>(duration_f32.count()) / duration_q8.count() << "x)" << std::endl;
    std::cout << "Q8_0 Max Error: " << q8_max_error << ", RMSE: " << q8_rmse << std::endl;
    
    std::cout << "Q4_0 MatMul Time: " << duration_q4.count() << " ms (speedup: " 
              << static_cast<float>(duration_f32.count()) / duration_q4.count() << "x)" << std::endl;
    std::cout << "Q4_0 Max Error: " << q4_max_error << ", RMSE: " << q4_rmse << std::endl;
}

// Test multi-dimensional tensor quantization
TEST_F(GGMLQuantizationTest, MultiDimensionalQuantization) {
    // Create a 3D tensor with known data
    std::vector<size_t> tensor_shape = {4, 5, 6};
    size_t tensor_size = 4 * 5 * 6;
    
    // Fill with pattern
    std::vector<float> tensor_data(tensor_size);
    for (size_t i = 0; i < tensor_size; i++) {
        tensor_data[i] = static_cast<float>(i) / tensor_size;
    }
    
    // Create tensor
    Tensor original = ggml_ctx->create_tensor(tensor_shape, DataType::F32);
    std::memcpy(original.data(), tensor_data.data(), tensor_size * sizeof(float));
    
    // Quantize to different formats
    Tensor q8 = ggml_ctx->cast(original, DataType::Q8_0);
    Tensor q4_0 = ggml_ctx->cast(original, DataType::Q4_0);
    Tensor q4_1 = ggml_ctx->cast(original, DataType::Q4_1);
    
    // Verify dtypes
    EXPECT_EQ(q8.dtype(), DataType::Q8_0);
    EXPECT_EQ(q4_0.dtype(), DataType::Q4_0);
    EXPECT_EQ(q4_1.dtype(), DataType::Q4_1);
    
    // Verify shapes
    EXPECT_EQ(q8.shape(), tensor_shape);
    EXPECT_EQ(q4_0.shape(), tensor_shape);
    EXPECT_EQ(q4_1.shape(), tensor_shape);
    
    // Cast back to F32 for verification
    Tensor dequant_q8 = ggml_ctx->cast(q8, DataType::F32);
    Tensor dequant_q4_0 = ggml_ctx->cast(q4_0, DataType::F32);
    Tensor dequant_q4_1 = ggml_ctx->cast(q4_1, DataType::F32);
    
    // Get data for comparison
    std::vector<float> result_q8(tensor_size);
    std::vector<float> result_q4_0(tensor_size);
    std::vector<float> result_q4_1(tensor_size);
    
    std::memcpy(result_q8.data(), dequant_q8.data(), tensor_size * sizeof(float));
    std::memcpy(result_q4_0.data(), dequant_q4_0.data(), tensor_size * sizeof(float));
    std::memcpy(result_q4_1.data(), dequant_q4_1.data(), tensor_size * sizeof(float));
    
    // Calculate errors
    double q8_max_error = 0.0;
    double q8_sum_squared_error = 0.0;
    double q4_0_max_error = 0.0;
    double q4_0_sum_squared_error = 0.0;
    double q4_1_max_error = 0.0;
    double q4_1_sum_squared_error = 0.0;
    
    for (size_t i = 0; i < tensor_size; i++) {
        double q8_error = std::abs(tensor_data[i] - result_q8[i]);
        double q4_0_error = std::abs(tensor_data[i] - result_q4_0[i]);
        double q4_1_error = std::abs(tensor_data[i] - result_q4_1[i]);
        
        q8_max_error = std::max(q8_max_error, q8_error);
        q8_sum_squared_error += q8_error * q8_error;
        
        q4_0_max_error = std::max(q4_0_max_error, q4_0_error);
        q4_0_sum_squared_error += q4_0_error * q4_0_error;
        
        q4_1_max_error = std::max(q4_1_max_error, q4_1_error);
        q4_1_sum_squared_error += q4_1_error * q4_1_error;
    }
    
    double q8_rmse = std::sqrt(q8_sum_squared_error / tensor_size);
    double q4_0_rmse = std::sqrt(q4_0_sum_squared_error / tensor_size);
    double q4_1_rmse = std::sqrt(q4_1_sum_squared_error / tensor_size);
    
    // Verify error limits
    EXPECT_LT(q8_max_error, 0.1);
    EXPECT_LT(q8_rmse, 0.02);
    
    EXPECT_LT(q4_0_max_error, 0.3);
    EXPECT_LT(q4_0_rmse, 0.1);
    
    EXPECT_LT(q4_1_max_error, 0.2);
    EXPECT_LT(q4_1_rmse, 0.05);
}

// Test quantization with type promotion
TEST_F(GGMLQuantizationTest, QuantizationWithTypePromotion) {
    // Create a random tensor
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    std::vector<float> tensor_data(test_size);
    for (size_t i = 0; i < test_size; i++) {
        tensor_data[i] = dist(gen);
    }
    
    // Create tensor
    Tensor original = ggml_ctx->create_tensor({test_size}, DataType::F32);
    std::memcpy(original.data(), tensor_data.data(), test_size * sizeof(float));
    
    // Quantize to different formats
    Tensor q8 = ggml_ctx->cast(original, DataType::Q8_0);
    Tensor q4_0 = ggml_ctx->cast(original, DataType::Q4_0);
    Tensor q4_1 = ggml_ctx->cast(original, DataType::Q4_1);
    
    // Test tensor operations with mixed types
    
    // 1. Add F32 and Q8_0
    Tensor result1 = ggml_ctx->add(original, q8);
    EXPECT_EQ(result1.dtype(), DataType::F32); // Should promote to F32
    
    // 2. Multiply Q8_0 and Q4_0
    Tensor result2 = ggml_ctx->multiply(q8, q4_0);
    EXPECT_EQ(result2.dtype(), DataType::F32); // Should promote to F32
    
    // 3. MatMul F32 and Q4_1
    // Create matrices for matmul
    Tensor mat1 = ggml_ctx->create_tensor({10, 20}, DataType::F32);
    Tensor mat2 = ggml_ctx->create_tensor({20, 5}, DataType::F32);
    
    // Fill with data
    std::vector<float> mat1_data(10 * 20);
    std::vector<float> mat2_data(20 * 5);
    
    for (size_t i = 0; i < 10 * 20; i++) {
        mat1_data[i] = dist(gen);
    }
    for (size_t i = 0; i < 20 * 5; i++) {
        mat2_data[i] = dist(gen);
    }
    
    std::memcpy(mat1.data(), mat1_data.data(), 10 * 20 * sizeof(float));
    std::memcpy(mat2.data(), mat2_data.data(), 20 * 5 * sizeof(float));
    
    // Quantize mat2
    Tensor mat2_q4 = ggml_ctx->cast(mat2, DataType::Q4_1);
    
    // Matrix multiplication
    Tensor result3 = ggml_ctx->matmul(mat1, mat2_q4);
    EXPECT_EQ(result3.dtype(), DataType::F32); // Should promote to F32
    EXPECT_EQ(result3.shape(0), 10);
    EXPECT_EQ(result3.shape(1), 5);
}

} // namespace
} // namespace ccsm