#include <ccsm/cpu/simd.h>
#include <gtest/gtest.h>
#include <random>
#include <vector>
#include <cmath>
#include <chrono>

namespace ccsm {
namespace simd {
namespace {

// Test fixture for quantization tests
class QuantizationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize random number generator
        std::mt19937 gen(42); // Fixed seed for reproducibility
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        
        // Generate random test data
        test_data.resize(test_size);
        for (size_t i = 0; i < test_size; i++) {
            test_data[i] = dist(gen);
        }
    }
    
    // Test data
    static constexpr size_t test_size = 1024;
    std::vector<float> test_data;
};

// Test Q8_0 quantization/dequantization
TEST_F(QuantizationTest, TestQ8_0_Roundtrip) {
    // Allocate memory for quantized data
    // Note: we add space for the scale factor at the end
    std::vector<int8_t> quantized_data(test_size + sizeof(float) / sizeof(int8_t));
    std::vector<float> dequantized_data(test_size);
    
    // Perform quantization
    quantize_q8_0<float>(quantized_data.data(), test_data.data(), test_size);
    
    // Get the scale factor
    const float* scale = reinterpret_cast<const float*>(quantized_data.data() + test_size);
    
    // Perform dequantization
    dequantize_q8_0<float>(dequantized_data.data(), quantized_data.data(), scale, test_size);
    
    // Verify the roundtrip: not exact due to quantization, but should be close
    double max_abs_error = 0.0;
    double sum_squared_error = 0.0;
    
    for (size_t i = 0; i < test_size; i++) {
        double error = std::abs(test_data[i] - dequantized_data[i]);
        max_abs_error = std::max(max_abs_error, error);
        sum_squared_error += error * error;
    }
    
    double rmse = std::sqrt(sum_squared_error / test_size);
    
    // Check that errors are within acceptable limits
    // For 8-bit quantization, we expect some error but not too much
    EXPECT_LT(max_abs_error, 0.1); // Max absolute error < 0.1
    EXPECT_LT(rmse, 0.02);        // Root mean square error < 0.02
    
    // Verify that quantized values are in the expected range [-127, 127]
    int8_t min_quantized = 127;
    int8_t max_quantized = -127;
    
    for (size_t i = 0; i < test_size; i++) {
        min_quantized = std::min(min_quantized, quantized_data[i]);
        max_quantized = std::max(max_quantized, quantized_data[i]);
    }
    
    // We should have at least some values reaching the extremes
    // of our quantization range
    EXPECT_LE(min_quantized, -120);
    EXPECT_GE(max_quantized, 120);
}

// Test matrix multiplication with Q8_0 quantized weights
TEST_F(QuantizationTest, TestMatrixMultiplyQ8_0) {
    // Create random matrices for testing
    constexpr size_t m = 16;
    constexpr size_t k = 32;
    constexpr size_t n = 24;
    
    // Matrix A: m x k
    std::vector<float> matrix_a(m * k);
    // Matrix B: k x n
    std::vector<float> matrix_b(k * n);
    
    // Fill with random data
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (size_t i = 0; i < m * k; i++) {
        matrix_a[i] = dist(gen);
    }
    
    for (size_t i = 0; i < k * n; i++) {
        matrix_b[i] = dist(gen);
    }
    
    // Create reference result using standard matrix multiplication
    std::vector<float> reference_result(m * n, 0.0f);
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (size_t l = 0; l < k; l++) {
                sum += matrix_a[i * k + l] * matrix_b[l * n + j];
            }
            reference_result[i * n + j] = sum;
        }
    }
    
    // Quantize matrix B
    std::vector<int8_t> quantized_b(k * n + sizeof(float) / sizeof(int8_t));
    quantize_q8_0<float>(quantized_b.data(), matrix_b.data(), k * n);
    
    // Get scale factor
    const float* scale = reinterpret_cast<const float*>(quantized_b.data() + k * n);
    
    // Perform matrix multiplication with quantized weights
    std::vector<float> result(m * n, 0.0f);
    matrix_mul_q8_0<float>(result.data(), matrix_a.data(), quantized_b.data(), scale, m, k, n);
    
    // Verify results - should be close but not exact due to quantization
    double max_abs_error = 0.0;
    double sum_squared_error = 0.0;
    
    for (size_t i = 0; i < m * n; i++) {
        double error = std::abs(reference_result[i] - result[i]);
        max_abs_error = std::max(max_abs_error, error);
        sum_squared_error += error * error;
    }
    
    double rmse = std::sqrt(sum_squared_error / (m * n));
    
    // Check that errors are within acceptable limits
    // For quantized matrix multiplication, we expect higher errors due to
    // accumulated quantization errors across many operations
    EXPECT_LT(max_abs_error, 1.0); // Max absolute error < 1.0
    EXPECT_LT(rmse, 0.2);          // Root mean square error < 0.2
}

// Speed test for matrix multiplication (disabled by default, only for benchmarking)
TEST_F(QuantizationTest, DISABLED_SpeedTestMatrixMultiplyQ8_0) {
    // Create larger matrices for benchmarking
    constexpr size_t m = 128;
    constexpr size_t k = 512;
    constexpr size_t n = 256;
    
    // Matrix A: m x k
    std::vector<float> matrix_a(m * k);
    // Matrix B: k x n
    std::vector<float> matrix_b(k * n);
    
    // Fill with random data
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (size_t i = 0; i < m * k; i++) {
        matrix_a[i] = dist(gen);
    }
    
    for (size_t i = 0; i < k * n; i++) {
        matrix_b[i] = dist(gen);
    }
    
    // Result matrices
    std::vector<float> result_f32(m * n, 0.0f);
    std::vector<float> result_q8_0(m * n, 0.0f);
    
    // Quantize matrix B
    std::vector<int8_t> quantized_b(k * n + sizeof(float) / sizeof(int8_t));
    quantize_q8_0<float>(quantized_b.data(), matrix_b.data(), k * n);
    
    // Get scale factor
    const float* scale = reinterpret_cast<const float*>(quantized_b.data() + k * n);
    
    // Time standard matrix multiplication
    auto start_f32 = std::chrono::high_resolution_clock::now();
    
    matrix_mul<float>(result_f32.data(), matrix_a.data(), matrix_b.data(), m, k, n);
    
    auto end_f32 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_f32 = end_f32 - start_f32;
    
    // Time quantized matrix multiplication
    auto start_q8_0 = std::chrono::high_resolution_clock::now();
    
    matrix_mul_q8_0<float>(result_q8_0.data(), matrix_a.data(), quantized_b.data(), scale, m, k, n);
    
    auto end_q8_0 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_q8_0 = end_q8_0 - start_q8_0;
    
    // Print results
    std::cout << "F32 MatMul Time: " << duration_f32.count() << " seconds" << std::endl;
    std::cout << "Q8_0 MatMul Time: " << duration_q8_0.count() << " seconds" << std::endl;
    std::cout << "Speedup: " << duration_f32.count() / duration_q8_0.count() << "x" << std::endl;
    
    // Calculate error metrics
    double max_abs_error = 0.0;
    double sum_squared_error = 0.0;
    
    for (size_t i = 0; i < m * n; i++) {
        double error = std::abs(result_f32[i] - result_q8_0[i]);
        max_abs_error = std::max(max_abs_error, error);
        sum_squared_error += error * error;
    }
    
    double rmse = std::sqrt(sum_squared_error / (m * n));
    
    std::cout << "Max Absolute Error: " << max_abs_error << std::endl;
    std::cout << "RMSE: " << rmse << std::endl;
}

} // namespace
} // namespace simd
} // namespace ccsm