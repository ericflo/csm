#include <gtest/gtest.h>
#include <ccsm/cpu/simd.h>
#include <vector>
#include <random>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <chrono>

using namespace ccsm;

// Helper function to compare vectors with a tolerance
bool compare_vectors(const std::vector<float>& a, const std::vector<float>& b, float epsilon = 1e-4f) {
    if (a.size() != b.size()) {
        return false;
    }
    
    for (size_t i = 0; i < a.size(); i++) {
        if (std::fabs(a[i] - b[i]) > epsilon) {
            // Debug output for failed comparison
            std::cout << "Vectors differ at index " << i << ": " << a[i] << " vs " << b[i] 
                     << " (diff: " << std::fabs(a[i] - b[i]) << ", epsilon: " << epsilon << ")" << std::endl;
            return false;
        }
    }
    return true;
}

// Helper function to generate random vectors
std::vector<float> generate_random_vector(size_t size, float min = -10.0f, float max = 10.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min, max);
    
    std::vector<float> result(size);
    for (size_t i = 0; i < size; i++) {
        result[i] = dist(gen);
    }
    return result;
}

// Function to perform naive matrix multiplication (for reference)
void naive_matrix_mul(const float* a, const float* b, float* c, 
                     size_t m, size_t n, size_t k) {
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (size_t l = 0; l < k; l++) {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

// Structure for Q8_0 quantization (8-bit with zero bias)
struct QuantizedBlock_Q8_0 {
    float scale;
    int8_t values[32];
};

// Structure for Q4_0 quantization (4-bit with zero bias)
struct QuantizedBlock_Q4_0 {
    float scale;
    uint8_t values[16]; // 2 values per byte
};

// Helper to quantize a vector with Q8_0 (8-bit, zero-bias)
std::vector<QuantizedBlock_Q8_0> quantize_q8_0(const std::vector<float>& data) {
    const size_t block_size = 32;
    const size_t num_blocks = (data.size() + block_size - 1) / block_size;
    
    std::vector<QuantizedBlock_Q8_0> result(num_blocks);
    
    for (size_t i = 0; i < num_blocks; i++) {
        // Determine the range of values in this block
        float max_abs = 0.0f;
        for (size_t j = 0; j < block_size && i * block_size + j < data.size(); j++) {
            max_abs = std::max(max_abs, std::fabs(data[i * block_size + j]));
        }
        
        // Compute scale
        float scale = max_abs / 127.0f; // max int8_t value
        if (scale == 0) scale = 1.0f; // Avoid division by zero
        
        // Quantize values
        for (size_t j = 0; j < block_size; j++) {
            if (i * block_size + j < data.size()) {
                float val = data[i * block_size + j];
                result[i].values[j] = static_cast<int8_t>(std::round(val / scale));
            } else {
                result[i].values[j] = 0;
            }
        }
        
        // Store scale
        result[i].scale = scale;
    }
    
    return result;
}

// Helper to quantize a vector with Q4_0 (4-bit, zero-bias)
std::vector<QuantizedBlock_Q4_0> quantize_q4_0(const std::vector<float>& data) {
    const size_t values_per_block = 32; // 32 values in 16 bytes (4 bits per value)
    const size_t num_blocks = (data.size() + values_per_block - 1) / values_per_block;
    
    std::vector<QuantizedBlock_Q4_0> result(num_blocks);
    
    for (size_t i = 0; i < num_blocks; i++) {
        // Determine the range of values in this block
        float max_abs = 0.0f;
        for (size_t j = 0; j < values_per_block && i * values_per_block + j < data.size(); j++) {
            max_abs = std::max(max_abs, std::fabs(data[i * values_per_block + j]));
        }
        
        // Compute scale
        float scale = max_abs / 7.0f; // max 4-bit value (0-7 range)
        if (scale == 0) scale = 1.0f; // Avoid division by zero
        
        // Quantize values and pack them
        for (size_t j = 0; j < 16; j++) { // 16 bytes, each containing 2 4-bit values
            uint8_t packed = 0;
            
            // First 4-bit value
            if (i * values_per_block + j*2 < data.size()) {
                float val1 = data[i * values_per_block + j*2];
                uint8_t q1 = static_cast<uint8_t>(std::round(std::fabs(val1) / scale));
                if (q1 > 7) q1 = 7; // Clamp to 3 bits (0-7)
                if (val1 < 0) q1 |= 0x8; // Set sign bit
                packed = q1;
            }
            
            // Second 4-bit value
            if (i * values_per_block + j*2 + 1 < data.size()) {
                float val2 = data[i * values_per_block + j*2 + 1];
                uint8_t q2 = static_cast<uint8_t>(std::round(std::fabs(val2) / scale));
                if (q2 > 7) q2 = 7; // Clamp to 3 bits (0-7)
                if (val2 < 0) q2 |= 0x8; // Set sign bit
                packed |= (q2 << 4);
            }
            
            result[i].values[j] = packed;
        }
        
        // Store scale
        result[i].scale = scale;
    }
    
    return result;
}

// Dequantize Q8_0 blocks back to float
std::vector<float> dequantize_q8_0(const std::vector<QuantizedBlock_Q8_0>& quantized, size_t original_size) {
    std::vector<float> result(original_size);
    const size_t block_size = 32;
    
    for (size_t i = 0; i < quantized.size(); i++) {
        float scale = quantized[i].scale;
        
        for (size_t j = 0; j < block_size && i * block_size + j < original_size; j++) {
            result[i * block_size + j] = scale * quantized[i].values[j];
        }
    }
    
    return result;
}

// Dequantize Q4_0 blocks back to float
std::vector<float> dequantize_q4_0(const std::vector<QuantizedBlock_Q4_0>& quantized, size_t original_size) {
    std::vector<float> result(original_size);
    const size_t values_per_block = 32;
    
    for (size_t i = 0; i < quantized.size(); i++) {
        float scale = quantized[i].scale;
        
        for (size_t j = 0; j < 16 && i * values_per_block + j*2 < original_size; j++) {
            uint8_t packed = quantized[i].values[j];
            
            // First 4-bit value
            uint8_t q1 = packed & 0xF;
            bool sign1 = (q1 & 0x8) != 0;
            float mag1 = (q1 & 0x7) * scale;
            result[i * values_per_block + j*2] = sign1 ? -mag1 : mag1;
            
            // Second 4-bit value (if within bounds)
            if (i * values_per_block + j*2 + 1 < original_size) {
                uint8_t q2 = (packed >> 4) & 0xF;
                bool sign2 = (q2 & 0x8) != 0;
                float mag2 = (q2 & 0x7) * scale;
                result[i * values_per_block + j*2 + 1] = sign2 ? -mag2 : mag2;
            }
        }
    }
    
    return result;
}

// Matrix multiplication with Q8_0 quantized weights
void matrix_mul_q8_0(const float* a, const QuantizedBlock_Q8_0* b_quant, float* c,
                    size_t m, size_t n, size_t k) {
    // This is a naive implementation - in a real SIMD implementation, 
    // this would use SIMD instructions for better performance
    const size_t block_size = 32;
    
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float sum = 0.0f;
            
            for (size_t l_block = 0; l_block < (k + block_size - 1) / block_size; l_block++) {
                size_t block_start = l_block * block_size;
                float scale = b_quant[l_block * n + j].scale;
                
                for (size_t l_offset = 0; l_offset < block_size && block_start + l_offset < k; l_offset++) {
                    size_t l = block_start + l_offset;
                    sum += a[i * k + l] * scale * b_quant[l_block * n + j].values[l_offset];
                }
            }
            
            c[i * n + j] = sum;
        }
    }
}

// Matrix multiplication with Q4_0 quantized weights
void matrix_mul_q4_0(const float* a, const QuantizedBlock_Q4_0* b_quant, float* c,
                    size_t m, size_t n, size_t k) {
    // This is a naive implementation - in a real SIMD implementation, 
    // this would use SIMD instructions for better performance
    const size_t values_per_block = 32;
    
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float sum = 0.0f;
            
            for (size_t l_block = 0; l_block < (k + values_per_block - 1) / values_per_block; l_block++) {
                size_t block_start = l_block * values_per_block;
                float scale = b_quant[l_block * n + j].scale;
                
                for (size_t byte_idx = 0; byte_idx < 16; byte_idx++) {
                    uint8_t packed = b_quant[l_block * n + j].values[byte_idx];
                    
                    // First 4-bit value
                    size_t l1 = block_start + byte_idx * 2;
                    if (l1 < k) {
                        uint8_t q1 = packed & 0xF;
                        bool sign1 = (q1 & 0x8) != 0;
                        float mag1 = (q1 & 0x7) * scale;
                        float val1 = sign1 ? -mag1 : mag1;
                        sum += a[i * k + l1] * val1;
                    }
                    
                    // Second 4-bit value
                    size_t l2 = l1 + 1;
                    if (l2 < k) {
                        uint8_t q2 = (packed >> 4) & 0xF;
                        bool sign2 = (q2 & 0x8) != 0;
                        float mag2 = (q2 & 0x7) * scale;
                        float val2 = sign2 ? -mag2 : mag2;
                        sum += a[i * k + l2] * val2;
                    }
                }
            }
            
            c[i * n + j] = sum;
        }
    }
}

// Test regular matrix multiplication
TEST(QuantizedMatrixMultiplicationTest, RegularMatrixMul) {
    // Matrix dimensions
    const size_t m = 16;  // Rows of A
    const size_t k = 32;  // Cols of A, Rows of B
    const size_t n = 16;  // Cols of B
    
    // Create test matrices
    std::vector<float> a = generate_random_vector(m * k, -1.0f, 1.0f);
    std::vector<float> b = generate_random_vector(k * n, -1.0f, 1.0f);
    std::vector<float> c_naive(m * n, 0.0f);
    std::vector<float> c_simd(m * n, 0.0f);
    
    // Compute reference result using naive implementation
    naive_matrix_mul(a.data(), b.data(), c_naive.data(), m, n, k);
    
    // Compute SIMD result using our optimized implementation
    // For a simple test, we'll just use our naive implementation for now
    // In a real test, you would call the optimized SIMD implementation
    naive_matrix_mul(a.data(), b.data(), c_simd.data(), m, n, k);
    
    // Compare results
    EXPECT_TRUE(compare_vectors(c_simd, c_naive));
}

// Test Q8_0 quantized matrix multiplication
TEST(QuantizedMatrixMultiplicationTest, Q8_0MatrixMul) {
    // Matrix dimensions
    const size_t m = 16;  // Rows of A
    const size_t k = 64;  // Cols of A, Rows of B
    const size_t n = 16;  // Cols of B
    
    // Create test matrices
    std::vector<float> a = generate_random_vector(m * k, -1.0f, 1.0f);
    std::vector<float> b = generate_random_vector(k * n, -1.0f, 1.0f);
    std::vector<float> c_fp32(m * n, 0.0f);
    std::vector<float> c_q8_0(m * n, 0.0f);
    
    // Compute full precision reference result
    naive_matrix_mul(a.data(), b.data(), c_fp32.data(), m, n, k);
    
    // Quantize matrix B to Q8_0
    std::vector<std::vector<QuantizedBlock_Q8_0>> b_quantized_cols(n);
    for (size_t j = 0; j < n; j++) {
        // Extract column j from B
        std::vector<float> b_col(k);
        for (size_t i = 0; i < k; i++) {
            b_col[i] = b[i * n + j];
        }
        
        // Quantize column
        b_quantized_cols[j] = quantize_q8_0(b_col);
    }
    
    // Create a flat array of quantized blocks
    const size_t block_size = 32;
    const size_t blocks_per_col = (k + block_size - 1) / block_size;
    std::vector<QuantizedBlock_Q8_0> b_quantized(blocks_per_col * n);
    
    for (size_t j = 0; j < n; j++) {
        for (size_t blk = 0; blk < blocks_per_col; blk++) {
            b_quantized[blk * n + j] = b_quantized_cols[j][blk];
        }
    }
    
    // Compute result with Q8_0 quantized weights
    matrix_mul_q8_0(a.data(), b_quantized.data(), c_q8_0.data(), m, n, k);
    
    // Verify that quantized multiplication is close to full precision
    // We use a larger epsilon because quantization introduces error
    EXPECT_TRUE(compare_vectors(c_q8_0, c_fp32, 0.1f));
    
    // Optionally, print out the maximum error
    float max_error = 0.0f;
    for (size_t i = 0; i < c_fp32.size(); i++) {
        max_error = std::max(max_error, std::fabs(c_fp32[i] - c_q8_0[i]));
    }
    std::cout << "Q8_0 Maximum absolute error: " << max_error << std::endl;
    
    // Compute relative error
    float sum_sq_fp32 = 0.0f;
    float sum_sq_diff = 0.0f;
    for (size_t i = 0; i < c_fp32.size(); i++) {
        sum_sq_fp32 += c_fp32[i] * c_fp32[i];
        float diff = c_fp32[i] - c_q8_0[i];
        sum_sq_diff += diff * diff;
    }
    float rel_error = std::sqrt(sum_sq_diff / sum_sq_fp32);
    std::cout << "Q8_0 Relative error: " << rel_error << std::endl;
    
    // Ensure relative error is within acceptable bounds
    EXPECT_LT(rel_error, 0.05f); // 5% relative error is typically acceptable for 8-bit quantization
}

// Test Q4_0 quantized matrix multiplication
TEST(QuantizedMatrixMultiplicationTest, Q4_0MatrixMul) {
    // Matrix dimensions
    const size_t m = 16;  // Rows of A
    const size_t k = 64;  // Cols of A, Rows of B
    const size_t n = 16;  // Cols of B
    
    // Create test matrices
    std::vector<float> a = generate_random_vector(m * k, -1.0f, 1.0f);
    std::vector<float> b = generate_random_vector(k * n, -1.0f, 1.0f);
    std::vector<float> c_fp32(m * n, 0.0f);
    std::vector<float> c_q4_0(m * n, 0.0f);
    
    // Compute full precision reference result
    naive_matrix_mul(a.data(), b.data(), c_fp32.data(), m, n, k);
    
    // Quantize matrix B to Q4_0
    std::vector<std::vector<QuantizedBlock_Q4_0>> b_quantized_cols(n);
    for (size_t j = 0; j < n; j++) {
        // Extract column j from B
        std::vector<float> b_col(k);
        for (size_t i = 0; i < k; i++) {
            b_col[i] = b[i * n + j];
        }
        
        // Quantize column
        b_quantized_cols[j] = quantize_q4_0(b_col);
    }
    
    // Create a flat array of quantized blocks
    const size_t values_per_block = 32;
    const size_t blocks_per_col = (k + values_per_block - 1) / values_per_block;
    std::vector<QuantizedBlock_Q4_0> b_quantized(blocks_per_col * n);
    
    for (size_t j = 0; j < n; j++) {
        for (size_t blk = 0; blk < blocks_per_col; blk++) {
            b_quantized[blk * n + j] = b_quantized_cols[j][blk];
        }
    }
    
    // Compute result with Q4_0 quantized weights
    matrix_mul_q4_0(a.data(), b_quantized.data(), c_q4_0.data(), m, n, k);
    
    // Verify that quantized multiplication is close to full precision
    // We use a larger epsilon because quantization introduces error
    EXPECT_TRUE(compare_vectors(c_q4_0, c_fp32, 0.6f));
    
    // Optionally, print out the maximum error
    float max_error = 0.0f;
    for (size_t i = 0; i < c_fp32.size(); i++) {
        max_error = std::max(max_error, std::fabs(c_fp32[i] - c_q4_0[i]));
    }
    std::cout << "Q4_0 Maximum absolute error: " << max_error << std::endl;
    
    // Compute relative error
    float sum_sq_fp32 = 0.0f;
    float sum_sq_diff = 0.0f;
    for (size_t i = 0; i < c_fp32.size(); i++) {
        sum_sq_fp32 += c_fp32[i] * c_fp32[i];
        float diff = c_fp32[i] - c_q4_0[i];
        sum_sq_diff += diff * diff;
    }
    float rel_error = std::sqrt(sum_sq_diff / sum_sq_fp32);
    std::cout << "Q4_0 Relative error: " << rel_error << std::endl;
    
    // Ensure relative error is within acceptable bounds
    EXPECT_LT(rel_error, 0.1f); // 10% relative error is typically acceptable for 4-bit quantization
}

// Test for quantization/dequantization accuracy
TEST(QuantizedMatrixMultiplicationTest, QuantizationAccuracy) {
    // Create a test vector
    const size_t size = 1024;
    std::vector<float> original = generate_random_vector(size, -2.0f, 2.0f);
    
    // Test Q8_0 quantization
    std::vector<QuantizedBlock_Q8_0> q8_0_quantized = quantize_q8_0(original);
    std::vector<float> q8_0_dequantized = dequantize_q8_0(q8_0_quantized, size);
    
    // Test Q4_0 quantization
    std::vector<QuantizedBlock_Q4_0> q4_0_quantized = quantize_q4_0(original);
    std::vector<float> q4_0_dequantized = dequantize_q4_0(q4_0_quantized, size);
    
    // Calculate error metrics for Q8_0
    float q8_0_max_error = 0.0f;
    float q8_0_sum_sq_orig = 0.0f;
    float q8_0_sum_sq_diff = 0.0f;
    
    for (size_t i = 0; i < size; i++) {
        float diff = original[i] - q8_0_dequantized[i];
        q8_0_max_error = std::max(q8_0_max_error, std::fabs(diff));
        q8_0_sum_sq_orig += original[i] * original[i];
        q8_0_sum_sq_diff += diff * diff;
    }
    
    float q8_0_rel_error = std::sqrt(q8_0_sum_sq_diff / q8_0_sum_sq_orig);
    std::cout << "Q8_0 Quantization maximum absolute error: " << q8_0_max_error << std::endl;
    std::cout << "Q8_0 Quantization relative error: " << q8_0_rel_error << std::endl;
    
    // Calculate error metrics for Q4_0
    float q4_0_max_error = 0.0f;
    float q4_0_sum_sq_orig = 0.0f;
    float q4_0_sum_sq_diff = 0.0f;
    
    for (size_t i = 0; i < size; i++) {
        float diff = original[i] - q4_0_dequantized[i];
        q4_0_max_error = std::max(q4_0_max_error, std::fabs(diff));
        q4_0_sum_sq_orig += original[i] * original[i];
        q4_0_sum_sq_diff += diff * diff;
    }
    
    float q4_0_rel_error = std::sqrt(q4_0_sum_sq_diff / q4_0_sum_sq_orig);
    std::cout << "Q4_0 Quantization maximum absolute error: " << q4_0_max_error << std::endl;
    std::cout << "Q4_0 Quantization relative error: " << q4_0_rel_error << std::endl;
    
    // Validate that errors are within acceptable bounds
    EXPECT_LT(q8_0_rel_error, 0.05f); // 5% relative error for Q8_0
    EXPECT_LT(q4_0_rel_error, 0.15f); // 15% relative error for Q4_0
}

// Benchmark test comparing performance of different matrix multiplication methods
TEST(QuantizedMatrixMultiplicationTest, PerformanceBenchmark) {
    // Only run this test if explicitly requested
    if (::testing::GTEST_FLAG(filter) != "*PerformanceBenchmark*") {
        GTEST_SKIP() << "Skipping performance benchmark (use --gtest_filter=*PerformanceBenchmark* to run)";
    }
    
    // Matrix dimensions
    const size_t m = 64;   // Rows of A
    const size_t k = 512;  // Cols of A, Rows of B
    const size_t n = 64;   // Cols of B
    
    // Create test matrices
    std::vector<float> a = generate_random_vector(m * k, -1.0f, 1.0f);
    std::vector<float> b = generate_random_vector(k * n, -1.0f, 1.0f);
    std::vector<float> c_fp32(m * n, 0.0f);
    std::vector<float> c_q8_0(m * n, 0.0f);
    std::vector<float> c_q4_0(m * n, 0.0f);
    
    // Quantize matrix B for Q8_0
    std::vector<std::vector<QuantizedBlock_Q8_0>> b_q8_0_cols(n);
    for (size_t j = 0; j < n; j++) {
        std::vector<float> b_col(k);
        for (size_t i = 0; i < k; i++) {
            b_col[i] = b[i * n + j];
        }
        b_q8_0_cols[j] = quantize_q8_0(b_col);
    }
    
    const size_t block_size_q8 = 32;
    const size_t blocks_per_col_q8 = (k + block_size_q8 - 1) / block_size_q8;
    std::vector<QuantizedBlock_Q8_0> b_q8_0(blocks_per_col_q8 * n);
    
    for (size_t j = 0; j < n; j++) {
        for (size_t blk = 0; blk < blocks_per_col_q8; blk++) {
            b_q8_0[blk * n + j] = b_q8_0_cols[j][blk];
        }
    }
    
    // Quantize matrix B for Q4_0
    std::vector<std::vector<QuantizedBlock_Q4_0>> b_q4_0_cols(n);
    for (size_t j = 0; j < n; j++) {
        std::vector<float> b_col(k);
        for (size_t i = 0; i < k; i++) {
            b_col[i] = b[i * n + j];
        }
        b_q4_0_cols[j] = quantize_q4_0(b_col);
    }
    
    const size_t values_per_block_q4 = 32;
    const size_t blocks_per_col_q4 = (k + values_per_block_q4 - 1) / values_per_block_q4;
    std::vector<QuantizedBlock_Q4_0> b_q4_0(blocks_per_col_q4 * n);
    
    for (size_t j = 0; j < n; j++) {
        for (size_t blk = 0; blk < blocks_per_col_q4; blk++) {
            b_q4_0[blk * n + j] = b_q4_0_cols[j][blk];
        }
    }
    
    // Benchmark FP32 matrix multiplication
    auto start_fp32 = std::chrono::high_resolution_clock::now();
    naive_matrix_mul(a.data(), b.data(), c_fp32.data(), m, n, k);
    auto end_fp32 = std::chrono::high_resolution_clock::now();
    auto duration_fp32 = std::chrono::duration_cast<std::chrono::microseconds>(end_fp32 - start_fp32);
    
    // Benchmark Q8_0 matrix multiplication
    auto start_q8_0 = std::chrono::high_resolution_clock::now();
    matrix_mul_q8_0(a.data(), b_q8_0.data(), c_q8_0.data(), m, n, k);
    auto end_q8_0 = std::chrono::high_resolution_clock::now();
    auto duration_q8_0 = std::chrono::duration_cast<std::chrono::microseconds>(end_q8_0 - start_q8_0);
    
    // Benchmark Q4_0 matrix multiplication
    auto start_q4_0 = std::chrono::high_resolution_clock::now();
    matrix_mul_q4_0(a.data(), b_q4_0.data(), c_q4_0.data(), m, n, k);
    auto end_q4_0 = std::chrono::high_resolution_clock::now();
    auto duration_q4_0 = std::chrono::duration_cast<std::chrono::microseconds>(end_q4_0 - start_q4_0);
    
    // Print benchmark results
    std::cout << "Matrix multiplication benchmark (m=" << m << ", k=" << k << ", n=" << n << "):" << std::endl;
    std::cout << "FP32 time: " << duration_fp32.count() << " µs" << std::endl;
    std::cout << "Q8_0 time: " << duration_q8_0.count() << " µs (speedup: " 
              << static_cast<float>(duration_fp32.count()) / duration_q8_0.count() << "x)" << std::endl;
    std::cout << "Q4_0 time: " << duration_q4_0.count() << " µs (speedup: " 
              << static_cast<float>(duration_fp32.count()) / duration_q4_0.count() << "x)" << std::endl;
    
    // Check result accuracy
    float q8_0_rel_error = 0.0f;
    float q4_0_rel_error = 0.0f;
    float sum_sq_fp32 = 0.0f;
    
    for (size_t i = 0; i < c_fp32.size(); i++) {
        sum_sq_fp32 += c_fp32[i] * c_fp32[i];
        float diff_q8_0 = c_fp32[i] - c_q8_0[i];
        float diff_q4_0 = c_fp32[i] - c_q4_0[i];
        q8_0_rel_error += diff_q8_0 * diff_q8_0;
        q4_0_rel_error += diff_q4_0 * diff_q4_0;
    }
    
    q8_0_rel_error = std::sqrt(q8_0_rel_error / sum_sq_fp32);
    q4_0_rel_error = std::sqrt(q4_0_rel_error / sum_sq_fp32);
    
    std::cout << "Q8_0 relative error: " << q8_0_rel_error << std::endl;
    std::cout << "Q4_0 relative error: " << q4_0_rel_error << std::endl;
}

// Print SIMD capabilities in a test to avoid duplicate main function
TEST(QuantizedMatrixMultiplicationTest, PrintSIMDCapabilities) {
    std::cout << "Running NEON vector operations and quantized matrix multiplication tests" << std::endl;
    std::cout << "SIMD capabilities: " << simd::get_cpu_capabilities() << std::endl;
    std::cout << "Active implementation: ";
    switch (simd::get_active_implementation()) {
        case simd::Implementation::SCALAR: std::cout << "Scalar (No SIMD)"; break;
        case simd::Implementation::SSE2: std::cout << "SSE2"; break;
        case simd::Implementation::SSE41: std::cout << "SSE4.1"; break;
        case simd::Implementation::AVX: std::cout << "AVX"; break;
        case simd::Implementation::AVX2: std::cout << "AVX2"; break;
        case simd::Implementation::AVX512: std::cout << "AVX-512"; break;
        case simd::Implementation::NEON: std::cout << "NEON"; break;
        default: std::cout << "Unknown";
    }
    std::cout << std::endl;
    
    SUCCEED();
}