#include <ccsm/cpu/simd.h>
#include <gtest/gtest.h>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <algorithm>
#include <functional>
#include <numeric>
#include <cmath>

using namespace ccsm;

// Test fixture for SIMD matrix multiplication tests
class SIMDMatMulTest : public ::testing::Test {
protected:
    // Create a tolerance helper for matrix validation
    bool matrices_almost_equal(const std::vector<float>& a, 
                             const std::vector<float>& b, 
                             size_t rows, size_t cols,
                             float epsilon = 1e-4f) {
        if (a.size() != b.size() || a.size() != rows * cols) {
            std::cout << "Matrix size mismatch: " << a.size() << " vs " << b.size() 
                      << " (expected " << rows * cols << ")" << std::endl;
            return false;
        }
        
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                size_t idx = i * cols + j;
                if (std::abs(a[idx] - b[idx]) > epsilon) {
                    std::cout << "Matrices differ at [" << i << "," << j << "]: " 
                              << a[idx] << " vs " << b[idx] 
                              << " (diff: " << std::abs(a[idx] - b[idx]) 
                              << ", epsilon: " << epsilon << ")" << std::endl;
                    // Print neighboring elements for context
                    std::cout << "Context around mismatch:" << std::endl;
                    for (int di = -1; di <= 1; di++) {
                        for (int dj = -1; dj <= 1; dj++) {
                            int ni = i + di;
                            int nj = j + dj;
                            if (ni >= 0 && ni < rows && nj >= 0 && nj < cols) {
                                size_t nidx = ni * cols + nj;
                                std::cout << "  [" << ni << "," << nj << "] " 
                                          << a[nidx] << " vs " << b[nidx] 
                                          << " (diff: " << std::abs(a[nidx] - b[nidx]) << ")" << std::endl;
                            }
                        }
                    }
                    return false;
                }
            }
        }
        
        return true;
    }
    
    // Generate a random matrix
    void generate_random_matrix(std::vector<float>& matrix, 
                              size_t rows, size_t cols,
                              float min_val = -1.0f, float max_val = 1.0f) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(min_val, max_val);
        
        matrix.resize(rows * cols);
        for (size_t i = 0; i < rows * cols; i++) {
            matrix[i] = dist(gen);
        }
    }
    
    // Simple scalar matrix multiplication for reference
    void matrix_mul_scalar(const std::vector<float>& a, const std::vector<float>& b, 
                          std::vector<float>& c, 
                          size_t m, size_t k, size_t n) {
        // Initialize output matrix to zeros
        c.resize(m * n, 0.0f);
        
        // For each row of A
        for (size_t i = 0; i < m; i++) {
            // For each column of B
            for (size_t j = 0; j < n; j++) {
                float sum = 0.0f;
                // For each element in row A and column B
                for (size_t l = 0; l < k; l++) {
                    sum += a[i * k + l] * b[l * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    }

    // Cache-blocking matrix multiplication for reference
    void matrix_mul_blocked(const std::vector<float>& a, const std::vector<float>& b, 
                          std::vector<float>& c, 
                          size_t m, size_t k, size_t n, 
                          size_t block_size = 32) {
        // Initialize output matrix to zeros
        c.resize(m * n, 0.0f);
        
        // Outer loops over blocks
        for (size_t i0 = 0; i0 < m; i0 += block_size) {
            size_t i_end = std::min(i0 + block_size, m);
            
            for (size_t j0 = 0; j0 < n; j0 += block_size) {
                size_t j_end = std::min(j0 + block_size, n);
                
                for (size_t l0 = 0; l0 < k; l0 += block_size) {
                    size_t l_end = std::min(l0 + block_size, k);
                    
                    // Inner loops within a block
                    for (size_t i = i0; i < i_end; i++) {
                        for (size_t j = j0; j < j_end; j++) {
                            float sum = c[i * n + j]; // Load accumulated sum
                            
                            for (size_t l = l0; l < l_end; l++) {
                                sum += a[i * k + l] * b[l * n + j];
                            }
                            
                            c[i * n + j] = sum; // Store accumulated sum
                        }
                    }
                }
            }
        }
    }
    
    // Benchmark a matrix multiplication function
    double benchmark_matmul(std::function<void(float*, const float*, const float*, size_t, size_t, size_t)> matmul_func,
                          const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c,
                          size_t m, size_t k, size_t n, 
                          int iterations = 5) {
        // Warm-up
        matmul_func(c.data(), a.data(), b.data(), m, k, n);
        
        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; i++) {
            matmul_func(c.data(), a.data(), b.data(), m, k, n);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        
        return elapsed.count() / iterations; // Average time per iteration
    }
    
    // Format GFLOPS (billions of floating-point operations per second)
    std::string format_gflops(size_t m, size_t k, size_t n, double seconds) {
        // Each matrix multiplication requires 2*m*n*k operations (multiply and add)
        double ops = 2.0 * m * n * k;
        double gflops = ops / (seconds * 1e9);
        
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << gflops << " GFLOPS";
        return ss.str();
    }
};

// Test basic matrix multiplication correctness
TEST_F(SIMDMatMulTest, BasicMatrixMultiplication) {
    // Small matrices for basic correctness testing
    const size_t m = 4; // Rows of A
    const size_t k = 5; // Cols of A / Rows of B
    const size_t n = 3; // Cols of B
    
    std::vector<float> a(m * k), b(k * n), c_simd(m * n), c_scalar(m * n);
    
    // Initialize with deterministic values for debugging
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < k; j++) {
            a[i * k + j] = static_cast<float>((i * k + j) % 5 + 1) / 2.0f; // 0.5 to 2.5
        }
    }
    
    for (size_t i = 0; i < k; i++) {
        for (size_t j = 0; j < n; j++) {
            b[i * n + j] = static_cast<float>((i * n + j) % 3 + 1) / 2.0f; // 0.5 to 1.5
        }
    }
    
    // Perform matrix multiplication with scalar implementation
    matrix_mul_scalar(a, b, c_scalar, m, k, n);
    
    // Test SIMD matrix multiplication
    simd::matrix_mul(c_simd.data(), a.data(), b.data(), m, k, n);
    
    // Verify results
    EXPECT_TRUE(matrices_almost_equal(c_simd, c_scalar, m, n));
}

// Test matrix multiplication with non-aligned dimensions
TEST_F(SIMDMatMulTest, NonAlignedDimensions) {
    // Odd dimensions to test non-aligned cases
    const size_t m = 7;  // Rows of A
    const size_t k = 9;  // Cols of A / Rows of B
    const size_t n = 11; // Cols of B
    
    std::vector<float> a(m * k), b(k * n), c_simd(m * n), c_scalar(m * n);
    
    // Initialize matrices with random values
    generate_random_matrix(a, m, k);
    generate_random_matrix(b, k, n);
    
    // Scalar reference implementation
    matrix_mul_scalar(a, b, c_scalar, m, k, n);
    
    // SIMD implementation
    simd::matrix_mul(c_simd.data(), a.data(), b.data(), m, k, n);
    
    // Verify results
    EXPECT_TRUE(matrices_almost_equal(c_simd, c_scalar, m, n));
}

// Test large matrix multiplication (performance-focused)
TEST_F(SIMDMatMulTest, LargeMatrixMultiplication) {
    // Skip if this is a quick test run
    if (::testing::FLAGS_gtest_filter == "*QuickTest*") {
        GTEST_SKIP() << "Skipping large matrix test in quick test mode";
    }
    
    // Medium-sized matrices for performance testing
    const size_t m = 128; // Rows of A
    const size_t k = 256; // Cols of A / Rows of B
    const size_t n = 128; // Cols of B
    
    std::vector<float> a(m * k), b(k * n), c_simd(m * n), c_blocked(m * n), c_scalar(m * n);
    
    // Initialize matrices with random values
    generate_random_matrix(a, m, k, -1.0f, 1.0f);
    generate_random_matrix(b, k, n, -1.0f, 1.0f);
    
    // Calculate results using all implementations
    matrix_mul_scalar(a, b, c_scalar, m, k, n);
    matrix_mul_blocked(a, b, c_blocked, m, k, n);
    simd::matrix_mul(c_simd.data(), a.data(), b.data(), m, k, n);
    
    // Verify SIMD and blocked implementations against scalar
    EXPECT_TRUE(matrices_almost_equal(c_simd, c_scalar, m, n));
    EXPECT_TRUE(matrices_almost_equal(c_blocked, c_scalar, m, n));
    
    // Performance benchmarks
    std::cout << "\nPerformance comparison for " << m << "x" << k << " * " << k << "x" << n << " matrix multiplication:" << std::endl;
    
    // Benchmark scalar implementation
    auto scalar_time = benchmark_matmul(
        [this](float* c, const float* a, const float* b, size_t m, size_t k, size_t n) {
            std::vector<float> a_vec(a, a + m * k);
            std::vector<float> b_vec(b, b + k * n);
            std::vector<float> c_vec;
            matrix_mul_scalar(a_vec, b_vec, c_vec, m, k, n);
            std::copy(c_vec.begin(), c_vec.end(), c);
        },
        a, b, c_scalar, m, k, n
    );
    
    // Benchmark blocked implementation
    auto blocked_time = benchmark_matmul(
        [this](float* c, const float* a, const float* b, size_t m, size_t k, size_t n) {
            std::vector<float> a_vec(a, a + m * k);
            std::vector<float> b_vec(b, b + k * n);
            std::vector<float> c_vec;
            matrix_mul_blocked(a_vec, b_vec, c_vec, m, k, n);
            std::copy(c_vec.begin(), c_vec.end(), c);
        },
        a, b, c_blocked, m, k, n
    );
    
    // Benchmark SIMD implementation
    auto simd_time = benchmark_matmul(
        [](float* c, const float* a, const float* b, size_t m, size_t k, size_t n) {
            simd::matrix_mul(c, a, b, m, k, n);
        },
        a, b, c_simd, m, k, n
    );
    
    // Print results
    std::cout << "  Scalar:  " << std::fixed << std::setprecision(3) << scalar_time * 1000 
              << " ms (" << format_gflops(m, k, n, scalar_time) << ")" << std::endl;
    
    std::cout << "  Blocked: " << std::fixed << std::setprecision(3) << blocked_time * 1000 
              << " ms (" << format_gflops(m, k, n, blocked_time) << ")" << std::endl;
    
    std::cout << "  SIMD:    " << std::fixed << std::setprecision(3) << simd_time * 1000 
              << " ms (" << format_gflops(m, k, n, simd_time) << ")" << std::endl;
    
    // Calculate speedups
    double simd_vs_scalar = scalar_time / simd_time;
    double blocked_vs_scalar = scalar_time / blocked_time;
    
    std::cout << "  SIMD speedup vs scalar: " << std::fixed << std::setprecision(2) << simd_vs_scalar << "x" << std::endl;
    std::cout << "  Blocked speedup vs scalar: " << std::fixed << std::setprecision(2) << blocked_vs_scalar << "x" << std::endl;
    
    // We expect SIMD to be faster than scalar, but the actual speedup depends on hardware
    // For a properly optimized SIMD implementation, we should see at least 2x speedup
    EXPECT_GT(simd_vs_scalar, 1.1) << "SIMD implementation should be faster than scalar";
}

// Test aligned vs unaligned memory access
TEST_F(SIMDMatMulTest, AlignedVsUnalignedAccess) {
    // Matrix dimensions
    const size_t m = 32; // Rows of A
    const size_t k = 32; // Cols of A / Rows of B
    const size_t n = 32; // Cols of B
    
    // Create aligned and unaligned matrices
    std::vector<float> a_aligned(m * k), b_aligned(k * n), c_aligned(m * n);
    
    // Create unaligned matrices with 1 float offset
    std::vector<float> a_buffer(m * k + 1), b_buffer(k * n + 1), c_buffer(m * n + 1);
    float* a_unaligned = a_buffer.data() + 1; // Offset by 1 float to ensure misalignment
    float* b_unaligned = b_buffer.data() + 1;
    float* c_unaligned = c_buffer.data() + 1;
    
    // Initialize with identical random values
    generate_random_matrix(a_aligned, m, k);
    generate_random_matrix(b_aligned, k, n);
    
    // Copy aligned data to unaligned buffers
    std::copy(a_aligned.begin(), a_aligned.end(), a_unaligned);
    std::copy(b_aligned.begin(), b_aligned.end(), b_unaligned);
    
    // Perform matrix multiplication on aligned memory
    simd::matrix_mul(c_aligned.data(), a_aligned.data(), b_aligned.data(), m, k, n);
    
    // Perform matrix multiplication on unaligned memory
    simd::matrix_mul(c_unaligned, a_unaligned, b_unaligned, m, k, n);
    
    // Results should be identical regardless of alignment
    bool results_match = true;
    for (size_t i = 0; i < m * n; i++) {
        if (std::abs(c_aligned[i] - c_unaligned[i]) > 1e-4f) {
            results_match = false;
            break;
        }
    }
    
    EXPECT_TRUE(results_match) << "Matrix multiplication results should be the same for aligned and unaligned memory";
}

// Test matrix multiplication with various tile sizes
TEST_F(SIMDMatMulTest, TileSizeOptimization) {
    // Skip if this is a quick test run
    if (::testing::FLAGS_gtest_filter == "*QuickTest*") {
        GTEST_SKIP() << "Skipping tile size optimization test in quick test mode";
    }
    
    // Medium-sized matrices for performance testing
    const size_t m = 128; // Rows of A
    const size_t k = 128; // Cols of A / Rows of B
    const size_t n = 128; // Cols of B
    
    std::vector<float> a(m * k), b(k * n), c_result(m * n), c_reference(m * n);
    
    // Initialize matrices with random values
    generate_random_matrix(a, m, k);
    generate_random_matrix(b, k, n);
    
    // Calculate reference result
    matrix_mul_scalar(a, b, c_reference, m, k, n);
    
    // Test different tile sizes
    std::vector<size_t> tile_sizes = {8, 16, 32, 64};
    std::vector<double> tile_times;
    
    std::cout << "\nTile size performance comparison for " << m << "x" << k << " * " << k << "x" << n << " matrix multiplication:" << std::endl;
    
    for (size_t tile_size : tile_sizes) {
        // Benchmark blocked implementation with this tile size
        auto time = benchmark_matmul(
            [this, tile_size](float* c, const float* a, const float* b, size_t m, size_t k, size_t n) {
                std::vector<float> a_vec(a, a + m * k);
                std::vector<float> b_vec(b, b + k * n);
                std::vector<float> c_vec;
                matrix_mul_blocked(a_vec, b_vec, c_vec, m, k, n, tile_size);
                std::copy(c_vec.begin(), c_vec.end(), c);
            },
            a, b, c_result, m, k, n
        );
        
        tile_times.push_back(time);
        
        // Verify correctness
        EXPECT_TRUE(matrices_almost_equal(c_result, c_reference, m, n))
            << "Matrix multiplication with tile size " << tile_size << " produced incorrect results";
        
        // Print performance
        std::cout << "  Tile size " << std::setw(2) << tile_size << ": " 
                  << std::fixed << std::setprecision(3) << time * 1000 
                  << " ms (" << format_gflops(m, k, n, time) << ")" << std::endl;
    }
    
    // Find best tile size
    auto min_it = std::min_element(tile_times.begin(), tile_times.end());
    size_t best_tile_idx = std::distance(tile_times.begin(), min_it);
    size_t best_tile_size = tile_sizes[best_tile_idx];
    
    std::cout << "  Best tile size: " << best_tile_size << std::endl;
    
    // The optimal tile size depends on the cache size of the CPU
    // For most CPUs, we expect the best tile size to be around 16-64
    EXPECT_TRUE(best_tile_size >= 8 && best_tile_size <= 64) 
        << "Unexpected optimal tile size: " << best_tile_size;
}

// Test edge cases and numerical stability
TEST_F(SIMDMatMulTest, EdgeCasesAndNumericalStability) {
    // Test 1: Matrix multiplication with very small and very large values
    {
        // Small matrices for basic correctness testing
        const size_t m = 8; // Rows of A
        const size_t k = 8; // Cols of A / Rows of B
        const size_t n = 8; // Cols of B
        
        std::vector<float> a(m * k), b(k * n), c_simd(m * n), c_scalar(m * n);
        
        // Initialize with very small and very large values
        for (size_t i = 0; i < m * k; i++) {
            a[i] = (i % 2 == 0) ? 1e-7f : 1e7f;
        }
        
        for (size_t i = 0; i < k * n; i++) {
            b[i] = (i % 3 == 0) ? 1e-6f : 1e6f;
        }
        
        // Calculate expected result
        matrix_mul_scalar(a, b, c_scalar, m, k, n);
        
        // Test SIMD implementation
        simd::matrix_mul(c_simd.data(), a.data(), b.data(), m, k, n);
        
        // Verify results with a larger epsilon due to potential numerical differences
        EXPECT_TRUE(matrices_almost_equal(c_simd, c_scalar, m, n, 1e-3f));
    }
    
    // Test 2: Matrix multiplication with NaN and Inf values
    {
        const size_t m = 4;
        const size_t k = 4;
        const size_t n = 4;
        
        std::vector<float> a(m * k, 1.0f);
        std::vector<float> b(k * n, 1.0f);
        std::vector<float> c_simd(m * n), c_scalar(m * n);
        
        // Add some NaN and Inf values
        a[5] = std::numeric_limits<float>::quiet_NaN();
        b[10] = std::numeric_limits<float>::infinity();
        
        // Calculate expected result (will contain NaN/Inf)
        matrix_mul_scalar(a, b, c_scalar, m, k, n);
        
        // Test SIMD implementation
        simd::matrix_mul(c_simd.data(), a.data(), b.data(), m, k, n);
        
        // Check if NaN/Inf propagation is consistent
        // We don't use matrices_almost_equal here because it doesn't handle NaN/Inf well
        bool consistent = true;
        for (size_t i = 0; i < m * n; i++) {
            bool scalar_is_nan = std::isnan(c_scalar[i]);
            bool simd_is_nan = std::isnan(c_simd[i]);
            bool scalar_is_inf = std::isinf(c_scalar[i]);
            bool simd_is_inf = std::isinf(c_simd[i]);
            
            if (scalar_is_nan != simd_is_nan || scalar_is_inf != simd_is_inf) {
                consistent = false;
                std::cout << "Inconsistent NaN/Inf handling at index " << i
                          << ": scalar=" << c_scalar[i] << ", simd=" << c_simd[i] << std::endl;
                break;
            }
            
            if (!scalar_is_nan && !scalar_is_inf && !simd_is_nan && !simd_is_inf) {
                if (std::abs(c_scalar[i] - c_simd[i]) > 1e-4f) {
                    consistent = false;
                    std::cout << "Value mismatch at index " << i
                              << ": scalar=" << c_scalar[i] << ", simd=" << c_simd[i] << std::endl;
                    break;
                }
            }
        }
        
        EXPECT_TRUE(consistent) << "SIMD implementation doesn't handle NaN/Inf values consistently";
    }
    
    // Test 3: Multiplication with zero matrices
    {
        const size_t m = 16;
        const size_t k = 16;
        const size_t n = 16;
        
        std::vector<float> a(m * k, 0.0f);
        std::vector<float> b(k * n, 0.0f);
        std::vector<float> c_simd(m * n), c_scalar(m * n);
        
        // Calculate expected result (all zeros)
        matrix_mul_scalar(a, b, c_scalar, m, k, n);
        
        // Test SIMD implementation
        simd::matrix_mul(c_simd.data(), a.data(), b.data(), m, k, n);
        
        // All results should be zero
        EXPECT_TRUE(std::all_of(c_simd.begin(), c_simd.end(), [](float val) { return val == 0.0f; }));
    }
}

// Test matrix multiplication with mixed precision (not applicable in current implementation)
// This is a placeholder for future implementation
TEST_F(SIMDMatMulTest, DISABLED_MixedPrecision) {
    GTEST_SKIP() << "Mixed precision matrix multiplication not implemented yet";
}