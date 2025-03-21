#include <ccsm/cpu/simd.h>
#include <ccsm/cpu/thread_pool.h>
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <chrono>

using namespace ccsm;

// Test fixture for SIMD operations
class SIMDTest : public ::testing::Test {
protected:
    // Simple test helpers
    bool almost_equal(float a, float b, float epsilon = 1e-5) {
        return std::fabs(a - b) < epsilon;
    }

    bool vector_almost_equal(const std::vector<float>& a, const std::vector<float>& b, float epsilon = 1e-5) {
        if (a.size() != b.size()) {
            return false;
        }
        
        for (size_t i = 0; i < a.size(); i++) {
            if (!almost_equal(a[i], b[i], epsilon)) {
                return false;
            }
        }
        
        return true;
    }
};

// Test vector operations
TEST_F(SIMDTest, VectorOperations) {
    const size_t n = 1024;
    
    // Create test vectors
    std::vector<float> a(n), b(n), c(n), expected(n);
    
    // Initialize test data
    for (size_t i = 0; i < n; i++) {
        a[i] = static_cast<float>(i) / 100.0f;
        b[i] = static_cast<float>(n - i) / 100.0f;
        expected[i] = a[i] + b[i];
    }
    
    // Test vector_add
    simd::vector_add(a.data(), b.data(), c.data(), n);
    EXPECT_TRUE(vector_almost_equal(c, expected));
    
    // Test vector_mul
    for (size_t i = 0; i < n; i++) {
        expected[i] = a[i] * b[i];
    }
    simd::vector_mul(a.data(), b.data(), c.data(), n);
    EXPECT_TRUE(vector_almost_equal(c, expected));
    
    // Test vector_scale
    const float scalar = 2.5f;
    for (size_t i = 0; i < n; i++) {
        expected[i] = a[i] * scalar;
    }
    simd::vector_scale(a.data(), scalar, c.data(), n);
    EXPECT_TRUE(vector_almost_equal(c, expected));
    
    // Test vector_dot
    float dot_expected = 0.0f;
    for (size_t i = 0; i < n; i++) {
        dot_expected += a[i] * b[i];
    }
    float dot_result = simd::vector_dot(a.data(), b.data(), n);
    EXPECT_TRUE(almost_equal(dot_result, dot_expected));
}

// Test activation functions
TEST_F(SIMDTest, ActivationFunctions) {
    const size_t n = 1024;
    
    // Create test vectors
    std::vector<float> a(n), c(n), expected(n);
    
    // Test relu
    for (size_t i = 0; i < n; i++) {
        a[i] = static_cast<float>(i - n/2) / 100.0f; // Mix of positive and negative values
        expected[i] = std::max(0.0f, a[i]);
    }
    simd::relu(a.data(), c.data(), n);
    EXPECT_TRUE(vector_almost_equal(c, expected));
    
    // Test softmax
    float sum = 0.0f;
    float max_val = a[0];
    for (size_t i = 1; i < n; i++) {
        max_val = std::max(max_val, a[i]);
    }
    for (size_t i = 0; i < n; i++) {
        expected[i] = std::exp(a[i] - max_val);
        sum += expected[i];
    }
    for (size_t i = 0; i < n; i++) {
        expected[i] /= sum;
    }
    simd::softmax(a.data(), c.data(), n);
    EXPECT_TRUE(vector_almost_equal(c, expected));
}

// Test matrix multiplication
TEST_F(SIMDTest, MatrixMultiplication) {
    // Create small matrices for testing
    const size_t m = 16, k = 16, p = 16;
    std::vector<float> mat_a(m*k), mat_b(k*p), mat_c(m*p), mat_expected(m*p);
    
    // Initialize test matrices
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < k; j++) {
            mat_a[i*k + j] = static_cast<float>(i*k + j) / 100.0f;
        }
    }
    
    for (size_t i = 0; i < k; i++) {
        for (size_t j = 0; j < p; j++) {
            mat_b[i*p + j] = static_cast<float>(i*p + j) / 100.0f;
        }
    }
    
    // Compute expected result
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < p; j++) {
            float sum = 0.0f;
            for (size_t l = 0; l < k; l++) {
                sum += mat_a[i*k + l] * mat_b[l*p + j];
            }
            mat_expected[i*p + j] = sum;
        }
    }
    
    // Test matrix_mul
    simd::matrix_mul(mat_a.data(), mat_b.data(), mat_c.data(), m, k, p);
    EXPECT_TRUE(vector_almost_equal(mat_c, mat_expected, 1e-3f)); // Larger epsilon due to potential FP precision differences
}

// Test fixture for thread pool
class ThreadPoolTest : public ::testing::Test {
};

// Test thread pool basic functionality
TEST_F(ThreadPoolTest, BasicThreadPoolFunctionality) {
    ThreadPool pool(4); // Create pool with 4 threads
    
    const int num_tasks = 100;
    std::vector<std::future<int>> results;
    
    // Enqueue tasks
    for (int i = 0; i < num_tasks; i++) {
        results.push_back(pool.enqueue([i]() {
            // Simulate some work
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            return i * i;
        }));
    }
    
    // Check results
    for (int i = 0; i < num_tasks; i++) {
        EXPECT_EQ(results[i].get(), i * i);
    }
}

// Test ParallelFor functionality
TEST_F(ThreadPoolTest, ParallelForFunctionality) {
    const int array_size = 10000;
    std::vector<int> array(array_size, 0);
    
    // Use ParallelFor to process array
    ParallelFor::exec(0, array_size, [&array](int i) {
        array[i] = i * i;
    });
    
    // Verify results
    for (int i = 0; i < array_size; i++) {
        EXPECT_EQ(array[i], i * i);
    }
}

// Test ParallelFor performance (this is a long-running test, so it's marked as DISABLED)
TEST_F(ThreadPoolTest, DISABLED_ParallelForPerformance) {
    const int perf_array_size = 50000000;
    std::vector<float> perf_array(perf_array_size, 1.0f);
    std::vector<float> result_array(perf_array_size, 0.0f);
    
    // Serial execution
    auto start_serial = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < perf_array_size; i++) {
        result_array[i] = std::sqrt(perf_array[i]);
    }
    auto end_serial = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> serial_time = end_serial - start_serial;
    
    // Parallel execution
    std::fill(result_array.begin(), result_array.end(), 0.0f);
    auto start_parallel = std::chrono::high_resolution_clock::now();
    ParallelFor::exec(0, perf_array_size, [&perf_array, &result_array](int i) {
        result_array[i] = std::sqrt(perf_array[i]);
    });
    auto end_parallel = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> parallel_time = end_parallel - start_parallel;
    
    // We should expect at least some speedup
    EXPECT_GT(serial_time.count() / parallel_time.count(), 1.0);
}