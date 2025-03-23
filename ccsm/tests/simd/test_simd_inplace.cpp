#include <ccsm/cpu/simd.h>
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

using namespace ccsm;

// Test fixture for in-place SIMD operations
class SIMDInplaceTest : public ::testing::Test {
protected:
    // Helper for comparing floats
    bool almost_equal(float a, float b, float epsilon = 1e-5) {
        return std::fabs(a - b) < epsilon;
    }

    bool vector_almost_equal(const std::vector<float>& a, const std::vector<float>& b, float epsilon = 1e-5) {
        if (a.size() != b.size()) {
            std::cout << "Vector size mismatch: " << a.size() << " vs " << b.size() << std::endl;
            return false;
        }
        
        for (size_t i = 0; i < a.size(); i++) {
            if (!almost_equal(a[i], b[i], epsilon)) {
                std::cout << "Vectors differ at index " << i << ": " << a[i] << " vs " << b[i] 
                          << " (diff: " << std::abs(a[i] - b[i]) << ", epsilon: " << epsilon << ")" << std::endl;
                return false;
            }
        }
        
        return true;
    }
};

// Test in-place vector addition
TEST_F(SIMDInplaceTest, InplaceVectorAdd) {
    const size_t n = 1024;
    
    // Create test vectors
    std::vector<float> a_inplace(n), a_copy(n), b(n), expected(n);
    
    // Initialize test data
    for (size_t i = 0; i < n; i++) {
        a_inplace[i] = a_copy[i] = static_cast<float>(i) / 100.0f;
        b[i] = static_cast<float>(n - i) / 100.0f;
        expected[i] = a_inplace[i] + b[i];
    }
    
    // Skip this test if in-place operations aren't implemented yet
    GTEST_SKIP() << "In-place vector operations not implemented yet";
    
    // Test in-place vector addition when implemented
    // simd::vector_add_inplace(a_inplace.data(), b.data(), n);
    
    // Check that in-place operation matches the expected result
    // EXPECT_TRUE(vector_almost_equal(a_inplace, expected));
    
    // For comparison, also do regular addition to verify both approaches give the same result
    // std::vector<float> regular_result(n);
    // simd::vector_add(regular_result.data(), a_copy.data(), b.data(), n);
    // EXPECT_TRUE(vector_almost_equal(a_inplace, regular_result));
}

// Test in-place vector multiplication
TEST_F(SIMDInplaceTest, InplaceVectorMul) {
    const size_t n = 1024;
    
    // Create test vectors
    std::vector<float> a_inplace(n), a_copy(n), b(n), expected(n);
    
    // Initialize test data
    for (size_t i = 0; i < n; i++) {
        a_inplace[i] = a_copy[i] = static_cast<float>(i) / 100.0f;
        b[i] = static_cast<float>(n - i) / 100.0f;
        expected[i] = a_inplace[i] * b[i];
    }
    
    // Skip this test if in-place operations aren't implemented yet
    GTEST_SKIP() << "In-place vector operations not implemented yet";
    
    // Test in-place vector multiplication when implemented
    // simd::vector_mul_inplace(a_inplace.data(), b.data(), n);
    
    // Check that in-place operation matches the expected result
    // EXPECT_TRUE(vector_almost_equal(a_inplace, expected));
    
    // For comparison, also do regular multiplication to verify both approaches give the same result
    // std::vector<float> regular_result(n);
    // simd::vector_mul(regular_result.data(), a_copy.data(), b.data(), n);
    // EXPECT_TRUE(vector_almost_equal(a_inplace, regular_result));
}

// Test in-place vector scaling
TEST_F(SIMDInplaceTest, InplaceVectorScale) {
    const size_t n = 1024;
    
    // Create test vectors
    std::vector<float> a_inplace(n), a_copy(n), expected(n);
    const float scalar = 2.5f;
    
    // Initialize test data
    for (size_t i = 0; i < n; i++) {
        a_inplace[i] = a_copy[i] = static_cast<float>(i) / 100.0f;
        expected[i] = a_inplace[i] * scalar;
    }
    
    // Skip this test if in-place operations aren't implemented yet
    GTEST_SKIP() << "In-place vector operations not implemented yet";
    
    // Test in-place vector scaling when implemented
    // simd::vector_scale_inplace(a_inplace.data(), scalar, n);
    
    // Check that in-place operation matches the expected result
    // EXPECT_TRUE(vector_almost_equal(a_inplace, expected));
    
    // For comparison, also do regular scaling to verify both approaches give the same result
    // std::vector<float> regular_result(n);
    // simd::vector_scale(regular_result.data(), a_copy.data(), scalar, n);
    // EXPECT_TRUE(vector_almost_equal(a_inplace, regular_result));
}

// Test in-place ReLU activation
TEST_F(SIMDInplaceTest, InplaceReLU) {
    const size_t n = 1024;
    
    // Create test vectors with both positive and negative values
    std::vector<float> a_inplace(n), a_copy(n), expected(n);
    
    // Initialize test data
    for (size_t i = 0; i < n; i++) {
        // Mix of positive and negative values
        a_inplace[i] = a_copy[i] = static_cast<float>(i - n/2) / 100.0f;
        expected[i] = std::max(0.0f, a_inplace[i]);
    }
    
    // Skip this test if in-place operations aren't implemented yet
    GTEST_SKIP() << "In-place activation functions not implemented yet";
    
    // Test in-place ReLU activation when implemented
    // simd::relu_inplace(a_inplace.data(), n);
    
    // Check that in-place operation matches the expected result
    // EXPECT_TRUE(vector_almost_equal(a_inplace, expected));
    
    // For comparison, also do regular ReLU to verify both approaches give the same result
    // std::vector<float> regular_result(n);
    // simd::relu(regular_result.data(), a_copy.data(), n);
    // EXPECT_TRUE(vector_almost_equal(a_inplace, regular_result));
}

// Test performance comparison between in-place and regular operations
TEST_F(SIMDInplaceTest, DISABLED_PerformanceComparison) {
    // This is a long-running test, so it's marked as DISABLED
    const size_t n = 10000000; // 10 million elements
    
    // Create large test vectors
    std::vector<float> a(n), b(n), result(n);
    
    // Initialize test data
    for (size_t i = 0; i < n; i++) {
        a[i] = static_cast<float>(i % 1000) / 100.0f;
        b[i] = static_cast<float>((n - i) % 1000) / 100.0f;
    }
    
    // Skip the performance test if in-place operations aren't implemented yet
    GTEST_SKIP() << "In-place vector operations not implemented yet";
    
    // When implemented, we should test performance like this:
    /*
    // Measure regular vector addition performance
    auto start_regular = std::chrono::high_resolution_clock::now();
    simd::vector_add(result.data(), a.data(), b.data(), n);
    auto end_regular = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> regular_time = end_regular - start_regular;
    
    // Measure in-place vector addition performance
    std::vector<float> a_copy = a; // Make a copy to use for in-place operation
    auto start_inplace = std::chrono::high_resolution_clock::now();
    simd::vector_add_inplace(a_copy.data(), b.data(), n);
    auto end_inplace = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> inplace_time = end_inplace - start_inplace;
    
    // Verify both operations give same result
    EXPECT_TRUE(vector_almost_equal(result, a_copy));
    
    // Log performance comparison
    std::cout << "Regular vector add time: " << regular_time.count() << " ms" << std::endl;
    std::cout << "In-place vector add time: " << inplace_time.count() << " ms" << std::endl;
    std::cout << "Performance ratio (regular/in-place): " << regular_time/inplace_time << std::endl;
    
    // We would expect in-place to be faster due to reduced memory traffic
    EXPECT_LT(inplace_time, regular_time) << "In-place operation should be faster than regular operation";
    */
}
