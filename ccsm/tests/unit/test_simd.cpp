#include <gtest/gtest.h>
#include <ccsm/cpu/simd.h>
#include <vector>
#include <random>
#include <iostream>
#include <iomanip>

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

// Helper function to compute dot product in naive way (for comparison)
float naive_dot_product(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector sizes must match");
    }
    
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

// Helper to print vectors for debugging
void print_vector(const std::vector<float>& v, const std::string& name) {
    std::cout << name << " = [";
    for (size_t i = 0; i < v.size(); i++) {
        std::cout << std::fixed << std::setprecision(6) << v[i];
        if (i < v.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}

// Test fixture for SIMD operations
class SIMDTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize test vectors with different sizes
        small_size = 10;  // Small enough to expose edge cases
        medium_size = 64; // Multiple of common SIMD widths
        large_size = 1025; // Odd size to test alignment and remainder handling
        
        // Generate test data
        small_vec1 = generate_random_vector(small_size);
        small_vec2 = generate_random_vector(small_size);
        medium_vec1 = generate_random_vector(medium_size);
        medium_vec2 = generate_random_vector(medium_size);
        large_vec1 = generate_random_vector(large_size);
        large_vec2 = generate_random_vector(large_size);
    }
    
    size_t small_size;
    size_t medium_size;
    size_t large_size;
    
    std::vector<float> small_vec1;
    std::vector<float> small_vec2;
    std::vector<float> medium_vec1;
    std::vector<float> medium_vec2;
    std::vector<float> large_vec1;
    std::vector<float> large_vec2;
};

// Test SIMD vector dot product with various sizes
TEST_F(SIMDTest, VectorDotProduct) {
    // Small vectors
    float result_small = simd::vector_dot(small_vec1.data(), small_vec2.data(), small_size);
    float expected_small = naive_dot_product(small_vec1, small_vec2);
    EXPECT_NEAR(result_small, expected_small, 1e-3f); // Using larger epsilon due to accumulated floating-point differences
    
    // Medium vectors (aligned size)
    float result_medium = simd::vector_dot(medium_vec1.data(), medium_vec2.data(), medium_size);
    float expected_medium = naive_dot_product(medium_vec1, medium_vec2);
    EXPECT_NEAR(result_medium, expected_medium, 1e-3f);
    
    // Large vectors (unaligned size)
    float result_large = simd::vector_dot(large_vec1.data(), large_vec2.data(), large_size);
    float expected_large = naive_dot_product(large_vec1, large_vec2);
    EXPECT_NEAR(result_large, expected_large, 1e-2f); // Even larger epsilon for large vectors
}

// Test that the SIMD implementation is using optimized path when available
TEST_F(SIMDTest, SIMDPathSelection) {
    // Get SIMD capabilities
    std::string simd_caps = simd::get_cpu_capabilities();
    std::cout << "Detected SIMD capabilities: " << simd_caps << std::endl;
    
    // Verify that some SIMD capabilities are detected
    EXPECT_FALSE(simd_caps.empty());
    
    // Check that an implementation is selected
    simd::Implementation impl = simd::get_active_implementation();
    EXPECT_NE(impl, simd::Implementation::UNKNOWN);
    
    std::cout << "Using SIMD implementation: ";
    switch (impl) {
        case simd::Implementation::SCALAR:
            std::cout << "Scalar (No SIMD)";
            break;
        case simd::Implementation::SSE2:
            std::cout << "SSE2";
            break;
        case simd::Implementation::SSE41:
            std::cout << "SSE4.1";
            break;
        case simd::Implementation::AVX:
            std::cout << "AVX";
            break;
        case simd::Implementation::AVX2:
            std::cout << "AVX2";
            break;
        case simd::Implementation::AVX512:
            std::cout << "AVX-512";
            break;
        case simd::Implementation::NEON:
            std::cout << "NEON";
            break;
        default:
            std::cout << "Unknown";
    }
    std::cout << std::endl;
}

// Test SIMD vector addition
TEST_F(SIMDTest, VectorAddition) {
    // Allocate result vectors
    std::vector<float> result_small(small_size);
    std::vector<float> result_medium(medium_size);
    std::vector<float> result_large(large_size);
    
    // Perform SIMD addition
    simd::vector_add(result_small.data(), small_vec1.data(), small_vec2.data(), small_size);
    simd::vector_add(result_medium.data(), medium_vec1.data(), medium_vec2.data(), medium_size);
    simd::vector_add(result_large.data(), large_vec1.data(), large_vec2.data(), large_size);
    
    // Compute expected results
    std::vector<float> expected_small(small_size);
    std::vector<float> expected_medium(medium_size);
    std::vector<float> expected_large(large_size);
    
    for (size_t i = 0; i < small_size; i++) {
        expected_small[i] = small_vec1[i] + small_vec2[i];
    }
    
    for (size_t i = 0; i < medium_size; i++) {
        expected_medium[i] = medium_vec1[i] + medium_vec2[i];
    }
    
    for (size_t i = 0; i < large_size; i++) {
        expected_large[i] = large_vec1[i] + large_vec2[i];
    }
    
    // Compare results
    EXPECT_TRUE(compare_vectors(result_small, expected_small));
    EXPECT_TRUE(compare_vectors(result_medium, expected_medium));
    EXPECT_TRUE(compare_vectors(result_large, expected_large));
}

// Test SIMD vector multiplication
TEST_F(SIMDTest, VectorMultiplication) {
    // Allocate result vectors
    std::vector<float> result_small(small_size);
    std::vector<float> result_medium(medium_size);
    std::vector<float> result_large(large_size);
    
    // Perform SIMD multiplication
    simd::vector_mul(result_small.data(), small_vec1.data(), small_vec2.data(), small_size);
    simd::vector_mul(result_medium.data(), medium_vec1.data(), medium_vec2.data(), medium_size);
    simd::vector_mul(result_large.data(), large_vec1.data(), large_vec2.data(), large_size);
    
    // Compute expected results
    std::vector<float> expected_small(small_size);
    std::vector<float> expected_medium(medium_size);
    std::vector<float> expected_large(large_size);
    
    for (size_t i = 0; i < small_size; i++) {
        expected_small[i] = small_vec1[i] * small_vec2[i];
    }
    
    for (size_t i = 0; i < medium_size; i++) {
        expected_medium[i] = medium_vec1[i] * medium_vec2[i];
    }
    
    for (size_t i = 0; i < large_size; i++) {
        expected_large[i] = large_vec1[i] * large_vec2[i];
    }
    
    // Compare results
    EXPECT_TRUE(compare_vectors(result_small, expected_small));
    EXPECT_TRUE(compare_vectors(result_medium, expected_medium));
    EXPECT_TRUE(compare_vectors(result_large, expected_large));
}

// Test SIMD scalar multiplication
TEST_F(SIMDTest, ScalarMultiplication) {
    const float scalar = 2.5f;
    
    // Allocate result vectors
    std::vector<float> result_small(small_size);
    std::vector<float> result_medium(medium_size);
    std::vector<float> result_large(large_size);
    
    // Perform scalar multiplication
    simd::vector_scale(result_small.data(), small_vec1.data(), scalar, small_size);
    simd::vector_scale(result_medium.data(), medium_vec1.data(), scalar, medium_size);
    simd::vector_scale(result_large.data(), large_vec1.data(), scalar, large_size);
    
    // Compute expected results
    std::vector<float> expected_small(small_size);
    std::vector<float> expected_medium(medium_size);
    std::vector<float> expected_large(large_size);
    
    for (size_t i = 0; i < small_size; i++) {
        expected_small[i] = small_vec1[i] * scalar;
    }
    
    for (size_t i = 0; i < medium_size; i++) {
        expected_medium[i] = medium_vec1[i] * scalar;
    }
    
    for (size_t i = 0; i < large_size; i++) {
        expected_large[i] = large_vec1[i] * scalar;
    }
    
    // Compare results
    EXPECT_TRUE(compare_vectors(result_small, expected_small));
    EXPECT_TRUE(compare_vectors(result_medium, expected_medium));
    EXPECT_TRUE(compare_vectors(result_large, expected_large));
}

// Test memory alignment utilities
TEST_F(SIMDTest, MemoryAlignment) {
    // Test pointer alignment
    float buffer[100];
    void* ptr = buffer;
    void* aligned_ptr = simd::align_ptr(ptr, 32);
    
    // Alignment should be a multiple of the specified alignment
    uintptr_t alignment = reinterpret_cast<uintptr_t>(aligned_ptr) % 32;
    EXPECT_EQ(alignment, 0u);
    
    // Test aligned allocation/deallocation
    size_t size = 1000;
    float* aligned_buffer = simd::aligned_alloc<float>(size, 64);
    EXPECT_NE(aligned_buffer, nullptr);
    
    // Verify the allocation is properly aligned
    alignment = reinterpret_cast<uintptr_t>(aligned_buffer) % 64;
    EXPECT_EQ(alignment, 0u);
    
    // Clean up
    simd::aligned_free(aligned_buffer);
}

// Test SIMD vector comparison operations
TEST_F(SIMDTest, VectorComparison) {
    // Generate test vectors with specific patterns for comparison testing
    std::vector<float> vec1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> vec2 = {1.0f, 0.0f, 5.0f, 4.0f, 1.0f, 9.0f, 7.0f, 0.0f};
    
    // Create mask for the expected result (1.0f where vec1 > vec2, 0.0f otherwise)
    std::vector<float> expected_gt = {0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f};
    
    // Perform comparison with SIMD
    std::vector<float> result(vec1.size());
    simd::vector_gt_mask(result.data(), vec1.data(), vec2.data(), vec1.size());
    
    // Compare results
    for (size_t i = 0; i < vec1.size(); i++) {
        EXPECT_FLOAT_EQ(result[i], expected_gt[i]);
    }
}