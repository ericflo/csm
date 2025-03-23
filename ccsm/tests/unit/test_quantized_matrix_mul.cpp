#include <gtest/gtest.h>
#include <ccsm/cpu/simd.h>
#include <vector>
#include <random>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>

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

TEST(VectorOperationsTest, VectorAddTest) {
    // Create test vectors
    const size_t size = 100;
    std::vector<float> a = generate_random_vector(size);
    std::vector<float> b = generate_random_vector(size);
    std::vector<float> result_simd(size);
    std::vector<float> result_naive(size);
    
    // Compute reference result
    for (size_t i = 0; i < size; i++) {
        result_naive[i] = a[i] + b[i];
    }
    
    // Compute SIMD result using our optimized implementation
    simd::vector_add(result_simd.data(), a.data(), b.data(), size);
    
    // Compare results
    EXPECT_TRUE(compare_vectors(result_simd, result_naive));
}

TEST(VectorOperationsTest, VectorMulTest) {
    // Create test vectors
    const size_t size = 100;
    std::vector<float> a = generate_random_vector(size);
    std::vector<float> b = generate_random_vector(size);
    std::vector<float> result_simd(size);
    std::vector<float> result_naive(size);
    
    // Compute reference result
    for (size_t i = 0; i < size; i++) {
        result_naive[i] = a[i] * b[i];
    }
    
    // Compute SIMD result
    simd::vector_mul(result_simd.data(), a.data(), b.data(), size);
    
    // Compare results
    EXPECT_TRUE(compare_vectors(result_simd, result_naive));
}

TEST(VectorOperationsTest, VectorFMATest) {
    // Create test vectors
    const size_t size = 100;
    std::vector<float> a = generate_random_vector(size);
    std::vector<float> b = generate_random_vector(size);
    std::vector<float> c = generate_random_vector(size);
    std::vector<float> result_simd(size);
    std::vector<float> result_naive(size);
    
    // Compute reference result
    for (size_t i = 0; i < size; i++) {
        result_naive[i] = a[i] * b[i] + c[i];
    }
    
    // Compute SIMD result
    simd::vector_fma(result_simd.data(), a.data(), b.data(), c.data(), size);
    
    // Compare results
    EXPECT_TRUE(compare_vectors(result_simd, result_naive));
}

TEST(VectorOperationsTest, VectorDotTest) {
    // Create test vectors
    const size_t size = 100;
    std::vector<float> a = generate_random_vector(size);
    std::vector<float> b = generate_random_vector(size);
    
    // Compute reference result
    float result_naive = 0.0f;
    for (size_t i = 0; i < size; i++) {
        result_naive += a[i] * b[i];
    }
    
    // Compute SIMD result
    float result_simd = simd::vector_dot(a.data(), b.data(), size);
    
    // Compare results with a tolerance
    EXPECT_NEAR(result_simd, result_naive, 1e-3);
}

int main(int argc, char **argv) {
    std::cout << "Running NEON vector operations tests" << std::endl;
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
    
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}