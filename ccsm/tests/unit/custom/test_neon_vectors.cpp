#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <random>
#include <gtest/gtest.h>
#include "include/ccsm/cpu/simd.h"

// Define program entry point explicitly
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

using namespace ccsm;

// Helper function to generate a random vector
std::vector<float> generate_vector(size_t size, int seed = 123) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    
    std::vector<float> vec(size);
    for (size_t i = 0; i < size; i++) {
        vec[i] = dist(gen);
    }
    return vec;
}

// Helper function to compare two vectors
bool compare_vectors(const std::vector<float>& a, const std::vector<float>& b, float epsilon = 1e-4f) {
    if (a.size() != b.size()) {
        return false;
    }
    
    for (size_t i = 0; i < a.size(); i++) {
        if (std::fabs(a[i] - b[i]) > epsilon) {
            return false;
        }
    }
    return true;
}

TEST(NEONVectorTest, VectorAdd) {
    const size_t size = 100;
    auto a = generate_vector(size);
    auto b = generate_vector(size, 456);
    
    // Compute reference result
    std::vector<float> expected(size);
    for (size_t i = 0; i < size; i++) {
        expected[i] = a[i] + b[i];
    }
    
    // Compute SIMD result
    std::vector<float> result(size);
    simd::vector_add(result.data(), a.data(), b.data(), size);
    
    // Compare results
    EXPECT_TRUE(compare_vectors(expected, result));
}

TEST(NEONVectorTest, VectorMul) {
    const size_t size = 100;
    auto a = generate_vector(size);
    auto b = generate_vector(size, 456);
    
    // Compute reference result
    std::vector<float> expected(size);
    for (size_t i = 0; i < size; i++) {
        expected[i] = a[i] * b[i];
    }
    
    // Compute SIMD result
    std::vector<float> result(size);
    simd::vector_mul(result.data(), a.data(), b.data(), size);
    
    // Compare results
    EXPECT_TRUE(compare_vectors(expected, result));
}

TEST(NEONVectorTest, VectorFMA) {
    const size_t size = 100;
    auto a = generate_vector(size);
    auto b = generate_vector(size, 456);
    auto c = generate_vector(size, 789);
    
    // Compute reference result
    std::vector<float> expected(size);
    for (size_t i = 0; i < size; i++) {
        expected[i] = a[i] * b[i] + c[i];
    }
    
    // Compute SIMD result
    std::vector<float> result(size);
    simd::vector_fma(result.data(), a.data(), b.data(), c.data(), size);
    
    // Compare results
    EXPECT_TRUE(compare_vectors(expected, result));
}

TEST(NEONVectorTest, VectorDot) {
    const size_t size = 100;
    auto a = generate_vector(size);
    auto b = generate_vector(size, 456);
    
    // Compute reference result
    float expected = 0.0f;
    for (size_t i = 0; i < size; i++) {
        expected += a[i] * b[i];
    }
    
    // Compute SIMD result
    float result = simd::vector_dot(a.data(), b.data(), size);
    
    // Compare results
    EXPECT_NEAR(expected, result, 1e-2f);
}

TEST(NEONVectorTest, PrintActiveImplementation) {
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
    std::cout << std::endl << std::endl;
    
    SUCCEED();
}
