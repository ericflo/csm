#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include "include/ccsm/cpu/simd.h"

using namespace ccsm;

// Helper function to generate a random vector
std::vector<float> generate_vector(size_t size) {
    std::vector<float> vec(size);
    for (size_t i = 0; i < size; i++) {
        vec[i] = (float)(rand() % 100) / 10.0f;
    }
    return vec;
}

// Helper function to compare two vectors
bool compare_vectors(const std::vector<float>& a, const std::vector<float>& b, float epsilon = 1e-4f) {
    if (a.size() != b.size()) {
        std::cout << "Vectors have different sizes: " << a.size() << " vs " << b.size() << std::endl;
        return false;
    }
    
    bool equal = true;
    for (size_t i = 0; i < a.size(); i++) {
        if (std::fabs(a[i] - b[i]) > epsilon) {
            std::cout << "Vectors differ at index " << i << ": " << a[i] << " vs " << b[i] 
                     << " (diff: " << std::fabs(a[i] - b[i]) << ")" << std::endl;
            equal = false;
        }
    }
    return equal;
}

// Test vector_add function
bool test_vector_add() {
    const size_t size = 100;
    auto a = generate_vector(size);
    auto b = generate_vector(size);
    
    // Compute reference result
    std::vector<float> expected(size);
    for (size_t i = 0; i < size; i++) {
        expected[i] = a[i] + b[i];
    }
    
    // Compute SIMD result
    std::vector<float> result(size);
    simd::vector_add(result.data(), a.data(), b.data(), size);
    
    // Compare results
    return compare_vectors(expected, result);
}

// Test vector_mul function
bool test_vector_mul() {
    const size_t size = 100;
    auto a = generate_vector(size);
    auto b = generate_vector(size);
    
    // Compute reference result
    std::vector<float> expected(size);
    for (size_t i = 0; i < size; i++) {
        expected[i] = a[i] * b[i];
    }
    
    // Compute SIMD result
    std::vector<float> result(size);
    simd::vector_mul(result.data(), a.data(), b.data(), size);
    
    // Compare results
    return compare_vectors(expected, result);
}

// Test vector_fma function
bool test_vector_fma() {
    const size_t size = 100;
    auto a = generate_vector(size);
    auto b = generate_vector(size);
    auto c = generate_vector(size);
    
    // Compute reference result
    std::vector<float> expected(size);
    for (size_t i = 0; i < size; i++) {
        expected[i] = a[i] * b[i] + c[i];
    }
    
    // Compute SIMD result
    std::vector<float> result(size);
    simd::vector_fma(result.data(), a.data(), b.data(), c.data(), size);
    
    // Compare results
    return compare_vectors(expected, result);
}

// Test vector_dot function
bool test_vector_dot() {
    const size_t size = 100;
    auto a = generate_vector(size);
    auto b = generate_vector(size);
    
    // Compute reference result
    float expected = 0.0f;
    for (size_t i = 0; i < size; i++) {
        expected += a[i] * b[i];
    }
    
    // Compute SIMD result
    float result = simd::vector_dot(a.data(), b.data(), size);
    
    // Compare results
    float diff = std::fabs(expected - result);
    if (diff > 1e-2f) {
        std::cout << "Dot product differs: " << expected << " vs " << result 
                 << " (diff: " << diff << ")" << std::endl;
        return false;
    }
    return true;
}

int main() {
    srand(123); // Seed for reproducibility
    
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
    
    bool all_passed = true;
    
    std::cout << "Testing vector_add... ";
    bool add_passed = test_vector_add();
    std::cout << (add_passed ? "PASSED" : "FAILED") << std::endl;
    all_passed &= add_passed;
    
    std::cout << "Testing vector_mul... ";
    bool mul_passed = test_vector_mul();
    std::cout << (mul_passed ? "PASSED" : "FAILED") << std::endl;
    all_passed &= mul_passed;
    
    std::cout << "Testing vector_fma... ";
    bool fma_passed = test_vector_fma();
    std::cout << (fma_passed ? "PASSED" : "FAILED") << std::endl;
    all_passed &= fma_passed;
    
    std::cout << "Testing vector_dot... ";
    bool dot_passed = test_vector_dot();
    std::cout << (dot_passed ? "PASSED" : "FAILED") << std::endl;
    all_passed &= dot_passed;
    
    std::cout << std::endl << "Overall result: " << (all_passed ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << std::endl;
    
    return all_passed ? 0 : 1;
}
