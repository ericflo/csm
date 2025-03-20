#ifndef CCSM_SIMD_H
#define CCSM_SIMD_H

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cassert>
#include <algorithm>

// CPU feature detection
#if defined(__x86_64__) || defined(_M_X64)
    #define CCSM_ARCH_X86_64 1
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define CCSM_ARCH_ARM64 1
#endif

// SIMD instruction sets
#if defined(CCSM_ARCH_X86_64)
    #include <immintrin.h>
    
    #if defined(__AVX512F__)
        #define CCSM_HAVE_AVX512 1
    #endif
    
    #if defined(__AVX2__)
        #define CCSM_HAVE_AVX2 1
    #endif
    
    #if defined(__AVX__)
        #define CCSM_HAVE_AVX 1
    #endif
    
    #if defined(__SSE4_2__)
        #define CCSM_HAVE_SSE4_2 1
    #endif
    
    #if defined(__SSE4_1__)
        #define CCSM_HAVE_SSE4_1 1
    #endif
    
    #if defined(__SSE3__)
        #define CCSM_HAVE_SSE3 1
    #endif
    
    #if defined(__SSE2__) || defined(_M_X64)
        #define CCSM_HAVE_SSE2 1
    #endif
#endif

#if defined(CCSM_ARCH_ARM64)
    #include <arm_neon.h>
    #define CCSM_HAVE_NEON 1
#endif

namespace ccsm {
namespace simd {

// Detect CPU features at runtime
struct CPUFeatures {
    bool avx512f = false;
    bool avx2 = false;
    bool avx = false;
    bool sse4_2 = false;
    bool sse4_1 = false;
    bool sse3 = false;
    bool sse2 = false;
    bool neon = false;
    
    static const CPUFeatures& get();
};

// Forward declarations for vectorized operations
template<typename T>
void vector_add(const T* a, const T* b, T* c, size_t n);

template<typename T>
void vector_mul(const T* a, const T* b, T* c, size_t n);

template<typename T>
void vector_fma(const T* a, const T* b, const T* c, T* d, size_t n); // d = a * b + c

template<typename T>
void vector_scale(const T* a, T scalar, T* b, size_t n);

template<typename T>
T vector_dot(const T* a, const T* b, size_t n);

template<typename T>
void matrix_mul(const T* a, const T* b, T* c, size_t m, size_t k, size_t n);

template<typename T>
void softmax(const T* input, T* output, size_t n);

template<typename T>
void relu(const T* input, T* output, size_t n);

template<typename T>
void silu(const T* input, T* output, size_t n);

// Helper functions to choose the best implementation at runtime
// These will be specialized based on architecture

// Implementation details namespace (not part of public API)
namespace detail {

// Vector add implementations
template<typename T>
void vector_add_scalar(const T* a, const T* b, T* c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

#if defined(CCSM_HAVE_AVX)
void vector_add_avx_f32(const float* a, const float* b, float* c, size_t n);
#endif

#if defined(CCSM_HAVE_AVX2)
void vector_add_avx2_f32(const float* a, const float* b, float* c, size_t n);
#endif

#if defined(CCSM_HAVE_NEON)
void vector_add_neon_f32(const float* a, const float* b, float* c, size_t n);
#endif

// Vector multiply implementations
template<typename T>
void vector_mul_scalar(const T* a, const T* b, T* c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        c[i] = a[i] * b[i];
    }
}

#if defined(CCSM_HAVE_AVX)
void vector_mul_avx_f32(const float* a, const float* b, float* c, size_t n);
#endif

#if defined(CCSM_HAVE_AVX2)
void vector_mul_avx2_f32(const float* a, const float* b, float* c, size_t n);
#endif

#if defined(CCSM_HAVE_NEON)
void vector_mul_neon_f32(const float* a, const float* b, float* c, size_t n);
#endif

// Vector FMA implementations
template<typename T>
void vector_fma_scalar(const T* a, const T* b, const T* c, T* d, size_t n) {
    for (size_t i = 0; i < n; i++) {
        d[i] = a[i] * b[i] + c[i];
    }
}

#if defined(CCSM_HAVE_AVX2)
void vector_fma_avx2_f32(const float* a, const float* b, const float* c, float* d, size_t n);
#endif

#if defined(CCSM_HAVE_NEON)
void vector_fma_neon_f32(const float* a, const float* b, const float* c, float* d, size_t n);
#endif

// Vector scale implementations
template<typename T>
void vector_scale_scalar(const T* a, T scalar, T* b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        b[i] = a[i] * scalar;
    }
}

#if defined(CCSM_HAVE_AVX)
void vector_scale_avx_f32(const float* a, float scalar, float* b, size_t n);
#endif

#if defined(CCSM_HAVE_AVX2)
void vector_scale_avx2_f32(const float* a, float scalar, float* b, size_t n);
#endif

#if defined(CCSM_HAVE_NEON)
void vector_scale_neon_f32(const float* a, float scalar, float* b, size_t n);
#endif

// Vector dot implementations
template<typename T>
T vector_dot_scalar(const T* a, const T* b, size_t n) {
    T sum = 0;
    for (size_t i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

#if defined(CCSM_HAVE_AVX)
float vector_dot_avx_f32(const float* a, const float* b, size_t n);
#endif

#if defined(CCSM_HAVE_AVX2)
float vector_dot_avx2_f32(const float* a, const float* b, size_t n);
#endif

#if defined(CCSM_HAVE_NEON)
float vector_dot_neon_f32(const float* a, const float* b, size_t n);
#endif

// Matrix multiply implementations
template<typename T>
void matrix_mul_scalar(const T* a, const T* b, T* c, size_t m, size_t k, size_t n) {
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            T sum = 0;
            for (size_t l = 0; l < k; l++) {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

#if defined(CCSM_HAVE_AVX)
void matrix_mul_avx_f32(const float* a, const float* b, float* c, size_t m, size_t k, size_t n);
#endif

#if defined(CCSM_HAVE_AVX2)
void matrix_mul_avx2_f32(const float* a, const float* b, float* c, size_t m, size_t k, size_t n);
#endif

#if defined(CCSM_HAVE_NEON)
void matrix_mul_neon_f32(const float* a, const float* b, float* c, size_t m, size_t k, size_t n);
#endif

// Softmax implementations
template<typename T>
void softmax_scalar(const T* input, T* output, size_t n) {
    // Find max for numerical stability
    T max_val = input[0];
    for (size_t i = 1; i < n; i++) {
        max_val = std::max(max_val, input[i]);
    }
    
    // Compute exp and sum
    T sum = 0;
    for (size_t i = 0; i < n; i++) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }
    
    // Normalize
    T inv_sum = 1 / sum;
    for (size_t i = 0; i < n; i++) {
        output[i] *= inv_sum;
    }
}

#if defined(CCSM_HAVE_AVX)
void softmax_avx_f32(const float* input, float* output, size_t n);
#endif

#if defined(CCSM_HAVE_NEON)
void softmax_neon_f32(const float* input, float* output, size_t n);
#endif

// ReLU implementations
template<typename T>
void relu_scalar(const T* input, T* output, size_t n) {
    for (size_t i = 0; i < n; i++) {
        output[i] = std::max(T(0), input[i]);
    }
}

#if defined(CCSM_HAVE_AVX)
void relu_avx_f32(const float* input, float* output, size_t n);
#endif

#if defined(CCSM_HAVE_NEON)
void relu_neon_f32(const float* input, float* output, size_t n);
#endif

// SiLU implementations
template<typename T>
void silu_scalar(const T* input, T* output, size_t n) {
    for (size_t i = 0; i < n; i++) {
        output[i] = input[i] / (1 + std::exp(-input[i]));
    }
}

#if defined(CCSM_HAVE_AVX)
void silu_avx_f32(const float* input, float* output, size_t n);
#endif

#if defined(CCSM_HAVE_NEON)
void silu_neon_f32(const float* input, float* output, size_t n);
#endif

} // namespace detail

// Template specializations for float - these will dispatch to the best available implementation
template<>
inline void vector_add<float>(const float* a, const float* b, float* c, size_t n) {
    const auto& features = CPUFeatures::get();
    
#if defined(CCSM_HAVE_AVX2)
    if (features.avx2) {
        detail::vector_add_avx2_f32(a, b, c, n);
        return;
    }
#endif

#if defined(CCSM_HAVE_AVX)
    if (features.avx) {
        detail::vector_add_avx_f32(a, b, c, n);
        return;
    }
#endif

#if defined(CCSM_HAVE_NEON)
    if (features.neon) {
        detail::vector_add_neon_f32(a, b, c, n);
        return;
    }
#endif

    // Fallback to scalar implementation
    detail::vector_add_scalar(a, b, c, n);
}

template<>
inline void vector_mul<float>(const float* a, const float* b, float* c, size_t n) {
    const auto& features = CPUFeatures::get();
    
#if defined(CCSM_HAVE_AVX2)
    if (features.avx2) {
        detail::vector_mul_avx2_f32(a, b, c, n);
        return;
    }
#endif

#if defined(CCSM_HAVE_AVX)
    if (features.avx) {
        detail::vector_mul_avx_f32(a, b, c, n);
        return;
    }
#endif

#if defined(CCSM_HAVE_NEON)
    if (features.neon) {
        detail::vector_mul_neon_f32(a, b, c, n);
        return;
    }
#endif

    // Fallback to scalar implementation
    detail::vector_mul_scalar(a, b, c, n);
}

template<>
inline void vector_fma<float>(const float* a, const float* b, const float* c, float* d, size_t n) {
    const auto& features = CPUFeatures::get();
    
#if defined(CCSM_HAVE_AVX2)
    if (features.avx2) {
        detail::vector_fma_avx2_f32(a, b, c, d, n);
        return;
    }
#endif

#if defined(CCSM_HAVE_NEON)
    if (features.neon) {
        detail::vector_fma_neon_f32(a, b, c, d, n);
        return;
    }
#endif

    // Fallback to scalar implementation
    detail::vector_fma_scalar(a, b, c, d, n);
}

template<>
inline void vector_scale<float>(const float* a, float scalar, float* b, size_t n) {
    const auto& features = CPUFeatures::get();
    
#if defined(CCSM_HAVE_AVX2)
    if (features.avx2) {
        detail::vector_scale_avx2_f32(a, scalar, b, n);
        return;
    }
#endif

#if defined(CCSM_HAVE_AVX)
    if (features.avx) {
        detail::vector_scale_avx_f32(a, scalar, b, n);
        return;
    }
#endif

#if defined(CCSM_HAVE_NEON)
    if (features.neon) {
        detail::vector_scale_neon_f32(a, scalar, b, n);
        return;
    }
#endif

    // Fallback to scalar implementation
    detail::vector_scale_scalar(a, scalar, b, n);
}

template<>
inline float vector_dot<float>(const float* a, const float* b, size_t n) {
    const auto& features = CPUFeatures::get();
    
#if defined(CCSM_HAVE_AVX2)
    if (features.avx2) {
        return detail::vector_dot_avx2_f32(a, b, n);
    }
#endif

#if defined(CCSM_HAVE_AVX)
    if (features.avx) {
        return detail::vector_dot_avx_f32(a, b, n);
    }
#endif

#if defined(CCSM_HAVE_NEON)
    if (features.neon) {
        return detail::vector_dot_neon_f32(a, b, n);
    }
#endif

    // Fallback to scalar implementation
    return detail::vector_dot_scalar(a, b, n);
}

template<>
inline void matrix_mul<float>(const float* a, const float* b, float* c, size_t m, size_t k, size_t n) {
    const auto& features = CPUFeatures::get();
    
#if defined(CCSM_HAVE_AVX2)
    if (features.avx2) {
        detail::matrix_mul_avx2_f32(a, b, c, m, k, n);
        return;
    }
#endif

#if defined(CCSM_HAVE_AVX)
    if (features.avx) {
        detail::matrix_mul_avx_f32(a, b, c, m, k, n);
        return;
    }
#endif

#if defined(CCSM_HAVE_NEON)
    if (features.neon) {
        detail::matrix_mul_neon_f32(a, b, c, m, k, n);
        return;
    }
#endif

    // Fallback to scalar implementation
    detail::matrix_mul_scalar(a, b, c, m, k, n);
}

template<>
inline void softmax<float>(const float* input, float* output, size_t n) {
    const auto& features = CPUFeatures::get();
    
#if defined(CCSM_HAVE_AVX)
    if (features.avx) {
        detail::softmax_avx_f32(input, output, n);
        return;
    }
#endif

#if defined(CCSM_HAVE_NEON)
    if (features.neon) {
        detail::softmax_neon_f32(input, output, n);
        return;
    }
#endif

    // Fallback to scalar implementation
    detail::softmax_scalar(input, output, n);
}

template<>
inline void relu<float>(const float* input, float* output, size_t n) {
    const auto& features = CPUFeatures::get();
    
#if defined(CCSM_HAVE_AVX)
    if (features.avx) {
        detail::relu_avx_f32(input, output, n);
        return;
    }
#endif

#if defined(CCSM_HAVE_NEON)
    if (features.neon) {
        detail::relu_neon_f32(input, output, n);
        return;
    }
#endif

    // Fallback to scalar implementation
    detail::relu_scalar(input, output, n);
}

template<>
inline void silu<float>(const float* input, float* output, size_t n) {
    const auto& features = CPUFeatures::get();
    
#if defined(CCSM_HAVE_AVX)
    if (features.avx) {
        detail::silu_avx_f32(input, output, n);
        return;
    }
#endif

#if defined(CCSM_HAVE_NEON)
    if (features.neon) {
        detail::silu_neon_f32(input, output, n);
        return;
    }
#endif

    // Fallback to scalar implementation
    detail::silu_scalar(input, output, n);
}

// Cache-aware memory layout utilities
inline size_t padded_size(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

// Memory alignment for SIMD operations
template<typename T>
bool is_aligned(const T* ptr, size_t alignment) {
    return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}

} // namespace simd
} // namespace ccsm

#endif // CCSM_SIMD_H