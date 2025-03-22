#ifndef CCSM_SIMD_H
#define CCSM_SIMD_H

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cassert>
#include <string>
#include <algorithm>
#include <vector>
#include <limits>

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

// SIMD implementation type - used for runtime detection
enum class Implementation {
    UNKNOWN,
    SCALAR,
    SSE2,
    SSE41,
    AVX,
    AVX2,
    AVX512,
    NEON
};

// Get current active implementation
Implementation get_active_implementation();

// Get CPU capabilities as string for debugging
std::string get_cpu_capabilities();

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

// Memory alignment utilities
template <typename T>
T* aligned_alloc(size_t n, size_t alignment) {
    void* ptr = nullptr;
#if defined(_MSC_VER)
    ptr = _aligned_malloc(n * sizeof(T), alignment);
#else
    int result = posix_memalign(&ptr, alignment, n * sizeof(T));
    if (result != 0) {
        ptr = nullptr;
    }
#endif
    return static_cast<T*>(ptr);
}

inline void aligned_free(void* ptr) {
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

inline void* align_ptr(void* ptr, size_t alignment) {
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t aligned = (addr + alignment - 1) & ~(alignment - 1);
    return reinterpret_cast<void*>(aligned);
}

// Vector comparison with mask
template<typename T>
void vector_gt_mask(T* result, const T* a, const T* b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        result[i] = (a[i] > b[i]) ? T(1) : T(0);
    }
}

// Forward declarations for vectorized operations
template<typename T>
void vector_add(T* result, const T* a, const T* b, size_t n);

template<typename T>
void vector_mul(T* result, const T* a, const T* b, size_t n);

template<typename T>
void vector_fma(T* result, const T* a, const T* b, const T* c, size_t n); // result = a * b + c

template<typename T>
void vector_scale(T* result, const T* a, T scalar, size_t n);

template<typename T>
T vector_dot(const T* a, const T* b, size_t n);

template<typename T>
void matrix_mul(T* result, const T* a, const T* b, size_t m, size_t k, size_t n);

template<typename T>
void softmax(T* output, const T* input, size_t n);

template<typename T>
void relu(T* output, const T* input, size_t n);

template<typename T>
void silu(T* output, const T* input, size_t n);

template<typename T>
void rms_norm(T* output, const T* input, const T* weight, T epsilon, size_t n);

template<typename T>
void layer_norm(T* output, const T* input, const T* weight, const T* bias, T epsilon, size_t n);

template<typename T>
void attention(
    T* output,              // [batch_size, seq_len, head_size]
    const T* query,         // [batch_size, seq_len, num_heads, head_size]
    const T* key,           // [batch_size, seq_len, num_heads, head_size]
    const T* value,         // [batch_size, seq_len, num_heads, head_size]
    const T* mask,          // [batch_size, 1, 1, seq_len] or nullptr
    size_t batch_size,
    size_t seq_len,
    size_t num_heads,
    size_t head_size,
    T scale = 1.0f          // Scaling factor (typically 1/sqrt(head_size))
);

// Fused operations for better performance
template<typename T>
void fused_rms_norm_silu(T* output, const T* input, const T* weight, T epsilon, size_t n);

template<typename T>
void fused_layer_norm_relu(T* output, const T* input, const T* weight, const T* bias, T epsilon, size_t n);

// Implementation details namespace (not part of public API)
namespace detail {

// Vector add implementations
template<typename T>
void vector_add_scalar(T* result, const T* a, const T* b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        result[i] = a[i] + b[i];
    }
}

#if defined(CCSM_HAVE_AVX)
void vector_add_avx_f32(float* result, const float* a, const float* b, size_t n);
#endif

#if defined(CCSM_HAVE_AVX2)
void vector_add_avx2_f32(float* result, const float* a, const float* b, size_t n);
#endif

#if defined(CCSM_HAVE_NEON)
void vector_add_neon_f32(float* result, const float* a, const float* b, size_t n);
#endif

// Vector multiply implementations
template<typename T>
void vector_mul_scalar(T* result, const T* a, const T* b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        result[i] = a[i] * b[i];
    }
}

#if defined(CCSM_HAVE_AVX)
void vector_mul_avx_f32(float* result, const float* a, const float* b, size_t n);
#endif

#if defined(CCSM_HAVE_AVX2)
void vector_mul_avx2_f32(float* result, const float* a, const float* b, size_t n);
#endif

#if defined(CCSM_HAVE_NEON)
void vector_mul_neon_f32(float* result, const float* a, const float* b, size_t n);
#endif

// Vector FMA implementations
template<typename T>
void vector_fma_scalar(T* result, const T* a, const T* b, const T* c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        result[i] = a[i] * b[i] + c[i];
    }
}

#if defined(CCSM_HAVE_AVX2)
void vector_fma_avx2_f32(float* result, const float* a, const float* b, const float* c, size_t n);
#endif

#if defined(CCSM_HAVE_NEON)
void vector_fma_neon_f32(float* result, const float* a, const float* b, const float* c, size_t n);
#endif

// Vector scale implementations
template<typename T>
void vector_scale_scalar(T* result, const T* a, T scalar, size_t n) {
    for (size_t i = 0; i < n; i++) {
        result[i] = a[i] * scalar;
    }
}

#if defined(CCSM_HAVE_AVX)
void vector_scale_avx_f32(float* result, const float* a, float scalar, size_t n);
#endif

#if defined(CCSM_HAVE_AVX2)
void vector_scale_avx2_f32(float* result, const float* a, float scalar, size_t n);
#endif

#if defined(CCSM_HAVE_NEON)
void vector_scale_neon_f32(float* result, const float* a, float scalar, size_t n);
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

// Vector comparison implementations
template<typename T>
void vector_gt_mask_scalar(T* result, const T* a, const T* b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        result[i] = (a[i] > b[i]) ? T(1) : T(0);
    }
}

#if defined(CCSM_HAVE_AVX)
void vector_gt_mask_avx_f32(float* result, const float* a, const float* b, size_t n);
#endif

#if defined(CCSM_HAVE_NEON)
void vector_gt_mask_neon_f32(float* result, const float* a, const float* b, size_t n);
#endif

// Matrix multiply implementations
template<typename T>
void matrix_mul_scalar(T* result, const T* a, const T* b, size_t m, size_t k, size_t n) {
    // Initialize result to zero
    for (size_t i = 0; i < m * n; i++) {
        result[i] = 0;
    }
    
    // Perform matrix multiplication
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            for (size_t l = 0; l < k; l++) {
                result[i * n + j] += a[i * k + l] * b[l * n + j];
            }
        }
    }
}

#if defined(CCSM_HAVE_AVX)
void matrix_mul_avx_f32(float* result, const float* a, const float* b, size_t m, size_t k, size_t n);
#endif

#if defined(CCSM_HAVE_AVX2)
void matrix_mul_avx2_f32(float* result, const float* a, const float* b, size_t m, size_t k, size_t n);
#endif

#if defined(CCSM_HAVE_NEON)
void matrix_mul_neon_f32(float* result, const float* a, const float* b, size_t m, size_t k, size_t n);
#endif

// Softmax implementations
template<typename T>
void softmax_scalar(T* output, const T* input, size_t n) {
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
void softmax_avx_f32(float* output, const float* input, size_t n);
#endif

#if defined(CCSM_HAVE_NEON)
void softmax_neon_f32(float* output, const float* input, size_t n);
#endif

// ReLU implementations
template<typename T>
void relu_scalar(T* output, const T* input, size_t n) {
    for (size_t i = 0; i < n; i++) {
        output[i] = std::max(T(0), input[i]);
    }
}

#if defined(CCSM_HAVE_AVX)
void relu_avx_f32(float* output, const float* input, size_t n);
#endif

#if defined(CCSM_HAVE_NEON)
void relu_neon_f32(float* output, const float* input, size_t n);
#endif

// SiLU implementations
template<typename T>
void silu_scalar(T* output, const T* input, size_t n) {
    for (size_t i = 0; i < n; i++) {
        output[i] = input[i] / (1 + std::exp(-input[i]));
    }
}

#if defined(CCSM_HAVE_AVX)
void silu_avx_f32(float* output, const float* input, size_t n);
#endif

#if defined(CCSM_HAVE_NEON)
void silu_neon_f32(float* output, const float* input, size_t n);
#endif

// RMS Normalization implementations
template<typename T>
void rms_norm_scalar(T* output, const T* input, const T* weight, T epsilon, size_t n) {
    // Calculate sum of squares
    T ss = 0;
    for (size_t i = 0; i < n; i++) {
        ss += input[i] * input[i];
    }
    
    // Calculate normalization factor
    T norm_factor = 1.0f / std::sqrt(ss / n + epsilon);
    
    // Apply normalization with weights
    for (size_t i = 0; i < n; i++) {
        output[i] = input[i] * norm_factor * weight[i];
    }
}

#if defined(CCSM_HAVE_AVX)
void rms_norm_avx_f32(float* output, const float* input, const float* weight, float epsilon, size_t n);
#endif

#if defined(CCSM_HAVE_NEON)
void rms_norm_neon_f32(float* output, const float* input, const float* weight, float epsilon, size_t n);
#endif

// Layer Normalization implementations
template<typename T>
void layer_norm_scalar(T* output, const T* input, const T* weight, const T* bias, T epsilon, size_t n) {
    // Calculate mean
    T sum = 0;
    for (size_t i = 0; i < n; i++) {
        sum += input[i];
    }
    T mean = sum / n;
    
    // Calculate variance
    T variance = 0;
    for (size_t i = 0; i < n; i++) {
        T diff = input[i] - mean;
        variance += diff * diff;
    }
    variance /= n;
    
    // Calculate normalization factor
    T norm_factor = 1.0f / std::sqrt(variance + epsilon);
    
    // Apply normalization with weights and bias
    for (size_t i = 0; i < n; i++) {
        output[i] = (input[i] - mean) * norm_factor * weight[i] + bias[i];
    }
}

#if defined(CCSM_HAVE_AVX)
void layer_norm_avx_f32(float* output, const float* input, const float* weight, const float* bias, float epsilon, size_t n);
#endif

#if defined(CCSM_HAVE_NEON)
void layer_norm_neon_f32(float* output, const float* input, const float* weight, const float* bias, float epsilon, size_t n);
#endif

// Attention implementations
template<typename T>
void attention_scalar(
    T* output,              // [batch_size, seq_len, head_size]
    const T* query,         // [batch_size, seq_len, num_heads, head_size]
    const T* key,           // [batch_size, seq_len, num_heads, head_size]
    const T* value,         // [batch_size, seq_len, num_heads, head_size]
    const T* mask,          // [batch_size, 1, 1, seq_len] or nullptr
    size_t batch_size,
    size_t seq_len,
    size_t num_heads,
    size_t head_size,
    T scale
) {
    // Allocate temporary buffers for attention scores and softmax results
    std::vector<T> scores(seq_len * seq_len);
    std::vector<T> softmax_out(seq_len);
    
    // Initialize output to zero
    const size_t output_size = batch_size * seq_len * head_size;
    for (size_t i = 0; i < output_size; i++) {
        output[i] = 0;
    }
    
    // Process each batch and head
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t h = 0; h < num_heads; h++) {
            // Step 1: Compute query-key attention scores
            for (size_t i = 0; i < seq_len; i++) {
                for (size_t j = 0; j < seq_len; j++) {
                    // Dot product of query[i] and key[j]
                    T dot = 0;
                    for (size_t k = 0; k < head_size; k++) {
                        size_t q_idx = ((b * seq_len + i) * num_heads + h) * head_size + k;
                        size_t k_idx = ((b * seq_len + j) * num_heads + h) * head_size + k;
                        dot += query[q_idx] * key[k_idx];
                    }
                    // Scale the dot product
                    scores[i * seq_len + j] = dot * scale;
                    
                    // Apply mask if provided
                    if (mask != nullptr) {
                        // Apply causal mask (if mask[b,1,1,j] == 0, then hide position j from position i)
                        size_t mask_idx = b * seq_len + j; // Simplified for causal mask
                        if (j > i || mask[mask_idx] == 0) {
                            scores[i * seq_len + j] = -std::numeric_limits<T>::infinity();
                        }
                    }
                }
                
                // Step 2: Apply softmax to get attention weights
                T max_val = scores[i * seq_len];
                for (size_t j = 1; j < seq_len; j++) {
                    max_val = std::max(max_val, scores[i * seq_len + j]);
                }
                
                T sum_exp = 0;
                for (size_t j = 0; j < seq_len; j++) {
                    T exp_val = std::exp(scores[i * seq_len + j] - max_val);
                    softmax_out[j] = exp_val;
                    sum_exp += exp_val;
                }
                
                // Normalize
                if (sum_exp > 0) {
                    for (size_t j = 0; j < seq_len; j++) {
                        softmax_out[j] /= sum_exp;
                    }
                }
                
                // Step 3: Apply attention weights to values
                for (size_t k = 0; k < head_size; k++) {
                    T weighted_sum = 0;
                    for (size_t j = 0; j < seq_len; j++) {
                        size_t v_idx = ((b * seq_len + j) * num_heads + h) * head_size + k;
                        weighted_sum += softmax_out[j] * value[v_idx];
                    }
                    
                    // Store in output
                    size_t out_idx = (b * seq_len + i) * head_size + k;
                    output[out_idx] += weighted_sum;
                }
            }
        }
    }
}

#if defined(CCSM_HAVE_AVX)
void attention_avx_f32(
    float* output,
    const float* query,
    const float* key,
    const float* value,
    const float* mask,
    size_t batch_size,
    size_t seq_len,
    size_t num_heads,
    size_t head_size,
    float scale
);
#endif

#if defined(CCSM_HAVE_NEON)
void attention_neon_f32(
    float* output,
    const float* query,
    const float* key,
    const float* value,
    const float* mask,
    size_t batch_size,
    size_t seq_len,
    size_t num_heads,
    size_t head_size,
    float scale
);
#endif

// Fused operations - scalar implementations
template<typename T>
void fused_rms_norm_silu_scalar(T* output, const T* input, const T* weight, T epsilon, size_t n) {
    // Calculate sum of squares
    T ss = 0;
    for (size_t i = 0; i < n; i++) {
        ss += input[i] * input[i];
    }
    
    // Calculate normalization factor
    T norm_factor = 1.0f / std::sqrt(ss / n + epsilon);
    
    // Apply normalization with weights and SiLU activation in one pass
    for (size_t i = 0; i < n; i++) {
        T normalized = input[i] * norm_factor * weight[i];
        // SiLU: x * sigmoid(x)
        output[i] = normalized / (1.0f + std::exp(-normalized)) * normalized;
    }
}

#if defined(CCSM_HAVE_AVX)
void fused_rms_norm_silu_avx_f32(float* output, const float* input, const float* weight, float epsilon, size_t n);
#endif

#if defined(CCSM_HAVE_NEON)
void fused_rms_norm_silu_neon_f32(float* output, const float* input, const float* weight, float epsilon, size_t n);
#endif

template<typename T>
void fused_layer_norm_relu_scalar(T* output, const T* input, const T* weight, const T* bias, T epsilon, size_t n) {
    // Calculate mean
    T sum = 0;
    for (size_t i = 0; i < n; i++) {
        sum += input[i];
    }
    T mean = sum / n;
    
    // Calculate variance
    T variance = 0;
    for (size_t i = 0; i < n; i++) {
        T diff = input[i] - mean;
        variance += diff * diff;
    }
    variance /= n;
    
    // Calculate normalization factor
    T norm_factor = 1.0f / std::sqrt(variance + epsilon);
    
    // Apply normalization with weights, bias, and ReLU activation in one pass
    for (size_t i = 0; i < n; i++) {
        T normalized = (input[i] - mean) * norm_factor * weight[i] + bias[i];
        // ReLU: max(0, x)
        output[i] = (normalized > 0) ? normalized : 0;
    }
}

#if defined(CCSM_HAVE_AVX)
void fused_layer_norm_relu_avx_f32(float* output, const float* input, const float* weight, const float* bias, float epsilon, size_t n);
#endif

#if defined(CCSM_HAVE_NEON)
void fused_layer_norm_relu_neon_f32(float* output, const float* input, const float* weight, const float* bias, float epsilon, size_t n);
#endif

} // namespace detail

// Template specializations for float - these will dispatch to the best available implementation
template<>
inline void vector_add<float>(float* result, const float* a, const float* b, size_t n) {
    const auto& features = CPUFeatures::get();
    
#if defined(CCSM_HAVE_AVX2)
    if (features.avx2) {
        detail::vector_add_avx2_f32(result, a, b, n);
        return;
    }
#endif

#if defined(CCSM_HAVE_AVX)
    if (features.avx) {
        detail::vector_add_avx_f32(result, a, b, n);
        return;
    }
#endif

#if defined(CCSM_HAVE_NEON)
    if (features.neon) {
        detail::vector_add_neon_f32(result, a, b, n);
        return;
    }
#endif

    // Fallback to scalar implementation
    detail::vector_add_scalar(result, a, b, n);
}

template<>
inline void vector_mul<float>(float* result, const float* a, const float* b, size_t n) {
    const auto& features = CPUFeatures::get();
    
#if defined(CCSM_HAVE_AVX2)
    if (features.avx2) {
        detail::vector_mul_avx2_f32(result, a, b, n);
        return;
    }
#endif

#if defined(CCSM_HAVE_AVX)
    if (features.avx) {
        detail::vector_mul_avx_f32(result, a, b, n);
        return;
    }
#endif

#if defined(CCSM_HAVE_NEON)
    if (features.neon) {
        detail::vector_mul_neon_f32(result, a, b, n);
        return;
    }
#endif

    // Fallback to scalar implementation
    detail::vector_mul_scalar(result, a, b, n);
}

template<>
inline void vector_fma<float>(float* result, const float* a, const float* b, const float* c, size_t n) {
    const auto& features = CPUFeatures::get();
    
#if defined(CCSM_HAVE_AVX2)
    if (features.avx2) {
        detail::vector_fma_avx2_f32(result, a, b, c, n);
        return;
    }
#endif

#if defined(CCSM_HAVE_NEON)
    if (features.neon) {
        detail::vector_fma_neon_f32(result, a, b, c, n);
        return;
    }
#endif

    // Fallback to scalar implementation
    detail::vector_fma_scalar(result, a, b, c, n);
}

template<>
inline void vector_scale<float>(float* result, const float* a, float scalar, size_t n) {
    const auto& features = CPUFeatures::get();
    
#if defined(CCSM_HAVE_AVX2)
    if (features.avx2) {
        detail::vector_scale_avx2_f32(result, a, scalar, n);
        return;
    }
#endif

#if defined(CCSM_HAVE_AVX)
    if (features.avx) {
        detail::vector_scale_avx_f32(result, a, scalar, n);
        return;
    }
#endif

#if defined(CCSM_HAVE_NEON)
    if (features.neon) {
        detail::vector_scale_neon_f32(result, a, scalar, n);
        return;
    }
#endif

    // Fallback to scalar implementation
    detail::vector_scale_scalar(result, a, scalar, n);
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
inline void vector_gt_mask<float>(float* result, const float* a, const float* b, size_t n) {
    const auto& features = CPUFeatures::get();
    
#if defined(CCSM_HAVE_AVX)
    if (features.avx) {
        detail::vector_gt_mask_avx_f32(result, a, b, n);
        return;
    }
#endif

#if defined(CCSM_HAVE_NEON)
    if (features.neon) {
        detail::vector_gt_mask_neon_f32(result, a, b, n);
        return;
    }
#endif

    // Fallback to scalar implementation
    detail::vector_gt_mask_scalar(result, a, b, n);
}

template<>
inline void matrix_mul<float>(float* result, const float* a, const float* b, size_t m, size_t k, size_t n) {
    const auto& features = CPUFeatures::get();
    
#if defined(CCSM_HAVE_AVX2)
    if (features.avx2) {
        detail::matrix_mul_avx2_f32(result, a, b, m, k, n);
        return;
    }
#endif

#if defined(CCSM_HAVE_AVX)
    if (features.avx) {
        detail::matrix_mul_avx_f32(result, a, b, m, k, n);
        return;
    }
#endif

#if defined(CCSM_HAVE_NEON)
    if (features.neon) {
        detail::matrix_mul_neon_f32(result, a, b, m, k, n);
        return;
    }
#endif

    // Fallback to scalar implementation
    detail::matrix_mul_scalar(result, a, b, m, k, n);
}

template<>
inline void softmax<float>(float* output, const float* input, size_t n) {
    const auto& features = CPUFeatures::get();
    
#if defined(CCSM_HAVE_AVX)
    if (features.avx) {
        detail::softmax_avx_f32(output, input, n);
        return;
    }
#endif

#if defined(CCSM_HAVE_NEON)
    if (features.neon) {
        detail::softmax_neon_f32(output, input, n);
        return;
    }
#endif

    // Fallback to scalar implementation
    detail::softmax_scalar(output, input, n);
}

template<>
inline void relu<float>(float* output, const float* input, size_t n) {
    const auto& features = CPUFeatures::get();
    
#if defined(CCSM_HAVE_AVX)
    if (features.avx) {
        detail::relu_avx_f32(output, input, n);
        return;
    }
#endif

#if defined(CCSM_HAVE_NEON)
    if (features.neon) {
        detail::relu_neon_f32(output, input, n);
        return;
    }
#endif

    // Fallback to scalar implementation
    detail::relu_scalar(output, input, n);
}

template<>
inline void silu<float>(float* output, const float* input, size_t n) {
    const auto& features = CPUFeatures::get();
    
#if defined(CCSM_HAVE_AVX)
    if (features.avx) {
        detail::silu_avx_f32(output, input, n);
        return;
    }
#endif

#if defined(CCSM_HAVE_NEON)
    if (features.neon) {
        detail::silu_neon_f32(output, input, n);
        return;
    }
#endif

    // Fallback to scalar implementation
    detail::silu_scalar(output, input, n);
}

template<>
inline void rms_norm<float>(float* output, const float* input, const float* weight, float epsilon, size_t n) {
    const auto& features = CPUFeatures::get();
    
#if defined(CCSM_HAVE_AVX)
    if (features.avx) {
        detail::rms_norm_avx_f32(output, input, weight, epsilon, n);
        return;
    }
#endif

#if defined(CCSM_HAVE_NEON)
    if (features.neon) {
        detail::rms_norm_neon_f32(output, input, weight, epsilon, n);
        return;
    }
#endif

    // Fallback to scalar implementation
    detail::rms_norm_scalar(output, input, weight, epsilon, n);
}

template<>
inline void layer_norm<float>(float* output, const float* input, const float* weight, const float* bias, float epsilon, size_t n) {
    const auto& features = CPUFeatures::get();
    
#if defined(CCSM_HAVE_AVX)
    if (features.avx) {
        detail::layer_norm_avx_f32(output, input, weight, bias, epsilon, n);
        return;
    }
#endif

#if defined(CCSM_HAVE_NEON)
    if (features.neon) {
        detail::layer_norm_neon_f32(output, input, weight, bias, epsilon, n);
        return;
    }
#endif

    // Fallback to scalar implementation
    detail::layer_norm_scalar(output, input, weight, bias, epsilon, n);
}

template<>
inline void attention<float>(
    float* output,
    const float* query,
    const float* key,
    const float* value,
    const float* mask,
    size_t batch_size,
    size_t seq_len,
    size_t num_heads,
    size_t head_size,
    float scale
) {
    const auto& features = CPUFeatures::get();
    
#if defined(CCSM_HAVE_AVX)
    if (features.avx) {
        detail::attention_avx_f32(output, query, key, value, mask, 
                                 batch_size, seq_len, num_heads, head_size, scale);
        return;
    }
#endif

#if defined(CCSM_HAVE_NEON)
    if (features.neon) {
        detail::attention_neon_f32(output, query, key, value, mask, 
                                  batch_size, seq_len, num_heads, head_size, scale);
        return;
    }
#endif

    // Fallback to scalar implementation
    detail::attention_scalar(output, query, key, value, mask, 
                           batch_size, seq_len, num_heads, head_size, scale);
}

// Fused operation specializations for float
template<>
inline void fused_rms_norm_silu<float>(float* output, const float* input, const float* weight, float epsilon, size_t n) {
    const auto& features = CPUFeatures::get();
    
#if defined(CCSM_HAVE_AVX)
    if (features.avx) {
        detail::fused_rms_norm_silu_avx_f32(output, input, weight, epsilon, n);
        return;
    }
#endif

#if defined(CCSM_HAVE_NEON)
    if (features.neon) {
        detail::fused_rms_norm_silu_neon_f32(output, input, weight, epsilon, n);
        return;
    }
#endif

    // Fallback to scalar implementation
    detail::fused_rms_norm_silu_scalar(output, input, weight, epsilon, n);
}

template<>
inline void fused_layer_norm_relu<float>(float* output, const float* input, const float* weight, const float* bias, float epsilon, size_t n) {
    const auto& features = CPUFeatures::get();
    
#if defined(CCSM_HAVE_AVX)
    if (features.avx) {
        detail::fused_layer_norm_relu_avx_f32(output, input, weight, bias, epsilon, n);
        return;
    }
#endif

#if defined(CCSM_HAVE_NEON)
    if (features.neon) {
        detail::fused_layer_norm_relu_neon_f32(output, input, weight, bias, epsilon, n);
        return;
    }
#endif

    // Fallback to scalar implementation
    detail::fused_layer_norm_relu_scalar(output, input, weight, bias, epsilon, n);
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