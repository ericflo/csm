#include <ccsm/cpu/simd.h>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <cmath>

#if defined(__x86_64__) || defined(_M_X64)
#define CCSM_HAVE_AVX
#include <immintrin.h>
#endif

#if defined(CCSM_HAVE_NEON)
#include <arm_neon.h>
#endif

namespace ccsm {
namespace simd {

// CPU features detection implementation
const CPUFeatures& CPUFeatures::get() {
    static CPUFeatures features;
    static bool initialized = false;
    
    if (!initialized) {
        // Detect CPU features at runtime
        #if defined(CCSM_ARCH_X86_64)
            // X86_64 feature detection
            #if defined(_MSC_VER)
                // MSVC implementation
                int cpuInfo[4];
                __cpuid(cpuInfo, 1);
                
                features.sse2 = (cpuInfo[3] & (1 << 26)) != 0;
                features.sse3 = (cpuInfo[2] & (1 << 0)) != 0;
                features.sse4_1 = (cpuInfo[2] & (1 << 19)) != 0;
                features.sse4_2 = (cpuInfo[2] & (1 << 20)) != 0;
                features.avx = (cpuInfo[2] & (1 << 28)) != 0;
                
                // Check for AVX2
                __cpuid(cpuInfo, 7);
                features.avx2 = (cpuInfo[1] & (1 << 5)) != 0;
                features.avx512f = (cpuInfo[1] & (1 << 16)) != 0;
            #else
                // GCC/Clang implementation
                features.sse2 = __builtin_cpu_supports("sse2");
                features.sse3 = __builtin_cpu_supports("sse3");
                features.sse4_1 = __builtin_cpu_supports("sse4.1");
                features.sse4_2 = __builtin_cpu_supports("sse4.2");
                features.avx = __builtin_cpu_supports("avx");
                features.avx2 = __builtin_cpu_supports("avx2");
                features.avx512f = __builtin_cpu_supports("avx512f");
            #endif
        #elif defined(CCSM_ARCH_ARM) || defined(CCSM_ARCH_ARM64)
            // ARM feature detection
            #if defined(__aarch64__)
                // ARM64 always has NEON
                features.neon = true;
            #elif defined(__ARM_NEON) || defined(__ARM_NEON__)
                // 32-bit ARM with NEON
                features.neon = true;
            #else
                // No NEON
                features.neon = false;
            #endif
        #endif
        
        initialized = true;
    }
    
    return features;
}

// Active implementation - determined at runtime based on CPU features
Implementation get_active_implementation() {
    static Implementation impl = Implementation::SCALAR;
    static bool initialized = false;
    
    if (!initialized) {
        const CPUFeatures& features = CPUFeatures::get();
        
        // Determine best implementation based on CPU features
        #if defined(CCSM_ARCH_X86_64)
            if (features.avx512f) {
                impl = Implementation::AVX512;
            } else if (features.avx2) {
                impl = Implementation::AVX2;
            } else if (features.avx) {
                impl = Implementation::AVX;
            } else if (features.sse4_1) {
                impl = Implementation::SSE41;
            } else if (features.sse2) {
                impl = Implementation::SSE2;
            }
        #elif defined(CCSM_ARCH_ARM) || defined(CCSM_ARCH_ARM64)
            if (features.neon) {
                impl = Implementation::NEON;
            }
        #endif
        
        initialized = true;
    }
    
    return impl;
}

// CPU capabilities string for diagnostics
std::string get_cpu_capabilities() {
    const CPUFeatures& features = CPUFeatures::get();
    std::string result;
    
    #if defined(CCSM_ARCH_X86_64)
        if (features.sse2) result += "SSE2 ";
        if (features.sse3) result += "SSE3 ";
        if (features.sse4_1) result += "SSE4.1 ";
        if (features.sse4_2) result += "SSE4.2 ";
        if (features.avx) result += "AVX ";
        if (features.avx2) result += "AVX2 ";
        if (features.avx512f) result += "AVX512F ";
    #elif defined(CCSM_ARCH_ARM) || defined(CCSM_ARCH_ARM64)
        if (features.neon) result += "NEON ";
    #endif
    
    if (result.empty()) {
        result = "None";
    } else {
        // Remove trailing space
        result.pop_back();
    }
    
    return result;
}

namespace detail {

// SIMD-optimized sigmoid activation function (AVX)
#if defined(CCSM_HAVE_AVX)
void sigmoid_avx(float* output, const float* input, size_t n) {
    // Constants for sigmoid approximation
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 neg_one = _mm256_set1_ps(-1.0f);
    const __m256 half = _mm256_set1_ps(0.5f);
    
    // Process 8 elements at a time (AVX register width)
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        // Load 8 input values
        __m256 x = _mm256_loadu_ps(input + i);
        
        // Compute sigmoid: 1.0f / (1.0f + exp(-x))
        // AVX doesn't have exp directly, so we use the following approximation:
        // 0.5 * (tanh(x/2) + 1)
        
        // Compute x/2
        __m256 x_half = _mm256_mul_ps(x, half);
        
        // Compute tanh(x/2) using rational approximation
        // tanh(x) = x * (27 + x^2) / (27 + 9 * x^2)
        __m256 x2 = _mm256_mul_ps(x_half, x_half);
        
        __m256 num = _mm256_mul_ps(x_half, _mm256_add_ps(_mm256_set1_ps(27.0f), x2));
        __m256 den = _mm256_add_ps(_mm256_set1_ps(27.0f), _mm256_mul_ps(_mm256_set1_ps(9.0f), x2));
        __m256 tanh_x_half = _mm256_div_ps(num, den);
        
        // Compute sigmoid: 0.5 * (tanh(x/2) + 1)
        __m256 result = _mm256_mul_ps(half, _mm256_add_ps(tanh_x_half, one));
        
        // Store the results
        _mm256_storeu_ps(output + i, result);
    }
    
    // Handle remaining elements
    for (; i < n; i++) {
        output[i] = 1.0f / (1.0f + std::exp(-input[i]));
    }
}
#endif

// SIMD-optimized sigmoid activation function (NEON)
#if defined(CCSM_HAVE_NEON)
void sigmoid_neon(float* output, const float* input, size_t n) {
    // Constants for sigmoid approximation
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t half = vdupq_n_f32(0.5f);
    
    // Process 4 elements at a time (NEON register width)
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        // Load 4 input values
        float32x4_t x = vld1q_f32(input + i);
        
        // Compute sigmoid using an approximation
        // 0.5 * (tanh(x/2) + 1)
        
        // Compute x/2
        float32x4_t x_half = vmulq_f32(x, half);
        
        // Compute tanh(x/2) using rational approximation
        // tanh(x) = x * (27 + x^2) / (27 + 9 * x^2)
        float32x4_t x2 = vmulq_f32(x_half, x_half);
        
        float32x4_t num = vmulq_f32(x_half, vaddq_f32(vdupq_n_f32(27.0f), x2));
        float32x4_t den = vaddq_f32(vdupq_n_f32(27.0f), vmulq_f32(vdupq_n_f32(9.0f), x2));
        float32x4_t tanh_x_half = vdivq_f32(num, den);
        
        // Compute sigmoid: 0.5 * (tanh(x/2) + 1)
        float32x4_t result = vmulq_f32(half, vaddq_f32(tanh_x_half, one));
        
        // Store the results
        vst1q_f32(output + i, result);
    }
    
    // Handle remaining elements
    for (; i < n; i++) {
        output[i] = 1.0f / (1.0f + std::exp(-input[i]));
    }
}
#endif

// Scalar implementation of sigmoid activation function
void sigmoid_scalar(float* output, const float* input, size_t n) {
    for (size_t i = 0; i < n; i++) {
        output[i] = 1.0f / (1.0f + std::exp(-input[i]));
    }
}

// SIMD-optimized SiLU activation function (AVX)
#if defined(CCSM_HAVE_AVX)
void silu_avx(float* output, const float* input, size_t n) {
    // Constants for sigmoid approximation
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 half = _mm256_set1_ps(0.5f);
    
    // Process 8 elements at a time (AVX register width)
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        // Load 8 input values
        __m256 x = _mm256_loadu_ps(input + i);
        
        // Compute sigmoid: 1.0f / (1.0f + exp(-x))
        // AVX doesn't have exp directly, so we use the following approximation:
        // 0.5 * (tanh(x/2) + 1)
        
        // Compute x/2
        __m256 x_half = _mm256_mul_ps(x, half);
        
        // Compute tanh(x/2) using rational approximation
        // tanh(x) = x * (27 + x^2) / (27 + 9 * x^2)
        __m256 x2 = _mm256_mul_ps(x_half, x_half);
        
        __m256 num = _mm256_mul_ps(x_half, _mm256_add_ps(_mm256_set1_ps(27.0f), x2));
        __m256 den = _mm256_add_ps(_mm256_set1_ps(27.0f), _mm256_mul_ps(_mm256_set1_ps(9.0f), x2));
        __m256 tanh_x_half = _mm256_div_ps(num, den);
        
        // Compute sigmoid: 0.5 * (tanh(x/2) + 1)
        __m256 sigmoid = _mm256_mul_ps(half, _mm256_add_ps(tanh_x_half, one));
        
        // Compute SiLU: x * sigmoid(x)
        __m256 result = _mm256_mul_ps(x, sigmoid);
        
        // Store the results
        _mm256_storeu_ps(output + i, result);
    }
    
    // Handle remaining elements
    for (; i < n; i++) {
        float sigmoid = 1.0f / (1.0f + std::exp(-input[i]));
        output[i] = input[i] * sigmoid;
    }
}
#endif

// SIMD-optimized SiLU activation function (NEON)
#if defined(CCSM_HAVE_NEON)
void silu_neon(float* output, const float* input, size_t n) {
    // Constants for sigmoid approximation
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t half = vdupq_n_f32(0.5f);
    
    // Process 4 elements at a time (NEON register width)
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        // Load 4 input values
        float32x4_t x = vld1q_f32(input + i);
        
        // Compute sigmoid using an approximation
        // 0.5 * (tanh(x/2) + 1)
        
        // Compute x/2
        float32x4_t x_half = vmulq_f32(x, half);
        
        // Compute tanh(x/2) using rational approximation
        // tanh(x) = x * (27 + x^2) / (27 + 9 * x^2)
        float32x4_t x2 = vmulq_f32(x_half, x_half);
        
        float32x4_t num = vmulq_f32(x_half, vaddq_f32(vdupq_n_f32(27.0f), x2));
        float32x4_t den = vaddq_f32(vdupq_n_f32(27.0f), vmulq_f32(vdupq_n_f32(9.0f), x2));
        float32x4_t tanh_x_half = vdivq_f32(num, den);
        
        // Compute sigmoid: 0.5 * (tanh(x/2) + 1)
        float32x4_t sigmoid = vmulq_f32(half, vaddq_f32(tanh_x_half, one));
        
        // Compute SiLU: x * sigmoid(x)
        float32x4_t result = vmulq_f32(x, sigmoid);
        
        // Store the results
        vst1q_f32(output + i, result);
    }
    
    // Handle remaining elements
    for (; i < n; i++) {
        float sigmoid = 1.0f / (1.0f + std::exp(-input[i]));
        output[i] = input[i] * sigmoid;
    }
}
#endif

// Scalar implementation of SiLU activation function
void silu_scalar(float* output, const float* input, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float sigmoid = 1.0f / (1.0f + std::exp(-input[i]));
        output[i] = input[i] * sigmoid;
    }
}

// SIMD-optimized ReLU activation function (AVX)
#if defined(CCSM_HAVE_AVX)
void relu_avx(float* output, const float* input, size_t n) {
    // Constants for ReLU
    const __m256 zero = _mm256_setzero_ps();
    
    // Process 8 elements at a time (AVX register width)
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        // Load 8 input values
        __m256 x = _mm256_loadu_ps(input + i);
        
        // Compute ReLU: max(0, x)
        __m256 result = _mm256_max_ps(zero, x);
        
        // Store the results
        _mm256_storeu_ps(output + i, result);
    }
    
    // Handle remaining elements
    for (; i < n; i++) {
        output[i] = std::max(0.0f, input[i]);
    }
}
#endif

// SIMD-optimized ReLU activation function (NEON)
#if defined(CCSM_HAVE_NEON)
void relu_neon(float* output, const float* input, size_t n) {
    // Constants for ReLU
    const float32x4_t zero = vdupq_n_f32(0.0f);
    
    // Process 4 elements at a time (NEON register width)
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        // Load 4 input values
        float32x4_t x = vld1q_f32(input + i);
        
        // Compute ReLU: max(0, x)
        float32x4_t result = vmaxq_f32(zero, x);
        
        // Store the results
        vst1q_f32(output + i, result);
    }
    
    // Handle remaining elements
    for (; i < n; i++) {
        output[i] = std::max(0.0f, input[i]);
    }
}
#endif

// Scalar implementation of ReLU activation function
void relu_scalar(float* output, const float* input, size_t n) {
    for (size_t i = 0; i < n; i++) {
        output[i] = std::max(0.0f, input[i]);
    }
}

// Fused RMSNorm + SiLU implementation (AVX)
#if defined(CCSM_HAVE_AVX)
void rms_norm_silu_avx(float* output, const float* input, const float* weight, size_t n, float eps) {
    // Constants
    const __m256 eps_vec = _mm256_set1_ps(eps);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 half = _mm256_set1_ps(0.5f);
    
    // First pass: calculate sum of squares
    __m256 sum_sq = _mm256_setzero_ps();
    size_t i = 0;
    
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);
        sum_sq = _mm256_add_ps(sum_sq, _mm256_mul_ps(x, x));
    }
    
    // Horizontal sum of sum_sq
    __m256 hsum = _mm256_hadd_ps(sum_sq, sum_sq);
    hsum = _mm256_hadd_ps(hsum, hsum);
    
    // Extract the sum from the first and fifth elements
    float sum_squares = _mm256_cvtss_f32(hsum) + _mm256_cvtss_f32(_mm256_permute_ps(hsum, 0x01));
    
    // Handle remaining elements with scalar code
    for (; i < n; i++) {
        sum_squares += input[i] * input[i];
    }
    
    // Calculate normalization factor
    float scale = 1.0f / std::sqrt(sum_squares / n + eps);
    __m256 scale_vec = _mm256_set1_ps(scale);
    
    // Second pass: apply RMSNorm and SiLU
    i = 0;
    for (; i + 8 <= n; i += 8) {
        // Load input and weights
        __m256 x = _mm256_loadu_ps(input + i);
        __m256 w = _mm256_loadu_ps(weight + i);
        
        // Apply RMSNorm: x * scale * weight
        __m256 normalized = _mm256_mul_ps(_mm256_mul_ps(x, scale_vec), w);
        
        // Compute sigmoid for SiLU
        // Using approximation: 0.5 * (tanh(x/2) + 1)
        __m256 x_half = _mm256_mul_ps(normalized, half);
        __m256 x2 = _mm256_mul_ps(x_half, x_half);
        
        __m256 num = _mm256_mul_ps(x_half, _mm256_add_ps(_mm256_set1_ps(27.0f), x2));
        __m256 den = _mm256_add_ps(_mm256_set1_ps(27.0f), _mm256_mul_ps(_mm256_set1_ps(9.0f), x2));
        __m256 tanh_x_half = _mm256_div_ps(num, den);
        
        __m256 sigmoid = _mm256_mul_ps(half, _mm256_add_ps(tanh_x_half, one));
        
        // Apply SiLU: x * sigmoid(x)
        __m256 result = _mm256_mul_ps(normalized, sigmoid);
        
        // Store result
        _mm256_storeu_ps(output + i, result);
    }
    
    // Handle remaining elements with scalar code
    for (; i < n; i++) {
        float normalized = input[i] * scale * weight[i];
        float sigmoid = 1.0f / (1.0f + std::exp(-normalized));
        output[i] = normalized * sigmoid;
    }
}
#endif

// Fused RMSNorm + SiLU implementation (NEON)
#if defined(CCSM_HAVE_NEON)
void rms_norm_silu_neon(float* output, const float* input, const float* weight, size_t n, float eps) {
    // Constants
    const float32x4_t eps_vec = vdupq_n_f32(eps);
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t half = vdupq_n_f32(0.5f);
    
    // First pass: calculate sum of squares
    float32x4_t sum_sq = vdupq_n_f32(0.0f);
    size_t i = 0;
    
    for (; i + 4 <= n; i += 4) {
        float32x4_t x = vld1q_f32(input + i);
        sum_sq = vaddq_f32(sum_sq, vmulq_f32(x, x));
    }
    
    // Horizontal sum of sum_sq
    float32x2_t sum_sq_2 = vadd_f32(vget_low_f32(sum_sq), vget_high_f32(sum_sq));
    sum_sq_2 = vpadd_f32(sum_sq_2, sum_sq_2);
    
    // Extract the sum
    float sum_squares = vget_lane_f32(sum_sq_2, 0);
    
    // Handle remaining elements with scalar code
    for (; i < n; i++) {
        sum_squares += input[i] * input[i];
    }
    
    // Calculate normalization factor
    float scale = 1.0f / std::sqrt(sum_squares / n + eps);
    float32x4_t scale_vec = vdupq_n_f32(scale);
    
    // Second pass: apply RMSNorm and SiLU
    i = 0;
    for (; i + 4 <= n; i += 4) {
        // Load input and weights
        float32x4_t x = vld1q_f32(input + i);
        float32x4_t w = vld1q_f32(weight + i);
        
        // Apply RMSNorm: x * scale * weight
        float32x4_t normalized = vmulq_f32(vmulq_f32(x, scale_vec), w);
        
        // Compute sigmoid for SiLU
        // Using approximation: 0.5 * (tanh(x/2) + 1)
        float32x4_t x_half = vmulq_f32(normalized, half);
        float32x4_t x2 = vmulq_f32(x_half, x_half);
        
        float32x4_t num = vmulq_f32(x_half, vaddq_f32(vdupq_n_f32(27.0f), x2));
        float32x4_t den = vaddq_f32(vdupq_n_f32(27.0f), vmulq_f32(vdupq_n_f32(9.0f), x2));
        float32x4_t tanh_x_half = vdivq_f32(num, den);
        
        float32x4_t sigmoid = vmulq_f32(half, vaddq_f32(tanh_x_half, one));
        
        // Apply SiLU: x * sigmoid(x)
        float32x4_t result = vmulq_f32(normalized, sigmoid);
        
        // Store result
        vst1q_f32(output + i, result);
    }
    
    // Handle remaining elements with scalar code
    for (; i < n; i++) {
        float normalized = input[i] * scale * weight[i];
        float sigmoid = 1.0f / (1.0f + std::exp(-normalized));
        output[i] = normalized * sigmoid;
    }
}
#endif

// Scalar implementation of fused RMSNorm + SiLU
void rms_norm_silu_scalar(float* output, const float* input, const float* weight, size_t n, float eps) {
    // First pass: calculate sum of squares
    float sum_squares = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum_squares += input[i] * input[i];
    }
    
    // Calculate normalization factor
    float scale = 1.0f / std::sqrt(sum_squares / n + eps);
    
    // Second pass: apply RMSNorm and SiLU
    for (size_t i = 0; i < n; i++) {
        float normalized = input[i] * scale * weight[i];
        float sigmoid = 1.0f / (1.0f + std::exp(-normalized));
        output[i] = normalized * sigmoid;
    }
}

// SIMD-optimized layer normalization + ReLU implementation (AVX)
#if defined(CCSM_HAVE_AVX)
void layer_norm_relu_avx(float* output, const float* input, const float* gamma, const float* beta, size_t n, float eps) {
    // Constants
    const __m256 eps_vec = _mm256_set1_ps(eps);
    const __m256 zero = _mm256_setzero_ps();
    
    // First pass: calculate mean
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;
    
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);
        sum = _mm256_add_ps(sum, x);
    }
    
    // Horizontal sum
    __m256 hsum = _mm256_hadd_ps(sum, sum);
    hsum = _mm256_hadd_ps(hsum, hsum);
    
    // Extract the sum from the first and fifth elements
    float sum_all = _mm256_cvtss_f32(hsum) + _mm256_cvtss_f32(_mm256_permute_ps(hsum, 0x01));
    
    // Handle remaining elements with scalar code
    for (; i < n; i++) {
        sum_all += input[i];
    }
    
    // Calculate mean
    float mean = sum_all / n;
    __m256 mean_vec = _mm256_set1_ps(mean);
    
    // Second pass: calculate variance
    __m256 var_sum = _mm256_setzero_ps();
    i = 0;
    
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);
        __m256 diff = _mm256_sub_ps(x, mean_vec);
        var_sum = _mm256_add_ps(var_sum, _mm256_mul_ps(diff, diff));
    }
    
    // Horizontal sum for variance
    __m256 hvar = _mm256_hadd_ps(var_sum, var_sum);
    hvar = _mm256_hadd_ps(hvar, hvar);
    
    // Extract the variance sum
    float var_sum_all = _mm256_cvtss_f32(hvar) + _mm256_cvtss_f32(_mm256_permute_ps(hvar, 0x01));
    
    // Handle remaining elements for variance calculation
    for (; i < n; i++) {
        float diff = input[i] - mean;
        var_sum_all += diff * diff;
    }
    
    // Calculate standard deviation
    float var = var_sum_all / n;
    float std_dev = std::sqrt(var + eps);
    __m256 inv_std_dev = _mm256_set1_ps(1.0f / std_dev);
    
    // Third pass: normalize, scale, shift, and apply ReLU
    i = 0;
    for (; i + 8 <= n; i += 8) {
        // Load input and normalization parameters
        __m256 x = _mm256_loadu_ps(input + i);
        __m256 g = _mm256_loadu_ps(gamma + i);
        __m256 b = _mm256_loadu_ps(beta + i);
        
        // Normalize: (x - mean) / std_dev
        __m256 normalized = _mm256_mul_ps(_mm256_sub_ps(x, mean_vec), inv_std_dev);
        
        // Scale and shift: gamma * normalized + beta
        __m256 scaled_shifted = _mm256_add_ps(_mm256_mul_ps(normalized, g), b);
        
        // Apply ReLU: max(0, x)
        __m256 result = _mm256_max_ps(zero, scaled_shifted);
        
        // Store result
        _mm256_storeu_ps(output + i, result);
    }
    
    // Handle remaining elements with scalar code
    for (; i < n; i++) {
        float normalized = (input[i] - mean) / std_dev;
        float scaled_shifted = gamma[i] * normalized + beta[i];
        output[i] = std::max(0.0f, scaled_shifted);
    }
}
#endif

// SIMD-optimized layer normalization + ReLU implementation (NEON)
#if defined(CCSM_HAVE_NEON)
void layer_norm_relu_neon(float* output, const float* input, const float* gamma, const float* beta, size_t n, float eps) {
    // Constants
    const float32x4_t eps_vec = vdupq_n_f32(eps);
    const float32x4_t zero = vdupq_n_f32(0.0f);
    
    // First pass: calculate mean
    float32x4_t sum = vdupq_n_f32(0.0f);
    size_t i = 0;
    
    for (; i + 4 <= n; i += 4) {
        float32x4_t x = vld1q_f32(input + i);
        sum = vaddq_f32(sum, x);
    }
    
    // Horizontal sum
    float32x2_t sum_2 = vadd_f32(vget_low_f32(sum), vget_high_f32(sum));
    sum_2 = vpadd_f32(sum_2, sum_2);
    
    // Extract the sum
    float sum_all = vget_lane_f32(sum_2, 0);
    
    // Handle remaining elements with scalar code
    for (; i < n; i++) {
        sum_all += input[i];
    }
    
    // Calculate mean
    float mean = sum_all / n;
    float32x4_t mean_vec = vdupq_n_f32(mean);
    
    // Second pass: calculate variance
    float32x4_t var_sum = vdupq_n_f32(0.0f);
    i = 0;
    
    for (; i + 4 <= n; i += 4) {
        float32x4_t x = vld1q_f32(input + i);
        float32x4_t diff = vsubq_f32(x, mean_vec);
        var_sum = vaddq_f32(var_sum, vmulq_f32(diff, diff));
    }
    
    // Horizontal sum for variance
    float32x2_t var_sum_2 = vadd_f32(vget_low_f32(var_sum), vget_high_f32(var_sum));
    var_sum_2 = vpadd_f32(var_sum_2, var_sum_2);
    
    // Extract the variance sum
    float var_sum_all = vget_lane_f32(var_sum_2, 0);
    
    // Handle remaining elements for variance calculation
    for (; i < n; i++) {
        float diff = input[i] - mean;
        var_sum_all += diff * diff;
    }
    
    // Calculate standard deviation
    float var = var_sum_all / n;
    float std_dev = std::sqrt(var + eps);
    float32x4_t inv_std_dev = vdupq_n_f32(1.0f / std_dev);
    
    // Third pass: normalize, scale, shift, and apply ReLU
    i = 0;
    for (; i + 4 <= n; i += 4) {
        // Load input and normalization parameters
        float32x4_t x = vld1q_f32(input + i);
        float32x4_t g = vld1q_f32(gamma + i);
        float32x4_t b = vld1q_f32(beta + i);
        
        // Normalize: (x - mean) / std_dev
        float32x4_t normalized = vmulq_f32(vsubq_f32(x, mean_vec), inv_std_dev);
        
        // Scale and shift: gamma * normalized + beta
        float32x4_t scaled_shifted = vaddq_f32(vmulq_f32(normalized, g), b);
        
        // Apply ReLU: max(0, x)
        float32x4_t result = vmaxq_f32(zero, scaled_shifted);
        
        // Store result
        vst1q_f32(output + i, result);
    }
    
    // Handle remaining elements with scalar code
    for (; i < n; i++) {
        float normalized = (input[i] - mean) / std_dev;
        float scaled_shifted = gamma[i] * normalized + beta[i];
        output[i] = std::max(0.0f, scaled_shifted);
    }
}
#endif

// Scalar implementation of layer normalization + ReLU
void layer_norm_relu_scalar(float* output, const float* input, const float* gamma, const float* beta, size_t n, float eps) {
    // First pass: calculate mean
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += input[i];
    }
    float mean = sum / n;
    
    // Second pass: calculate variance
    float var_sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float diff = input[i] - mean;
        var_sum += diff * diff;
    }
    float var = var_sum / n;
    float std_dev = std::sqrt(var + eps);
    
    // Third pass: normalize, scale, shift, and apply ReLU
    for (size_t i = 0; i < n; i++) {
        float normalized = (input[i] - mean) / std_dev;
        float scaled_shifted = gamma[i] * normalized + beta[i];
        output[i] = std::max(0.0f, scaled_shifted);
    }
}

void matrix_mul_mma_f32(float* result, const float* a, const float* b, size_t m, size_t k, size_t n) {
    // A simple matrix-matrix multiplication implementation
    // A is mxk, B is kxn, result is mxn
    
    // Zero the result matrix
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            result[i * n + j] = 0.0f;
        }
    }
    
    // Perform matrix multiplication
    for (size_t i = 0; i < m; i++) {
        for (size_t l = 0; l < k; l++) {
            for (size_t j = 0; j < n; j++) {
                result[i * n + j] += a[i * k + l] * b[l * n + j];
            }
        }
    }
}

// Quantized matrix multiplication operation with Q8_0 format
void matrix_mul_q8_0_scalar(float* result, const float* a, const int8_t* b, const float* b_scale, 
                           size_t m, size_t k, size_t n) {
    const float scale = *b_scale;
    
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (size_t l = 0; l < k; l++) {
                sum += a[i * k + l] * (static_cast<float>(b[l * n + j]) * scale);
            }
            result[i * n + j] = sum;
        }
    }
}

#if defined(CCSM_HAVE_AVX)
// Implementation for matrix_mul_q8_0 (AVX)
void matrix_mul_q8_0_avx_f32(float* result, const float* a, const int8_t* b, const float* b_scale, 
                         size_t m, size_t k, size_t n) {
    const float scale = *b_scale;
    
    // Process each row of the result matrix
    for (size_t i = 0; i < m; i++) {
        // Process columns in blocks of 8 (AVX register width)
        for (size_t j = 0; j < n; j += 8) {
            // Handle boundary condition
            size_t block_size = std::min(size_t(8), n - j);
            
            // If we don't have a full AVX vector, use scalar code
            if (block_size < 8) {
                for (size_t jj = j; jj < j + block_size; jj++) {
                    float sum = 0.0f;
                    for (size_t l = 0; l < k; l++) {
                        sum += a[i * k + l] * (b[l * n + jj] * scale);
                    }
                    result[i * n + jj] = sum;
                }
                continue;
            }
            
            // Full AVX vector case - process 8 output elements at once
            __m256 vsum = _mm256_setzero_ps();
            const float* a_row = a + i * k;
            
            // Accumulate dot products for 8 columns
            for (size_t l = 0; l < k; l++) {
                // Load a single scalar value from a and broadcast to all elements
                __m256 va = _mm256_set1_ps(a_row[l]);
                
                // Load 8 int8_t values from b, convert to int32, then to float
                // Note: we're directly accessing the column-wise layout
                const int8_t* b_ptr = b + l * n + j;
                
                // Load 8 int8_t values
                __m128i b_i8 = _mm_loadl_epi64((__m128i const*)b_ptr);
                
                // Sign-extend int8 to int16
                __m128i b_i16 = _mm_cvtepi8_epi16(b_i8);
                
                // Sign-extend low 4 int16 to int32
                __m128i b_i32_low = _mm_cvtepi16_epi32(b_i16);
                
                // Extract high 4 int16 and sign-extend to int32
                __m128i b_i16_high = _mm_shuffle_epi32(b_i16, 0x0E); // Shuffle to get high 4 elements
                __m128i b_i32_high = _mm_cvtepi16_epi32(b_i16_high);
                
                // Convert int32 to float
                __m256 vb_f = _mm256_insertf128_ps(
                    _mm256_castps128_ps256(_mm_cvtepi32_ps(b_i32_low)),
                    _mm_cvtepi32_ps(b_i32_high),
                    1
                );
                
                // Apply scale
                __m256 vscale = _mm256_set1_ps(scale);
                vb_f = _mm256_mul_ps(vb_f, vscale);
                
                // Multiply and accumulate
                vsum = _mm256_add_ps(vsum, _mm256_mul_ps(va, vb_f));
            }
            
            // Store the accumulated results
            _mm256_storeu_ps(result + i * n + j, vsum);
        }
    }
}
#endif

#if defined(CCSM_HAVE_NEON)
// Implementation for matrix_mul_q8_0 (NEON)
void matrix_mul_q8_0_neon_f32(float* result, const float* a, const int8_t* b, const float* b_scale, 
                          size_t m, size_t k, size_t n) {
    // Multiply a[m,k] * b[k,n] = result[m,n]
    const float scale = *b_scale;
    
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float32x4_t vsum = vdupq_n_f32(0.0f);
            size_t l = 0;
            
            // Process in blocks of 4
            for (; l + 4 <= k; l += 4) {
                float32x4_t va = vld1q_f32(a + i * k + l);
                
                // Load and convert 4 int8_t values from B
                int8x8_t vb = vdup_n_s8(0);
                vb = vset_lane_s8(b[(l + 0) * n + j], vb, 0);
                vb = vset_lane_s8(b[(l + 1) * n + j], vb, 1);
                vb = vset_lane_s8(b[(l + 2) * n + j], vb, 2);
                vb = vset_lane_s8(b[(l + 3) * n + j], vb, 3);
                
                // Convert to int16, then int32, then float32
                int16x8_t vb16 = vmovl_s8(vb);
                int32x4_t vb32 = vmovl_s16(vget_low_s16(vb16));
                float32x4_t vbf = vcvtq_f32_s32(vb32);
                
                // Apply scale
                vbf = vmulq_n_f32(vbf, scale);
                
                // Multiply and accumulate
                vsum = vmlaq_f32(vsum, va, vbf);
            }
            
            // Horizontal sum
            float32x2_t vsum2 = vadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
            vsum2 = vpadd_f32(vsum2, vsum2);
            float sum = vget_lane_f32(vsum2, 0);
            
            // Handle remaining elements
            for (; l < k; l++) {
                sum += a[i * k + l] * (b[l * n + j] * scale);
            }
            
            result[i * n + j] = sum;
        }
    }
}
#endif

// Scalar implementation for matrix_mul_q4_0
void matrix_mul_q4_0_scalar(float* result, const float* a, const uint8_t* b, const float* b_scale, 
                            size_t m, size_t k, size_t n) {
    const float scale = *b_scale;
    
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (size_t l = 0; l < k; l++) {
                // Calculate index for accessing b
                size_t b_idx = (l / 2) * n + j;
                uint8_t packed = b[b_idx];
                
                // Extract appropriate 4-bit value (upper or lower half)
                int8_t val;
                if (l % 2 == 0) {
                    // Lower 4 bits
                    val = static_cast<int8_t>(packed & 0xF);
                    // Sign extend if needed
                    if (val & 0x8) val |= 0xF0;
                } else {
                    // Upper 4 bits
                    val = static_cast<int8_t>((packed >> 4) & 0xF);
                    // Sign extend if needed
                    if (val & 0x8) val |= 0xF0;
                }
                
                // Dequantize and multiply
                sum += a[i * k + l] * (val * scale);
            }
            result[i * n + j] = sum;
        }
    }
}

#if defined(CCSM_HAVE_AVX)
// Implementation for matrix_mul_q4_0 (AVX)
void matrix_mul_q4_0_avx_f32(float* result, const float* a, const uint8_t* b, const float* b_scale, 
                          size_t m, size_t k, size_t n) {
    const float scale = *b_scale;
    
    // Process each row of the result matrix
    for (size_t i = 0; i < m; i++) {
        // Process columns in blocks of 8 (AVX register width)
        for (size_t j = 0; j < n; j += 8) {
            // Handle boundary condition
            size_t block_size = std::min(size_t(8), n - j);
            
            // If we don't have a full AVX vector, use scalar code
            if (block_size < 8) {
                for (size_t jj = j; jj < j + block_size; jj++) {
                    float sum = 0.0f;
                    for (size_t l = 0; l < k; l++) {
                        // Calculate index for accessing b
                        size_t b_idx = (l / 2) * n + jj;
                        uint8_t packed = b[b_idx];
                        
                        // Extract appropriate 4-bit value (upper or lower half)
                        int8_t val;
                        if (l % 2 == 0) {
                            // Lower 4 bits
                            val = static_cast<int8_t>(packed & 0xF);
                            // Sign extend if needed
                            if (val & 0x8) val |= 0xF0;
                        } else {
                            // Upper 4 bits
                            val = static_cast<int8_t>((packed >> 4) & 0xF);
                            // Sign extend if needed
                            if (val & 0x8) val |= 0xF0;
                        }
                        
                        // Dequantize and multiply
                        sum += a[i * k + l] * (val * scale);
                    }
                    result[i * n + jj] = sum;
                }
                continue;
            }
            
            // Initialize the result vector to zero
            __m256 vsum = _mm256_setzero_ps();
            const float* a_row = a + i * k;
            
            for (size_t l = 0; l < k; l++) {
                // Load and broadcast a single value from a
                __m256 va = _mm256_set1_ps(a_row[l]);
                
                // Handle the 4-bit packed format
                // Each byte contains two 4-bit values
                int8_t unpacked_values[8] = {0};
                for (size_t jj = 0; jj < 8; jj++) {
                    // Calculate index for the packed byte
                    size_t b_idx = (l / 2) * n + j + jj;
                    uint8_t packed = b[b_idx];
                    
                    // Extract the 4-bit value (upper or lower)
                    int8_t val;
                    if (l % 2 == 0) {
                        // Lower 4 bits
                        val = static_cast<int8_t>(packed & 0xF);
                        // Sign extend if needed
                        if (val & 0x8) val |= 0xF0;
                    } else {
                        // Upper 4 bits
                        val = static_cast<int8_t>((packed >> 4) & 0xF);
                        // Sign extend if needed
                        if (val & 0x8) val |= 0xF0;
                    }
                    
                    unpacked_values[jj] = val;
                }
                
                // Load the unpacked values into an AVX register
                __m128i vb_i8 = _mm_loadl_epi64((__m128i const*)unpacked_values);
                __m128i vb_i16 = _mm_cvtepi8_epi16(vb_i8);
                __m128i vb_i32_low = _mm_cvtepi16_epi32(vb_i16);
                __m128i vb_i16_high = _mm_shuffle_epi32(vb_i16, 0x0E);
                __m128i vb_i32_high = _mm_cvtepi16_epi32(vb_i16_high);
                
                // Convert to float
                __m256 vb_f = _mm256_insertf128_ps(
                    _mm256_castps128_ps256(_mm_cvtepi32_ps(vb_i32_low)),
                    _mm_cvtepi32_ps(vb_i32_high),
                    1
                );
                
                // Apply scale
                __m256 vscale = _mm256_set1_ps(scale);
                vb_f = _mm256_mul_ps(vb_f, vscale);
                
                // Multiply and accumulate
                vsum = _mm256_add_ps(vsum, _mm256_mul_ps(va, vb_f));
            }
            
            // Store the results
            _mm256_storeu_ps(result + i * n + j, vsum);
        }
    }
}
#endif

#if defined(CCSM_HAVE_NEON)
// Implementation for matrix_mul_q4_0 (NEON)
void matrix_mul_q4_0_neon_f32(float* result, const float* a, const uint8_t* b, const float* b_scale, 
                           size_t m, size_t k, size_t n) {
    const float scale = *b_scale;
    
    // Process each row of the result matrix
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j += 4) {
            // Handle boundary condition
            size_t block_size = std::min(size_t(4), n - j);
            
            // If we don't have a full NEON vector, use scalar code
            if (block_size < 4) {
                for (size_t jj = j; jj < j + block_size; jj++) {
                    float sum = 0.0f;
                    for (size_t l = 0; l < k; l++) {
                        // Calculate index for accessing b
                        size_t b_idx = (l / 2) * n + jj;
                        uint8_t packed = b[b_idx];
                        
                        // Extract appropriate 4-bit value (upper or lower half)
                        int8_t val;
                        if (l % 2 == 0) {
                            // Lower 4 bits
                            val = static_cast<int8_t>(packed & 0xF);
                            // Sign extend if needed
                            if (val & 0x8) val |= 0xF0;
                        } else {
                            // Upper 4 bits
                            val = static_cast<int8_t>((packed >> 4) & 0xF);
                            // Sign extend if needed
                            if (val & 0x8) val |= 0xF0;
                        }
                        
                        // Dequantize and multiply
                        sum += a[i * k + l] * (val * scale);
                    }
                    result[i * n + jj] = sum;
                }
                continue;
            }
            
            // Use NEON for 4 elements at a time
            float32x4_t vsum = vdupq_n_f32(0.0f);
            const float* a_row = a + i * k;
            
            for (size_t l = 0; l < k; l++) {
                // Load and broadcast single a value
                float32x4_t va = vdupq_n_f32(a_row[l]);
                
                // Handle the 4-bit packed format
                int8_t unpacked_values[4] = {0};
                for (size_t jj = 0; jj < 4; jj++) {
                    // Calculate index for the packed byte
                    size_t b_idx = (l / 2) * n + j + jj;
                    uint8_t packed = b[b_idx];
                    
                    // Extract the 4-bit value (upper or lower)
                    int8_t val;
                    if (l % 2 == 0) {
                        // Lower 4 bits
                        val = static_cast<int8_t>(packed & 0xF);
                        // Sign extend if needed
                        if (val & 0x8) val |= 0xF0;
                    } else {
                        // Upper 4 bits
                        val = static_cast<int8_t>((packed >> 4) & 0xF);
                        // Sign extend if needed
                        if (val & 0x8) val |= 0xF0;
                    }
                    
                    unpacked_values[jj] = val;
                }
                
                // Load the unpacked values into a NEON register
                int8x8_t b_i8 = vld1_s8(unpacked_values);
                int16x8_t b_i16 = vmovl_s8(b_i8);
                int32x4_t b_i32 = vmovl_s16(vget_low_s16(b_i16));
                float32x4_t vb = vcvtq_f32_s32(b_i32);
                
                // Apply scale
                vb = vmulq_n_f32(vb, scale);
                
                // Multiply and accumulate
                vsum = vmlaq_f32(vsum, va, vb);
            }
            
            // Store the result
            vst1q_f32(result + i * n + j, vsum);
        }
    }
}
#endif

// Scalar implementation for matrix_mul_q4_1
void matrix_mul_q4_1_scalar(float* result, const float* a, const uint8_t* b, const float* b_scale, 
                            const float* b_bias, size_t m, size_t k, size_t n) {
    const float scale = *b_scale;
    const float bias = *b_bias;
    
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (size_t l = 0; l < k; l++) {
                // Calculate index for accessing b
                size_t b_idx = (l / 2) * n + j;
                uint8_t packed = b[b_idx];
                
                // Extract appropriate 4-bit value (upper or lower half)
                uint8_t val_unsigned;
                if (l % 2 == 0) {
                    // Lower 4 bits
                    val_unsigned = packed & 0xF;
                } else {
                    // Upper 4 bits
                    val_unsigned = (packed >> 4) & 0xF;
                }
                
                // Dequantize and multiply
                float val_float = static_cast<float>(val_unsigned) * scale + bias;
                sum += a[i * k + l] * val_float;
            }
            result[i * n + j] = sum;
        }
    }
}

#if defined(CCSM_HAVE_AVX)
// Implementation for matrix_mul_q4_1 (AVX)
void matrix_mul_q4_1_avx_f32(float* result, const float* a, const uint8_t* b, const float* b_scale, const float* b_bias, 
                          size_t m, size_t k, size_t n) {
    const float scale = *b_scale;
    const float bias = *b_bias;
    
    // Process each row of the result matrix
    for (size_t i = 0; i < m; i++) {
        // Process columns in blocks of 8 (AVX register width)
        for (size_t j = 0; j < n; j += 8) {
            // Handle boundary condition
            size_t block_size = std::min(size_t(8), n - j);
            
            // If we don't have a full AVX vector, use scalar code
            if (block_size < 8) {
                for (size_t jj = j; jj < j + block_size; jj++) {
                    float sum = 0.0f;
                    for (size_t l = 0; l < k; l++) {
                        // Calculate index for accessing b
                        size_t b_idx = (l / 2) * n + jj;
                        uint8_t packed = b[b_idx];
                        
                        // Extract appropriate 4-bit value (upper or lower half)
                        uint8_t val_unsigned;
                        if (l % 2 == 0) {
                            // Lower 4 bits
                            val_unsigned = packed & 0xF;
                        } else {
                            // Upper 4 bits
                            val_unsigned = (packed >> 4) & 0xF;
                        }
                        
                        // Dequantize and multiply
                        float val_float = static_cast<float>(val_unsigned) * scale + bias;
                        sum += a[i * k + l] * val_float;
                    }
                    result[i * n + jj] = sum;
                }
                continue;
            }
            
            // Initialize the result vector to zero
            __m256 vsum = _mm256_setzero_ps();
            const float* a_row = a + i * k;
            
            // Process each element of a
            for (size_t l = 0; l < k; l++) {
                // Load and broadcast a single value from a
                __m256 va = _mm256_set1_ps(a_row[l]);
                
                // Handle the 4-bit packed format
                // Each byte contains two 4-bit values
                uint8_t unpacked_values[8] = {0};
                for (size_t jj = 0; jj < 8; jj++) {
                    // Calculate index for the packed byte
                    size_t b_idx = (l / 2) * n + j + jj;
                    uint8_t packed = b[b_idx];
                    
                    // Extract the 4-bit value (upper or lower)
                    uint8_t val;
                    if (l % 2 == 0) {
                        // Lower 4 bits
                        val = packed & 0xF;
                    } else {
                        // Upper 4 bits
                        val = (packed >> 4) & 0xF;
                    }
                    
                    unpacked_values[jj] = val;
                }
                
                // Load the unpacked values into an AVX register
                __m128i vb_u8 = _mm_loadl_epi64((__m128i const*)unpacked_values);
                __m128i vb_u16 = _mm_cvtepu8_epi16(vb_u8);
                __m128i vb_u32_low = _mm_cvtepu16_epi32(vb_u16);
                __m128i vb_u16_high = _mm_shuffle_epi32(vb_u16, 0x0E);
                __m128i vb_u32_high = _mm_cvtepu16_epi32(vb_u16_high);
                
                // Convert to float
                __m256 vb_f = _mm256_insertf128_ps(
                    _mm256_castps128_ps256(_mm_cvtepi32_ps(vb_u32_low)),
                    _mm_cvtepi32_ps(vb_u32_high),
                    1
                );
                
                // Apply scale and bias
                __m256 vscale = _mm256_set1_ps(scale);
                __m256 vbias = _mm256_set1_ps(bias);
                vb_f = _mm256_add_ps(_mm256_mul_ps(vb_f, vscale), vbias);
                
                // Multiply and accumulate
                vsum = _mm256_add_ps(vsum, _mm256_mul_ps(va, vb_f));
            }
            
            // Store the results
            _mm256_storeu_ps(result + i * n + j, vsum);
        }
    }
}
#endif

#if defined(CCSM_HAVE_NEON)
// Implementation for matrix_mul_q4_1 (NEON)
void matrix_mul_q4_1_neon_f32(float* result, const float* a, const uint8_t* b, const float* b_scale, const float* b_bias, 
                          size_t m, size_t k, size_t n) {
    const float scale = *b_scale;
    const float bias = *b_bias;
    
    // Process each row of the result matrix
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j += 4) {
            // Handle boundary condition
            size_t block_size = std::min(size_t(4), n - j);
            
            // If we don't have a full NEON vector, use scalar code
            if (block_size < 4) {
                for (size_t jj = j; jj < j + block_size; jj++) {
                    float sum = 0.0f;
                    for (size_t l = 0; l < k; l++) {
                        // Calculate index for accessing b
                        size_t b_idx = (l / 2) * n + jj;
                        uint8_t packed = b[b_idx];
                        
                        // Extract appropriate 4-bit value (upper or lower half)
                        uint8_t val_unsigned;
                        if (l % 2 == 0) {
                            // Lower 4 bits
                            val_unsigned = packed & 0xF;
                        } else {
                            // Upper 4 bits
                            val_unsigned = (packed >> 4) & 0xF;
                        }
                        
                        // Dequantize and multiply
                        float val_float = static_cast<float>(val_unsigned) * scale + bias;
                        sum += a[i * k + l] * val_float;
                    }
                    result[i * n + jj] = sum;
                }
                continue;
            }
            
            // Use NEON for 4 elements at a time
            float32x4_t vsum = vdupq_n_f32(0.0f);
            const float* a_row = a + i * k;
            
            for (size_t l = 0; l < k; l++) {
                // Load and broadcast single a value
                float32x4_t va = vdupq_n_f32(a_row[l]);
                
                // Handle the 4-bit packed format
                uint8_t unpacked_values[4] = {0};
                for (size_t jj = 0; jj < 4; jj++) {
                    // Calculate index for the packed byte
                    size_t b_idx = (l / 2) * n + j + jj;
                    uint8_t packed = b[b_idx];
                    
                    // Extract the 4-bit value (upper or lower)
                    uint8_t val;
                    if (l % 2 == 0) {
                        // Lower 4 bits
                        val = packed & 0xF;
                    } else {
                        // Upper 4 bits
                        val = (packed >> 4) & 0xF;
                    }
                    
                    unpacked_values[jj] = val;
                }
                
                // Load the unpacked values into a NEON register
                uint8x8_t b_u8 = vld1_u8(unpacked_values);
                uint16x8_t b_u16 = vmovl_u8(b_u8);
                uint32x4_t b_u32 = vmovl_u16(vget_low_u16(b_u16));
                float32x4_t vb = vcvtq_f32_u32(b_u32);
                
                // Apply scale and bias
                float32x4_t vbias = vdupq_n_f32(bias);
                vb = vaddq_f32(vmulq_n_f32(vb, scale), vbias);
                
                // Multiply and accumulate
                vsum = vmlaq_f32(vsum, va, vb);
            }
            
            // Store the result
            vst1q_f32(result + i * n + j, vsum);
        }
    }
}
#endif

// NOTE: The implementation of NEON matrix multiplication functions has been removed
// because they are already defined elsewhere in the file. This was causing redefinition errors.
// The existing implementations for matrix_mul_q8_0_neon_f32, matrix_mul_q4_0_neon_f32, and
// matrix_mul_q4_1_neon_f32 are used instead.

#if defined(CCSM_HAVE_NEON)
// NEON implementations of vector operations (these are new and not defined elsewhere)
void vector_add_neon_f32(float* result, const float* a, const float* b, size_t n) {
    size_t i = 0;
    // Process 4 elements at a time
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vr = vaddq_f32(va, vb);
        vst1q_f32(result + i, vr);
    }
    // Handle remaining elements
    for (; i < n; i++) {
        result[i] = a[i] + b[i];
    }
}

void vector_mul_neon_f32(float* result, const float* a, const float* b, size_t n) {
    size_t i = 0;
    // Process 4 elements at a time
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vr = vmulq_f32(va, vb);
        vst1q_f32(result + i, vr);
    }
    // Handle remaining elements
    for (; i < n; i++) {
        result[i] = a[i] * b[i];
    }
}

void vector_fma_neon_f32(float* result, const float* a, const float* b, const float* c, size_t n) {
    size_t i = 0;
    // Process 4 elements at a time
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vc = vld1q_f32(c + i);
        float32x4_t vr = vmlaq_f32(vc, va, vb); // a * b + c
        vst1q_f32(result + i, vr);
    }
    // Handle remaining elements
    for (; i < n; i++) {
        result[i] = a[i] * b[i] + c[i];
    }
}

float vector_dot_neon_f32(const float* a, const float* b, size_t n) {
    float32x4_t vsum = vdupq_n_f32(0.0f);
    size_t i = 0;
    
    // Process 4 elements at a time
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vsum = vmlaq_f32(vsum, va, vb);
    }
    
    // Horizontal sum
    float32x2_t vsum2 = vadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
    vsum2 = vpadd_f32(vsum2, vsum2);
    float result = vget_lane_f32(vsum2, 0);
    
    // Handle remaining elements
    for (; i < n; i++) {
        result += a[i] * b[i];
    }
    
    return result;
}
#endif // defined(CCSM_HAVE_NEON)

} // namespace detail

// Matrix multiply implementation with dispatch to appropriate backend
template<>
void matrix_mul<float>(float* result, const float* a, const float* b, size_t m, size_t k, size_t n) {
    // Simple matrix multiplication
    detail::matrix_mul_mma_f32(result, a, b, m, k, n);
}

// Quantized matrix multiplication with Q8_0 format (8-bit signed integers with scale)
template<>
void matrix_mul_q8_0<float>(float* result, const float* a, const int8_t* b, const float* b_scale, 
                         size_t m, size_t k, size_t n) {
    // Get active implementation
    Implementation impl = get_active_implementation();
    
#if defined(CCSM_HAVE_AVX)
    if (impl >= Implementation::AVX) {
        detail::matrix_mul_q8_0_avx_f32(result, a, b, b_scale, m, k, n);
        return;
    }
#elif defined(CCSM_HAVE_NEON)
    if (impl == Implementation::NEON) {
        detail::matrix_mul_q8_0_neon_f32(result, a, b, b_scale, m, k, n);
        return;
    }
#endif
    
    // Fallback scalar implementation
    detail::matrix_mul_q8_0_scalar(result, a, b, b_scale, m, k, n);
}

// Q4_0 matrix multiplication (4-bit signed integers with scale)
template<>
void matrix_mul_q4_0<float>(float* result, const float* a, const uint8_t* b, const float* b_scale, 
                         size_t m, size_t k, size_t n) {
    // Get active implementation
    Implementation impl = get_active_implementation();
    
#if defined(CCSM_HAVE_AVX)
    if (impl >= Implementation::AVX) {
        detail::matrix_mul_q4_0_avx_f32(result, a, b, b_scale, m, k, n);
        return;
    }
#elif defined(CCSM_HAVE_NEON)
    if (impl == Implementation::NEON) {
        detail::matrix_mul_q4_0_neon_f32(result, a, b, b_scale, m, k, n);
        return;
    }
#endif
    
    // Fallback scalar implementation
    detail::matrix_mul_q4_0_scalar(result, a, b, b_scale, m, k, n);
}

// Q4_1 matrix multiplication (4-bit unsigned integers with scale and bias)
template<>
void matrix_mul_q4_1<float>(float* result, const float* a, const uint8_t* b, const float* b_scale, 
                         const float* b_bias, size_t m, size_t k, size_t n) {
    // Get active implementation
    Implementation impl = get_active_implementation();
    
#if defined(CCSM_HAVE_AVX)
    if (impl >= Implementation::AVX) {
        detail::matrix_mul_q4_1_avx_f32(result, a, b, b_scale, b_bias, m, k, n);
        return;
    }
#elif defined(CCSM_HAVE_NEON)
    if (impl == Implementation::NEON) {
        detail::matrix_mul_q4_1_neon_f32(result, a, b, b_scale, b_bias, m, k, n);
        return;
    }
#endif
    
    // Fallback scalar implementation
    detail::matrix_mul_q4_1_scalar(result, a, b, b_scale, b_bias, m, k, n);
}

// Matrix-vector multiplication (matmul with n=1)
template<>
void matrix_vector_mul<float>(float* result, const float* a, const float* v, size_t m, size_t k) {
    // Using matrix_mul implementation with n=1
    matrix_mul<float>(result, a, v, m, k, 1);
}

// Sigmoid activation function
template<>
void sigmoid<float>(float* output, const float* input, size_t n) {
    // Get active implementation
    Implementation impl = get_active_implementation();
    
#if defined(CCSM_HAVE_AVX)
    if (impl >= Implementation::AVX) {
        detail::sigmoid_avx(output, input, n);
        return;
    }
#elif defined(CCSM_HAVE_NEON)
    if (impl == Implementation::NEON) {
        detail::sigmoid_neon(output, input, n);
        return;
    }
#endif
    
    // Fallback scalar implementation
    detail::sigmoid_scalar(output, input, n);
}

// SiLU activation function (x * sigmoid(x))
template<>
void silu<float>(float* output, const float* input, size_t n) {
    // Get active implementation
    Implementation impl = get_active_implementation();
    
#if defined(CCSM_HAVE_AVX)
    if (impl >= Implementation::AVX) {
        detail::silu_avx(output, input, n);
        return;
    }
#elif defined(CCSM_HAVE_NEON)
    if (impl == Implementation::NEON) {
        detail::silu_neon(output, input, n);
        return;
    }
#endif
    
    // Fallback scalar implementation
    detail::silu_scalar(output, input, n);
}

// ReLU activation function
template<>
void relu<float>(float* output, const float* input, size_t n) {
    // Get active implementation
    Implementation impl = get_active_implementation();
    
#if defined(CCSM_HAVE_AVX)
    if (impl >= Implementation::AVX) {
        detail::relu_avx(output, input, n);
        return;
    }
#elif defined(CCSM_HAVE_NEON)
    if (impl == Implementation::NEON) {
        detail::relu_neon(output, input, n);
        return;
    }
#endif
    
    // Fallback scalar implementation
    detail::relu_scalar(output, input, n);
}

// Fused RMSNorm + SiLU implementation
template<>
void rms_norm_silu<float>(float* output, const float* input, const float* weight, size_t n, float eps) {
    // Get active implementation
    Implementation impl = get_active_implementation();
    
#if defined(CCSM_HAVE_AVX)
    if (impl >= Implementation::AVX) {
        detail::rms_norm_silu_avx(output, input, weight, n, eps);
        return;
    }
#elif defined(CCSM_HAVE_NEON)
    if (impl == Implementation::NEON) {
        detail::rms_norm_silu_neon(output, input, weight, n, eps);
        return;
    }
#endif
    
    // Fallback scalar implementation
    detail::rms_norm_silu_scalar(output, input, weight, n, eps);
}

// Fused Layer Normalization + ReLU implementation
template<>
void layer_norm_relu<float>(float* output, const float* input, const float* gamma, const float* beta, 
                            size_t n, float eps) {
    // Get active implementation
    Implementation impl = get_active_implementation();
    
#if defined(CCSM_HAVE_AVX)
    if (impl >= Implementation::AVX) {
        detail::layer_norm_relu_avx(output, input, gamma, beta, n, eps);
        return;
    }
#elif defined(CCSM_HAVE_NEON)
    if (impl == Implementation::NEON) {
        detail::layer_norm_relu_neon(output, input, gamma, beta, n, eps);
        return;
    }
#endif
    
    // Fallback scalar implementation
    detail::layer_norm_relu_scalar(output, input, gamma, beta, n, eps);
}

// Implementation of mixed precision operations
namespace detail {

// F32 to F16 conversion implementation
void convert_f32_to_f16_scalar(const float* src, uint16_t* dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float value = src[i];
        
        // Extract components from the float
        uint32_t f32_bits;
        std::memcpy(&f32_bits, &value, sizeof(float));
        
        uint16_t sign = (f32_bits >> 31) & 0x1;
        int32_t exponent = ((f32_bits >> 23) & 0xFF) - 127;
        uint32_t mantissa = f32_bits & 0x7FFFFF;
        
        // Handle special cases
        if (std::isnan(value)) {
            // NaN
            dst[i] = 0x7E00;
            continue;
        }
        
        if (std::isinf(value)) {
            // Infinity
            dst[i] = sign ? 0xFC00 : 0x7C00;
            continue;
        }
        
        if (value == 0.0f) {
            // Zero (preserve sign)
            dst[i] = sign << 15;
            continue;
        }
        
        // Adjust for float16 bias and range
        uint16_t f16_bits;
        
        if (exponent > 15) {
            // Overflow, return infinity
            f16_bits = (sign << 15) | 0x7C00;
        } else if (exponent < -14) {
            // Underflow or denormal
            // For simplicity, we flush denormals to zero
            f16_bits = sign << 15;
        } else {
            // Normal value
            uint16_t f16_exponent = (exponent + 15) & 0x1F;
            uint16_t f16_mantissa = (mantissa >> 13) & 0x3FF;
            
            f16_bits = (sign << 15) | (f16_exponent << 10) | f16_mantissa;
        }
        
        dst[i] = f16_bits;
    }
}

// F16 to F32 conversion implementation
void convert_f16_to_f32_scalar(const uint16_t* src, float* dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        uint16_t value = src[i];
        
        // Extract components
        uint16_t sign = (value >> 15) & 0x1;
        uint16_t exponent = (value >> 10) & 0x1F;
        uint16_t mantissa = value & 0x3FF;
        
        // Convert to IEEE 754 float
        uint32_t f32_bits;
        
        if (exponent == 0x1F) {
            // Infinity or NaN
            if (mantissa == 0) {
                // Infinity
                f32_bits = (sign << 31) | 0x7F800000;
            } else {
                // NaN (preserve a portion of the mantissa)
                f32_bits = (sign << 31) | 0x7FC00000 | (mantissa << 13);
            }
        } else if (exponent == 0) {
            if (mantissa == 0) {
                // Zero
                f32_bits = sign << 31;
            } else {
                // Denormal in F16, convert to normal in F32
                // Find leading bit position in mantissa
                int leading_zeros = 0;
                uint16_t temp = mantissa;
                for (int j = 9; j >= 0; j--) {
                    if ((temp >> j) & 0x1) {
                        break;
                    }
                    leading_zeros++;
                }
                
                int32_t f32_exponent = -14 - leading_zeros;
                uint32_t f32_mantissa = (mantissa << (23 - 10 + leading_zeros + 1)) & 0x7FFFFF;
                f32_bits = (sign << 31) | ((f32_exponent + 127) << 23) | f32_mantissa;
            }
        } else {
            // Normal value
            int32_t f32_exponent = exponent - 15 + 127;
            uint32_t f32_mantissa = mantissa << 13;
            f32_bits = (sign << 31) | (f32_exponent << 23) | f32_mantissa;
        }
        
        std::memcpy(&dst[i], &f32_bits, sizeof(float));
    }
}

// F32 to BF16 conversion implementation
void convert_f32_to_bf16_scalar(const float* src, uint16_t* dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        // BF16 is just the top 16 bits of F32
        uint32_t f32_bits;
        std::memcpy(&f32_bits, &src[i], sizeof(float));
        
        // Round to nearest even when truncating
        uint32_t rounding_bit = (f32_bits >> 15) & 0x1;
        uint32_t tie_bit = ((f32_bits >> 16) & 0x1);
        uint32_t lower_bits = f32_bits & 0x7FFF;
        
        // Apply rounding
        if (rounding_bit && (lower_bits > 0 || tie_bit)) {
            f32_bits += 0x8000; // Round up
        }
        
        dst[i] = static_cast<uint16_t>(f32_bits >> 16);
    }
}

// BF16 to F32 conversion implementation
void convert_bf16_to_f32_scalar(const uint16_t* src, float* dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        uint32_t f32_bits = static_cast<uint32_t>(src[i]) << 16;
        std::memcpy(&dst[i], &f32_bits, sizeof(float));
    }
}

// Edge case handling implementations
// Enumerate possible special values
enum class FloatSpecialValue {
    NORMAL,     // Regular floating-point value
    NAN_VALUE,  // Not-a-Number
    INF_VALUE,  // Infinity
    NEG_INF,    // Negative Infinity
    DENORMAL    // Denormalized value
};

// Check for special floating point values
FloatSpecialValue check_float_value(float value) {
    // Check for NaN
    if (std::isnan(value)) {
        return FloatSpecialValue::NAN_VALUE;
    }
    
    // Check for positive/negative infinity
    if (std::isinf(value)) {
        return value > 0 ? FloatSpecialValue::INF_VALUE : FloatSpecialValue::NEG_INF;
    }
    
    // Check for denormals
    if (std::fpclassify(value) == FP_SUBNORMAL) {
        return FloatSpecialValue::DENORMAL;
    }
    
    // Normal value
    return FloatSpecialValue::NORMAL;
}

// Handle denormals (flush to zero for performance in contexts where it's safe)
float handle_denormal(float value, bool flush_to_zero) {
    if (flush_to_zero && std::fpclassify(value) == FP_SUBNORMAL) {
        return std::copysign(0.0f, value); // Preserve sign of zero
    }
    return value;
}

// Handle NaN values in operations
float handle_nan(float value, float replacement) {
    if (std::isnan(value)) {
        return replacement;
    }
    return value;
}

// Handle Infinity values in operations
float handle_infinity(float value, 
                     float pos_replacement,
                     float neg_replacement) {
    if (std::isinf(value)) {
        return value > 0 ? pos_replacement : neg_replacement;
    }
    return value;
}

// Vector operations with safe handling for NaN/Inf/Denormals
void vector_add_safe_scalar(float* result, const float* a, const float* b, size_t n,
                          bool flush_denormals, bool replace_nan) {
    for (size_t i = 0; i < n; i++) {
        float val_a = a[i];
        float val_b = b[i];
        
        // Handle special values if needed
        if (flush_denormals) {
            val_a = handle_denormal(val_a, true);
            val_b = handle_denormal(val_b, true);
        }
        
        if (replace_nan) {
            val_a = handle_nan(val_a, 0.0f);
            val_b = handle_nan(val_b, 0.0f);
        }
        
        // Perform the addition
        result[i] = val_a + val_b;
        
        // Handle special cases in the result if necessary
        if (flush_denormals) {
            result[i] = handle_denormal(result[i], true);
        }
    }
}

// Safe ReLU implementation with special value handling
void relu_safe_scalar(float* output, const float* input, size_t n,
                    bool flush_denormals, bool replace_nan) {
    for (size_t i = 0; i < n; i++) {
        float value = input[i];
        
        // Check for special values
        auto special = check_float_value(value);
        
        if (special == FloatSpecialValue::NAN_VALUE && replace_nan) {
            // Replace NaN with 0
            output[i] = 0.0f;
        } else if (special == FloatSpecialValue::DENORMAL && flush_denormals) {
            // Flush denormals to 0
            output[i] = 0.0f;
        } else if (special == FloatSpecialValue::NEG_INF) {
            // Negative infinity becomes 0 with ReLU
            output[i] = 0.0f;
        } else if (special == FloatSpecialValue::INF_VALUE) {
            // Positive infinity stays infinity
            output[i] = std::numeric_limits<float>::infinity();
        } else {
            // Normal case - regular ReLU
            output[i] = std::max(0.0f, value);
        }
    }
}

// In-place vector addition
template<typename T>
void vector_add_inplace_scalar(T* a, const T* b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        a[i] += b[i];
    }
}

// In-place vector multiplication
template<typename T>
void vector_mul_inplace_scalar(T* a, const T* b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        a[i] *= b[i];
    }
}

// In-place vector scaling
template<typename T>
void vector_scale_inplace_scalar(T* a, T scalar, size_t n) {
    for (size_t i = 0; i < n; i++) {
        a[i] *= scalar;
    }
}

// In-place ReLU activation
template<typename T>
void relu_inplace_scalar(T* data, size_t n) {
    for (size_t i = 0; i < n; i++) {
        data[i] = std::max(T(0), data[i]);
    }
}

} // namespace detail

// Public safe operation implementations
void vector_add_safe(float* result, const float* a, const float* b, size_t n,
                   bool flush_denormals, bool replace_nan) {
    detail::vector_add_safe_scalar(result, a, b, n, flush_denormals, replace_nan);
}

void relu_safe(float* output, const float* input, size_t n,
             bool flush_denormals, bool replace_nan) {
    // Fall back to scalar implementation
    detail::relu_safe_scalar(output, input, n, flush_denormals, replace_nan);
}

// Public in-place operation implementations
// Template implementations for vector_add_inplace
template<typename T>
void vector_add_inplace(T* a, const T* b, size_t n) {
    detail::vector_add_inplace_scalar(a, b, n);
}

// Explicit instantiations for common types
template void vector_add_inplace<float>(float* a, const float* b, size_t n);
template void vector_add_inplace<double>(double* a, const double* b, size_t n);
template void vector_add_inplace<int32_t>(int32_t* a, const int32_t* b, size_t n);

// Template implementations for vector_mul_inplace
template<typename T>
void vector_mul_inplace(T* a, const T* b, size_t n) {
    detail::vector_mul_inplace_scalar(a, b, n);
}

// Explicit instantiations for common types
template void vector_mul_inplace<float>(float* a, const float* b, size_t n);
template void vector_mul_inplace<double>(double* a, const double* b, size_t n);
template void vector_mul_inplace<int32_t>(int32_t* a, const int32_t* b, size_t n);

// Template implementations for vector_scale_inplace
template<typename T>
void vector_scale_inplace(T* a, T scalar, size_t n) {
    detail::vector_scale_inplace_scalar(a, scalar, n);
}

// Explicit instantiations for common types
template void vector_scale_inplace<float>(float* a, float scalar, size_t n);
template void vector_scale_inplace<double>(double* a, double scalar, size_t n);
template void vector_scale_inplace<int32_t>(int32_t* a, int32_t scalar, size_t n);

// In-place activation functions
template<typename T>
void relu_inplace(T* data, size_t n) {
    detail::relu_inplace_scalar(data, n);
}

// Explicit instantiations for common types
template void relu_inplace<float>(float* data, size_t n);
template void relu_inplace<double>(double* data, size_t n);

} // namespace simd
} // namespace ccsm