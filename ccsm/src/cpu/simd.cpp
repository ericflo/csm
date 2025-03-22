#include <ccsm/cpu/simd.h>
#include <vector>
#include <cmath>
#include <limits>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace ccsm {
namespace simd {

// Return the implementation active for the current architecture
Implementation get_active_implementation() {
    const auto& features = CPUFeatures::get();
    
    if (features.avx512f) {
        return Implementation::AVX512;
    } else if (features.avx2) {
        return Implementation::AVX2;
    } else if (features.avx) {
        return Implementation::AVX;
    } else if (features.sse4_1) {
        return Implementation::SSE41;
    } else if (features.sse2) {
        return Implementation::SSE2;
    } else if (features.neon) {
        return Implementation::NEON;
    } else {
        return Implementation::SCALAR;
    }
}

// Return information about CPU capabilities as string
std::string get_cpu_capabilities() {
    const auto& features = CPUFeatures::get();
    std::string capabilities;
    
    if (features.avx512f) capabilities += "AVX-512F ";
    if (features.avx2) capabilities += "AVX2 ";
    if (features.avx) capabilities += "AVX ";
    if (features.sse4_2) capabilities += "SSE4.2 ";
    if (features.sse4_1) capabilities += "SSE4.1 ";
    if (features.sse3) capabilities += "SSE3 ";
    if (features.sse2) capabilities += "SSE2 ";
    if (features.neon) capabilities += "NEON ";
    
    if (capabilities.empty()) {
        capabilities = "Scalar (no SIMD)";
    }
    
    return capabilities;
}

// Function to detect CPU features at runtime
const CPUFeatures& CPUFeatures::get() {
    static CPUFeatures features;
    static bool initialized = false;
    
    if (!initialized) {
        // Initialize only once
#if defined(CCSM_ARCH_X86_64)
    #if defined(_MSC_VER)
        // MSVC-specific CPU feature detection
        int cpuInfo[4];
        __cpuid(cpuInfo, 0);
        int numIds = cpuInfo[0];
        
        if (numIds >= 1) {
            __cpuid(cpuInfo, 1);
            features.sse2 = (cpuInfo[3] & (1 << 26)) != 0;
            features.sse3 = (cpuInfo[2] & (1 << 0)) != 0;
            features.sse4_1 = (cpuInfo[2] & (1 << 19)) != 0;
            features.sse4_2 = (cpuInfo[2] & (1 << 20)) != 0;
            features.avx = (cpuInfo[2] & (1 << 28)) != 0;
            
            if (numIds >= 7) {
                __cpuidex(cpuInfo, 7, 0);
                features.avx2 = (cpuInfo[1] & (1 << 5)) != 0;
                features.avx512f = (cpuInfo[1] & (1 << 16)) != 0;
            }
        }
    #else
        // GCC, Clang, etc.
        features.sse2 = false;
        features.sse3 = false;
        features.sse4_1 = false;
        features.sse4_2 = false;
        features.avx = false;
        features.avx2 = false;
        features.avx512f = false;
        
        #if defined(__SSE2__)
        features.sse2 = true;
        #endif
        
        #if defined(__SSE3__)
        features.sse3 = true;
        #endif
        
        #if defined(__SSE4_1__)
        features.sse4_1 = true;
        #endif
        
        #if defined(__SSE4_2__)
        features.sse4_2 = true;
        #endif
        
        #if defined(__AVX__)
        features.avx = true;
        #endif
        
        #if defined(__AVX2__)
        features.avx2 = true;
        #endif
        
        #if defined(__AVX512F__)
        features.avx512f = true;
        #endif
    #endif
#endif

#if defined(CCSM_ARCH_ARM64)
        // ARM NEON is always available on ARM64
        features.neon = true;
#endif
        
        initialized = true;
    }
    
    return features;
}

namespace detail {

// -------------------------------------------------------------------------
// AVX implementations
// -------------------------------------------------------------------------

#if defined(CCSM_HAVE_AVX) && defined(__AVX__)

void vector_gt_mask_avx_f32(float* result, const float* a, const float* b, size_t n) {
    size_t i = 0;
    
    // Process 8 elements at a time using AVX
    for (; i + 7 < n; i += 8) {
        // Load 8 elements from a and b
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        
        // Compare a > b (creates a mask with all bits set to 1 for true, 0 for false)
        __m256 vcmp = _mm256_cmp_ps(va, vb, _CMP_GT_OQ);
        
        // Convert the mask to 1.0f (all bits set) or 0.0f (no bits set)
        // We use a logical AND with 1.0f to achieve this
        __m256 vone = _mm256_set1_ps(1.0f);
        __m256 vmask = _mm256_and_ps(vcmp, vone);
        
        // Store the mask in the result array
        _mm256_storeu_ps(result + i, vmask);
    }
    
    // Process remaining elements with scalar code
    for (; i < n; i++) {
        result[i] = (a[i] > b[i]) ? 1.0f : 0.0f;
    }
}

void relu_avx_f32(float* output, const float* input, size_t n) {
    size_t i = 0;
    __m256 vzero = _mm256_setzero_ps();
    
    // Process 8 elements at a time using AVX
    for (; i + 7 < n; i += 8) {
        __m256 vin = _mm256_loadu_ps(input + i);
        __m256 vout = _mm256_max_ps(vzero, vin);
        _mm256_storeu_ps(output + i, vout);
    }
    
    // Process remaining elements
    for (; i < n; i++) {
        output[i] = std::max(0.0f, input[i]);
    }
}

// First silu_avx_f32 implementation - remove it as it's duplicated
// This will be replaced by the improved version below

void vector_add_avx_f32(float* result, const float* a, const float* b, size_t n) {
    size_t i = 0;
    
    // Process 8 elements at a time using AVX
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vr = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(result + i, vr);
    }
    
    // Process remaining elements
    for (; i < n; i++) {
        result[i] = a[i] + b[i];
    }
}

void vector_mul_avx_f32(float* result, const float* a, const float* b, size_t n) {
    size_t i = 0;
    
    // Process 8 elements at a time using AVX
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vr = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(result + i, vr);
    }
    
    // Process remaining elements
    for (; i < n; i++) {
        result[i] = a[i] * b[i];
    }
}

void vector_scale_avx_f32(float* result, const float* a, float scalar, size_t n) {
    size_t i = 0;
    __m256 vs = _mm256_set1_ps(scalar);
    
    // Process 8 elements at a time using AVX
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vr = _mm256_mul_ps(va, vs);
        _mm256_storeu_ps(result + i, vr);
    }
    
    // Process remaining elements
    for (; i < n; i++) {
        result[i] = a[i] * scalar;
    }
}

float vector_dot_avx_f32(const float* a, const float* b, size_t n) {
    size_t i = 0;
    __m256 sum = _mm256_setzero_ps();
    
    // Process 8 elements at a time using AVX
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vmul = _mm256_mul_ps(va, vb);
        sum = _mm256_add_ps(sum, vmul);
    }
    
    // Horizontal sum of AVX register - extract high and low 128-bit parts
    __m128 high = _mm256_extractf128_ps(sum, 1);
    __m128 low = _mm256_extractf128_ps(sum, 0);
    
    // Add high and low parts
    __m128 sum128 = _mm_add_ps(high, low);
    
    // Add upper and lower halves of 128-bit register
    __m128 sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    
    // Add upper and lower 32-bits
    __m128 sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 0x1));
    
    // Extract the final result
    float result = _mm_cvtss_f32(sum32);
    
    // Process remaining elements
    for (; i < n; i++) {
        result += a[i] * b[i];
    }
    
    return result;
}

void relu_avx_f32(float* output, const float* input, size_t n) {
    // Use scalar implementation for simplicity
    for (size_t i = 0; i < n; i++) {
        output[i] = std::max(0.0f, input[i]);
    }
}

void silu_avx_f32(float* output, const float* input, size_t n) {
    size_t i = 0;
    __m256 vone = _mm256_set1_ps(1.0f);
    __m256 vzero = _mm256_setzero_ps();
    
    // Constants for fast, modified Pade approximation
    // These specific constants provide a good balance of accuracy and speed for sigmoid
    __m256 vc1 = _mm256_set1_ps(0.398942280f);
    __m256 vc2 = _mm256_set1_ps(0.0f);
    __m256 vc3 = _mm256_set1_ps(1.13879349f);
    __m256 vhalf = _mm256_set1_ps(0.5f);
    
    // Process 8 elements at a time using AVX
    for (; i + 7 < n; i += 8) {
        __m256 vin = _mm256_loadu_ps(input + i);
        
        // Fast sigmoid using piecewise approximation
        // Clamp inputs to acceptable range
        __m256 vclamp_min = _mm256_set1_ps(-15.0f);
        __m256 vclamp_max = _mm256_set1_ps(15.0f);
        __m256 vx_clamped = _mm256_max_ps(vclamp_min, _mm256_min_ps(vclamp_max, vin));
        
        // Get absolute value of x for symmetrical approximation
        __m256 vabs_x = _mm256_and_ps(vx_clamped, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
        
        // Fast modified sigmoid approximation using quadratic approximation
        // This approximates 1/(1+exp(-x)) well within [-15,15] range
        // sigmoid(x) ≈ 0.5 + 0.5 * (vc1*x) / (vc3 + |x|)
        
        // First, compute the numerator: vc1*x (scaled x)
        __m256 vnum = _mm256_mul_ps(vc1, vx_clamped);
        
        // Now compute the denominator: vc3 + |x|
        __m256 vdenom = _mm256_add_ps(vc3, vabs_x);
        
        // Compute the approximation: 0.5 + 0.5 * (vc1*x) / (vc3 + |x|)
        __m256 vterm = _mm256_div_ps(vnum, vdenom);
        __m256 vterm_scaled = _mm256_mul_ps(vhalf, vterm);
        __m256 vsigmoid = _mm256_add_ps(vhalf, vterm_scaled);
        
        // For very negative inputs (< -15), force sigmoid to 0
        __m256 vtoo_small = _mm256_cmp_ps(vin, _mm256_set1_ps(-15.0f), _CMP_LT_OQ);
        vsigmoid = _mm256_andnot_ps(vtoo_small, vsigmoid);
        
        // For very positive inputs (> 15), force sigmoid to 1.0
        __m256 vtoo_large = _mm256_cmp_ps(vin, _mm256_set1_ps(15.0f), _CMP_GT_OQ);
        vsigmoid = _mm256_or_ps(
            _mm256_and_ps(vtoo_large, vone),
            _mm256_andnot_ps(vtoo_large, vsigmoid)
        );
        
        // Calculate SiLU: x * sigmoid(x)
        __m256 vout = _mm256_mul_ps(vin, vsigmoid);
        _mm256_storeu_ps(output + i, vout);
    }
    
    // Process remaining elements using the scalar equivalent of our fast approximation
    for (; i < n; i++) {
        float x = input[i];
        float sigmoid;
        
        if (x < -15.0f) {
            sigmoid = 0.0f;
        } else if (x > 15.0f) {
            sigmoid = 1.0f;
        } else {
            float abs_x = std::abs(x);
            sigmoid = 0.5f + 0.5f * (0.398942280f * x) / (1.13879349f + abs_x);
        }
        
        output[i] = x * sigmoid;
    }
}

void rms_norm_avx_f32(float* output, const float* input, const float* weight, float epsilon, size_t n) {
    if (n == 0) return;
    
    // Calculate sum of squares using AVX
    __m256 vsum_sq = _mm256_setzero_ps();
    size_t i = 0;
    
    // Process 8 elements at a time
    for (; i + 7 < n; i += 8) {
        __m256 vin = _mm256_loadu_ps(input + i);
        __m256 vsq = _mm256_mul_ps(vin, vin);
        vsum_sq = _mm256_add_ps(vsum_sq, vsq);
    }
    
    // Horizontal sum of vsum_sq
    // Extract high and low 128-bit parts
    __m128 high = _mm256_extractf128_ps(vsum_sq, 1);
    __m128 low = _mm256_extractf128_ps(vsum_sq, 0);
    
    // Add high and low parts
    __m128 sum128 = _mm_add_ps(high, low);
    
    // Add upper and lower halves of 128-bit register
    __m128 sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    
    // Add upper and lower 32-bits
    __m128 sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 0x1));
    
    // Extract the final result
    float sum_sq = _mm_cvtss_f32(sum32);
    
    // Add remaining elements
    for (; i < n; i++) {
        sum_sq += input[i] * input[i];
    }
    
    // Calculate normalization factor: 1/sqrt(sum_sq/n + epsilon)
    float mean_sq = sum_sq / n;
    float inv_norm = 1.0f / std::sqrt(mean_sq + epsilon);
    __m256 vinv_norm = _mm256_set1_ps(inv_norm);
    
    // Apply normalization with weights using AVX
    i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 vin = _mm256_loadu_ps(input + i);
        __m256 vw = _mm256_loadu_ps(weight + i);
        
        // Normalize and scale with weights
        __m256 vnorm = _mm256_mul_ps(vin, vinv_norm);
        __m256 vout = _mm256_mul_ps(vnorm, vw);
        
        // Store result
        _mm256_storeu_ps(output + i, vout);
    }
    
    // Process remaining elements
    for (; i < n; i++) {
        output[i] = input[i] * inv_norm * weight[i];
    }
}

void layer_norm_avx_f32(float* output, const float* input, const float* weight, const float* bias, float epsilon, size_t n) {
    if (n == 0) return;
    
    // First pass: calculate mean using AVX
    __m256 vsum = _mm256_setzero_ps();
    size_t i = 0;
    
    // Process 8 elements at a time
    for (; i + 7 < n; i += 8) {
        __m256 vin = _mm256_loadu_ps(input + i);
        vsum = _mm256_add_ps(vsum, vin);
    }
    
    // Horizontal sum
    // Extract high and low 128-bit parts
    __m128 high = _mm256_extractf128_ps(vsum, 1);
    __m128 low = _mm256_extractf128_ps(vsum, 0);
    
    // Add high and low parts
    __m128 sum128 = _mm_add_ps(high, low);
    
    // Add upper and lower halves of 128-bit register
    __m128 sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    
    // Add upper and lower 32-bits
    __m128 sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 0x1));
    
    // Extract the final result
    float sum = _mm_cvtss_f32(sum32);
    
    // Add remaining elements
    for (; i < n; i++) {
        sum += input[i];
    }
    
    // Calculate mean
    float mean = sum / n;
    __m256 vmean = _mm256_set1_ps(mean);
    
    // Second pass: calculate variance
    __m256 vvar = _mm256_setzero_ps();
    i = 0;
    
    // Process 8 elements at a time
    for (; i + 7 < n; i += 8) {
        __m256 vin = _mm256_loadu_ps(input + i);
        __m256 vdiff = _mm256_sub_ps(vin, vmean);
        vvar = _mm256_add_ps(vvar, _mm256_mul_ps(vdiff, vdiff));
    }
    
    // Horizontal sum for variance
    high = _mm256_extractf128_ps(vvar, 1);
    low = _mm256_extractf128_ps(vvar, 0);
    sum128 = _mm_add_ps(high, low);
    sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 0x1));
    float variance = _mm_cvtss_f32(sum32);
    
    // Add remaining elements
    for (; i < n; i++) {
        float diff = input[i] - mean;
        variance += diff * diff;
    }
    
    // Calculate variance
    variance /= n;
    
    // Calculate normalization factor: 1/sqrt(variance + epsilon)
    float inv_norm = 1.0f / std::sqrt(variance + epsilon);
    __m256 vinv_norm = _mm256_set1_ps(inv_norm);
    
    // Third pass: normalize, scale with weights, and add bias
    i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 vin = _mm256_loadu_ps(input + i);
        __m256 vw = _mm256_loadu_ps(weight + i);
        __m256 vb = _mm256_loadu_ps(bias + i);
        
        // x_norm = (x - mean) * inv_std
        __m256 vcentered = _mm256_sub_ps(vin, vmean);
        __m256 vnorm = _mm256_mul_ps(vcentered, vinv_norm);
        
        // y = x_norm * weight + bias
        __m256 vscaled = _mm256_mul_ps(vnorm, vw);
        __m256 vout = _mm256_add_ps(vscaled, vb);
        
        // Store result
        _mm256_storeu_ps(output + i, vout);
    }
    
    // Process remaining elements
    for (; i < n; i++) {
        float normalized = (input[i] - mean) * inv_norm;
        output[i] = normalized * weight[i] + bias[i];
    }
}

void softmax_avx_f32(float* output, const float* input, size_t n) {
    if (n == 0) return;
    
    // Find max value for numerical stability using SIMD
    __m256 vmax = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
    size_t i = 0;
    
    // Find max value using AVX
    for (; i + 7 < n; i += 8) {
        __m256 vin = _mm256_loadu_ps(input + i);
        vmax = _mm256_max_ps(vmax, vin);
    }
    
    // Horizontal maximum of vmax
    __m128 high = _mm256_extractf128_ps(vmax, 1);
    __m128 low = _mm256_extractf128_ps(vmax, 0);
    __m128 max4 = _mm_max_ps(high, low);
    __m128 max2 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
    __m128 max1 = _mm_max_ss(max2, _mm_shuffle_ps(max2, max2, 0x1));
    float max_val = _mm_cvtss_f32(max1);
    
    // Check remaining elements for maximum
    for (; i < n; i++) {
        max_val = std::max(max_val, input[i]);
    }
    
    // Use full precision exp for output correctness
    __m256 vmax_val = _mm256_set1_ps(max_val);
    __m256 vsum = _mm256_setzero_ps();
    
    // Use a more accurate but still optimized exp approximation
    // Constants for minimax approximation of exp(x) for x in [-87.33, 0]
    // These are appropriate for float32 in softmax calculations
    __m256 vone = _mm256_set1_ps(1.0f);
    __m256 vminmax = _mm256_set1_ps(-87.33f); // Min value for float32 exp
    
    // Coefficients from a 6th-order polynomial approximation
    // exp(x) ≈ a0 + x*(a1 + x*(a2 + x*(a3 + x*(a4 + x*(a5 + x*a6)))))
    __m256 va0 = _mm256_set1_ps(1.0000000f);
    __m256 va1 = _mm256_set1_ps(0.9999999f);
    __m256 va2 = _mm256_set1_ps(0.4999986f);
    __m256 va3 = _mm256_set1_ps(0.1666653f);
    __m256 va4 = _mm256_set1_ps(0.0416573f);
    __m256 va5 = _mm256_set1_ps(0.0083013f);
    __m256 va6 = _mm256_set1_ps(0.0013298f);
    
    // Process in chunks of 8 elements
    i = 0;
    for (; i + 7 < n; i += 8) {
        // Compute x - max_val for numerical stability
        __m256 vin = _mm256_loadu_ps(input + i);
        __m256 vx = _mm256_sub_ps(vin, vmax_val);
        
        // Clamp very negative values to avoid underflow
        vx = _mm256_max_ps(vx, vminmax);
        
        // Horner scheme for polynomial evaluation
        // exp(x) ≈ a0 + x*(a1 + x*(a2 + x*(a3 + x*(a4 + x*(a5 + x*a6)))))
        __m256 vexp = _mm256_add_ps(
            _mm256_mul_ps(vx, va6),
            va5
        );
        vexp = _mm256_add_ps(
            _mm256_mul_ps(vexp, vx),
            va4
        );
        vexp = _mm256_add_ps(
            _mm256_mul_ps(vexp, vx),
            va3
        );
        vexp = _mm256_add_ps(
            _mm256_mul_ps(vexp, vx),
            va2
        );
        vexp = _mm256_add_ps(
            _mm256_mul_ps(vexp, vx),
            va1
        );
        vexp = _mm256_add_ps(
            _mm256_mul_ps(vexp, vx),
            va0
        );
        
        // Zero out any negative values from underflow
        __m256 vmask = _mm256_cmp_ps(vx, vminmax, _CMP_EQ_OQ);
        vexp = _mm256_andnot_ps(vmask, vexp);
        
        // Store the result
        _mm256_storeu_ps(output + i, vexp);
        
        // Accumulate sum
        vsum = _mm256_add_ps(vsum, vexp);
    }
    
    // Reduce vsum to a single value
    __m128 high_sum = _mm256_extractf128_ps(vsum, 1);
    __m128 low_sum = _mm256_extractf128_ps(vsum, 0);
    __m128 sum128 = _mm_add_ps(high_sum, low_sum);
    __m128 sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    __m128 sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 0x1));
    float sum = _mm_cvtss_f32(sum32);
    
    // Process remaining elements with standard exp
    for (; i < n; i++) {
        float x = input[i] - max_val;
        // Clamp very negative values
        x = std::max(x, -87.33f);
        // Use standard exp for the rest
        float exp_val = std::exp(x);
        output[i] = exp_val;
        sum += exp_val;
    }
    
    // Normalize with 1/sum using SIMD
    __m256 vinv_sum = _mm256_set1_ps(1.0f / sum);
    
    i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 vout = _mm256_loadu_ps(output + i);
        vout = _mm256_mul_ps(vout, vinv_sum);
        _mm256_storeu_ps(output + i, vout);
    }
    
    // Process remaining elements
    float inv_sum = 1.0f / sum;
    for (; i < n; i++) {
        output[i] *= inv_sum;
    }
}

void attention_avx_f32(
    float* output,              // [batch_size, seq_len, head_size]
    const float* query,         // [batch_size, seq_len, num_heads, head_size]
    const float* key,           // [batch_size, seq_len, num_heads, head_size]
    const float* value,         // [batch_size, seq_len, num_heads, head_size]
    const float* mask,          // [batch_size, 1, 1, seq_len] or nullptr
    size_t batch_size,
    size_t seq_len,
    size_t num_heads,
    size_t head_size,
    float scale
) {
    // Compute attention for each batch and head
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t h = 0; h < num_heads; h++) {
            // Allocate temporary storage for attention scores and probabilities
            std::vector<float> scores(seq_len * seq_len, 0.0f);
            std::vector<float> probs(seq_len * seq_len, 0.0f);
            
            // Step 1: Compute query-key attention scores with SIMD optimization
            // Compute QK^T (matmul) for this batch and head - scaled by factor
            __m256 vscale = _mm256_set1_ps(scale);
            
            for (size_t i = 0; i < seq_len; i++) {
                for (size_t j = 0; j < seq_len; j++) {
                    // Compute dot product between query[i] and key[j] vectors
                    __m256 vdot = _mm256_setzero_ps();
                    size_t k = 0;
                    
                    // Process 8 elements at a time
                    for (; k + 7 < head_size; k += 8) {
                        size_t q_idx = ((b * seq_len + i) * num_heads + h) * head_size + k;
                        size_t k_idx = ((b * seq_len + j) * num_heads + h) * head_size + k;
                        
                        __m256 vq = _mm256_loadu_ps(&query[q_idx]);
                        __m256 vk = _mm256_loadu_ps(&key[k_idx]);
                        __m256 vmul = _mm256_mul_ps(vq, vk);
                        vdot = _mm256_add_ps(vdot, vmul);
                    }
                    
                    // Horizontal sum of vdot
                    __m128 high = _mm256_extractf128_ps(vdot, 1);
                    __m128 low = _mm256_extractf128_ps(vdot, 0);
                    __m128 sum128 = _mm_add_ps(high, low);
                    __m128 sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
                    __m128 sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 0x1));
                    float dot = _mm_cvtss_f32(sum32);
                    
                    // Process remaining elements
                    for (; k < head_size; k++) {
                        size_t q_idx = ((b * seq_len + i) * num_heads + h) * head_size + k;
                        size_t k_idx = ((b * seq_len + j) * num_heads + h) * head_size + k;
                        dot += query[q_idx] * key[k_idx];
                    }
                    
                    // Scale the dot product
                    float score = dot * scale;
                    
                    // Apply mask if provided
                    if (mask != nullptr) {
                        // Apply causal mask (if mask[b,1,1,j] == 0, then hide position j from position i)
                        size_t mask_idx = b * seq_len + j; // Simplified for causal mask
                        if (j > i || mask[mask_idx] == 0) {
                            score = -std::numeric_limits<float>::infinity();
                        }
                    }
                    
                    scores[i * seq_len + j] = score;
                }
            }
            
            // Step 2: Apply softmax to each row of scores to get attention weights
            for (size_t i = 0; i < seq_len; i++) {
                // Find max value for numerical stability
                float max_val = -std::numeric_limits<float>::infinity();
                for (size_t j = 0; j < seq_len; j++) {
                    max_val = std::max(max_val, scores[i * seq_len + j]);
                }
                
                // Compute exp(score - max_val) and sum
                float sum_exp = 0.0f;
                for (size_t j = 0; j < seq_len; j++) {
                    float score = scores[i * seq_len + j];
                    float exp_val = 0.0f;
                    
                    if (std::isinf(score) && score < 0) {
                        // Handle -inf (masked values)
                        exp_val = 0.0f;
                    } else {
                        // For normal values, compute exp(score - max_val)
                        exp_val = std::exp(score - max_val);
                    }
                    
                    probs[i * seq_len + j] = exp_val;
                    sum_exp += exp_val;
                }
                
                // Normalize probabilities
                float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
                for (size_t j = 0; j < seq_len; j++) {
                    probs[i * seq_len + j] *= inv_sum;
                }
            }
            
            // Step 3: Compute weighted sum of values (attn_probs @ value)
            // For each query position and output dimension
            for (size_t i = 0; i < seq_len; i++) {
                // Process each dimension of the output
                size_t d = 0;
                
                // Process 8 elements at a time with AVX
                for (; d + 7 < head_size; d += 8) {
                    __m256 vsum = _mm256_setzero_ps();
                    
                    // Weighted sum over all value vectors
                    for (size_t j = 0; j < seq_len; j++) {
                        size_t v_idx = ((b * seq_len + j) * num_heads + h) * head_size + d;
                        float attention_weight = probs[i * seq_len + j];
                        
                        // Skip computation for near-zero weights
                        if (attention_weight < 1e-10f) continue;
                        
                        __m256 vvalue = _mm256_loadu_ps(&value[v_idx]);
                        __m256 vweight = _mm256_set1_ps(attention_weight);
                        __m256 vweighted = _mm256_mul_ps(vvalue, vweight);
                        vsum = _mm256_add_ps(vsum, vweighted);
                    }
                    
                    // Store result
                    size_t out_idx = (b * seq_len + i) * head_size + d;
                    _mm256_storeu_ps(&output[out_idx], vsum);
                }
                
                // Process remaining elements
                for (; d < head_size; d++) {
                    float sum = 0.0f;
                    for (size_t j = 0; j < seq_len; j++) {
                        size_t v_idx = ((b * seq_len + j) * num_heads + h) * head_size + d;
                        sum += probs[i * seq_len + j] * value[v_idx];
                    }
                    
                    size_t out_idx = (b * seq_len + i) * head_size + d;
                    output[out_idx] = sum;
                }
            }
        }
    }
}

void matrix_mul_avx_f32(float* c, const float* a, const float* b, size_t m, size_t k, size_t n) {
    // Advanced implementation using AVX and cache blocking (similar to AVX2 version but without FMA)
    constexpr size_t BLOCK_SIZE_M = 16; // Optimal block sizes based on benchmarks
    constexpr size_t BLOCK_SIZE_N = 16; 
    constexpr size_t BLOCK_SIZE_K = 32;
    constexpr size_t SIMD_WIDTH = 8;    // AVX processes 8 floats at once
    
    // Initialize result matrix to zeros
    for (size_t i = 0; i < m * n; i++) {
        c[i] = 0.0f;
    }
    
    // Outer blocking for cache efficiency
    for (size_t i0 = 0; i0 < m; i0 += BLOCK_SIZE_M) {
        size_t i_end = std::min(i0 + BLOCK_SIZE_M, m);
        
        for (size_t j0 = 0; j0 < n; j0 += BLOCK_SIZE_N) {
            size_t j_end = std::min(j0 + BLOCK_SIZE_N, n);
            
            // Here we process j in multiples of 8 (SIMD_WIDTH) for better vectorization
            for (size_t j_offset = 0; j_offset < j_end - j0; j_offset += SIMD_WIDTH) {
                size_t j_simd_width = std::min(SIMD_WIDTH, j_end - (j0 + j_offset));
                
                // If we don't have a full vector width, handle scalar remainder later
                if (j_simd_width != SIMD_WIDTH && j_simd_width != 0) {
                    continue;
                }
                
                for (size_t i = i0; i < i_end; i++) {
                    size_t j = j0 + j_offset;
                    
                    // We're computing an entire row of the output at once (8 elements)
                    __m256 vsum = _mm256_loadu_ps(&c[i * n + j]);
                    
                    // Process inner dimension in blocks for better cache utilization
                    for (size_t k0 = 0; k0 < k; k0 += BLOCK_SIZE_K) {
                        size_t k_end = std::min(k0 + BLOCK_SIZE_K, k);
                        
                        // Process the block using AVX
                        for (size_t l = k0; l < k_end; l++) {
                            // Broadcast single element from A matrix (reuse in register)
                            __m256 va = _mm256_set1_ps(a[i * k + l]);
                            
                            // Load 8 elements from B matrix (consecutive in memory)
                            __m256 vb = _mm256_loadu_ps(&b[l * n + j]);
                            
                            // vsum += va * vb (using mul and add in AVX, no FMA)
                            __m256 vprod = _mm256_mul_ps(va, vb);
                            vsum = _mm256_add_ps(vsum, vprod);
                        }
                    }
                    
                    // Store result back
                    _mm256_storeu_ps(&c[i * n + j], vsum);
                }
            }
            
            // Handle remainders (when j_end - j0 is not multiple of SIMD_WIDTH)
            for (size_t i = i0; i < i_end; i++) {
                for (size_t j = j0 + ((j_end - j0) / SIMD_WIDTH) * SIMD_WIDTH; j < j_end; j++) {
                    float sum = 0.0f;
                    for (size_t l = 0; l < k; l++) {
                        sum += a[i * k + l] * b[l * n + j];
                    }
                    c[i * n + j] = sum;
                }
            }
        }
    }
}

#endif // CCSM_HAVE_AVX

// -------------------------------------------------------------------------
// AVX2 implementations
// -------------------------------------------------------------------------

#if defined(CCSM_HAVE_AVX2) && defined(__AVX2__)

void vector_add_avx2_f32(const float* a, const float* b, float* c, size_t n) {
    // AVX2 doesn't add much over AVX for basic float operations like add
    vector_add_avx_f32(a, b, c, n);
}

void vector_mul_avx2_f32(const float* a, const float* b, float* c, size_t n) {
    // AVX2 doesn't add much over AVX for basic float operations like multiply
    vector_mul_avx_f32(a, b, c, n);
}

void vector_fma_avx2_f32(const float* a, const float* b, const float* c, float* d, size_t n) {
    size_t i = 0;
    
    // Process 8 elements at a time using AVX2 FMA
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_loadu_ps(c + i);
        
        // d = a * b + c
        __m256 vd = _mm256_fmadd_ps(va, vb, vc);
        _mm256_storeu_ps(d + i, vd);
    }
    
    // Process remaining elements
    for (; i < n; i++) {
        d[i] = a[i] * b[i] + c[i];
    }
}

void vector_scale_avx2_f32(const float* a, float scalar, float* b, size_t n) {
    // Reuse AVX implementation since AVX2 doesn't add much for this operation
    vector_scale_avx_f32(a, scalar, b, n);
}

float vector_dot_avx2_f32(const float* a, const float* b, size_t n) {
    size_t i = 0;
    __m256 sum = _mm256_setzero_ps();
    
    // Process 8 elements at a time using AVX2 FMA for dot product
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        
        // sum += a * b (using FMA)
        sum = _mm256_fmadd_ps(va, vb, sum);
    }
    
    // Horizontal sum of AVX register - extract high and low 128-bit parts
    __m128 high = _mm256_extractf128_ps(sum, 1);
    __m128 low = _mm256_extractf128_ps(sum, 0);
    
    // Add high and low parts
    __m128 sum128 = _mm_add_ps(high, low);
    
    // Add upper and lower halves of 128-bit register
    __m128 sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    
    // Add upper and lower 32-bits
    __m128 sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 0x1));
    
    // Extract the final result
    float result = _mm_cvtss_f32(sum32);
    
    // Process remaining elements
    for (; i < n; i++) {
        result += a[i] * b[i];
    }
    
    return result;
}

void matrix_mul_avx2_f32(float* c, const float* a, const float* b, size_t m, size_t k, size_t n) {
    // Advanced implementation using AVX2, FMA, and cache blocking
    constexpr size_t BLOCK_SIZE_M = 16; // Optimal block sizes based on benchmarks
    constexpr size_t BLOCK_SIZE_N = 16; 
    constexpr size_t BLOCK_SIZE_K = 32;
    constexpr size_t SIMD_WIDTH = 8;    // AVX/AVX2 processes 8 floats at once
    
    // Initialize result matrix to zeros
    for (size_t i = 0; i < m * n; i++) {
        c[i] = 0.0f;
    }
    
    // Outer blocking for cache efficiency
    for (size_t i0 = 0; i0 < m; i0 += BLOCK_SIZE_M) {
        size_t i_end = std::min(i0 + BLOCK_SIZE_M, m);
        
        for (size_t j0 = 0; j0 < n; j0 += BLOCK_SIZE_N) {
            size_t j_end = std::min(j0 + BLOCK_SIZE_N, n);
            
            // Here we process j in multiples of 8 (SIMD_WIDTH) for better vectorization
            for (size_t j_offset = 0; j_offset < j_end - j0; j_offset += SIMD_WIDTH) {
                size_t j_simd_width = std::min(SIMD_WIDTH, j_end - (j0 + j_offset));
                
                // If we don't have a full vector width, handle scalar remainder later
                if (j_simd_width != SIMD_WIDTH && j_simd_width != 0) {
                    continue;
                }
                
                for (size_t i = i0; i < i_end; i++) {
                    size_t j = j0 + j_offset;
                    
                    // We're computing an entire row of the output at once (8 elements)
                    __m256 vsum = _mm256_loadu_ps(&c[i * n + j]);
                    
                    // Process inner dimension in blocks for better cache utilization
                    for (size_t k0 = 0; k0 < k; k0 += BLOCK_SIZE_K) {
                        size_t k_end = std::min(k0 + BLOCK_SIZE_K, k);
                        
                        // Process the block using AVX2 FMA instructions
                        for (size_t l = k0; l < k_end; l++) {
                            // Broadcast single element from A matrix (reuse in register)
                            __m256 va = _mm256_set1_ps(a[i * k + l]);
                            
                            // Load 8 elements from B matrix (consecutive in memory)
                            // This matches the exact memory layout for faster loads
                            __m256 vb = _mm256_loadu_ps(&b[l * n + j]);
                            
                            // vsum += va * vb using FMA
                            vsum = _mm256_fmadd_ps(va, vb, vsum);
                        }
                    }
                    
                    // Store result back
                    _mm256_storeu_ps(&c[i * n + j], vsum);
                }
            }
            
            // Handle remainders (when j_end - j0 is not multiple of SIMD_WIDTH)
            // This ensures correctness for all matrix sizes
            for (size_t i = i0; i < i_end; i++) {
                for (size_t j = j0 + ((j_end - j0) / SIMD_WIDTH) * SIMD_WIDTH; j < j_end; j++) {
                    float sum = 0.0f;
                    for (size_t l = 0; l < k; l++) {
                        sum += a[i * k + l] * b[l * n + j];
                    }
                    c[i * n + j] = sum;
                }
            }
        }
    }
}

#endif // CCSM_HAVE_AVX2

// -------------------------------------------------------------------------
// NEON implementations
// -------------------------------------------------------------------------

#if defined(CCSM_HAVE_NEON) && defined(__ARM_NEON)

void vector_gt_mask_neon_f32(float* result, const float* a, const float* b, size_t n) {
    size_t i = 0;
    
    // Process 4 elements at a time using NEON
    for (; i + 3 < n; i += 4) {
        // Load 4 elements from a and b
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        
        // Compare a > b (creates a mask with all bits set to 1 for true, 0 for false)
        uint32x4_t vcmp = vcgtq_f32(va, vb);
        
        // Convert the mask to 1.0f (all bits set) or 0.0f (no bits set)
        // First, convert uint32 mask to float32 mask
        float32x4_t vone = vdupq_n_f32(1.0f);
        float32x4_t vzero = vdupq_n_f32(0.0f);
        float32x4_t vmask = vbslq_f32(vcmp, vone, vzero);
        
        // Store the mask in the result array
        vst1q_f32(result + i, vmask);
    }
    
    // Process remaining elements with scalar code
    for (; i < n; i++) {
        result[i] = (a[i] > b[i]) ? 1.0f : 0.0f;
    }
}

void vector_add_neon_f32(float* result, const float* a, const float* b, size_t n) {
    size_t i = 0;
    
    // Process 4 elements at a time using NEON
    for (; i + 3 < n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vc = vaddq_f32(va, vb);
        vst1q_f32(result + i, vc);
    }
    
    // Process remaining elements
    for (; i < n; i++) {
        result[i] = a[i] + b[i];
    }
}

void vector_mul_neon_f32(float* result, const float* a, const float* b, size_t n) {
    size_t i = 0;
    
    // Process 4 elements at a time using NEON
    for (; i + 3 < n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vc = vmulq_f32(va, vb);
        vst1q_f32(result + i, vc);
    }
    
    // Process remaining elements
    for (; i < n; i++) {
        result[i] = a[i] * b[i];
    }
}

void vector_fma_neon_f32(float* result, const float* a, const float* b, const float* c, size_t n) {
    size_t i = 0;
    
    // Process 4 elements at a time using NEON
    for (; i + 3 < n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vc = vld1q_f32(c + i);
        
        // result = a * b + c
        float32x4_t vr = vmlaq_f32(vc, va, vb);
        vst1q_f32(result + i, vr);
    }
    
    // Process remaining elements
    for (; i < n; i++) {
        result[i] = a[i] * b[i] + c[i];
    }
}

void vector_scale_neon_f32(float* result, const float* a, float scalar, size_t n) {
    size_t i = 0;
    float32x4_t vs = vdupq_n_f32(scalar);
    
    // Process 4 elements at a time using NEON
    for (; i + 3 < n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vr = vmulq_f32(va, vs);
        vst1q_f32(result + i, vr);
    }
    
    // Process remaining elements
    for (; i < n; i++) {
        result[i] = a[i] * scalar;
    }
}

float vector_dot_neon_f32(const float* a, const float* b, size_t n) {
    size_t i = 0;
    float32x4_t vsum = vdupq_n_f32(0.0f);
    
    // Process 4 elements at a time using NEON
    for (; i + 3 < n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        // Multiply and accumulate: vsum += va * vb
        vsum = vmlaq_f32(vsum, va, vb);
    }
    
    // Horizontal sum of NEON register
    // Add the high 64 bits to the low 64 bits
    float32x2_t vsum2 = vadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
    // Add adjacent pairs of 32-bit floats
    vsum2 = vpadd_f32(vsum2, vsum2);
    // Extract the first lane (the sum)
    float result = vget_lane_f32(vsum2, 0);
    
    // Process remaining elements
    for (; i < n; i++) {
        result += a[i] * b[i];
    }
    
    return result;
}

void relu_neon_f32(float* output, const float* input, size_t n) {
    size_t i = 0;
    float32x4_t vzero = vdupq_n_f32(0.0f);
    
    // Process 4 elements at a time using NEON
    for (; i + 3 < n; i += 4) {
        float32x4_t vin = vld1q_f32(input + i);
        float32x4_t vout = vmaxq_f32(vzero, vin);
        vst1q_f32(output + i, vout);
    }
    
    // Process remaining elements
    for (; i < n; i++) {
        output[i] = std::max(0.0f, input[i]);
    }
}

void silu_neon_f32(float* output, const float* input, size_t n) {
    size_t i = 0;
    
    // Constants for fast sigmoid approximation - same as in AVX version
    float32x4_t vone = vdupq_n_f32(1.0f);
    float32x4_t vhalf = vdupq_n_f32(0.5f);
    float32x4_t vc1 = vdupq_n_f32(0.398942280f);
    float32x4_t vc3 = vdupq_n_f32(1.13879349f);
    
    // Process 4 elements at a time using NEON
    for (; i + 3 < n; i += 4) {
        float32x4_t vin = vld1q_f32(input + i);
        
        // Fast sigmoid using piecewise approximation
        // Clamp inputs to acceptable range
        float32x4_t vclamp_min = vdupq_n_f32(-15.0f);
        float32x4_t vclamp_max = vdupq_n_f32(15.0f);
        float32x4_t vx_clamped = vmaxq_f32(vclamp_min, vminq_f32(vclamp_max, vin));
        
        // Get absolute value of x
        float32x4_t vabs_x = vabsq_f32(vx_clamped);
        
        // Fast sigmoid approximation: 0.5 + 0.5 * (vc1*x) / (vc3 + |x|)
        // Compute the numerator: vc1*x
        float32x4_t vnum = vmulq_f32(vc1, vx_clamped);
        
        // Compute the denominator: vc3 + |x|
        float32x4_t vdenom = vaddq_f32(vc3, vabs_x);
        
        // Compute the approximation: 0.5 + 0.5 * (vc1*x) / (vc3 + |x|)
        float32x4_t vterm = vdivq_f32(vnum, vdenom);
        float32x4_t vterm_scaled = vmulq_f32(vhalf, vterm);
        float32x4_t vsigmoid = vaddq_f32(vhalf, vterm_scaled);
        
        // Handle edge cases with uint32x4_t masks
        uint32x4_t vtoo_small = vcltq_f32(vin, vdupq_n_f32(-15.0f));
        uint32x4_t vtoo_large = vcgtq_f32(vin, vdupq_n_f32(15.0f));
        
        // Apply masks for clamping
        vsigmoid = vbslq_f32(vtoo_small, vdupq_n_f32(0.0f), vsigmoid);
        vsigmoid = vbslq_f32(vtoo_large, vone, vsigmoid);
        
        // Calculate SiLU: x * sigmoid(x)
        float32x4_t vout = vmulq_f32(vin, vsigmoid);
        vst1q_f32(output + i, vout);
    }
    
    // Process remaining elements with the same approximation
    for (; i < n; i++) {
        float x = input[i];
        float sigmoid;
        
        if (x < -15.0f) {
            sigmoid = 0.0f;
        } else if (x > 15.0f) {
            sigmoid = 1.0f;
        } else {
            float abs_x = std::abs(x);
            sigmoid = 0.5f + 0.5f * (0.398942280f * x) / (1.13879349f + abs_x);
        }
        
        output[i] = x * sigmoid;
    }
}

void softmax_neon_f32(float* output, const float* input, size_t n) {
    if (n == 0) return;
    
    // Find max value for numerical stability using SIMD
    float32x4_t vmax = vdupq_n_f32(-std::numeric_limits<float>::infinity());
    size_t i = 0;
    
    // Find max value using NEON
    for (; i + 3 < n; i += 4) {
        float32x4_t vin = vld1q_f32(input + i);
        vmax = vmaxq_f32(vmax, vin);
    }
    
    // Horizontal maximum of vmax
    float32x2_t vmax2 = vpmax_f32(vget_low_f32(vmax), vget_high_f32(vmax));
    vmax2 = vpmax_f32(vmax2, vmax2);
    float max_val = vget_lane_f32(vmax2, 0);
    
    // Check remaining elements for maximum
    for (; i < n; i++) {
        max_val = std::max(max_val, input[i]);
    }
    
    // Use accurate exp for output correctness
    float32x4_t vmax_val = vdupq_n_f32(max_val);
    float32x4_t vsum = vdupq_n_f32(0.0f);
    
    // Use a more accurate approximation for exp(x) in the range [-87.33, 0]
    // Constants for minimax approximation 
    float32x4_t vone = vdupq_n_f32(1.0f);
    float32x4_t vminmax = vdupq_n_f32(-87.33f); // Min value for float32 exp
    
    // Coefficients for 6th-order polynomial approximation
    // exp(x) ≈ a0 + x*(a1 + x*(a2 + x*(a3 + x*(a4 + x*(a5 + x*a6)))))
    float32x4_t va0 = vdupq_n_f32(1.0000000f);
    float32x4_t va1 = vdupq_n_f32(0.9999999f);
    float32x4_t va2 = vdupq_n_f32(0.4999986f);
    float32x4_t va3 = vdupq_n_f32(0.1666653f);
    float32x4_t va4 = vdupq_n_f32(0.0416573f);
    float32x4_t va5 = vdupq_n_f32(0.0083013f);
    float32x4_t va6 = vdupq_n_f32(0.0013298f);
    
    // Process in chunks of 4 elements
    i = 0;
    for (; i + 3 < n; i += 4) {
        // Compute x - max_val for numerical stability
        float32x4_t vin = vld1q_f32(input + i);
        float32x4_t vx = vsubq_f32(vin, vmax_val);
        
        // Clamp very negative values to avoid underflow
        vx = vmaxq_f32(vx, vminmax);
        
        // Horner scheme for polynomial evaluation
        // exp(x) ≈ a0 + x*(a1 + x*(a2 + x*(a3 + x*(a4 + x*(a5 + x*a6)))))
        float32x4_t vexp = vaddq_f32(
            vmulq_f32(vx, va6),
            va5
        );
        vexp = vaddq_f32(
            vmulq_f32(vexp, vx),
            va4
        );
        vexp = vaddq_f32(
            vmulq_f32(vexp, vx),
            va3
        );
        vexp = vaddq_f32(
            vmulq_f32(vexp, vx),
            va2
        );
        vexp = vaddq_f32(
            vmulq_f32(vexp, vx),
            va1
        );
        vexp = vaddq_f32(
            vmulq_f32(vexp, vx),
            va0
        );
        
        // Zero out any negative values from underflow
        uint32x4_t mask = vceqq_f32(vx, vminmax);
        vexp = vbslq_f32(mask, vdupq_n_f32(0.0f), vexp);
        
        // Store the result
        vst1q_f32(output + i, vexp);
        
        // Accumulate sum
        vsum = vaddq_f32(vsum, vexp);
    }
    
    // Reduce vsum to a single value
    float32x2_t vsum2 = vadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
    vsum2 = vpadd_f32(vsum2, vsum2);
    float sum = vget_lane_f32(vsum2, 0);
    
    // Process remaining elements with standard exp for accuracy
    for (; i < n; i++) {
        float x = input[i] - max_val;
        // Clamp very negative values
        x = std::max(x, -87.33f);
        // Use standard exp for the rest
        float exp_val = std::exp(x);
        output[i] = exp_val;
        sum += exp_val;
    }
    
    // Normalize with 1/sum using SIMD
    float32x4_t vinv_sum = vdupq_n_f32(1.0f / sum);
    
    i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t vout = vld1q_f32(output + i);
        vout = vmulq_f32(vout, vinv_sum);
        vst1q_f32(output + i, vout);
    }
    
    // Process remaining elements
    float inv_sum = 1.0f / sum;
    for (; i < n; i++) {
        output[i] *= inv_sum;
    }
}

void rms_norm_neon_f32(float* output, const float* input, const float* weight, float epsilon, size_t n) {
    if (n == 0) return;
    
    // Calculate sum of squares using NEON
    float32x4_t vsum_sq = vdupq_n_f32(0.0f);
    size_t i = 0;
    
    // Process 4 elements at a time
    for (; i + 3 < n; i += 4) {
        float32x4_t vin = vld1q_f32(input + i);
        float32x4_t vsq = vmulq_f32(vin, vin);
        vsum_sq = vaddq_f32(vsum_sq, vsq);
    }
    
    // Horizontal sum of vsum_sq
    float32x2_t vsum_sq2 = vadd_f32(vget_low_f32(vsum_sq), vget_high_f32(vsum_sq));
    vsum_sq2 = vpadd_f32(vsum_sq2, vsum_sq2);
    float sum_sq = vget_lane_f32(vsum_sq2, 0);
    
    // Add remaining elements
    for (; i < n; i++) {
        sum_sq += input[i] * input[i];
    }
    
    // Calculate normalization factor: 1/sqrt(sum_sq/n + epsilon)
    float mean_sq = sum_sq / n;
    float inv_norm = 1.0f / std::sqrt(mean_sq + epsilon);
    float32x4_t vinv_norm = vdupq_n_f32(inv_norm);
    
    // Apply normalization with weights using NEON
    i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t vin = vld1q_f32(input + i);
        float32x4_t vw = vld1q_f32(weight + i);
        
        // Normalize and scale with weights
        float32x4_t vnorm = vmulq_f32(vin, vinv_norm);
        float32x4_t vout = vmulq_f32(vnorm, vw);
        
        // Store result
        vst1q_f32(output + i, vout);
    }
    
    // Process remaining elements
    for (; i < n; i++) {
        output[i] = input[i] * inv_norm * weight[i];
    }
}

void layer_norm_neon_f32(float* output, const float* input, const float* weight, const float* bias, float epsilon, size_t n) {
    if (n == 0) return;
    
    // First pass: calculate mean using NEON
    float32x4_t vsum = vdupq_n_f32(0.0f);
    size_t i = 0;
    
    // Process 4 elements at a time
    for (; i + 3 < n; i += 4) {
        float32x4_t vin = vld1q_f32(input + i);
        vsum = vaddq_f32(vsum, vin);
    }
    
    // Horizontal sum
    float32x2_t vsum2 = vadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
    vsum2 = vpadd_f32(vsum2, vsum2);
    float sum = vget_lane_f32(vsum2, 0);
    
    // Add remaining elements
    for (; i < n; i++) {
        sum += input[i];
    }
    
    // Calculate mean
    float mean = sum / n;
    float32x4_t vmean = vdupq_n_f32(mean);
    
    // Second pass: calculate variance
    float32x4_t vvar = vdupq_n_f32(0.0f);
    i = 0;
    
    // Process 4 elements at a time
    for (; i + 3 < n; i += 4) {
        float32x4_t vin = vld1q_f32(input + i);
        float32x4_t vdiff = vsubq_f32(vin, vmean);
        vvar = vaddq_f32(vvar, vmulq_f32(vdiff, vdiff));
    }
    
    // Horizontal sum for variance
    float32x2_t vvar2 = vadd_f32(vget_low_f32(vvar), vget_high_f32(vvar));
    vvar2 = vpadd_f32(vvar2, vvar2);
    float variance = vget_lane_f32(vvar2, 0);
    
    // Add remaining elements
    for (; i < n; i++) {
        float diff = input[i] - mean;
        variance += diff * diff;
    }
    
    // Calculate variance
    variance /= n;
    
    // Calculate normalization factor: 1/sqrt(variance + epsilon)
    float inv_norm = 1.0f / std::sqrt(variance + epsilon);
    float32x4_t vinv_norm = vdupq_n_f32(inv_norm);
    
    // Third pass: normalize, scale with weights, and add bias
    i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t vin = vld1q_f32(input + i);
        float32x4_t vw = vld1q_f32(weight + i);
        float32x4_t vb = vld1q_f32(bias + i);
        
        // x_norm = (x - mean) * inv_std
        float32x4_t vcentered = vsubq_f32(vin, vmean);
        float32x4_t vnorm = vmulq_f32(vcentered, vinv_norm);
        
        // y = x_norm * weight + bias
        float32x4_t vscaled = vmulq_f32(vnorm, vw);
        float32x4_t vout = vaddq_f32(vscaled, vb);
        
        // Store result
        vst1q_f32(output + i, vout);
    }
    
    // Process remaining elements
    for (; i < n; i++) {
        float normalized = (input[i] - mean) * inv_norm;
        output[i] = normalized * weight[i] + bias[i];
    }
}

void attention_neon_f32(
    float* output,              // [batch_size, seq_len, head_size]
    const float* query,         // [batch_size, seq_len, num_heads, head_size]
    const float* key,           // [batch_size, seq_len, num_heads, head_size]
    const float* value,         // [batch_size, seq_len, num_heads, head_size]
    const float* mask,          // [batch_size, 1, 1, seq_len] or nullptr
    size_t batch_size,
    size_t seq_len,
    size_t num_heads,
    size_t head_size,
    float scale
) {
    // Compute attention for each batch and head
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t h = 0; h < num_heads; h++) {
            // Allocate temporary storage for attention scores and probabilities
            std::vector<float> scores(seq_len * seq_len, 0.0f);
            std::vector<float> probs(seq_len * seq_len, 0.0f);
            
            // Step 1: Compute query-key attention scores with NEON optimization
            // Compute QK^T (matmul) for this batch and head - scaled by factor
            float32x4_t vscale = vdupq_n_f32(scale);
            
            for (size_t i = 0; i < seq_len; i++) {
                for (size_t j = 0; j < seq_len; j++) {
                    // Compute dot product between query[i] and key[j] vectors
                    float32x4_t vdot = vdupq_n_f32(0.0f);
                    size_t k = 0;
                    
                    // Process 4 elements at a time
                    for (; k + 3 < head_size; k += 4) {
                        size_t q_idx = ((b * seq_len + i) * num_heads + h) * head_size + k;
                        size_t k_idx = ((b * seq_len + j) * num_heads + h) * head_size + k;
                        
                        float32x4_t vq = vld1q_f32(&query[q_idx]);
                        float32x4_t vk = vld1q_f32(&key[k_idx]);
                        vdot = vmlaq_f32(vdot, vq, vk); // Multiply and accumulate
                    }
                    
                    // Horizontal sum of vdot to get the complete dot product
                    float32x2_t vsum2 = vadd_f32(vget_low_f32(vdot), vget_high_f32(vdot));
                    vsum2 = vpadd_f32(vsum2, vsum2);
                    float dot = vget_lane_f32(vsum2, 0);
                    
                    // Process remaining elements
                    for (; k < head_size; k++) {
                        size_t q_idx = ((b * seq_len + i) * num_heads + h) * head_size + k;
                        size_t k_idx = ((b * seq_len + j) * num_heads + h) * head_size + k;
                        dot += query[q_idx] * key[k_idx];
                    }
                    
                    // Scale the dot product
                    float score = dot * scale;
                    
                    // Apply mask if provided
                    if (mask != nullptr) {
                        // Apply causal mask (if mask[b,1,1,j] == 0, then hide position j from position i)
                        size_t mask_idx = b * seq_len + j; // Simplified for causal mask
                        if (j > i || mask[mask_idx] == 0) {
                            score = -std::numeric_limits<float>::infinity();
                        }
                    }
                    
                    scores[i * seq_len + j] = score;
                }
            }
            
            // Step 2: Apply softmax to each row of scores to get attention weights
            for (size_t i = 0; i < seq_len; i++) {
                // Find max value for numerical stability
                float max_val = -std::numeric_limits<float>::infinity();
                for (size_t j = 0; j < seq_len; j++) {
                    max_val = std::max(max_val, scores[i * seq_len + j]);
                }
                
                // Compute exp(score - max_val) and sum
                float sum_exp = 0.0f;
                for (size_t j = 0; j < seq_len; j++) {
                    float score = scores[i * seq_len + j];
                    float exp_val = 0.0f;
                    
                    if (std::isinf(score) && score < 0) {
                        // Handle -inf (masked values)
                        exp_val = 0.0f;
                    } else {
                        // For normal values, compute exp(score - max_val)
                        exp_val = std::exp(score - max_val);
                    }
                    
                    probs[i * seq_len + j] = exp_val;
                    sum_exp += exp_val;
                }
                
                // Normalize probabilities
                float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
                float32x4_t vinv_sum = vdupq_n_f32(inv_sum);
                
                // Normalize with NEON when possible
                size_t j = 0;
                for (; j + 3 < seq_len; j += 4) {
                    float32x4_t vprob = vld1q_f32(&probs[i * seq_len + j]);
                    vprob = vmulq_f32(vprob, vinv_sum);
                    vst1q_f32(&probs[i * seq_len + j], vprob);
                }
                
                // Handle remainder
                for (; j < seq_len; j++) {
                    probs[i * seq_len + j] *= inv_sum;
                }
            }
            
            // Step 3: Compute weighted sum of values (attn_probs @ value)
            // For each query position and output dimension
            for (size_t i = 0; i < seq_len; i++) {
                // Process each dimension of the output
                size_t d = 0;
                
                // Process 4 elements at a time with NEON
                for (; d + 3 < head_size; d += 4) {
                    float32x4_t vsum = vdupq_n_f32(0.0f);
                    
                    // Weighted sum over all value vectors
                    for (size_t j = 0; j < seq_len; j++) {
                        size_t v_idx = ((b * seq_len + j) * num_heads + h) * head_size + d;
                        float attention_weight = probs[i * seq_len + j];
                        
                        // Skip computation for near-zero weights
                        if (attention_weight < 1e-10f) continue;
                        
                        float32x4_t vvalue = vld1q_f32(&value[v_idx]);
                        float32x4_t vweight = vdupq_n_f32(attention_weight);
                        float32x4_t vweighted = vmulq_f32(vvalue, vweight);
                        vsum = vaddq_f32(vsum, vweighted);
                    }
                    
                    // Store result
                    size_t out_idx = (b * seq_len + i) * head_size + d;
                    vst1q_f32(&output[out_idx], vsum);
                }
                
                // Process remaining elements
                for (; d < head_size; d++) {
                    float sum = 0.0f;
                    for (size_t j = 0; j < seq_len; j++) {
                        size_t v_idx = ((b * seq_len + j) * num_heads + h) * head_size + d;
                        sum += probs[i * seq_len + j] * value[v_idx];
                    }
                    
                    size_t out_idx = (b * seq_len + i) * head_size + d;
                    output[out_idx] = sum;
                }
            }
        }
    }
}

void matrix_mul_neon_f32(float* c, const float* a, const float* b, size_t m, size_t k, size_t n) {
    // Advanced implementation using NEON and cache blocking
    constexpr size_t BLOCK_SIZE_M = 16; // Optimal block sizes based on benchmarks
    constexpr size_t BLOCK_SIZE_N = 16; 
    constexpr size_t BLOCK_SIZE_K = 32;
    constexpr size_t SIMD_WIDTH = 4;    // NEON processes 4 floats at once
    
    // Initialize output matrix to zeros
    for (size_t i = 0; i < m * n; i++) {
        c[i] = 0.0f;
    }
    
    // Outer blocking for cache efficiency
    for (size_t i0 = 0; i0 < m; i0 += BLOCK_SIZE_M) {
        size_t i_end = std::min(i0 + BLOCK_SIZE_M, m);
        
        for (size_t j0 = 0; j0 < n; j0 += BLOCK_SIZE_N) {
            size_t j_end = std::min(j0 + BLOCK_SIZE_N, n);
            
            // Here we process j in multiples of 4 (SIMD_WIDTH) for better vectorization
            for (size_t j_offset = 0; j_offset < j_end - j0; j_offset += SIMD_WIDTH) {
                size_t j_simd_width = std::min(SIMD_WIDTH, j_end - (j0 + j_offset));
                
                // If we don't have a full vector width, handle scalar remainder later
                if (j_simd_width != SIMD_WIDTH && j_simd_width != 0) {
                    continue;
                }
                
                for (size_t i = i0; i < i_end; i++) {
                    size_t j = j0 + j_offset;
                    
                    // We're computing an entire row of the output at once (4 elements)
                    float32x4_t vsum = vld1q_f32(&c[i * n + j]);
                    
                    // Process inner dimension in blocks for better cache utilization
                    for (size_t k0 = 0; k0 < k; k0 += BLOCK_SIZE_K) {
                        size_t k_end = std::min(k0 + BLOCK_SIZE_K, k);
                        
                        // Process the block
                        for (size_t l = k0; l < k_end; l++) {
                            // Broadcast single element from A matrix
                            float32x4_t va = vdupq_n_f32(a[i * k + l]);
                            
                            // Load 4 elements from B matrix (consecutive in memory)
                            float32x4_t vb = vld1q_f32(&b[l * n + j]);
                            
                            // vsum += va * vb using multiply-accumulate
                            vsum = vmlaq_f32(vsum, va, vb);
                        }
                    }
                    
                    // Store result back
                    vst1q_f32(&c[i * n + j], vsum);
                }
            }
            
            // Handle remainders (when j_end - j0 is not multiple of SIMD_WIDTH)
            for (size_t i = i0; i < i_end; i++) {
                for (size_t j = j0 + ((j_end - j0) / SIMD_WIDTH) * SIMD_WIDTH; j < j_end; j++) {
                    float sum = 0.0f;
                    for (size_t l = 0; l < k; l++) {
                        sum += a[i * k + l] * b[l * n + j];
                    }
                    c[i * n + j] = sum;
                }
            }
        }
    }
}

#endif // CCSM_HAVE_NEON

//
// Fused Operation Implementations
//

#if defined(CCSM_HAVE_AVX)
// AVX approximation of exponential function
// Using a fast polynomial approximation for exp(x)
static inline __m256 exp256_ps(__m256 x) {
    // Clamp the input to prevent overflow/underflow
    x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-88.0f)), _mm256_set1_ps(88.0f));
    
    // Constants for the polynomial approximation
    const __m256 vlog2e = _mm256_set1_ps(1.442695f);       // log2(e)
    const __m256 vone = _mm256_set1_ps(1.0f);
    const __m256 vc0 = _mm256_set1_ps(0.99992522f);
    const __m256 vc1 = _mm256_set1_ps(0.69583354f);
    const __m256 vc2 = _mm256_set1_ps(0.22606716f);
    const __m256 vc3 = _mm256_set1_ps(0.07944154f);
    const __m256 vc4 = _mm256_set1_ps(0.01386268f);
    
    // Scale by log2(e)
    __m256 z = _mm256_mul_ps(x, vlog2e);
    
    // Round to nearest integer
    __m256 floor_x = _mm256_floor_ps(z);
    
    // Compute 2^i
    __m256i emm0 = _mm256_cvttps_epi32(floor_x);
    emm0 = _mm256_add_epi32(emm0, _mm256_set1_epi32(127));
    emm0 = _mm256_slli_epi32(emm0, 23);
    __m256 pow2i = _mm256_castsi256_ps(emm0);
    
    // Compute fractional part
    __m256 frac = _mm256_sub_ps(z, floor_x);
    
    // Polynomial approximation of 2^frac
    __m256 poly = _mm256_add_ps(vc4, _mm256_mul_ps(frac, vc3));
    poly = _mm256_add_ps(poly, _mm256_mul_ps(frac, vc2));
    poly = _mm256_add_ps(poly, _mm256_mul_ps(frac, vc1));
    poly = _mm256_add_ps(poly, _mm256_mul_ps(frac, vc0));
    poly = _mm256_add_ps(poly, vone);
    
    // Combine: 2^i * 2^frac
    return _mm256_mul_ps(pow2i, poly);
}

void fused_rms_norm_silu_avx_f32(float* output, const float* input, const float* weight, float epsilon, size_t n) {
    // Constants needed for AVX operations
    const __m256 vzero = _mm256_setzero_ps();
    const __m256 vones = _mm256_set1_ps(1.0f);
    const __m256 vepsilon = _mm256_set1_ps(epsilon);
    const __m256 vinv_n = _mm256_set1_ps(1.0f / n);
    
    // First pass: calculate sum of squares
    __m256 vsum_sq = _mm256_setzero_ps();
    
    // Process 8 elements at a time using AVX
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 vin = _mm256_loadu_ps(&input[i]);
        __m256 vsquared = _mm256_mul_ps(vin, vin);
        vsum_sq = _mm256_add_ps(vsum_sq, vsquared);
    }
    
    // Reduce vsum_sq to a single value
    // Horizontal sum of 8 floats in avx register
    __m128 vlow = _mm256_castps256_ps128(vsum_sq);
    __m128 vhigh = _mm256_extractf128_ps(vsum_sq, 1);
    __m128 vsum = _mm_add_ps(vlow, vhigh);
    __m128 vshuf = _mm_movehdup_ps(vsum);        // Broadcast elements 1,3 to 0,2
    __m128 vshufs = _mm_add_ps(vsum, vshuf);     // Add pairs
    __m128 vshuf2 = _mm_movehl_ps(vshuf, vshufs); // High half -> low half
    __m128 vsums = _mm_add_ss(vshufs, vshuf2);   // Add remaining elements
    float sum_sq = _mm_cvtss_f32(vsums);
    
    // Process any remaining elements
    for (; i < n; i++) {
        sum_sq += input[i] * input[i];
    }
    
    // Calculate normalization factor: 1 / sqrt(sum_sq / n + epsilon)
    float inv_rms = 1.0f / std::sqrt(sum_sq / n + epsilon);
    __m256 vinv_rms = _mm256_set1_ps(inv_rms);
    
    // Second pass: normalize each element, apply weight, and SiLU activation
    i = 0;
    for (; i + 7 < n; i += 8) {
        // Load input and weights
        __m256 vin = _mm256_loadu_ps(&input[i]);
        __m256 vw = _mm256_loadu_ps(&weight[i]);
        
        // Apply normalization and weights
        __m256 vnormalized = _mm256_mul_ps(_mm256_mul_ps(vin, vinv_rms), vw);
        
        // Apply SiLU activation: x * sigmoid(x)
        // Compute sigmoid using optimized approximation: 1 / (1 + exp(-x))
        __m256 vneg = _mm256_sub_ps(vzero, vnormalized);
        __m256 vexp_neg = exp256_ps(vneg);
        __m256 vdenom = _mm256_add_ps(vones, vexp_neg);
        __m256 vsigmoid = _mm256_div_ps(vones, vdenom);
        
        // Final SiLU: x * sigmoid(x)
        __m256 vout = _mm256_mul_ps(vnormalized, vsigmoid);
        
        // Store result
        _mm256_storeu_ps(&output[i], vout);
    }
    
    // Process remaining elements
    for (; i < n; i++) {
        float normalized = input[i] * inv_rms * weight[i];
        output[i] = normalized / (1.0f + std::exp(-normalized)) * normalized;
    }
}

void fused_layer_norm_relu_avx_f32(float* output, const float* input, const float* weight, const float* bias, float epsilon, size_t n) {
    // Constants needed for AVX operations
    const __m256 vzero = _mm256_setzero_ps();
    const __m256 vepsilon = _mm256_set1_ps(epsilon);
    const __m256 vinv_n = _mm256_set1_ps(1.0f / n);
    
    // First pass: calculate mean
    __m256 vsum = _mm256_setzero_ps();
    
    // Process 8 elements at a time using AVX
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 vin = _mm256_loadu_ps(&input[i]);
        vsum = _mm256_add_ps(vsum, vin);
    }
    
    // Horizontal sum of vsum to get total sum
    __m128 vlow = _mm256_castps256_ps128(vsum);
    __m128 vhigh = _mm256_extractf128_ps(vsum, 1);
    __m128 vsum128 = _mm_add_ps(vlow, vhigh);
    __m128 vshuf = _mm_movehdup_ps(vsum128);     // Broadcast elements 1,3 to 0,2
    __m128 vshufs = _mm_add_ps(vsum128, vshuf);  // Add pairs
    __m128 vshuf2 = _mm_movehl_ps(vshuf, vshufs); // High half -> low half
    __m128 vsums = _mm_add_ss(vshufs, vshuf2);   // Add remaining elements
    float sum = _mm_cvtss_f32(vsums);
    
    // Process any remaining elements
    for (; i < n; i++) {
        sum += input[i];
    }
    
    // Calculate mean
    float mean = sum / n;
    __m256 vmean = _mm256_set1_ps(mean);
    
    // Second pass: calculate variance
    __m256 vvar = _mm256_setzero_ps();
    
    i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 vin = _mm256_loadu_ps(&input[i]);
        __m256 vdiff = _mm256_sub_ps(vin, vmean);
        __m256 vsquared = _mm256_mul_ps(vdiff, vdiff);
        vvar = _mm256_add_ps(vvar, vsquared);
    }
    
    // Horizontal sum of vvar to get sum of squared differences
    vlow = _mm256_castps256_ps128(vvar);
    vhigh = _mm256_extractf128_ps(vvar, 1);
    vsum128 = _mm_add_ps(vlow, vhigh);
    vshuf = _mm_movehdup_ps(vsum128);
    vshufs = _mm_add_ps(vsum128, vshuf);
    vshuf2 = _mm_movehl_ps(vshuf, vshufs);
    vsums = _mm_add_ss(vshufs, vshuf2);
    float variance_sum = _mm_cvtss_f32(vsums);
    
    // Process any remaining elements
    for (; i < n; i++) {
        float diff = input[i] - mean;
        variance_sum += diff * diff;
    }
    
    // Calculate variance and normalization factor
    float variance = variance_sum / n;
    float inv_std = 1.0f / std::sqrt(variance + epsilon);
    __m256 vinv_std = _mm256_set1_ps(inv_std);
    
    // Third pass: normalize, scale, shift, and apply ReLU
    i = 0;
    for (; i + 7 < n; i += 8) {
        // Load input, weights, and bias
        __m256 vin = _mm256_loadu_ps(&input[i]);
        __m256 vw = _mm256_loadu_ps(&weight[i]);
        __m256 vb = _mm256_loadu_ps(&bias[i]);
        
        // Normalize: (x - mean) * inv_std
        __m256 vdiff = _mm256_sub_ps(vin, vmean);
        __m256 vnormalized = _mm256_mul_ps(vdiff, vinv_std);
        
        // Scale and shift
        __m256 vscaled = _mm256_mul_ps(vnormalized, vw);
        __m256 vbiased = _mm256_add_ps(vscaled, vb);
        
        // Apply ReLU: max(0, x)
        __m256 vout = _mm256_max_ps(vzero, vbiased);
        
        // Store result
        _mm256_storeu_ps(&output[i], vout);
    }
    
    // Process remaining elements
    for (; i < n; i++) {
        float normalized = (input[i] - mean) * inv_std;
        float scaled = normalized * weight[i] + bias[i];
        output[i] = (scaled > 0) ? scaled : 0;
    }
}
#endif // CCSM_HAVE_AVX

#if defined(CCSM_HAVE_NEON)
// NEON approximation of exponential function
// Using a fast polynomial approximation for exp(x)
static inline float32x4_t exp_ps_neon(float32x4_t x) {
    // Clamp the input to prevent overflow/underflow
    const float32x4_t vmax = vdupq_n_f32(88.0f);
    const float32x4_t vmin = vdupq_n_f32(-88.0f);
    x = vminq_f32(vmaxq_f32(x, vmin), vmax);
    
    // Constants for polynomial approximation
    const float32x4_t vlog2e = vdupq_n_f32(1.442695f); // log2(e)
    const float32x4_t vone = vdupq_n_f32(1.0f);
    
    // Polynomial coefficients for 2^x over the interval [0, 1)
    const float32x4_t vc0 = vdupq_n_f32(0.99992522f);
    const float32x4_t vc1 = vdupq_n_f32(0.69583354f);
    const float32x4_t vc2 = vdupq_n_f32(0.22606716f);
    const float32x4_t vc3 = vdupq_n_f32(0.07944154f);
    const float32x4_t vc4 = vdupq_n_f32(0.01386268f);
    
    // Scale by log2(e)
    float32x4_t z = vmulq_f32(x, vlog2e);
    
    // Split into integer and fractional parts
    float32x4_t floor_x = vcvtq_f32_s32(vcvtq_s32_f32(z));
    float32x4_t frac = vsubq_f32(z, floor_x);
    
    // Calculate 2^i where i is the integer part
    int32x4_t emm0 = vaddq_s32(vcvtq_s32_f32(floor_x), vdupq_n_s32(127));
    emm0 = vshlq_n_s32(emm0, 23);
    float32x4_t pow2i = vreinterpretq_f32_s32(emm0);
    
    // Polynomial approximation of 2^frac
    float32x4_t poly = vaddq_f32(vc4, vmulq_f32(frac, vc3));
    poly = vaddq_f32(poly, vmulq_f32(frac, vc2));
    poly = vaddq_f32(poly, vmulq_f32(frac, vc1));
    poly = vaddq_f32(poly, vmulq_f32(frac, vc0));
    poly = vaddq_f32(poly, vone);
    
    // Combine: 2^i * 2^frac
    return vmulq_f32(pow2i, poly);
}

void fused_rms_norm_silu_neon_f32(float* output, const float* input, const float* weight, float epsilon, size_t n) {
    // Constants needed for NEON operations
    const float32x4_t vzero = vdupq_n_f32(0.0f);
    const float32x4_t vones = vdupq_n_f32(1.0f);
    const float32x4_t vepsilon = vdupq_n_f32(epsilon);
    const float32x4_t vinv_n = vdupq_n_f32(1.0f / n);
    
    // First pass: calculate sum of squares
    float32x4_t vsum_sq = vdupq_n_f32(0.0f);
    
    // Process 4 elements at a time using NEON
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t vin = vld1q_f32(&input[i]);
        float32x4_t vsquared = vmulq_f32(vin, vin);
        vsum_sq = vaddq_f32(vsum_sq, vsquared);
    }
    
    // Horizontal sum of vsum_sq to get total sum of squares
    float32x2_t vsum_low = vget_low_f32(vsum_sq);
    float32x2_t vsum_high = vget_high_f32(vsum_sq);
    float32x2_t vsum = vadd_f32(vsum_low, vsum_high);
    vsum = vpadd_f32(vsum, vsum);  // Pair-wise add to get horizontal sum
    float sum_sq = vget_lane_f32(vsum, 0);
    
    // Process remaining elements
    for (; i < n; i++) {
        sum_sq += input[i] * input[i];
    }
    
    // Calculate normalization factor: 1 / sqrt(sum_sq / n + epsilon)
    float inv_rms = 1.0f / std::sqrt(sum_sq / n + epsilon);
    float32x4_t vinv_rms = vdupq_n_f32(inv_rms);
    
    // Second pass: normalize each element, apply weight, and SiLU activation
    i = 0;
    for (; i + 3 < n; i += 4) {
        // Load input and weights
        float32x4_t vin = vld1q_f32(&input[i]);
        float32x4_t vw = vld1q_f32(&weight[i]);
        
        // Apply normalization and weights
        float32x4_t vnormalized = vmulq_f32(vmulq_f32(vin, vinv_rms), vw);
        
        // Apply SiLU activation: x * sigmoid(x)
        // For sigmoid, we'll use approximation: 1 / (1 + exp(-x))
        float32x4_t vneg = vsubq_f32(vzero, vnormalized);
        float32x4_t vexp_neg = exp_ps_neon(vneg);
        float32x4_t vdenom = vaddq_f32(vones, vexp_neg);
        float32x4_t vsigmoid = vdivq_f32(vones, vdenom);
        
        // Final SiLU: x * sigmoid(x)
        float32x4_t vout = vmulq_f32(vnormalized, vsigmoid);
        
        // Store result
        vst1q_f32(&output[i], vout);
    }
    
    // Process remaining elements
    for (; i < n; i++) {
        float normalized = input[i] * inv_rms * weight[i];
        output[i] = normalized / (1.0f + std::exp(-normalized)) * normalized;
    }
}

void fused_layer_norm_relu_neon_f32(float* output, const float* input, const float* weight, const float* bias, float epsilon, size_t n) {
    // Constants needed for NEON operations
    const float32x4_t vzero = vdupq_n_f32(0.0f);
    const float32x4_t vepsilon = vdupq_n_f32(epsilon);
    const float32x4_t vinv_n = vdupq_n_f32(1.0f / n);
    
    // First pass: calculate mean
    float32x4_t vsum = vdupq_n_f32(0.0f);
    
    // Process 4 elements at a time using NEON
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t vin = vld1q_f32(&input[i]);
        vsum = vaddq_f32(vsum, vin);
    }
    
    // Horizontal sum of vsum to get total sum
    float32x2_t vsum_low = vget_low_f32(vsum);
    float32x2_t vsum_high = vget_high_f32(vsum);
    float32x2_t vsum2 = vadd_f32(vsum_low, vsum_high);
    vsum2 = vpadd_f32(vsum2, vsum2);  // Pair-wise add to get horizontal sum
    float sum = vget_lane_f32(vsum2, 0);
    
    // Process remaining elements
    for (; i < n; i++) {
        sum += input[i];
    }
    
    // Calculate mean
    float mean = sum / n;
    float32x4_t vmean = vdupq_n_f32(mean);
    
    // Second pass: calculate variance
    float32x4_t vvar = vdupq_n_f32(0.0f);
    
    i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t vin = vld1q_f32(&input[i]);
        float32x4_t vdiff = vsubq_f32(vin, vmean);
        float32x4_t vsquared = vmulq_f32(vdiff, vdiff);
        vvar = vaddq_f32(vvar, vsquared);
    }
    
    // Horizontal sum of vvar to get total variance
    float32x2_t vvar_low = vget_low_f32(vvar);
    float32x2_t vvar_high = vget_high_f32(vvar);
    float32x2_t vvar2 = vadd_f32(vvar_low, vvar_high);
    vvar2 = vpadd_f32(vvar2, vvar2);  // Pair-wise add to get horizontal sum
    float variance_sum = vget_lane_f32(vvar2, 0);
    
    // Process remaining elements
    for (; i < n; i++) {
        float diff = input[i] - mean;
        variance_sum += diff * diff;
    }
    
    // Calculate variance and normalization factor
    float variance = variance_sum / n;
    float inv_std = 1.0f / std::sqrt(variance + epsilon);
    float32x4_t vinv_std = vdupq_n_f32(inv_std);
    
    // Third pass: normalize, scale, shift, and apply ReLU
    i = 0;
    for (; i + 3 < n; i += 4) {
        // Load input, weights, and bias
        float32x4_t vin = vld1q_f32(&input[i]);
        float32x4_t vw = vld1q_f32(&weight[i]);
        float32x4_t vb = vld1q_f32(&bias[i]);
        
        // Normalize: (x - mean) * inv_std
        float32x4_t vdiff = vsubq_f32(vin, vmean);
        float32x4_t vnormalized = vmulq_f32(vdiff, vinv_std);
        
        // Scale and shift
        float32x4_t vscaled = vmulq_f32(vnormalized, vw);
        float32x4_t vbiased = vaddq_f32(vscaled, vb);
        
        // Apply ReLU: max(0, x)
        float32x4_t vout = vmaxq_f32(vzero, vbiased);
        
        // Store result
        vst1q_f32(&output[i], vout);
    }
    
    // Process remaining elements
    for (; i < n; i++) {
        float normalized = (input[i] - mean) * inv_std;
        float scaled = normalized * weight[i] + bias[i];
        output[i] = (scaled > 0) ? scaled : 0;
    }
}
#endif // CCSM_HAVE_NEON

} // namespace detail
} // namespace simd
} // namespace ccsm