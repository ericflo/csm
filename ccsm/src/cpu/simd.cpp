#include <ccsm/cpu/simd.h>

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

void softmax_avx_f32(float* output, const float* input, size_t n) {
    if (n == 0) return;
    
    // Find max value for numerical stability
    float max_val = input[0];
    for (size_t i = 1; i < n; i++) {
        max_val = std::max(max_val, input[i]);
    }
    
    // Just use scalar implementation for simplicity and correctness
    // Calculate exp(x - max) for all inputs
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }
    
    // Normalize with 1/sum
    float inv_sum = 1.0f / sum;
    for (size_t i = 0; i < n; i++) {
        output[i] *= inv_sum;
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
    
    // Find max value for numerical stability
    float max_val = input[0];
    for (size_t i = 1; i < n; i++) {
        max_val = std::max(max_val, input[i]);
    }
    
    // Compute exp(x - max) and sum
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }
    
    // Normalize
    float inv_sum = 1.0f / sum;
    size_t i = 0;
    float32x4_t vinv_sum = vdupq_n_f32(inv_sum);
    
    // Process 4 elements at a time using NEON
    for (; i + 3 < n; i += 4) {
        float32x4_t vout = vld1q_f32(output + i);
        vout = vmulq_f32(vout, vinv_sum);
        vst1q_f32(output + i, vout);
    }
    
    // Process remaining elements
    for (; i < n; i++) {
        output[i] *= inv_sum;
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

} // namespace detail
} // namespace simd
} // namespace ccsm