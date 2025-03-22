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

void silu_avx_f32(float* output, const float* input, size_t n) {
    size_t i = 0;
    __m256 vone = _mm256_set1_ps(1.0f);
    
    // Process 8 elements at a time using AVX
    for (; i + 7 < n; i += 8) {
        // Load input values
        __m256 vin = _mm256_loadu_ps(input + i);
        
        // Calculate -x for exp(-x)
        __m256 vneg = _mm256_mul_ps(vin, _mm256_set1_ps(-1.0f));
        
        // Calculate sigmoid(x) = 1 / (1 + exp(-x))
        // For each element, we need to compute exp(-x)
        __m256 vexp_neg = vneg;
        
        // Approximate exp using a simplified approach
        // This approximation works reasonably well for small values of x
        // For better accuracy in a real implementation, we would use a higher-order polynomial
        
        // exp(x) ≈ 1 + x + x²/2 for small x
        __m256 vsquared = _mm256_mul_ps(vneg, vneg);
        __m256 vhalf_squared = _mm256_mul_ps(vsquared, _mm256_set1_ps(0.5f));
        
        vexp_neg = _mm256_add_ps(vone, vneg);
        vexp_neg = _mm256_add_ps(vexp_neg, vhalf_squared);
        
        // Calculate 1 / (1 + exp(-x))
        __m256 vdenom = _mm256_add_ps(vone, vexp_neg);
        __m256 vsigmoid = _mm256_div_ps(vone, vdenom);
        
        // Calculate x * sigmoid(x)
        __m256 vout = _mm256_mul_ps(vin, vsigmoid);
        _mm256_storeu_ps(output + i, vout);
    }
    
    // Process remaining elements using the scalar implementation
    for (; i < n; i++) {
        output[i] = input[i] / (1.0f + std::exp(-input[i]));
    }
}

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
    
    // Process 8 elements at a time using AVX
    for (; i + 7 < n; i += 8) {
        __m256 vin = _mm256_loadu_ps(input + i);
        
        // Compute SiLU: x * sigmoid(x)
        // Approximation of sigmoid using AVX
        __m256 vneg = _mm256_mul_ps(vin, _mm256_set1_ps(-1.0f));
        
        // Approximation of exp(-x) using polynomial expansion
        // This is a simplified approximation for illustration; a more accurate
        // implementation would use a better approximation or lookup table
        __m256 vexp = _mm256_set1_ps(1.0f);
        __m256 vtmp = _mm256_set1_ps(1.0f);
        __m256 vfac = _mm256_set1_ps(1.0f);
        
        for (int j = 1; j <= 4; j++) {
            vfac = _mm256_mul_ps(vfac, _mm256_set1_ps(1.0f / j));
            vtmp = _mm256_mul_ps(vtmp, vneg);
            __m256 vterm = _mm256_mul_ps(vtmp, vfac);
            vexp = _mm256_add_ps(vexp, vterm);
        }
        
        // Compute sigmoid: 1 / (1 + exp(-x))
        __m256 vsigmoid = _mm256_div_ps(vone, _mm256_add_ps(vone, vexp));
        
        // Compute SiLU: x * sigmoid(x)
        __m256 vout = _mm256_mul_ps(vin, vsigmoid);
        _mm256_storeu_ps(output + i, vout);
    }
    
    // Process remaining elements
    for (; i < n; i++) {
        output[i] = input[i] / (1.0f + std::exp(-input[i]));
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

void matrix_mul_avx_f32(float* result, const float* a, const float* b, size_t m, size_t k, size_t n) {
    constexpr size_t BLOCK_SIZE_M = 32; // Adjust based on L1 cache size
    constexpr size_t BLOCK_SIZE_N = 32; 
    constexpr size_t BLOCK_SIZE_K = 32;
    
    // Initialize result matrix to zeros
    for (size_t i = 0; i < m * n; i++) {
        result[i] = 0.0f;
    }
    
    // Cache-blocking matrix multiplication
    for (size_t i0 = 0; i0 < m; i0 += BLOCK_SIZE_M) {
        size_t i_end = std::min(i0 + BLOCK_SIZE_M, m);
        
        for (size_t j0 = 0; j0 < n; j0 += BLOCK_SIZE_N) {
            size_t j_end = std::min(j0 + BLOCK_SIZE_N, n);
            
            for (size_t k0 = 0; k0 < k; k0 += BLOCK_SIZE_K) {
                size_t k_end = std::min(k0 + BLOCK_SIZE_K, k);
                
                // Process blocks using AVX
                for (size_t i = i0; i < i_end; i++) {
                    for (size_t j = j0; j < j_end; j += 8) {
                        if (j + 8 <= j_end) {
                            // Load 8 elements from result (C matrix)
                            __m256 vc = _mm256_loadu_ps(&result[i * n + j]);
                            
                            // Process inner dimension in chunks
                            for (size_t l = k0; l < k_end; l++) {
                                // Broadcast single element from A matrix
                                __m256 va = _mm256_set1_ps(a[i * k + l]);
                                
                                // Load 8 elements from B matrix
                                __m256 vb = _mm256_loadu_ps(&b[l * n + j]);
                                
                                // Multiply and accumulate: vc += va * vb
                                vc = _mm256_add_ps(vc, _mm256_mul_ps(va, vb));
                            }
                            
                            // Store result back to C matrix
                            _mm256_storeu_ps(&result[i * n + j], vc);
                        } else {
                            // Handle remainder (< 8 columns)
                            for (size_t j1 = j; j1 < j_end; j1++) {
                                float sum = result[i * n + j1];
                                for (size_t l = k0; l < k_end; l++) {
                                    sum += a[i * k + l] * b[l * n + j1];
                                }
                                result[i * n + j1] = sum;
                            }
                        }
                    }
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

void matrix_mul_avx2_f32(const float* a, const float* b, float* c, size_t m, size_t k, size_t n) {
    // This is a simple implementation for illustration
    // For production, consider using a highly optimized library like OpenBLAS
    
    // Initialize output matrix to zeros
    for (size_t i = 0; i < m * n; i++) {
        c[i] = 0.0f;
    }
    
    // For each row of A
    for (size_t i = 0; i < m; i++) {
        // For each column of B
        for (size_t j = 0; j < n; j++) {
            // For each element in row A and column B in chunks of 8
            __m256 vsum = _mm256_setzero_ps();
            
            for (size_t l = 0; l < k; l += 8) {
                // Handle bounds check
                if (l + 8 <= k) {
                    // Load 8 elements from A row i
                    __m256 va = _mm256_loadu_ps(&a[i * k + l]);
                    
                    // Load 8 elements from B column j
                    float bCol[8];
                    for (int z = 0; z < 8; z++) {
                        bCol[z] = b[(l + z) * n + j];
                    }
                    __m256 vb = _mm256_loadu_ps(bCol);
                    
                    // vsum += va * vb (using FMA)
                    vsum = _mm256_fmadd_ps(va, vb, vsum);
                } else {
                    // Handle remaining elements (< 8)
                    for (size_t z = 0; z < k - l; z++) {
                        c[i * n + j] += a[i * k + l + z] * b[(l + z) * n + j];
                    }
                    break;
                }
            }
            
            // Horizontal sum of vsum and add to c[i, j]
            __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(vsum, 0), _mm256_extractf128_ps(vsum, 1));
            __m128 sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
            __m128 sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 0x55));
            c[i * n + j] += _mm_cvtss_f32(sum32);
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
    
    // Process remaining elements with scalar code
    // SiLU is complex to vectorize efficiently on NEON without a good exp approximation
    for (i = 0; i < n; i++) {
        output[i] = input[i] / (1.0f + std::exp(-input[i]));
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

void matrix_mul_neon_f32(float* result, const float* a, const float* b, size_t m, size_t k, size_t n) {
    // Initialize output matrix to zeros
    for (size_t i = 0; i < m * n; i++) {
        result[i] = 0.0f;
    }
    
    // For each row of A
    for (size_t i = 0; i < m; i++) {
        // For each column of B
        for (size_t j = 0; j < n; j++) {
            float32x4_t vsum = vdupq_n_f32(0.0f);
            size_t l = 0;
            
            // Process 4 elements at a time from row A and column B
            for (; l + 3 < k; l += 4) {
                float32x4_t va = vld1q_f32(&a[i * k + l]);
                
                // Load 4 elements from column j of B
                float bCol[4] = {
                    b[l * n + j],
                    b[(l + 1) * n + j],
                    b[(l + 2) * n + j],
                    b[(l + 3) * n + j]
                };
                float32x4_t vb = vld1q_f32(bCol);
                
                // vsum += va * vb
                vsum = vmlaq_f32(vsum, va, vb);
            }
            
            // Horizontal sum of vsum
            float32x2_t vsum2 = vadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
            vsum2 = vpadd_f32(vsum2, vsum2);
            result[i * n + j] += vget_lane_f32(vsum2, 0);
            
            // Process remaining elements
            for (; l < k; l++) {
                result[i * n + j] += a[i * k + l] * b[l * n + j];
            }
        }
    }
}

#endif // CCSM_HAVE_NEON

} // namespace detail
} // namespace simd
} // namespace ccsm