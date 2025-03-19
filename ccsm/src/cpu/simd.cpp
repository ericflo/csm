#include <ccsm/cpu/simd.h>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace ccsm {
namespace simd {

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

#if defined(CCSM_HAVE_AVX)

void vector_add_avx_f32(const float* a, const float* b, float* c, size_t n) {
    size_t i = 0;
    
    // Process 8 elements at a time using AVX
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(c + i, vc);
    }
    
    // Process remaining elements
    for (; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

void vector_mul_avx_f32(const float* a, const float* b, float* c, size_t n) {
    size_t i = 0;
    
    // Process 8 elements at a time using AVX
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(c + i, vc);
    }
    
    // Process remaining elements
    for (; i < n; i++) {
        c[i] = a[i] * b[i];
    }
}

void vector_scale_avx_f32(const float* a, float scalar, float* b, size_t n) {
    size_t i = 0;
    __m256 vs = _mm256_set1_ps(scalar);
    
    // Process 8 elements at a time using AVX
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_mul_ps(va, vs);
        _mm256_storeu_ps(b + i, vb);
    }
    
    // Process remaining elements
    for (; i < n; i++) {
        b[i] = a[i] * scalar;
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
    
    // Horizontal sum of AVX register
    __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum, 0), _mm256_extractf128_ps(sum, 1));
    __m128 sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    __m128 sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 0x55));
    float result = _mm_cvtss_f32(sum32);
    
    // Process remaining elements
    for (; i < n; i++) {
        result += a[i] * b[i];
    }
    
    return result;
}

void relu_avx_f32(const float* input, float* output, size_t n) {
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

void silu_avx_f32(const float* input, float* output, size_t n) {
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

void softmax_avx_f32(const float* input, float* output, size_t n) {
    if (n == 0) return;
    
    // Find max value for numerical stability
    float max_val = input[0];
    for (size_t i = 1; i < n; i++) {
        max_val = std::max(max_val, input[i]);
    }
    
    __m256 vmax = _mm256_set1_ps(max_val);
    __m256 vsum = _mm256_setzero_ps();
    
    size_t i = 0;
    
    // First pass: compute exp(x - max) for each element
    for (; i + 7 < n; i += 8) {
        __m256 vin = _mm256_loadu_ps(input + i);
        __m256 vcentered = _mm256_sub_ps(vin, vmax);
        
        // Approximate exp using polynomial or other method
        // This is a simplified approximation for illustration
        __m256 vexp = _mm256_set1_ps(1.0f);
        __m256 vtmp = _mm256_set1_ps(1.0f);
        __m256 vfac = _mm256_set1_ps(1.0f);
        
        for (int j = 1; j <= 6; j++) {
            vfac = _mm256_mul_ps(vfac, _mm256_set1_ps(1.0f / j));
            vtmp = _mm256_mul_ps(vtmp, vcentered);
            __m256 vterm = _mm256_mul_ps(vtmp, vfac);
            vexp = _mm256_add_ps(vexp, vterm);
        }
        
        _mm256_storeu_ps(output + i, vexp);
        vsum = _mm256_add_ps(vsum, vexp);
    }
    
    // Process any remaining elements
    float sum_scalar = 0.0f;
    for (; i < n; i++) {
        output[i] = std::exp(input[i] - max_val);
        sum_scalar += output[i];
    }
    
    // Horizontal sum of vsum
    __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(vsum, 0), _mm256_extractf128_ps(vsum, 1));
    __m128 sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    __m128 sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 0x55));
    float sum = _mm_cvtss_f32(sum32) + sum_scalar;
    
    // Second pass: normalize
    __m256 vinv_sum = _mm256_set1_ps(1.0f / sum);
    
    i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 vout = _mm256_loadu_ps(output + i);
        vout = _mm256_mul_ps(vout, vinv_sum);
        _mm256_storeu_ps(output + i, vout);
    }
    
    // Normalize remaining elements
    for (; i < n; i++) {
        output[i] /= sum;
    }
}

void matrix_mul_avx_f32(const float* a, const float* b, float* c, size_t m, size_t k, size_t n) {
    // Simple AVX implementation of matrix multiplication
    // For production use, consider using a highly optimized library like OpenBLAS
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            c[i * n + j] = 0.0f;
        }
        
        for (size_t l = 0; l < k; l += 8) {
            size_t kRemaining = std::min(size_t(8), k - l);
            if (kRemaining < 8) {
                // Handle edge case with scalar code
                for (size_t j = 0; j < n; j++) {
                    for (size_t lOff = 0; lOff < kRemaining; lOff++) {
                        c[i * n + j] += a[i * k + l + lOff] * b[(l + lOff) * n + j];
                    }
                }
            } else {
                // Process full AVX block
                for (size_t j = 0; j < n; j++) {
                    __m256 vsum = _mm256_setzero_ps();
                    
                    // Load 8 elements from row i of A starting at column l
                    __m256 va = _mm256_loadu_ps(&a[i * k + l]);
                    
                    // Load 8 elements from column j of B starting at row l
                    // Note: This assumes B is row-major; for column-major, different addressing would be used
                    float bCol[8];
                    for (int lOff = 0; lOff < 8; lOff++) {
                        bCol[lOff] = b[(l + lOff) * n + j];
                    }
                    __m256 vb = _mm256_loadu_ps(bCol);
                    
                    // Compute dot product and add to c[i, j]
                    vsum = _mm256_mul_ps(va, vb);
                    
                    // Horizontal sum
                    __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(vsum, 0), _mm256_extractf128_ps(vsum, 1));
                    __m128 sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
                    __m128 sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 0x55));
                    c[i * n + j] += _mm_cvtss_f32(sum32);
                }
            }
        }
    }
}

#endif // CCSM_HAVE_AVX

// -------------------------------------------------------------------------
// AVX2 implementations
// -------------------------------------------------------------------------

#if defined(CCSM_HAVE_AVX2)

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
    
    // Horizontal sum of AVX register
    __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum, 0), _mm256_extractf128_ps(sum, 1));
    __m128 sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    __m128 sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 0x55));
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

#if defined(CCSM_HAVE_NEON)

void vector_add_neon_f32(const float* a, const float* b, float* c, size_t n) {
    size_t i = 0;
    
    // Process 4 elements at a time using NEON
    for (; i + 3 < n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vc = vaddq_f32(va, vb);
        vst1q_f32(c + i, vc);
    }
    
    // Process remaining elements
    for (; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

void vector_mul_neon_f32(const float* a, const float* b, float* c, size_t n) {
    size_t i = 0;
    
    // Process 4 elements at a time using NEON
    for (; i + 3 < n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vc = vmulq_f32(va, vb);
        vst1q_f32(c + i, vc);
    }
    
    // Process remaining elements
    for (; i < n; i++) {
        c[i] = a[i] * b[i];
    }
}

void vector_fma_neon_f32(const float* a, const float* b, const float* c, float* d, size_t n) {
    size_t i = 0;
    
    // Process 4 elements at a time using NEON
    for (; i + 3 < n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vc = vld1q_f32(c + i);
        
        // d = a * b + c
        float32x4_t vd = vmlaq_f32(vc, va, vb);
        vst1q_f32(d + i, vd);
    }
    
    // Process remaining elements
    for (; i < n; i++) {
        d[i] = a[i] * b[i] + c[i];
    }
}

void vector_scale_neon_f32(const float* a, float scalar, float* b, size_t n) {
    size_t i = 0;
    float32x4_t vs = vdupq_n_f32(scalar);
    
    // Process 4 elements at a time using NEON
    for (; i + 3 < n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vmulq_f32(va, vs);
        vst1q_f32(b + i, vb);
    }
    
    // Process remaining elements
    for (; i < n; i++) {
        b[i] = a[i] * scalar;
    }
}

float vector_dot_neon_f32(const float* a, const float* b, size_t n) {
    size_t i = 0;
    float32x4_t vsum = vdupq_n_f32(0.0f);
    
    // Process 4 elements at a time using NEON
    for (; i + 3 < n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vsum = vmlaq_f32(vsum, va, vb);
    }
    
    // Horizontal sum of NEON register
    float32x2_t vsum2 = vadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
    vsum2 = vpadd_f32(vsum2, vsum2);
    float result = vget_lane_f32(vsum2, 0);
    
    // Process remaining elements
    for (; i < n; i++) {
        result += a[i] * b[i];
    }
    
    return result;
}

void relu_neon_f32(const float* input, float* output, size_t n) {
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

void silu_neon_f32(const float* input, float* output, size_t n) {
    size_t i = 0;
    
    // Process remaining elements with scalar code
    // SiLU is complex to vectorize efficiently on NEON without a good exp approximation
    for (i = 0; i < n; i++) {
        output[i] = input[i] / (1.0f + std::exp(-input[i]));
    }
}

void softmax_neon_f32(const float* input, float* output, size_t n) {
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

void matrix_mul_neon_f32(const float* a, const float* b, float* c, size_t m, size_t k, size_t n) {
    // Initialize output matrix to zeros
    for (size_t i = 0; i < m * n; i++) {
        c[i] = 0.0f;
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
            c[i * n + j] += vget_lane_f32(vsum2, 0);
            
            // Process remaining elements
            for (; l < k; l++) {
                c[i * n + j] += a[i * k + l] * b[l * n + j];
            }
        }
    }
}

#endif // CCSM_HAVE_NEON

} // namespace detail
} // namespace simd
} // namespace ccsm