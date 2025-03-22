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
        #elif defined(CCSM_ARCH_ARM64)
            // All ARM64 processors should have NEON
            features.neon = true;
        #endif
        
        initialized = true;
    }
    
    return features;
}

// Get highest available SIMD implementation
Implementation get_active_implementation() {
    const CPUFeatures& features = CPUFeatures::get();
    
    #if defined(CCSM_ARCH_X86_64)
        if (features.avx512f) return Implementation::AVX512;
        if (features.avx2) return Implementation::AVX2;
        if (features.avx) return Implementation::AVX;
        if (features.sse4_1) return Implementation::SSE41;
        if (features.sse2) return Implementation::SSE2;
    #elif defined(CCSM_ARCH_ARM64)
        if (features.neon) return Implementation::NEON;
    #endif
    
    return Implementation::SCALAR;
}

// Get CPU capabilities as string
std::string get_cpu_capabilities() {
    const CPUFeatures& features = CPUFeatures::get();
    std::string result = "CPU Features: ";
    
    if (features.avx512f) result += "AVX-512 ";
    if (features.avx2) result += "AVX2 ";
    if (features.avx) result += "AVX ";
    if (features.sse4_2) result += "SSE4.2 ";
    if (features.sse4_1) result += "SSE4.1 ";
    if (features.sse3) result += "SSE3 ";
    if (features.sse2) result += "SSE2 ";
    if (features.neon) result += "NEON ";
    
    if (result == "CPU Features: ") {
        result += "None";
    }
    
    return result;
}

namespace detail {

// Implementation of exp256_ps for AVX
#ifdef CCSM_HAVE_AVX
__m256 exp256_ps(__m256 x) {
    // Simple exponential approximation for AVX
    // Note: This is a very basic implementation and could be improved for accuracy
    
    // Constants for exponential approximation
    const __m256 c1 = _mm256_set1_ps(1.0f);
    const __m256 c2 = _mm256_set1_ps(1.0f);
    const __m256 c3 = _mm256_set1_ps(0.5f);
    const __m256 c4 = _mm256_set1_ps(0.1666666f);
    const __m256 c5 = _mm256_set1_ps(0.0416666f);
    
    // Clamp x to prevent overflow/underflow
    const __m256 max_x = _mm256_set1_ps(88.3762626647949f);
    const __m256 min_x = _mm256_set1_ps(-88.3762626647949f);
    x = _mm256_max_ps(_mm256_min_ps(x, max_x), min_x);
    
    // Calculate exp(x) using Taylor series: 1 + x + x^2/2 + x^3/6 + x^4/24
    __m256 result = c1;
    __m256 term = x;
    result = _mm256_add_ps(result, term);
    
    term = _mm256_mul_ps(term, x);
    term = _mm256_mul_ps(term, c3);
    result = _mm256_add_ps(result, term);
    
    term = _mm256_mul_ps(term, x);
    term = _mm256_mul_ps(term, c4);
    result = _mm256_add_ps(result, term);
    
    term = _mm256_mul_ps(term, x);
    term = _mm256_mul_ps(term, c5);
    result = _mm256_add_ps(result, term);
    
    return result;
}
#endif

// Fused matrix multiplication with Q4_0 quantized weights and ReLU activation
void fused_matmul_relu_q4_0_avx_f32(float* output, const float* a, const uint8_t* b, const float* b_scale, 
                                  size_t m, size_t k, size_t n) {
    // Get scale factor for dequantization
    const float inv_scale = *b_scale;
    
    // Initialize result matrix to zeros
    for (size_t i = 0; i < m * n; i++) {
        output[i] = 0.0f;
    }
    
#ifdef CCSM_HAVE_AVX
    // AVX-specific variables
    const __m256 vscale = _mm256_set1_ps(inv_scale);
    const __m256 vzero = _mm256_setzero_ps();
#endif
    
    // Process each row of A and corresponding output row
    for (size_t i = 0; i < m; i++) {
        // Process output elements in blocks of 8 (AVX register width)
        for (size_t j = 0; j < n; j += 8) {
            // Handle boundary condition
            size_t block_size = std::min(size_t(8), n - j);
            
#if !defined(CCSM_HAVE_AVX)
            // If AVX is not available, always use scalar code
            {
#else
            // With AVX, only use scalar for partial vectors
            if (block_size < 8) {
#endif
                // If we don't have a full vector or AVX is not available, 
                // process using scalar code
                for (size_t jj = j; jj < j + block_size; jj++) {
                    float sum = 0.0f;
                    for (size_t l = 0; l < k; l++) {
                        // Calculate index for accessing b
                        size_t b_idx = l * n + jj;
                        size_t byte_idx = b_idx / 2;
                        bool is_upper = (b_idx % 2) != 0;
                        
                        // Extract the appropriate 4-bit value
                        int8_t q4_val;
                        if (is_upper) {
                            q4_val = static_cast<int8_t>((b[byte_idx] >> 4) & 0xF);
                            // Sign extend if the high bit is set
                            if (q4_val & 0x8) {
                                q4_val |= 0xF0; // Extend to 8 bits
                            }
                        } else {
                            q4_val = static_cast<int8_t>(b[byte_idx] & 0xF);
                            // Sign extend if the high bit is set
                            if (q4_val & 0x8) {
                                q4_val |= 0xF0; // Extend to 8 bits
                            }
                        }
                        
                        // Dequantize and multiply
                        float b_val = static_cast<float>(q4_val) * inv_scale;
                        sum += a[i * k + l] * b_val;
                    }
                    // Apply ReLU activation: max(0, x)
                    output[i * n + jj] = (sum > 0.0f) ? sum : 0.0f;
                }
                continue;
#if !defined(CCSM_HAVE_AVX)
            }
#else
            }
            
            // With AVX, continue with vector processing
            // Use 8 accumulators for better instruction-level parallelism
            __m256 sum = _mm256_setzero_ps();
            
            // Compute dot product of row i of A with corresponding columns of B
            for (size_t l = 0; l < k; l++) {
                // For Q4_0, we need to process 2 values per byte
                // We'll unpack 4 bytes (8 values) at a time, which matches our AVX vector width
                
                // Calculate the starting byte index for this row/column combination
                size_t start_idx = l * n + j;
                size_t byte_start = start_idx / 2;
                
                // Load 4 bytes that contain 8 quantized values
                // For simplicity, we'll handle each 4-bit value individually
                uint32_t packed_bytes = 0;
                for (size_t byte_idx = 0; byte_idx < 4; byte_idx++) {
                    if (byte_start + byte_idx < (k*n+1)/2) { // Ensure we don't read past the end
                        packed_bytes |= (static_cast<uint32_t>(b[byte_start + byte_idx]) << (byte_idx * 8));
                    }
                }
                
                // Extract each 4-bit value, sign extend to 32 bits, and convert to float
                int32_t q4_vals[8];
                for (size_t v = 0; v < 8; v++) {
                    // Determine if we're extracting upper or lower 4 bits
                    bool is_upper = (v % 2) != 0;
                    uint8_t byte_val = (packed_bytes >> ((v / 2) * 8)) & 0xFF;
                    
                    // Extract the 4-bit value
                    int8_t q4_val;
                    if (is_upper) {
                        q4_val = static_cast<int8_t>((byte_val >> 4) & 0xF);
                    } else {
                        q4_val = static_cast<int8_t>(byte_val & 0xF);
                    }
                    
                    // Sign extend if the high bit is set
                    if (q4_val & 0x8) {
                        q4_val |= 0xF0; // Extend to 8 bits
                    }
                    
                    q4_vals[v] = static_cast<int32_t>(q4_val);
                }
                
                // Convert int32 array to __m256i
                __m256i i32_vals = _mm256_setr_epi32(
                    q4_vals[0], q4_vals[1], q4_vals[2], q4_vals[3],
                    q4_vals[4], q4_vals[5], q4_vals[6], q4_vals[7]
                );
                
                // Convert int32 to float32
                __m256 f32_vals = _mm256_cvtepi32_ps(i32_vals);
                
                // Dequantize by multiplying with scale
                __m256 dequantized = _mm256_mul_ps(f32_vals, vscale);
                
                // Broadcast A value to all lanes
                __m256 a_val = _mm256_set1_ps(a[i * k + l]);
                
                // Multiply and accumulate
                sum = _mm256_add_ps(sum, _mm256_mul_ps(a_val, dequantized));
            }
            
            // Apply ReLU activation: max(0, x)
            __m256 result = _mm256_max_ps(sum, vzero);
            
            // Store the result
            _mm256_storeu_ps(&output[i * n + j], result);
#endif // CCSM_HAVE_AVX
        }
    }
}

// Fused matrix multiplication with Q4_0 quantized weights and SiLU activation
void fused_matmul_silu_q4_0_avx_f32(float* output, const float* a, const uint8_t* b, const float* b_scale, 
                                   size_t m, size_t k, size_t n) {
    // Get scale factor for dequantization
    const float inv_scale = *b_scale;
    
    // Initialize result matrix to zeros
    for (size_t i = 0; i < m * n; i++) {
        output[i] = 0.0f;
    }
    
#ifdef CCSM_HAVE_AVX
    // Constants for SiLU calculation
    const __m256 vscale = _mm256_set1_ps(inv_scale);
    const __m256 vones = _mm256_set1_ps(1.0f);
    const __m256 vzero = _mm256_setzero_ps();
#endif
    
    // Process each row of A and corresponding output row
    for (size_t i = 0; i < m; i++) {
        // Process output elements in blocks of 8 (AVX register width)
        for (size_t j = 0; j < n; j += 8) {
            // Handle boundary condition
            size_t block_size = std::min(size_t(8), n - j);
            
#if !defined(CCSM_HAVE_AVX)
            // If AVX is not available, always use scalar code
            {
#else
            // With AVX, only use scalar for partial vectors
            if (block_size < 8) {
#endif
                // If we don't have a full vector or AVX is not available,
                // process using scalar code
                for (size_t jj = j; jj < j + block_size; jj++) {
                    float sum = 0.0f;
                    for (size_t l = 0; l < k; l++) {
                        // Calculate index for accessing b
                        size_t b_idx = l * n + jj;
                        size_t byte_idx = b_idx / 2;
                        bool is_upper = (b_idx % 2) != 0;
                        
                        // Extract the appropriate 4-bit value
                        int8_t q4_val;
                        if (is_upper) {
                            q4_val = static_cast<int8_t>((b[byte_idx] >> 4) & 0xF);
                            // Sign extend if the high bit is set
                            if (q4_val & 0x8) {
                                q4_val |= 0xF0; // Extend to 8 bits
                            }
                        } else {
                            q4_val = static_cast<int8_t>(b[byte_idx] & 0xF);
                            // Sign extend if the high bit is set
                            if (q4_val & 0x8) {
                                q4_val |= 0xF0; // Extend to 8 bits
                            }
                        }
                        
                        // Dequantize and multiply
                        float b_val = static_cast<float>(q4_val) * inv_scale;
                        sum += a[i * k + l] * b_val;
                    }
                    // Apply SiLU activation: x * sigmoid(x)
                    output[i * n + jj] = sum / (1.0f + std::exp(-sum));
                }
                continue;
#if !defined(CCSM_HAVE_AVX)
            }
#else
            }
            
            // With AVX, continue with vector processing
            // Use 8 accumulators for better instruction-level parallelism
            __m256 sum = _mm256_setzero_ps();
            
            // Compute dot product of row i of A with corresponding columns of B
            for (size_t l = 0; l < k; l++) {
                // For Q4_0, we need to process 2 values per byte
                // We'll unpack 4 bytes (8 values) at a time, which matches our AVX vector width
                
                // Calculate the starting byte index for this row/column combination
                size_t start_idx = l * n + j;
                size_t byte_start = start_idx / 2;
                
                // Load 4 bytes that contain 8 quantized values
                // For simplicity, we'll handle each 4-bit value individually
                uint32_t packed_bytes = 0;
                for (size_t byte_idx = 0; byte_idx < 4; byte_idx++) {
                    if (byte_start + byte_idx < (k*n+1)/2) { // Ensure we don't read past the end
                        packed_bytes |= (static_cast<uint32_t>(b[byte_start + byte_idx]) << (byte_idx * 8));
                    }
                }
                
                // Extract each 4-bit value, sign extend to 32 bits, and convert to float
                int32_t q4_vals[8];
                for (size_t v = 0; v < 8; v++) {
                    // Determine if we're extracting upper or lower 4 bits
                    bool is_upper = (v % 2) != 0;
                    uint8_t byte_val = (packed_bytes >> ((v / 2) * 8)) & 0xFF;
                    
                    // Extract the 4-bit value
                    int8_t q4_val;
                    if (is_upper) {
                        q4_val = static_cast<int8_t>((byte_val >> 4) & 0xF);
                    } else {
                        q4_val = static_cast<int8_t>(byte_val & 0xF);
                    }
                    
                    // Sign extend if the high bit is set
                    if (q4_val & 0x8) {
                        q4_val |= 0xF0; // Extend to 8 bits
                    }
                    
                    q4_vals[v] = static_cast<int32_t>(q4_val);
                }
                
                // Convert int32 array to __m256i
                __m256i i32_vals = _mm256_setr_epi32(
                    q4_vals[0], q4_vals[1], q4_vals[2], q4_vals[3],
                    q4_vals[4], q4_vals[5], q4_vals[6], q4_vals[7]
                );
                
                // Convert int32 to float32
                __m256 f32_vals = _mm256_cvtepi32_ps(i32_vals);
                
                // Dequantize by multiplying with scale
                __m256 dequantized = _mm256_mul_ps(f32_vals, vscale);
                
                // Broadcast A value to all lanes
                __m256 a_val = _mm256_set1_ps(a[i * k + l]);
                
                // Multiply and accumulate
                sum = _mm256_add_ps(sum, _mm256_mul_ps(a_val, dequantized));
            }
            
            // Apply SiLU activation: x * sigmoid(x)
            // Compute sigmoid using optimized approximation: 1 / (1 + exp(-x))
            __m256 neg_sum = _mm256_sub_ps(vzero, sum);
            __m256 exp_neg = exp256_ps(neg_sum);  // Using the existing exp approximation function
            __m256 denom = _mm256_add_ps(vones, exp_neg);
            __m256 sigmoid = _mm256_div_ps(vones, denom);
            
            // Final SiLU: x * sigmoid(x)
            __m256 result = _mm256_mul_ps(sum, sigmoid);
            
            // Store the result
            _mm256_storeu_ps(&output[i * n + j], result);
#endif // CCSM_HAVE_AVX
        }
    }
}

// Implementation of Q8_0 quantization with AVX support
#ifdef CCSM_HAVE_AVX
void quantize_q8_0_avx(int8_t* output, const float* input, size_t n) {
    // Find absolute maximum value for scaling
    __m256 vmax_abs = _mm256_setzero_ps();
    
    // Process 8 float values at a time
    for (size_t i = 0; i < n; i += 8) {
        // Handle boundary condition
        if (i + 8 <= n) {
            // Load 8 float values
            __m256 vx = _mm256_loadu_ps(&input[i]);
            
            // Compute absolute values
            __m256 vabs = _mm256_and_ps(vx, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
            
            // Update max_abs
            vmax_abs = _mm256_max_ps(vmax_abs, vabs);
        } else {
            // Handle remaining elements (less than 8)
            for (size_t j = i; j < n; j++) {
                float abs_val = std::abs(input[j]);
                vmax_abs = _mm256_max_ps(vmax_abs, _mm256_set1_ps(abs_val));
            }
        }
    }
    
    // Reduce max across vector lanes
    __m128 vmax128 = _mm_max_ps(_mm256_extractf128_ps(vmax_abs, 0), _mm256_extractf128_ps(vmax_abs, 1));
    vmax128 = _mm_max_ps(vmax128, _mm_permute_ps(vmax128, 0x4E)); // 0x4E = 01 00 11 10
    vmax128 = _mm_max_ps(vmax128, _mm_permute_ps(vmax128, 0xB1)); // 0xB1 = 10 11 00 01
    
    // Extract the maximum value
    float max_abs;
    _mm_store_ss(&max_abs, vmax128);
    
    // Calculate scale to map range [-max_abs, max_abs] to [-127, 127]
    float scale = max_abs > 0 ? 127.0f / max_abs : 1.0f;
    
    // Store scale value at the end of the block (inverse scale for dequantization)
    float* scale_ptr = reinterpret_cast<float*>(output + n);
    *scale_ptr = 1.0f / scale;
    
    // Broadcast scale to vector
    __m256 vscale = _mm256_set1_ps(scale);
    
    // Constants for clamping to [-127, 127]
    __m256 vmax_clamp = _mm256_set1_ps(127.0f);
    __m256 vmin_clamp = _mm256_set1_ps(-127.0f);
    
    // Process 8 values at a time
    for (size_t i = 0; i < n; i += 8) {
        if (i + 8 <= n) {
            // Load 8 float values
            __m256 vx = _mm256_loadu_ps(&input[i]);
            
            // Scale the values
            __m256 vscaled = _mm256_mul_ps(vx, vscale);
            
            // Clamp the values to [-127, 127]
            vscaled = _mm256_min_ps(_mm256_max_ps(vscaled, vmin_clamp), vmax_clamp);
            
            // Convert to int32
            __m256i vi32 = _mm256_cvtps_epi32(vscaled);
            
            // Extract and store int8 values
            // First convert to 4 int32 values in the lower lane
            __m128i vi32_low = _mm256_extractf128_si256(vi32, 0);
            __m128i vi32_high = _mm256_extractf128_si256(vi32, 1);
            
            // Pack into 8 int16 values
            __m128i vi16 = _mm_packs_epi32(vi32_low, vi32_high);
            
            // Pack into 16 int8 values (we only care about the first 8)
            __m128i vi8 = _mm_packs_epi16(vi16, vi16);
            
            // Store the 8 int8 values
            _mm_storel_epi64(reinterpret_cast<__m128i*>(&output[i]), vi8);
        } else {
            // Handle remaining elements (less than 8)
            for (size_t j = i; j < n; j++) {
                // Scale and clamp
                float val = input[j] * scale;
                val = std::min(127.0f, std::max(-127.0f, val));
                output[j] = static_cast<int8_t>(val);
            }
        }
    }
}
#endif

#ifdef CCSM_HAVE_NEON
void quantize_q8_0_neon(int8_t* output, const float* input, size_t n) {
    // Find absolute maximum value for scaling
    float32x4_t vmax_abs = vdupq_n_f32(0.0f);
    
    // Process 4 float values at a time
    for (size_t i = 0; i < n; i += 4) {
        // Handle boundary condition
        if (i + 4 <= n) {
            // Load 4 float values
            float32x4_t vx = vld1q_f32(&input[i]);
            
            // Compute absolute values
            float32x4_t vabs = vabsq_f32(vx);
            
            // Update max_abs
            vmax_abs = vmaxq_f32(vmax_abs, vabs);
        } else {
            // Handle remaining elements (less than 4)
            for (size_t j = i; j < n; j++) {
                float abs_val = std::abs(input[j]);
                float32x4_t vabs = vdupq_n_f32(abs_val);
                vmax_abs = vmaxq_f32(vmax_abs, vabs);
            }
        }
    }
    
    // Reduce max across vector lanes
    float32x2_t vmax2 = vpmax_f32(vget_low_f32(vmax_abs), vget_high_f32(vmax_abs));
    vmax2 = vpmax_f32(vmax2, vmax2);
    
    // Extract the maximum value
    float max_abs = vget_lane_f32(vmax2, 0);
    
    // Calculate scale to map range [-max_abs, max_abs] to [-127, 127]
    float scale = max_abs > 0 ? 127.0f / max_abs : 1.0f;
    
    // Store scale value at the end of the block (inverse scale for dequantization)
    float* scale_ptr = reinterpret_cast<float*>(output + n);
    *scale_ptr = 1.0f / scale;
    
    // Broadcast scale to vector
    float32x4_t vscale = vdupq_n_f32(scale);
    
    // Constants for clamping to [-127, 127]
    float32x4_t vmax_clamp = vdupq_n_f32(127.0f);
    float32x4_t vmin_clamp = vdupq_n_f32(-127.0f);
    
    // Process 4 values at a time
    for (size_t i = 0; i < n; i += 4) {
        if (i + 4 <= n) {
            // Load 4 float values
            float32x4_t vx = vld1q_f32(&input[i]);
            
            // Scale the values
            float32x4_t vscaled = vmulq_f32(vx, vscale);
            
            // Clamp the values to [-127, 127]
            vscaled = vminq_f32(vmaxq_f32(vscaled, vmin_clamp), vmax_clamp);
            
            // Convert to int32
            int32x4_t vi32 = vcvtq_s32_f32(vscaled);
            
            // Convert to int16
            int16x4_t vi16 = vmovn_s32(vi32);
            
            // Convert to int8
            int8x8_t vi8 = vmovn_s16(vcombine_s16(vi16, vi16));
            
            // Store the 4 int8 values
            vst1_lane_s32(reinterpret_cast<int32_t*>(&output[i]), vreinterpret_s32_s8(vi8), 0);
        } else {
            // Handle remaining elements (less than 4)
            for (size_t j = i; j < n; j++) {
                // Scale and clamp
                float val = input[j] * scale;
                val = std::min(127.0f, std::max(-127.0f, val));
                output[j] = static_cast<int8_t>(val);
            }
        }
    }
}
#endif

// Implementation of Q4_0 quantization with AVX support
#ifdef CCSM_HAVE_AVX
void quantize_q4_0_avx(uint8_t* output, const float* input, size_t n) {
    // Q4_0 packs 2 values into a single byte, with a single scale for the block
    
    // Calculate number of bytes needed (ceil(n/2))
    size_t n_bytes = (n + 1) / 2;
    
    // Find absolute maximum value for scaling
    __m256 vmax_abs = _mm256_setzero_ps();
    
    // Process 8 float values at a time
    for (size_t i = 0; i < n; i += 8) {
        // Handle boundary condition
        if (i + 8 <= n) {
            // Load 8 float values
            __m256 vx = _mm256_loadu_ps(&input[i]);
            
            // Compute absolute values
            __m256 vabs = _mm256_and_ps(vx, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
            
            // Update max_abs
            vmax_abs = _mm256_max_ps(vmax_abs, vabs);
        } else {
            // Handle remaining elements (less than 8)
            for (size_t j = i; j < n; j++) {
                float abs_val = std::abs(input[j]);
                vmax_abs = _mm256_max_ps(vmax_abs, _mm256_set1_ps(abs_val));
            }
        }
    }
    
    // Reduce max across vector lanes
    __m128 vmax128 = _mm_max_ps(_mm256_extractf128_ps(vmax_abs, 0), _mm256_extractf128_ps(vmax_abs, 1));
    vmax128 = _mm_max_ps(vmax128, _mm_permute_ps(vmax128, 0x4E)); // 0x4E = 01 00 11 10
    vmax128 = _mm_max_ps(vmax128, _mm_permute_ps(vmax128, 0xB1)); // 0xB1 = 10 11 00 01
    
    // Extract the maximum value
    float max_abs;
    _mm_store_ss(&max_abs, vmax128);
    
    // Calculate scale to map range [-max_abs, max_abs] to [-7, 7]
    float scale = max_abs > 0 ? 7.0f / max_abs : 1.0f;
    
    // Store scale value at the end of the block (inverse scale for dequantization)
    float* scale_ptr = reinterpret_cast<float*>(output + n_bytes);
    *scale_ptr = 1.0f / scale;
    
    // Process pairs of values and pack them into bytes
    for (size_t i = 0; i < n; i += 2) {
        // Quantize first value (0-7 bits)
        float val1 = input[i] * scale;
        val1 = std::min(7.0f, std::max(-7.0f, val1));
        int32_t q_val1 = static_cast<int32_t>(val1) & 0xF;
        
        // Quantize second value if it exists (8-15 bits)
        int32_t q_val2 = 0;
        if (i + 1 < n) {
            float val2 = input[i + 1] * scale;
            val2 = std::min(7.0f, std::max(-7.0f, val2));
            q_val2 = static_cast<int32_t>(val2) & 0xF;
        }
        
        // Pack two 4-bit values into a single byte
        output[i / 2] = static_cast<uint8_t>(q_val1 | (q_val2 << 4));
    }
}
#endif

#ifdef CCSM_HAVE_NEON
void quantize_q4_0_neon(uint8_t* output, const float* input, size_t n) {
    // Q4_0 packs 2 values into a single byte, with a single scale for the block
    
    // Calculate number of bytes needed (ceil(n/2))
    size_t n_bytes = (n + 1) / 2;
    
    // Find absolute maximum value for scaling
    float32x4_t vmax_abs = vdupq_n_f32(0.0f);
    
    // Process 4 float values at a time
    for (size_t i = 0; i < n; i += 4) {
        // Handle boundary condition
        if (i + 4 <= n) {
            // Load 4 float values
            float32x4_t vx = vld1q_f32(&input[i]);
            
            // Compute absolute values
            float32x4_t vabs = vabsq_f32(vx);
            
            // Update max_abs
            vmax_abs = vmaxq_f32(vmax_abs, vabs);
        } else {
            // Handle remaining elements (less than 4)
            for (size_t j = i; j < n; j++) {
                float abs_val = std::abs(input[j]);
                float32x4_t vabs = vdupq_n_f32(abs_val);
                vmax_abs = vmaxq_f32(vmax_abs, vabs);
            }
        }
    }
    
    // Reduce max across vector lanes
    float32x2_t vmax2 = vpmax_f32(vget_low_f32(vmax_abs), vget_high_f32(vmax_abs));
    vmax2 = vpmax_f32(vmax2, vmax2);
    
    // Extract the maximum value
    float max_abs = vget_lane_f32(vmax2, 0);
    
    // Calculate scale to map range [-max_abs, max_abs] to [-7, 7]
    float scale = max_abs > 0 ? 7.0f / max_abs : 1.0f;
    
    // Store scale value at the end of the block (inverse scale for dequantization)
    float* scale_ptr = reinterpret_cast<float*>(output + n_bytes);
    *scale_ptr = 1.0f / scale;
    
    // Process pairs of values and pack them into bytes
    for (size_t i = 0; i < n; i += 2) {
        // Quantize first value (0-7 bits)
        float val1 = input[i] * scale;
        val1 = std::min(7.0f, std::max(-7.0f, val1));
        int32_t q_val1 = static_cast<int32_t>(val1) & 0xF;
        
        // Quantize second value if it exists (8-15 bits)
        int32_t q_val2 = 0;
        if (i + 1 < n) {
            float val2 = input[i + 1] * scale;
            val2 = std::min(7.0f, std::max(-7.0f, val2));
            q_val2 = static_cast<int32_t>(val2) & 0xF;
        }
        
        // Pack two 4-bit values into a single byte
        output[i / 2] = static_cast<uint8_t>(q_val1 | (q_val2 << 4));
    }
}
#endif

// Implementation of Q4_1 quantization (4-bit with non-zero bias)
#ifdef CCSM_HAVE_AVX
void quantize_q4_1_avx(uint8_t* output, const float* input, size_t n) {
    // Q4_1 packs 2 values into a single byte, with a scale and non-zero bias for the block
    
    // Calculate number of bytes needed (ceil(n/2))
    size_t n_bytes = (n + 1) / 2;
    
    // Find min and max values for scaling
    __m256 vmin_val = _mm256_set1_ps(std::numeric_limits<float>::max());
    __m256 vmax_val = _mm256_set1_ps(std::numeric_limits<float>::lowest());
    
    // Process 8 float values at a time
    for (size_t i = 0; i < n; i += 8) {
        // Handle boundary condition
        if (i + 8 <= n) {
            // Load 8 float values
            __m256 vx = _mm256_loadu_ps(&input[i]);
            
            // Update min/max
            vmin_val = _mm256_min_ps(vmin_val, vx);
            vmax_val = _mm256_max_ps(vmax_val, vx);
        } else {
            // Handle remaining elements (less than 8)
            for (size_t j = i; j < n; j++) {
                float val = input[j];
                vmin_val = _mm256_min_ps(vmin_val, _mm256_set1_ps(val));
                vmax_val = _mm256_max_ps(vmax_val, _mm256_set1_ps(val));
            }
        }
    }
    
    // Reduce min across vector lanes
    __m128 vmin128 = _mm_min_ps(_mm256_extractf128_ps(vmin_val, 0), _mm256_extractf128_ps(vmin_val, 1));
    vmin128 = _mm_min_ps(vmin128, _mm_permute_ps(vmin128, 0x4E)); // 0x4E = 01 00 11 10
    vmin128 = _mm_min_ps(vmin128, _mm_permute_ps(vmin128, 0xB1)); // 0xB1 = 10 11 00 01
    
    // Reduce max across vector lanes
    __m128 vmax128 = _mm_max_ps(_mm256_extractf128_ps(vmax_val, 0), _mm256_extractf128_ps(vmax_val, 1));
    vmax128 = _mm_max_ps(vmax128, _mm_permute_ps(vmax128, 0x4E));
    vmax128 = _mm_max_ps(vmax128, _mm_permute_ps(vmax128, 0xB1));
    
    // Extract min and max values
    float min_val, max_val;
    _mm_store_ss(&min_val, vmin128);
    _mm_store_ss(&max_val, vmax128);
    
    // Calculate scale and bias
    float range = max_val - min_val;
    float scale = range > 0 ? 15.0f / range : 1.0f;
    float bias = min_val;
    
    // Store scale and bias values at the end of the block
    float* params = reinterpret_cast<float*>(output + n_bytes);
    params[0] = 1.0f / scale; // Store inverse scale for dequantization
    params[1] = bias;         // Store bias for dequantization
    
    // Process pairs of values and pack them into bytes
    for (size_t i = 0; i < n; i += 2) {
        // Quantize first value (0-7 bits)
        float val1 = (input[i] - bias) * scale;
        val1 = std::min(15.0f, std::max(0.0f, val1));
        int32_t q_val1 = static_cast<int32_t>(val1) & 0xF;
        
        // Quantize second value if it exists (8-15 bits)
        int32_t q_val2 = 0;
        if (i + 1 < n) {
            float val2 = (input[i + 1] - bias) * scale;
            val2 = std::min(15.0f, std::max(0.0f, val2));
            q_val2 = static_cast<int32_t>(val2) & 0xF;
        }
        
        // Pack two 4-bit values into a single byte
        output[i / 2] = static_cast<uint8_t>(q_val1 | (q_val2 << 4));
    }
}
#endif

#ifdef CCSM_HAVE_NEON
void quantize_q4_1_neon(uint8_t* output, const float* input, size_t n) {
    // Q4_1 packs 2 values into a single byte, with a scale and non-zero bias for the block
    
    // Calculate number of bytes needed (ceil(n/2))
    size_t n_bytes = (n + 1) / 2;
    
    // Find min and max values for scaling
    float32x4_t vmin_val = vdupq_n_f32(std::numeric_limits<float>::max());
    float32x4_t vmax_val = vdupq_n_f32(std::numeric_limits<float>::lowest());
    
    // Process 4 float values at a time
    for (size_t i = 0; i < n; i += 4) {
        // Handle boundary condition
        if (i + 4 <= n) {
            // Load 4 float values
            float32x4_t vx = vld1q_f32(&input[i]);
            
            // Update min/max
            vmin_val = vminq_f32(vmin_val, vx);
            vmax_val = vmaxq_f32(vmax_val, vx);
        } else {
            // Handle remaining elements (less than 4)
            for (size_t j = i; j < n; j++) {
                float val = input[j];
                float32x4_t vx = vdupq_n_f32(val);
                vmin_val = vminq_f32(vmin_val, vx);
                vmax_val = vmaxq_f32(vmax_val, vx);
            }
        }
    }
    
    // Reduce min/max across vector lanes
    float32x2_t vmin2 = vpmin_f32(vget_low_f32(vmin_val), vget_high_f32(vmin_val));
    vmin2 = vpmin_f32(vmin2, vmin2);
    
    float32x2_t vmax2 = vpmax_f32(vget_low_f32(vmax_val), vget_high_f32(vmax_val));
    vmax2 = vpmax_f32(vmax2, vmax2);
    
    // Extract min and max values
    float min_val = vget_lane_f32(vmin2, 0);
    float max_val = vget_lane_f32(vmax2, 0);
    
    // Calculate scale and bias
    float range = max_val - min_val;
    float scale = range > 0 ? 15.0f / range : 1.0f;
    float bias = min_val;
    
    // Store scale and bias values at the end of the block
    float* params = reinterpret_cast<float*>(output + n_bytes);
    params[0] = 1.0f / scale; // Store inverse scale for dequantization
    params[1] = bias;         // Store bias for dequantization
    
    // Process pairs of values and pack them into bytes
    for (size_t i = 0; i < n; i += 2) {
        // Quantize first value (0-7 bits)
        float val1 = (input[i] - bias) * scale;
        val1 = std::min(15.0f, std::max(0.0f, val1));
        int32_t q_val1 = static_cast<int32_t>(val1) & 0xF;
        
        // Quantize second value if it exists (8-15 bits)
        int32_t q_val2 = 0;
        if (i + 1 < n) {
            float val2 = (input[i + 1] - bias) * scale;
            val2 = std::min(15.0f, std::max(0.0f, val2));
            q_val2 = static_cast<int32_t>(val2) & 0xF;
        }
        
        // Pack two 4-bit values into a single byte
        output[i / 2] = static_cast<uint8_t>(q_val1 | (q_val2 << 4));
    }
}
#endif

// Dequantization implementations with SIMD support
#ifdef CCSM_HAVE_AVX
void dequantize_q8_0_avx(float* output, const int8_t* input, const float* scale, size_t n) {
    const float inv_scale = *scale;
    __m256 vscale = _mm256_set1_ps(inv_scale);
    
    // Process 8 values at a time
    for (size_t i = 0; i < n; i += 8) {
        if (i + 8 <= n) {
            // Load 8 int8 values and convert to int32
            __m128i vi8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&input[i]));
            __m128i vi16_low = _mm_cvtepi8_epi16(vi8);
            __m128i vi32_low = _mm_cvtepi16_epi32(vi16_low);
            __m128i vi16_high = _mm_unpackhi_epi8(vi8, _mm_cmpgt_epi8(_mm_setzero_si128(), vi8));
            __m128i vi32_high = _mm_cvtepi16_epi32(vi16_high);
            
            // Convert to float and scale
            __m256 vf_low = _mm256_cvtepi32_ps(_mm256_setr_m128i(vi32_low, vi32_high));
            __m256 vf = _mm256_mul_ps(vf_low, vscale);
            
            // Store result
            _mm256_storeu_ps(&output[i], vf);
        } else {
            // Handle remaining elements (less than 8)
            for (size_t j = i; j < n; j++) {
                output[j] = static_cast<float>(input[j]) * inv_scale;
            }
        }
    }
}
#endif

#ifdef CCSM_HAVE_NEON
void dequantize_q8_0_neon(float* output, const int8_t* input, const float* scale, size_t n) {
    const float inv_scale = *scale;
    float32x4_t vscale = vdupq_n_f32(inv_scale);
    
    // Process 4 values at a time
    for (size_t i = 0; i < n; i += 4) {
        if (i + 4 <= n) {
            // Load 4 int8 values
            int8x8_t vi8 = vld1_s8(&input[i]);
            
            // Convert to int16
            int16x8_t vi16 = vmovl_s8(vi8);
            
            // Convert to int32
            int32x4_t vi32_low = vmovl_s16(vget_low_s16(vi16));
            
            // Convert to float and scale
            float32x4_t vf = vmulq_f32(vcvtq_f32_s32(vi32_low), vscale);
            
            // Store result
            vst1q_f32(&output[i], vf);
        } else {
            // Handle remaining elements (less than 4)
            for (size_t j = i; j < n; j++) {
                output[j] = static_cast<float>(input[j]) * inv_scale;
            }
        }
    }
}
#endif

#ifdef CCSM_HAVE_AVX
void dequantize_q4_0_avx(float* output, const uint8_t* input, const float* scale, size_t n) {
    const float inv_scale = *scale;
    
    for (size_t i = 0; i < n; i += 2) {
        // Extract from packed 4-bit representation
        uint8_t byte = input[i / 2];
        
        // Extract first 4-bit value
        int8_t val1 = static_cast<int8_t>(byte & 0xF);
        // Sign extend if the high bit is set
        if (val1 & 0x8) {
            val1 |= 0xF0; // Extend to 8 bits
        }
        output[i] = static_cast<float>(val1) * inv_scale;
        
        // Extract second 4-bit value if we have one
        if (i + 1 < n) {
            int8_t val2 = static_cast<int8_t>((byte >> 4) & 0xF);
            // Sign extend if the high bit is set
            if (val2 & 0x8) {
                val2 |= 0xF0; // Extend to 8 bits
            }
            output[i + 1] = static_cast<float>(val2) * inv_scale;
        }
    }
}
#endif

#ifdef CCSM_HAVE_NEON
void dequantize_q4_0_neon(float* output, const uint8_t* input, const float* scale, size_t n) {
    const float inv_scale = *scale;
    
    for (size_t i = 0; i < n; i += 2) {
        // Extract from packed 4-bit representation
        uint8_t byte = input[i / 2];
        
        // Extract first 4-bit value
        int8_t val1 = static_cast<int8_t>(byte & 0xF);
        // Sign extend if the high bit is set
        if (val1 & 0x8) {
            val1 |= 0xF0; // Extend to 8 bits
        }
        output[i] = static_cast<float>(val1) * inv_scale;
        
        // Extract second 4-bit value if we have one
        if (i + 1 < n) {
            int8_t val2 = static_cast<int8_t>((byte >> 4) & 0xF);
            // Sign extend if the high bit is set
            if (val2 & 0x8) {
                val2 |= 0xF0; // Extend to 8 bits
            }
            output[i + 1] = static_cast<float>(val2) * inv_scale;
        }
    }
}
#endif

#ifdef CCSM_HAVE_AVX
void dequantize_q4_1_avx(float* output, const uint8_t* input, const float* scale, const float* bias, size_t n) {
    const float inv_scale = scale[0];
    const float bias_val = bias[0];
    
    for (size_t i = 0; i < n; i += 2) {
        // Extract from packed 4-bit representation
        uint8_t byte = input[i / 2];
        
        // Extract first 4-bit value
        uint8_t val1 = byte & 0xF;
        output[i] = static_cast<float>(val1) * inv_scale + bias_val;
        
        // Extract second 4-bit value if we have one
        if (i + 1 < n) {
            uint8_t val2 = (byte >> 4) & 0xF;
            output[i + 1] = static_cast<float>(val2) * inv_scale + bias_val;
        }
    }
}
#endif

#ifdef CCSM_HAVE_NEON
void dequantize_q4_1_neon(float* output, const uint8_t* input, const float* scale, const float* bias, size_t n) {
    const float inv_scale = scale[0];
    const float bias_val = bias[0];
    
    for (size_t i = 0; i < n; i += 2) {
        // Extract from packed 4-bit representation
        uint8_t byte = input[i / 2];
        
        // Extract first 4-bit value
        uint8_t val1 = byte & 0xF;
        output[i] = static_cast<float>(val1) * inv_scale + bias_val;
        
        // Extract second 4-bit value if we have one
        if (i + 1 < n) {
            uint8_t val2 = (byte >> 4) & 0xF;
            output[i + 1] = static_cast<float>(val2) * inv_scale + bias_val;
        }
    }
}
#endif

} // namespace detail

// Template specializations for quantization functions
template<>
void quantize_q8_0<float>(int8_t* output, const float* input, size_t n) {
    Implementation impl = get_active_implementation();
    
    #if defined(CCSM_HAVE_AVX)
    if (impl >= Implementation::AVX) {
        detail::quantize_q8_0_avx(output, input, n);
        return;
    }
    #endif
    
    #if defined(CCSM_HAVE_NEON)
    if (impl == Implementation::NEON) {
        detail::quantize_q8_0_neon(output, input, n);
        return;
    }
    #endif
    
    // Fallback to scalar implementation
    detail::quantize_q8_0_scalar(output, input, n);
}

template<>
void dequantize_q8_0<float>(float* output, const int8_t* input, const float* scale, size_t n) {
    Implementation impl = get_active_implementation();
    
    #if defined(CCSM_HAVE_AVX)
    if (impl >= Implementation::AVX) {
        detail::dequantize_q8_0_avx(output, input, scale, n);
        return;
    }
    #endif
    
    #if defined(CCSM_HAVE_NEON)
    if (impl == Implementation::NEON) {
        detail::dequantize_q8_0_neon(output, input, scale, n);
        return;
    }
    #endif
    
    // Fallback to scalar implementation
    detail::dequantize_q8_0_scalar(output, input, scale, n);
}

template<>
void quantize_q4_0<float>(uint8_t* output, const float* input, size_t n) {
    Implementation impl = get_active_implementation();
    
    #if defined(CCSM_HAVE_AVX)
    if (impl >= Implementation::AVX) {
        detail::quantize_q4_0_avx(output, input, n);
        return;
    }
    #endif
    
    #if defined(CCSM_HAVE_NEON)
    if (impl == Implementation::NEON) {
        detail::quantize_q4_0_neon(output, input, n);
        return;
    }
    #endif
    
    // Fallback to scalar implementation
    detail::quantize_q4_0_scalar(output, input, n);
}

template<>
void dequantize_q4_0<float>(float* output, const uint8_t* input, const float* scale, size_t n) {
    Implementation impl = get_active_implementation();
    
    #if defined(CCSM_HAVE_AVX)
    if (impl >= Implementation::AVX) {
        detail::dequantize_q4_0_avx(output, input, scale, n);
        return;
    }
    #endif
    
    #if defined(CCSM_HAVE_NEON)
    if (impl == Implementation::NEON) {
        detail::dequantize_q4_0_neon(output, input, scale, n);
        return;
    }
    #endif
    
    // Fallback to scalar implementation
    detail::dequantize_q4_0_scalar(output, input, scale, n);
}

template<>
void quantize_q4_1<float>(uint8_t* output, const float* input, size_t n) {
    Implementation impl = get_active_implementation();
    
    #if defined(CCSM_HAVE_AVX)
    if (impl >= Implementation::AVX) {
        detail::quantize_q4_1_avx(output, input, n);
        return;
    }
    #endif
    
    #if defined(CCSM_HAVE_NEON)
    if (impl == Implementation::NEON) {
        detail::quantize_q4_1_neon(output, input, n);
        return;
    }
    #endif
    
    // Fallback to scalar implementation
    detail::quantize_q4_1_scalar(output, input, n);
}

template<>
void dequantize_q4_1<float>(float* output, const uint8_t* input, const float* scale, const float* bias, size_t n) {
    Implementation impl = get_active_implementation();
    
    #if defined(CCSM_HAVE_AVX)
    if (impl >= Implementation::AVX) {
        detail::dequantize_q4_1_avx(output, input, scale, bias, n);
        return;
    }
    #endif
    
    #if defined(CCSM_HAVE_NEON)
    if (impl == Implementation::NEON) {
        detail::dequantize_q4_1_neon(output, input, scale, bias, n);
        return;
    }
    #endif
    
    // Fallback to scalar implementation
    detail::dequantize_q4_1_scalar(output, input, scale, bias, n);
}

// Matrix multiplication specializations for quantized operations
template<>
void matrix_mul_q8_0<float>(float* result, const float* a, const int8_t* b, const float* b_scale, 
                            size_t m, size_t k, size_t n) {
    Implementation impl = get_active_implementation();
    
    // Use a scalar implementation for now - can be optimized with SIMD in the future
    detail::matrix_mul_q8_0_scalar(result, a, b, b_scale, m, k, n);
}

template<>
void matrix_mul_q4_0<float>(float* result, const float* a, const uint8_t* b, const float* b_scale, 
                            size_t m, size_t k, size_t n) {
    Implementation impl = get_active_implementation();
    
    // Use a scalar implementation for now - can be optimized with SIMD in the future
    detail::matrix_mul_q4_0_scalar(result, a, b, b_scale, m, k, n);
}

template<>
void matrix_mul_q4_1<float>(float* result, const float* a, const uint8_t* b, const float* b_scale, 
                            const float* b_bias, size_t m, size_t k, size_t n) {
    Implementation impl = get_active_implementation();
    
    // Use a scalar implementation for now - can be optimized with SIMD in the future
    detail::matrix_mul_q4_1_scalar(result, a, b, b_scale, b_bias, m, k, n);
}

} // namespace simd
} // namespace ccsm