// Fused matrix multiplication with Q4_0 quantized weights and ReLU activation
void fused_matmul_relu_q4_0_avx_f32(float* output, const float* a, const uint8_t* b, const float* b_scale, 
                                  size_t m, size_t k, size_t n) {
    // Get scale factor for dequantization
    const float inv_scale = *b_scale;
    const __m256 vscale = _mm256_set1_ps(inv_scale);
    
    // Zero vector for ReLU
    const __m256 vzero = _mm256_setzero_ps();
    
    // Initialize result matrix to zeros
    for (size_t i = 0; i < m * n; i++) {
        output[i] = 0.0f;
    }
    
    // Process each row of A and corresponding output row
    for (size_t i = 0; i < m; i++) {
        // Process output elements in blocks of 8 (AVX register width)
        for (size_t j = 0; j < n; j += 8) {
            // Handle boundary condition
            size_t block_size = std::min(size_t(8), n - j);
            
            if (block_size < 8) {
                // If we don't have a full vector, process the rest using scalar code
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
            }
            
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
        }
    }
}

// Fused matrix multiplication with Q4_0 quantized weights and SiLU activation
void fused_matmul_silu_q4_0_avx_f32(float* output, const float* a, const uint8_t* b, const float* b_scale, 
                                   size_t m, size_t k, size_t n) {
    // Get scale factor for dequantization
    const float inv_scale = *b_scale;
    const __m256 vscale = _mm256_set1_ps(inv_scale);
    
    // Constants for SiLU calculation
    const __m256 vones = _mm256_set1_ps(1.0f);
    const __m256 vzero = _mm256_setzero_ps();
    
    // Initialize result matrix to zeros
    for (size_t i = 0; i < m * n; i++) {
        output[i] = 0.0f;
    }
    
    // Process each row of A and corresponding output row
    for (size_t i = 0; i < m; i++) {
        // Process output elements in blocks of 8 (AVX register width)
        for (size_t j = 0; j < n; j += 8) {
            // Handle boundary condition
            size_t block_size = std::min(size_t(8), n - j);
            
            if (block_size < 8) {
                // If we don't have a full vector, process the rest using scalar code
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
            }
            
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
        }
    }
}