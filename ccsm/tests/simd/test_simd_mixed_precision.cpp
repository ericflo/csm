#include <ccsm/cpu/simd.h>
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <limits>

using namespace ccsm;

// Test fixture for SIMD mixed precision tests
class SIMDMixedPrecisionTest : public ::testing::Test {
protected:
    // Helper for comparing floats
    bool almost_equal(float a, float b, float epsilon = 1e-5) {
        return std::fabs(a - b) < epsilon;
    }

    bool vector_almost_equal(const std::vector<float>& a, const std::vector<float>& b, float epsilon = 1e-5) {
        if (a.size() != b.size()) {
            std::cout << "Vector size mismatch: " << a.size() << " vs " << b.size() << std::endl;
            return false;
        }
        
        for (size_t i = 0; i < a.size(); i++) {
            if (!almost_equal(a[i], b[i], epsilon)) {
                std::cout << "Vectors differ at index " << i << ": " << a[i] << " vs " << b[i] 
                          << " (diff: " << std::abs(a[i] - b[i]) << ", epsilon: " << epsilon << ")" << std::endl;
                return false;
            }
        }
        
        return true;
    }

    // Helper to convert between f32 and f16 for testing
    uint16_t f32_to_f16(float value) {
        // Simple conversion for testing purposes
        // IEEE 754 float16 format: 1 sign bit, 5 exponent bits, 10 mantissa bits
        uint32_t f32_bits;
        std::memcpy(&f32_bits, &value, sizeof(float));
        
        uint16_t sign = (f32_bits >> 31) & 0x1;
        int32_t exponent = ((f32_bits >> 23) & 0xFF) - 127;
        uint32_t mantissa = f32_bits & 0x7FFFFF;
        
        // Handle special cases
        if (std::isnan(value)) return 0x7E00; // NaN
        if (std::isinf(value)) return sign ? 0xFC00 : 0x7C00; // +/- Inf
        if (value == 0.0f) return sign << 15; // +/- Zero
        
        // Adjust for float16 bias and range
        if (exponent > 15) {
            // Overflow, return infinity
            return (sign << 15) | 0x7C00;
        } else if (exponent < -14) {
            // Underflow or denormal
            return (sign << 15);
        }
        
        uint16_t f16_exponent = (exponent + 15) & 0x1F;
        uint16_t f16_mantissa = (mantissa >> 13) & 0x3FF;
        
        return (sign << 15) | (f16_exponent << 10) | f16_mantissa;
    }

    float f16_to_f32(uint16_t value) {
        // Simple conversion for testing purposes
        uint16_t sign = (value >> 15) & 0x1;
        uint16_t exponent = (value >> 10) & 0x1F;
        uint16_t mantissa = value & 0x3FF;
        
        // Handle special cases
        if (exponent == 0x1F) {
            if (mantissa == 0) {
                // Infinity
                return sign ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
            } else {
                // NaN
                return std::numeric_limits<float>::quiet_NaN();
            }
        }
        
        if (exponent == 0 && mantissa == 0) {
            // Zero
            return sign ? -0.0f : 0.0f;
        }
        
        // Convert to IEEE 754 float
        int32_t f32_exponent;
        uint32_t f32_mantissa;
        
        if (exponent == 0) {
            // Denormalized
            f32_exponent = -14;
            f32_mantissa = mantissa;
        } else {
            // Normalized
            f32_exponent = exponent - 15;
            f32_mantissa = mantissa | 0x400; // Add implicit leading 1
        }
        
        uint32_t f32_bits = (sign << 31) | ((f32_exponent + 127) << 23) | (f32_mantissa << 13);
        float result;
        std::memcpy(&result, &f32_bits, sizeof(float));
        return result;
    }
    
    // Simple BF16 simulation - truncate float32 to 16 bits (1 sign, 8 exponent, 7 mantissa)
    uint16_t f32_to_bf16(float value) {
        uint32_t f32_bits;
        std::memcpy(&f32_bits, &value, sizeof(float));
        return f32_bits >> 16;
    }
    
    float bf16_to_f32(uint16_t value) {
        uint32_t f32_bits = static_cast<uint32_t>(value) << 16;
        float result;
        std::memcpy(&result, &f32_bits, sizeof(float));
        return result;
    }
};

// Test mixed precision vector operations
TEST_F(SIMDMixedPrecisionTest, MixedPrecisionVectorOperations) {
    const size_t n = 128; // Small enough for quick test, large enough for SIMD
    
    // Create test data in FP32 format
    std::vector<float> a_f32(n), b_f32(n), c_f32(n), expected_f32(n);
    
    // Create corresponding FP16 arrays
    std::vector<uint16_t> a_f16(n), b_f16(n), c_f16(n);
    std::vector<uint16_t> a_bf16(n), b_bf16(n), c_bf16(n);
    
    // Initialize with test values
    for (size_t i = 0; i < n; i++) {
        a_f32[i] = static_cast<float>(i) / 100.0f - 0.5f; // Mix of positive and negative
        b_f32[i] = static_cast<float>(i % 17) / 10.0f;    // Some different values
        
        // Convert to FP16/BF16
        a_f16[i] = f32_to_f16(a_f32[i]);
        b_f16[i] = f32_to_f16(b_f32[i]);
        
        a_bf16[i] = f32_to_bf16(a_f32[i]);
        b_bf16[i] = f32_to_bf16(b_f32[i]);
    }
    
    // Test vector addition
    // First compute expected result in full FP32 precision
    for (size_t i = 0; i < n; i++) {
        expected_f32[i] = a_f32[i] + b_f32[i];
    }
    
    // Run mixed precision vector add test if available, otherwise skip
    GTEST_SKIP() << "Mixed precision test not implemented yet";
    
    // Hypothetical API: (f32 + f16 -> f32)
    // simd::vector_add_mixed(c_f32.data(), a_f32.data(), a_f16.data(), n, simd::DataType::F16);
    // EXPECT_TRUE(vector_almost_equal(c_f32, expected_f32, 1e-2f)); // Lower precision expected
}

TEST_F(SIMDMixedPrecisionTest, F16Precision) {
    // Test that our F16/BF16 simulation routines work properly
    std::vector<float> test_values = {
        0.0f, 1.0f, -1.0f, 0.5f, -0.5f,
        65504.0f, -65504.0f, // Max representable in FP16
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN()
    };
    
    for (float val : test_values) {
        uint16_t f16 = f32_to_f16(val);
        float roundtrip = f16_to_f32(f16);
        
        // Skip NaN comparison (NaN != NaN)
        if (std::isnan(val)) {
            EXPECT_TRUE(std::isnan(roundtrip)) << "Expected NaN after roundtrip conversion";
        } else if (std::isinf(val)) {
            EXPECT_TRUE(std::isinf(roundtrip)) << "Expected Inf after roundtrip conversion";
            EXPECT_EQ(std::signbit(val), std::signbit(roundtrip)) << "Sign of Inf should be preserved";
        } else {
            // Regular values have limited precision, so we need a larger epsilon
            EXPECT_NEAR(val, roundtrip, std::max(0.001f, std::abs(val * 0.01f))) 
                << "Expected approximately equal value after F16 roundtrip conversion";
        }
    }
}

TEST_F(SIMDMixedPrecisionTest, BF16Precision) {
    // Test BF16 conversion with various values
    std::vector<float> test_values = {
        0.0f, 1.0f, -1.0f, 0.5f, -0.5f,
        1.0e20f, -1.0e20f, // Much larger range than FP16
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN()
    };
    
    for (float val : test_values) {
        uint16_t bf16 = f32_to_bf16(val);
        float roundtrip = bf16_to_f32(bf16);
        
        // Skip NaN comparison
        if (std::isnan(val)) {
            EXPECT_TRUE(std::isnan(roundtrip)) << "Expected NaN after BF16 roundtrip conversion";
        } else if (std::isinf(val)) {
            EXPECT_TRUE(std::isinf(roundtrip)) << "Expected Inf after BF16 roundtrip conversion";
            EXPECT_EQ(std::signbit(val), std::signbit(roundtrip)) << "Sign of Inf should be preserved";
        } else if (val == 0.0f) {
            EXPECT_EQ(val, roundtrip) << "Zero should be preserved exactly in BF16";
            EXPECT_EQ(std::signbit(val), std::signbit(roundtrip)) << "Sign of zero should be preserved";
        } else {
            // BF16 loses precision in the mantissa but keeps the exponent range
            // We can have significant relative error for regular values
            EXPECT_NEAR(val, roundtrip, std::max(0.01f, std::abs(val * 0.01f))) 
                << "Expected approximately equal value after BF16 roundtrip conversion";
        }
    }
}
