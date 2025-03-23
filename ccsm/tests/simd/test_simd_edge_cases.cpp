#include <ccsm/cpu/simd.h>
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <limits>

using namespace ccsm;

// Test fixture for SIMD edge case handling
class SIMDEdgeCaseTest : public ::testing::Test {
protected:
    // Generate special floating point values for testing
    std::vector<float> create_special_values(size_t n) {
        std::vector<float> values(n);
        
        // Fill with various special values
        size_t i = 0;
        
        // NaN values
        values[i++] = std::numeric_limits<float>::quiet_NaN();
        if (i < n) values[i++] = -std::numeric_limits<float>::quiet_NaN();
        if (i < n) values[i++] = std::numeric_limits<float>::signaling_NaN();
        
        // Infinity values
        if (i < n) values[i++] = std::numeric_limits<float>::infinity();
        if (i < n) values[i++] = -std::numeric_limits<float>::infinity();
        
        // Zero values
        if (i < n) values[i++] = 0.0f;
        if (i < n) values[i++] = -0.0f;
        
        // Denormal values
        if (i < n) values[i++] = std::numeric_limits<float>::denorm_min();
        if (i < n) values[i++] = -std::numeric_limits<float>::denorm_min();
        
        // Normal values but close to limits
        if (i < n) values[i++] = std::numeric_limits<float>::min();
        if (i < n) values[i++] = -std::numeric_limits<float>::min();
        if (i < n) values[i++] = std::numeric_limits<float>::max();
        if (i < n) values[i++] = -std::numeric_limits<float>::max();
        
        // Fill the rest with regular values
        for (; i < n; i++) {
            values[i] = static_cast<float>(i) - n/2;
        }
        
        return values;
    }
    
    // Helper for checking if a value is denormal
    bool is_denormal(float x) {
        return std::fpclassify(x) == FP_SUBNORMAL;
    }
};

// Test activation functions with special values
TEST_F(SIMDEdgeCaseTest, ActivationFunctionsWithSpecialValues) {
    const size_t n = 32;  // Enough for all our special cases plus regular values
    
    // Create test data with special values
    std::vector<float> input = create_special_values(n);
    std::vector<float> output(n);
    
    // Test ReLU activation with special values
    simd::relu(output.data(), input.data(), n);
    
    // Verify ReLU behavior with special values
    for (size_t i = 0; i < n; i++) {
        // NaN handling - ReLU should propagate NaN by default 
        if (std::isnan(input[i])) {
            EXPECT_TRUE(std::isnan(output[i])) << "ReLU should propagate NaN by default at index " << i;
            continue;
        }
        
        // Negative infinity should become 0 in ReLU
        if (std::isinf(input[i]) && input[i] < 0) {
            EXPECT_FLOAT_EQ(output[i], 0.0f) << "ReLU should map negative infinity to zero at index " << i;
            continue;
        }
        
        // Positive infinity should remain infinity
        if (std::isinf(input[i]) && input[i] > 0) {
            EXPECT_TRUE(std::isinf(output[i]) && output[i] > 0) 
                << "ReLU should preserve positive infinity at index " << i;
            continue;
        }
        
        // Negative values should become 0
        if (input[i] < 0) {
            EXPECT_FLOAT_EQ(output[i], 0.0f) << "ReLU should map negative values to zero at index " << i;
            continue;
        }
        
        // Positive values should remain unchanged
        EXPECT_FLOAT_EQ(output[i], input[i]) << "ReLU should preserve positive values at index " << i;
    }
    
    // Test SiLU activation with special values
    simd::silu(output.data(), input.data(), n);
    
    // Verify SiLU behavior with special values
    for (size_t i = 0; i < n; i++) {
        // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
        
        // NaN handling - SiLU should propagate NaN
        if (std::isnan(input[i])) {
            EXPECT_TRUE(std::isnan(output[i])) << "SiLU should propagate NaN at index " << i;
            continue;
        }
        
        // Negative infinity: SiLU(-inf) = -inf * sigmoid(-inf) = -inf * 0 = -0
        if (std::isinf(input[i]) && input[i] < 0) {
            EXPECT_FLOAT_EQ(output[i], 0.0f) << "SiLU should map negative infinity close to zero at index " << i;
            EXPECT_TRUE(std::signbit(output[i])) << "SiLU of negative infinity should be negative zero at index " << i;
            continue;
        }
        
        // Positive infinity: SiLU(+inf) = +inf * sigmoid(+inf) = +inf * 1 = +inf
        if (std::isinf(input[i]) && input[i] > 0) {
            EXPECT_TRUE(std::isinf(output[i]) && output[i] > 0) 
                << "SiLU should preserve positive infinity at index " << i;
            continue;
        }
        
        // For regular values, just verify it's not NaN or Inf
        if (std::isfinite(input[i])) {
            EXPECT_FALSE(std::isnan(output[i])) << "SiLU should not produce NaN for finite input at index " << i;
            EXPECT_FALSE(std::isinf(output[i])) << "SiLU should not produce Inf for finite input at index " << i;
            
            // Very simple check on SiLU behavior - should have same sign as input except near zero
            if (std::abs(input[i]) > 0.1f) {
                EXPECT_EQ(std::signbit(input[i]), std::signbit(output[i])) 
                    << "SiLU should preserve sign for values away from zero at index " << i;
            }
        }
    }
}

// Test safe operations with denormal handling disabled
TEST_F(SIMDEdgeCaseTest, DenormalHandling) {
    // Skip the test if safe operations aren't implemented yet
    GTEST_SKIP() << "Safe operations with denormal handling not implemented yet";
    
    const size_t n = 16;
    
    // Create values with denormals
    std::vector<float> input(n);
    std::vector<float> output(n);
    
    // Fill with denormals and small normal values
    input[0] = std::numeric_limits<float>::denorm_min();
    input[1] = std::numeric_limits<float>::denorm_min() * 2.0f;
    input[2] = std::numeric_limits<float>::denorm_min() * 10.0f;
    input[3] = -std::numeric_limits<float>::denorm_min();
    input[4] = std::numeric_limits<float>::min(); // Smallest normal
    input[5] = -std::numeric_limits<float>::min();
    
    // Fill the rest with regular values
    for (size_t i = 6; i < n; i++) {
        input[i] = static_cast<float>(i) - n/2;
    }
    
    // Test denormal handling with safe operations
    // Expect this API once implemented:
    // simd::vector_add_safe(output.data(), input.data(), input.data(), n, true, true);
    
    // Verification logic would check that denormals are properly flushed to zero
    /*
    for (size_t i = 0; i < n; i++) {
        if (is_denormal(input[i])) {
            EXPECT_FLOAT_EQ(output[i], 0.0f) 
                << "Safe operation should flush denormal to zero at index " << i;
        } else {
            // For normal values, output should be twice the input (a + a)
            EXPECT_FLOAT_EQ(output[i], input[i] * 2.0f) 
                << "Safe add should give correct results for normal values at index " << i;
        }
    }
    */
}

// Test NaN handling options
TEST_F(SIMDEdgeCaseTest, NaNHandling) {
    // Skip the test if safe operations aren't implemented yet
    GTEST_SKIP() << "Safe operations with NaN handling not implemented yet";
    
    const size_t n = 16;
    
    // Create test vectors with some NaN values
    std::vector<float> a(n), b(n), c(n);
    
    a[0] = std::numeric_limits<float>::quiet_NaN();
    a[1] = 1.0f;
    a[2] = std::numeric_limits<float>::quiet_NaN();
    a[3] = 3.0f;
    
    b[0] = 5.0f;
    b[1] = std::numeric_limits<float>::quiet_NaN();
    b[2] = std::numeric_limits<float>::quiet_NaN();
    b[3] = 7.0f;
    
    // Fill the rest with regular values
    for (size_t i = 4; i < n; i++) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i) + 5.0f;
    }
    
    // Test vector operations with NaN handling
    // Expect this API once implemented:
    // simd::vector_add_safe(c.data(), a.data(), b.data(), n, false, true);
    
    // Verification logic would check that NaNs are replaced with 0
    /*
    for (size_t i = 0; i < n; i++) {
        if (std::isnan(a[i]) || std::isnan(b[i])) {
            // NaN values should be replaced with 0, so result should be the other value
            // or 0 if both are NaN
            float expected = std::isnan(a[i]) ? (std::isnan(b[i]) ? 0.0f : b[i]) : a[i];
            EXPECT_FLOAT_EQ(c[i], expected) 
                << "Safe operation should handle NaN properly at index " << i;
        } else {
            // For regular values, result should be a + b
            EXPECT_FLOAT_EQ(c[i], a[i] + b[i]) 
                << "Safe add should give correct results for normal values at index " << i;
        }
    }
    */
}

// Test infinity handling
TEST_F(SIMDEdgeCaseTest, InfinityHandling) {
    const size_t n = 16;
    
    // Create test vectors with infinity values
    std::vector<float> a(n), b(n), c(n);
    
    a[0] = std::numeric_limits<float>::infinity();
    a[1] = 1.0f;
    a[2] = -std::numeric_limits<float>::infinity();
    a[3] = 3.0f;
    
    b[0] = 5.0f;
    b[1] = std::numeric_limits<float>::infinity();
    b[2] = std::numeric_limits<float>::infinity();
    b[3] = -std::numeric_limits<float>::infinity();
    
    // Fill the rest with regular values
    for (size_t i = 4; i < n; i++) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i) + 5.0f;
    }
    
    // Test regular vector add with infinity
    simd::vector_add(c.data(), a.data(), b.data(), n);
    
    // Verify infinity handling in regular operations
    for (size_t i = 0; i < n; i++) {
        if (std::isinf(a[i]) && std::isinf(b[i])) {
            if ((a[i] > 0 && b[i] > 0) || (a[i] < 0 && b[i] < 0)) {
                // Same sign infinity added together should be infinity with same sign
                EXPECT_TRUE(std::isinf(c[i])) << "Adding same-sign infinities should give infinity at index " << i;
                EXPECT_EQ(std::signbit(a[i]), std::signbit(c[i])) << "Sign should be preserved at index " << i;
            } else {
                // Positive and negative infinity should give NaN
                EXPECT_TRUE(std::isnan(c[i])) 
                    << "Adding infinities of opposite signs should give NaN at index " << i;
            }
        } else if (std::isinf(a[i])) {
            // Infinity plus any finite value should be infinity
            EXPECT_TRUE(std::isinf(c[i])) << "Infinity plus finite should be infinity at index " << i;
            EXPECT_EQ(std::signbit(a[i]), std::signbit(c[i])) << "Sign should be preserved at index " << i;
        } else if (std::isinf(b[i])) {
            // Any finite value plus infinity should be infinity
            EXPECT_TRUE(std::isinf(c[i])) << "Finite plus infinity should be infinity at index " << i;
            EXPECT_EQ(std::signbit(b[i]), std::signbit(c[i])) << "Sign should be preserved at index " << i;
        } else {
            // Regular values
            EXPECT_FLOAT_EQ(c[i], a[i] + b[i]) << "Regular addition should work at index " << i;
        }
    }
}
