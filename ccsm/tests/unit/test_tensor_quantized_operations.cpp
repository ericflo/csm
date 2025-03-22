#include <gtest/gtest.h>
#include <ccsm/tensor.h>
#include <vector>
#include <memory>
#include <cmath>
#include <stdexcept>
#include <limits>

using namespace ccsm;

// Test fixture for quantized tensor operations
class QuantizedTensorOperationsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up default tensor shapes for testing
        vector_shape = {10};
        matrix_shape = {5, 10};
        
        // Create context
        context = ContextFactory::create();
    }
    
    std::vector<size_t> vector_shape;
    std::vector<size_t> matrix_shape;
    std::shared_ptr<Context> context;
    
    // Helper to create a tensor with specific data
    // For float tensors, fill with these values
    template <typename T>
    Tensor createTensorWithValues(const std::vector<size_t>& shape, DataType dtype, const std::vector<T>& values) {
        // Create a tensor of the specified type
        Tensor tensor = TensorFactory::zeros(shape, dtype);
        
        // Calculate total size
        size_t total_size = 1;
        for (auto dim : shape) {
            total_size *= dim;
        }
        
        // Ensure we have enough values or can repeat them
        size_t values_size = values.size();
        if (values_size == 0) {
            return tensor;
        }
        
        // Fill tensor with values, repeating if necessary
        if (dtype == DataType::F32) {
            float* data = static_cast<float*>(tensor.data());
            for (size_t i = 0; i < total_size; ++i) {
                data[i] = static_cast<float>(values[i % values_size]);
            }
        } else if (dtype == DataType::I32) {
            int32_t* data = static_cast<int32_t*>(tensor.data());
            for (size_t i = 0; i < total_size; ++i) {
                data[i] = static_cast<int32_t>(values[i % values_size]);
            }
        }
        // Other types would need special handling for quantization
        
        return tensor;
    }
    
    // Specialized version for quantized tensors that need to go through the cast operation
    Tensor createQuantizedTensor(const std::vector<size_t>& shape, DataType dtype, const std::vector<float>& values) {
        // First create an F32 tensor
        Tensor f32_tensor = createTensorWithValues(shape, DataType::F32, values);
        
        // Then cast to the desired quantized format
        return context->cast(f32_tensor, dtype);
    }
    
    // Helper to compare tensors of different types
    bool compareTensors(const Tensor& a, const Tensor& b, float tolerance = 1e-5f) {
        // Check shapes match
        if (a.ndim() != b.ndim()) {
            return false;
        }
        
        for (int i = 0; i < a.ndim(); ++i) {
            if (a.shape(i) != b.shape(i)) {
                return false;
            }
        }
        
        // For real implementation, we would need to compare the actual values
        // by dequantizing if necessary. This is a placeholder.
        return true;
    }
};

// Test basic creation of quantized tensors
TEST_F(QuantizedTensorOperationsTest, QuantizedTensorCreation) {
    // Test creating tensors with different quantized types
    Tensor q8_0_tensor = TensorFactory::zeros(vector_shape, DataType::Q8_0);
    Tensor q4_0_tensor = TensorFactory::zeros(vector_shape, DataType::Q4_0);
    Tensor q4_1_tensor = TensorFactory::zeros(vector_shape, DataType::Q4_1);
    
    // Verify types
    EXPECT_EQ(q8_0_tensor.dtype(), DataType::Q8_0);
    EXPECT_EQ(q4_0_tensor.dtype(), DataType::Q4_0);
    EXPECT_EQ(q4_1_tensor.dtype(), DataType::Q4_1);
    
    // Verify shapes
    EXPECT_EQ(q8_0_tensor.shape(), vector_shape);
    EXPECT_EQ(q4_0_tensor.shape(), vector_shape);
    EXPECT_EQ(q4_1_tensor.shape(), vector_shape);
}

// Test casting between regular and quantized types
TEST_F(QuantizedTensorOperationsTest, QuantizationAndDequantization) {
    // Create a float tensor with known values
    std::vector<float> values = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f};
    Tensor f32_tensor = createTensorWithValues(vector_shape, DataType::F32, values);
    
    // Quantize to different formats
    Tensor q8_0_tensor = context->cast(f32_tensor, DataType::Q8_0);
    Tensor q4_0_tensor = context->cast(f32_tensor, DataType::Q4_0);
    Tensor q4_1_tensor = context->cast(f32_tensor, DataType::Q4_1);
    
    // Verify types
    EXPECT_EQ(q8_0_tensor.dtype(), DataType::Q8_0);
    EXPECT_EQ(q4_0_tensor.dtype(), DataType::Q4_0);
    EXPECT_EQ(q4_1_tensor.dtype(), DataType::Q4_1);
    
    // Dequantize back to F32
    Tensor f32_from_q8_0 = context->cast(q8_0_tensor, DataType::F32);
    Tensor f32_from_q4_0 = context->cast(q4_0_tensor, DataType::F32);
    Tensor f32_from_q4_1 = context->cast(q4_1_tensor, DataType::F32);
    
    // Verify types of dequantized tensors
    EXPECT_EQ(f32_from_q8_0.dtype(), DataType::F32);
    EXPECT_EQ(f32_from_q4_0.dtype(), DataType::F32);
    EXPECT_EQ(f32_from_q4_1.dtype(), DataType::F32);
    
    // In a real implementation, we would check quantization error here
    // This is a placeholder
}

// Test basic operations with quantized tensors
TEST_F(QuantizedTensorOperationsTest, BasicQuantizedOperations) {
    // Create quantized tensors with data
    std::vector<float> values1 = {-1.0f, 0.0f, 1.0f, 2.0f, 3.0f};
    std::vector<float> values2 = {0.5f, 1.0f, 1.5f, 2.0f, 2.5f};
    
    // We need to create smaller tensors for this test to avoid timeouts
    std::vector<size_t> small_shape = {5};
    
    Tensor q8_0_tensor1 = createQuantizedTensor(small_shape, DataType::Q8_0, values1);
    Tensor q8_0_tensor2 = createQuantizedTensor(small_shape, DataType::Q8_0, values2);
    
    // Test addition of quantized tensors
    Tensor add_result = context->add(q8_0_tensor1, q8_0_tensor2);
    EXPECT_EQ(add_result.dtype(), DataType::Q8_0);
    
    // Test multiplication of quantized tensors
    Tensor mul_result = context->multiply(q8_0_tensor1, q8_0_tensor2);
    EXPECT_EQ(mul_result.dtype(), DataType::Q8_0);
    
    // Test operations between quantized and float tensors
    Tensor f32_tensor = createTensorWithValues(small_shape, DataType::F32, values2);
    
    // Quantized + Float should result in Float
    Tensor mixed_add = context->add(q8_0_tensor1, f32_tensor);
    EXPECT_EQ(mixed_add.dtype(), DataType::F32);
    
    // Quantized * Float should result in Float
    Tensor mixed_mul = context->multiply(q8_0_tensor1, f32_tensor);
    EXPECT_EQ(mixed_mul.dtype(), DataType::F32);
}

// Test mixed operations between different quantized formats
TEST_F(QuantizedTensorOperationsTest, MixedQuantizedFormats) {
    // Create tensors with different quantized formats
    std::vector<float> values = {-1.0f, 0.0f, 1.0f, 2.0f, 3.0f};
    std::vector<size_t> small_shape = {5};
    
    Tensor q8_0_tensor = createQuantizedTensor(small_shape, DataType::Q8_0, values);
    Tensor q4_0_tensor = createQuantizedTensor(small_shape, DataType::Q4_0, values);
    Tensor q4_1_tensor = createQuantizedTensor(small_shape, DataType::Q4_1, values);
    
    // Test operations between different quantized formats
    // Q8_0 + Q4_0 should result in Q8_0 (higher precision)
    Tensor add_result1 = context->add(q8_0_tensor, q4_0_tensor);
    EXPECT_EQ(add_result1.dtype(), DataType::Q8_0);
    
    // Q4_1 + Q4_0 should result in Q4_1 (better precision with bias)
    Tensor add_result2 = context->add(q4_1_tensor, q4_0_tensor);
    EXPECT_EQ(add_result2.dtype(), DataType::Q4_1);
    
    // Q8_0 * Q4_1 should result in Q8_0 (higher precision)
    Tensor mul_result = context->multiply(q8_0_tensor, q4_1_tensor);
    EXPECT_EQ(mul_result.dtype(), DataType::Q8_0);
}

// Test matrix operations with quantized tensors
TEST_F(QuantizedTensorOperationsTest, QuantizedMatrixOperations) {
    // Create matrices with quantized formats
    std::vector<float> values1;
    std::vector<float> values2;
    
    // Generate some test values
    for (int i = 0; i < 15; ++i) {
        values1.push_back((i - 7) / 7.0f); // Range: -1.0 to 1.0
        values2.push_back((i - 5) / 5.0f); // Range: -1.0 to 1.8
    }
    
    // Create smaller matrices for this test
    std::vector<size_t> matrix_shape1 = {3, 5};
    std::vector<size_t> matrix_shape2 = {5, 2};
    
    Tensor f32_matrix1 = createTensorWithValues(matrix_shape1, DataType::F32, values1);
    Tensor q8_0_matrix = createQuantizedTensor(matrix_shape2, DataType::Q8_0, values2);
    
    // Test matrix multiplication with mixed precision
    // F32 @ Q8_0 should result in F32
    Tensor mm_result = context->matmul(f32_matrix1, q8_0_matrix);
    EXPECT_EQ(mm_result.dtype(), DataType::F32);
    EXPECT_EQ(mm_result.shape(0), 3);
    EXPECT_EQ(mm_result.shape(1), 2);
    
    // Create quantized matrices for matmul
    Tensor q8_0_matrix1 = createQuantizedTensor(matrix_shape1, DataType::Q8_0, values1);
    
    // Q8_0 @ Q8_0 should result in Q8_0
    Tensor mm_result2 = context->matmul(q8_0_matrix1, q8_0_matrix);
    EXPECT_EQ(mm_result2.dtype(), DataType::Q8_0);
}

// Test activation functions with quantized tensors
TEST_F(QuantizedTensorOperationsTest, QuantizedActivations) {
    // Create a quantized tensor
    std::vector<float> values = {-2.0f, -1.5f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f};
    Tensor q8_0_tensor = createQuantizedTensor(vector_shape, DataType::Q8_0, values);
    
    // Test ReLU (should be able to be done in quantized format)
    Tensor relu_result = context->relu(q8_0_tensor);
    EXPECT_EQ(relu_result.dtype(), DataType::Q8_0);
    
    // Test SiLU (implementation dependent, might dequantize)
    Tensor silu_result = context->silu(q8_0_tensor);
    // We don't assert on the exact type since it's implementation-dependent
    
    // Test GELU (implementation dependent, might dequantize)
    Tensor gelu_result = context->gelu(q8_0_tensor);
    // We don't assert on the exact type since it's implementation-dependent
    
    // Test softmax (implementation dependent, might dequantize)
    Tensor softmax_result = context->softmax(q8_0_tensor, 0);
    // We don't assert on the exact type since it's implementation-dependent
}

// Test quantized tensor with edge case values
TEST_F(QuantizedTensorOperationsTest, QuantizedEdgeCases) {
    // Test with extreme values that might be challenging for quantization
    std::vector<float> edge_values = {
        std::numeric_limits<float>::lowest(),  // Extremely negative
        -1e5f,                                // Large negative
        -1.0f,                                // Typical negative
        -1e-5f,                               // Small negative
        0.0f,                                 // Zero
        1e-5f,                                // Small positive
        1.0f,                                 // Typical positive
        1e5f,                                 // Large positive
        std::numeric_limits<float>::max()     // Extremely positive
    };
    
    std::vector<size_t> small_shape = {9};  // To match the number of edge values
    
    // Create F32 tensor with edge values
    Tensor f32_tensor = createTensorWithValues(small_shape, DataType::F32, edge_values);
    
    // Quantize to different formats
    Tensor q8_0_tensor = context->cast(f32_tensor, DataType::Q8_0);
    Tensor q4_0_tensor = context->cast(f32_tensor, DataType::Q4_0);
    Tensor q4_1_tensor = context->cast(f32_tensor, DataType::Q4_1);
    
    // Dequantize back to F32
    Tensor f32_from_q8_0 = context->cast(q8_0_tensor, DataType::F32);
    Tensor f32_from_q4_0 = context->cast(q4_0_tensor, DataType::F32);
    Tensor f32_from_q4_1 = context->cast(q4_1_tensor, DataType::F32);
    
    // In a real implementation, we would check quantization error for each value
    // This is a placeholder
}

// Test quantized tensor reductions
TEST_F(QuantizedTensorOperationsTest, QuantizedReductions) {
    // Create a quantized tensor with structured values
    std::vector<float> values;
    for (int i = 0; i < 20; ++i) {
        values.push_back(i / 10.0f);  // Range: 0.0 to 1.9
    }
    
    std::vector<size_t> matrix_shape = {4, 5};  // 4x5 matrix
    
    Tensor q8_0_tensor = createQuantizedTensor(matrix_shape, DataType::Q8_0, values);
    
    // Test sum reduction along rows
    Tensor sum_rows = context->sum(q8_0_tensor, 0);  // Result should be a 5-element vector
    EXPECT_EQ(sum_rows.ndim(), 1);
    EXPECT_EQ(sum_rows.shape(0), 5);
    
    // Test sum reduction along columns
    Tensor sum_cols = context->sum(q8_0_tensor, 1);  // Result should be a 4-element vector
    EXPECT_EQ(sum_cols.ndim(), 1);
    EXPECT_EQ(sum_cols.shape(0), 4);
    
    // Test mean reduction
    Tensor mean_result = context->mean(q8_0_tensor, 0);
    EXPECT_EQ(mean_result.ndim(), 1);
    EXPECT_EQ(mean_result.shape(0), 5);
}

// Test performance comparison between quantized and full precision operations
TEST_F(QuantizedTensorOperationsTest, QuantizedPerformance) {
    // This test is mostly a placeholder for benchmark-style tests
    // In a real test, we would measure and compare performance
    
    // Create larger tensors for performance testing
    std::vector<size_t> large_shape = {100, 100};
    std::vector<float> values;
    for (int i = 0; i < 100; ++i) {
        values.push_back(i / 50.0f - 1.0f);  // Range: -1.0 to 1.0
    }
    
    Tensor f32_tensor = createTensorWithValues(large_shape, DataType::F32, values);
    
    // Skip performance testing if context has a simple implementation
    // This is just a placeholder
    
    // In a real test, we would:
    // 1. Create F32 and quantized versions
    // 2. Measure time for various operations
    // 3. Compare and assert that quantized ops are faster
}