#include <gtest/gtest.h>
#include <ccsm/tensor.h>
#include <ccsm/cpu/ggml_tensor.h>
#include <vector>
#include <memory>
#include <cmath>

using namespace ccsm;

// Test fixture for tensor tests
class TensorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a GGML context for testing
        context = std::make_shared<GGMLContext>();
    }
    
    std::shared_ptr<GGMLContext> context;
};

// Test tensor creation and basic properties
TEST_F(TensorTest, CreateTensor) {
    // Create a tensor
    Tensor tensor = context->zeros({2, 3, 4}, DataType::F32);
    
    // Check properties
    EXPECT_TRUE(tensor.is_valid());
    EXPECT_EQ(tensor.ndim(), 3);
    EXPECT_EQ(tensor.shape(0), 2);
    EXPECT_EQ(tensor.shape(1), 3);
    EXPECT_EQ(tensor.shape(2), 4);
    EXPECT_EQ(tensor.size(), 24);
    EXPECT_EQ(tensor.dtype(), DataType::F32);
    EXPECT_EQ(tensor.dtype_str(), "F32");
    
    // Check shape vector
    std::vector<size_t> shape = tensor.shape();
    EXPECT_EQ(shape.size(), 3);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    EXPECT_EQ(shape[2], 4);
}

// Test tensor initialization with zeros
TEST_F(TensorTest, ZerosTensor) {
    // Create a tensor filled with zeros
    Tensor tensor = context->zeros({2, 3}, DataType::F32);
    
    // Check that all elements are zero
    float* data = static_cast<float*>(tensor.data());
    for (size_t i = 0; i < tensor.size(); i++) {
        EXPECT_FLOAT_EQ(data[i], 0.0f);
    }
}

// Test tensor initialization with ones
TEST_F(TensorTest, OnesTensor) {
    // Create a tensor filled with ones
    Tensor tensor = context->ones({2, 3}, DataType::F32);
    
    // Check that all elements are one
    float* data = static_cast<float*>(tensor.data());
    for (size_t i = 0; i < tensor.size(); i++) {
        EXPECT_FLOAT_EQ(data[i], 1.0f);
    }
}

// Test tensor reshape operation
TEST_F(TensorTest, ReshapeTensor) {
    // Create a tensor
    Tensor tensor = context->ones({2, 3, 4}, DataType::F32);
    
    // Reshape tensor
    Tensor reshaped = tensor.reshape({4, 6});
    
    // Check properties of reshaped tensor
    EXPECT_TRUE(reshaped.is_valid());
    EXPECT_EQ(reshaped.ndim(), 2);
    EXPECT_EQ(reshaped.shape(0), 4);
    EXPECT_EQ(reshaped.shape(1), 6);
    EXPECT_EQ(reshaped.size(), 24);
    EXPECT_EQ(reshaped.dtype(), DataType::F32);
    
    // Check that data is preserved
    float* data = static_cast<float*>(reshaped.data());
    for (size_t i = 0; i < reshaped.size(); i++) {
        EXPECT_FLOAT_EQ(data[i], 1.0f);
    }
}

// Test tensor slice operation
TEST_F(TensorTest, SliceTensor) {
    // Create a tensor
    Tensor tensor = context->zeros({5, 5}, DataType::F32);
    
    // Fill with increasing values
    float* data = static_cast<float*>(tensor.data());
    for (size_t i = 0; i < tensor.size(); i++) {
        data[i] = static_cast<float>(i);
    }
    
    // Slice tensor
    Tensor sliced = tensor.slice(0, 1, 4);
    
    // Check properties of sliced tensor
    EXPECT_TRUE(sliced.is_valid());
    EXPECT_EQ(sliced.ndim(), 2);
    EXPECT_EQ(sliced.shape(0), 3); // 4 - 1 = 3 elements
    EXPECT_EQ(sliced.shape(1), 5);
    EXPECT_EQ(sliced.size(), 15);
    EXPECT_EQ(sliced.dtype(), DataType::F32);
    
    // Check that correct portion was sliced
    // TODO: The actual check will depend on how tensor data is laid out in memory
    // This is a simplified check assuming row-major order
    float* sliced_data = static_cast<float*>(sliced.data());
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 5; j++) {
            float expected = (i + 1) * 5 + j; // Adjusted for slice starting at 1
            EXPECT_FLOAT_EQ(sliced_data[i * 5 + j], expected);
        }
    }
}

// Test basic tensor operations
TEST_F(TensorTest, BasicOperations) {
    // Create tensors
    Tensor a = context->ones({2, 3}, DataType::F32);
    Tensor b = context->ones({2, 3}, DataType::F32);
    
    // Fill with specific values
    float* a_data = static_cast<float*>(a.data());
    float* b_data = static_cast<float*>(b.data());
    for (size_t i = 0; i < a.size(); i++) {
        a_data[i] = 2.0f;
        b_data[i] = 3.0f;
    }
    
    // Test add
    Tensor sum = context->add(a, b);
    EXPECT_TRUE(sum.is_valid());
    EXPECT_EQ(sum.shape(0), 2);
    EXPECT_EQ(sum.shape(1), 3);
    
    // Test subtract
    Tensor diff = context->subtract(a, b);
    EXPECT_TRUE(diff.is_valid());
    EXPECT_EQ(diff.shape(0), 2);
    EXPECT_EQ(diff.shape(1), 3);
    
    // Test multiply
    Tensor prod = context->multiply(a, b);
    EXPECT_TRUE(prod.is_valid());
    EXPECT_EQ(prod.shape(0), 2);
    EXPECT_EQ(prod.shape(1), 3);
    
    // Test divide
    Tensor quot = context->divide(a, b);
    EXPECT_TRUE(quot.is_valid());
    EXPECT_EQ(quot.shape(0), 2);
    EXPECT_EQ(quot.shape(1), 3);
}

// Test matrix multiplication
TEST_F(TensorTest, MatrixMultiplication) {
    // Create matrices
    Tensor a = context->ones({2, 3}, DataType::F32);
    Tensor b = context->ones({3, 4}, DataType::F32);
    
    // Fill with values
    float* a_data = static_cast<float*>(a.data());
    float* b_data = static_cast<float*>(b.data());
    for (size_t i = 0; i < a.size(); i++) {
        a_data[i] = 1.0f;
    }
    for (size_t i = 0; i < b.size(); i++) {
        b_data[i] = 1.0f;
    }
    
    // Matrix multiply
    Tensor c = context->matmul(a, b);
    
    // Check result shape
    EXPECT_TRUE(c.is_valid());
    EXPECT_EQ(c.shape(0), 2);
    EXPECT_EQ(c.shape(1), 4);
    
    // TODO: Verify values when computation graph is working
}

// Test activation functions
TEST_F(TensorTest, ActivationFunctions) {
    // Create tensor
    Tensor x = context->zeros({2, 3}, DataType::F32);
    
    // Fill with values from -2 to 2
    float* data = static_cast<float*>(x.data());
    for (size_t i = 0; i < x.size(); i++) {
        data[i] = -2.0f + 4.0f * static_cast<float>(i) / static_cast<float>(x.size() - 1);
    }
    
    // Test ReLU
    Tensor relu_result = context->relu(x);
    EXPECT_TRUE(relu_result.is_valid());
    EXPECT_EQ(relu_result.shape(0), 2);
    EXPECT_EQ(relu_result.shape(1), 3);
    
    // Test GELU
    Tensor gelu_result = context->gelu(x);
    EXPECT_TRUE(gelu_result.is_valid());
    EXPECT_EQ(gelu_result.shape(0), 2);
    EXPECT_EQ(gelu_result.shape(1), 3);
    
    // Test SiLU
    Tensor silu_result = context->silu(x);
    EXPECT_TRUE(silu_result.is_valid());
    EXPECT_EQ(silu_result.shape(0), 2);
    EXPECT_EQ(silu_result.shape(1), 3);
    
    // Test softmax
    Tensor softmax_result = context->softmax(x, 1);
    EXPECT_TRUE(softmax_result.is_valid());
    EXPECT_EQ(softmax_result.shape(0), 2);
    EXPECT_EQ(softmax_result.shape(1), 3);
    
    // TODO: Verify values when computation graph is working
}

// Main function for running tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}