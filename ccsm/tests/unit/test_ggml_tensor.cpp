#include <gtest/gtest.h>
#include <ccsm/tensor.h>
#include <ccsm/cpu/ggml_tensor.h>
#include <vector>
#include <memory>
#include <cmath>
#include <stdexcept>
#include <random>

using namespace ccsm;

// Test fixture for GGML tensor tests
class GGMLTensorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up default tensor shapes for testing
        scalar_shape = {};
        vector_shape = {10};
        matrix_shape = {5, 10};
        tensor3d_shape = {3, 5, 10};
        
        // Default data type
        default_dtype = DataType::F32;
        
        // Create context
        context = std::make_shared<GGMLContext>();
    }
    
    // Helper to create test data
    template <typename T>
    std::vector<T> generate_test_data(size_t size) {
        std::vector<T> data(size);
        for (size_t i = 0; i < size; i++) {
            data[i] = static_cast<T>(i);
        }
        return data;
    }
    
    // Helper to compare tensor data with expected values
    template <typename T>
    bool compare_tensor_data(const Tensor& tensor, const std::vector<T>& expected) {
        if (tensor.size() != expected.size()) {
            return false;
        }
        
        const T* data = static_cast<const T*>(tensor.data());
        for (size_t i = 0; i < expected.size(); i++) {
            if (std::fabs(data[i] - expected[i]) > 1e-5) {
                return false;
            }
        }
        return true;
    }
    
    std::vector<size_t> scalar_shape;
    std::vector<size_t> vector_shape;
    std::vector<size_t> matrix_shape;
    std::vector<size_t> tensor3d_shape;
    DataType default_dtype;
    std::shared_ptr<GGMLContext> context;
};

// Test GGML tensor creation
TEST_F(GGMLTensorTest, TensorCreation) {
    // Test vector creation
    Tensor vector = context->zeros(vector_shape, DataType::F32);
    EXPECT_TRUE(vector.is_valid());
    EXPECT_EQ(vector.ndim(), 1);
    EXPECT_EQ(vector.shape(0), 10);
    EXPECT_EQ(vector.size(), 10);
    EXPECT_EQ(vector.dtype(), DataType::F32);
    EXPECT_EQ(vector.dtype_str(), "F32");
    
    // Test matrix creation
    Tensor matrix = context->zeros(matrix_shape, DataType::F16);
    EXPECT_TRUE(matrix.is_valid());
    EXPECT_EQ(matrix.ndim(), 2);
    EXPECT_EQ(matrix.shape(0), 5);
    EXPECT_EQ(matrix.shape(1), 10);
    EXPECT_EQ(matrix.size(), 50);
    EXPECT_EQ(matrix.dtype(), DataType::F16);
    EXPECT_EQ(matrix.dtype_str(), "F16");
    
    // Test 3D tensor creation
    Tensor tensor3d = context->zeros(tensor3d_shape, DataType::F32);
    EXPECT_TRUE(tensor3d.is_valid());
    EXPECT_EQ(tensor3d.ndim(), 3);
    EXPECT_EQ(tensor3d.shape(0), 3);
    EXPECT_EQ(tensor3d.shape(1), 5);
    EXPECT_EQ(tensor3d.shape(2), 10);
    EXPECT_EQ(tensor3d.size(), 150);
    EXPECT_EQ(tensor3d.dtype(), DataType::F32);
    EXPECT_EQ(tensor3d.dtype_str(), "F32");
    
    // Test ones tensor
    Tensor ones = context->ones(vector_shape, DataType::F32);
    EXPECT_TRUE(ones.is_valid());
    EXPECT_EQ(ones.dtype(), DataType::F32);
    
    // Check that data pointers are accessible
    EXPECT_NE(ones.data(), nullptr);
    EXPECT_NE(static_cast<const Tensor&>(ones).data(), nullptr);
    
    // Verify ones contains all 1.0
    const float* data = static_cast<const float*>(ones.data());
    for (size_t i = 0; i < ones.size(); i++) {
        EXPECT_FLOAT_EQ(data[i], 1.0f);
    }
}

// Test GGML tensor data type conversions
TEST_F(GGMLTensorTest, DataTypeConversion) {
    // Test conversion from ccsm::DataType to ggml_type
    EXPECT_EQ(GGMLTensorImpl::to_ggml_type(DataType::F32), GGML_TYPE_F32);
    EXPECT_EQ(GGMLTensorImpl::to_ggml_type(DataType::F16), GGML_TYPE_F16);
    EXPECT_EQ(GGMLTensorImpl::to_ggml_type(DataType::I32), GGML_TYPE_I32);
    EXPECT_EQ(GGMLTensorImpl::to_ggml_type(DataType::I16), GGML_TYPE_I16);
    EXPECT_EQ(GGMLTensorImpl::to_ggml_type(DataType::I8), GGML_TYPE_I8);
    EXPECT_EQ(GGMLTensorImpl::to_ggml_type(DataType::Q8_0), GGML_TYPE_Q8_0);
    EXPECT_EQ(GGMLTensorImpl::to_ggml_type(DataType::Q4_0), GGML_TYPE_Q4_0);
    EXPECT_EQ(GGMLTensorImpl::to_ggml_type(DataType::Q4_1), GGML_TYPE_Q4_1);
    EXPECT_EQ(GGMLTensorImpl::to_ggml_type(DataType::BF16), GGML_TYPE_F16); // BF16 fallback to F16
    
    // Test conversion from ggml_type to ccsm::DataType
    EXPECT_EQ(GGMLTensorImpl::from_ggml_type(GGML_TYPE_F32), DataType::F32);
    EXPECT_EQ(GGMLTensorImpl::from_ggml_type(GGML_TYPE_F16), DataType::F16);
    EXPECT_EQ(GGMLTensorImpl::from_ggml_type(GGML_TYPE_I32), DataType::I32);
    EXPECT_EQ(GGMLTensorImpl::from_ggml_type(GGML_TYPE_I16), DataType::I16);
    EXPECT_EQ(GGMLTensorImpl::from_ggml_type(GGML_TYPE_I8), DataType::I8);
    EXPECT_EQ(GGMLTensorImpl::from_ggml_type(GGML_TYPE_Q8_0), DataType::Q8_0);
    EXPECT_EQ(GGMLTensorImpl::from_ggml_type(GGML_TYPE_Q4_0), DataType::Q4_0);
    EXPECT_EQ(GGMLTensorImpl::from_ggml_type(GGML_TYPE_Q4_1), DataType::Q4_1);
    
    // Test invalid conversions
    EXPECT_THROW(GGMLTensorImpl::from_ggml_type(GGML_TYPE_COUNT), std::runtime_error);
    
    // Check that tensors created with different types have correct dtype
    Tensor t_f32 = context->zeros({5}, DataType::F32);
    EXPECT_EQ(t_f32.dtype(), DataType::F32);
    
    Tensor t_f16 = context->zeros({5}, DataType::F16);
    EXPECT_EQ(t_f16.dtype(), DataType::F16);
    
    Tensor t_i32 = context->zeros({5}, DataType::I32);
    EXPECT_EQ(t_i32.dtype(), DataType::I32);
}

// Test GGML tensor reshape and view operations
TEST_F(GGMLTensorTest, ReshapeAndView) {
    // Create a 3D tensor with known data
    Tensor original = context->zeros(tensor3d_shape, default_dtype);
    EXPECT_TRUE(original.is_valid());
    EXPECT_EQ(original.ndim(), 3);
    EXPECT_EQ(original.size(), 150);
    
    // Fill with pattern
    float* data = static_cast<float*>(original.data());
    for (size_t i = 0; i < original.size(); i++) {
        data[i] = static_cast<float>(i);
    }
    
    // Reshape to 2D
    std::vector<size_t> new_shape = {15, 10};
    Tensor reshaped = original.reshape(new_shape);
    EXPECT_TRUE(reshaped.is_valid());
    EXPECT_EQ(reshaped.ndim(), 2);
    EXPECT_EQ(reshaped.shape(0), 15);
    EXPECT_EQ(reshaped.shape(1), 10);
    EXPECT_EQ(reshaped.size(), 150);
    
    // Verify data is preserved in reshape
    const float* reshaped_data = static_cast<const float*>(reshaped.data());
    for (size_t i = 0; i < original.size(); i++) {
        EXPECT_FLOAT_EQ(reshaped_data[i], static_cast<float>(i));
    }
    
    // Reshape to 1D
    new_shape = {150};
    Tensor reshaped_flat = original.reshape(new_shape);
    EXPECT_TRUE(reshaped_flat.is_valid());
    EXPECT_EQ(reshaped_flat.ndim(), 1);
    EXPECT_EQ(reshaped_flat.shape(0), 150);
    EXPECT_EQ(reshaped_flat.size(), 150);
    
    // Verify data is preserved in flatten
    const float* flat_data = static_cast<const float*>(reshaped_flat.data());
    for (size_t i = 0; i < original.size(); i++) {
        EXPECT_FLOAT_EQ(flat_data[i], static_cast<float>(i));
    }
    
    // View as 2D (should behave the same as reshape for GGML)
    new_shape = {15, 10};
    Tensor viewed = original.view(new_shape);
    EXPECT_TRUE(viewed.is_valid());
    EXPECT_EQ(viewed.ndim(), 2);
    EXPECT_EQ(viewed.shape(0), 15);
    EXPECT_EQ(viewed.shape(1), 10);
    EXPECT_EQ(viewed.size(), 150);
    
    // Invalid reshape (size mismatch) should throw
    new_shape = {10, 10};
    EXPECT_THROW(original.reshape(new_shape), std::invalid_argument);
}

// Test GGML tensor slice operations
TEST_F(GGMLTensorTest, Slicing) {
    // Create a 3D tensor with known data
    Tensor original = context->zeros(tensor3d_shape, default_dtype);
    EXPECT_TRUE(original.is_valid());
    
    // Fill with pattern
    float* data = static_cast<float*>(original.data());
    for (size_t i = 0; i < original.size(); i++) {
        data[i] = static_cast<float>(i);
    }
    
    // Slice along dim 0
    Tensor slice0 = original.slice(0, 1, 2);
    EXPECT_TRUE(slice0.is_valid());
    EXPECT_EQ(slice0.ndim(), 3);
    EXPECT_EQ(slice0.shape(0), 1);  // sliced dimension
    EXPECT_EQ(slice0.shape(1), 5);  // unchanged
    EXPECT_EQ(slice0.shape(2), 10); // unchanged
    
    // Verify sliced data
    const float* slice0_data = static_cast<const float*>(slice0.data());
    for (size_t i = 0; i < slice0.size(); i++) {
        // Calculate the expected value based on the original tensor pattern
        size_t offset = 1 * (5 * 10); // 1 is the start index in dim 0
        EXPECT_FLOAT_EQ(slice0_data[i], static_cast<float>(offset + i));
    }
    
    // Slice along dim 1
    Tensor slice1 = original.slice(1, 2, 4);
    EXPECT_TRUE(slice1.is_valid());
    EXPECT_EQ(slice1.ndim(), 3);
    EXPECT_EQ(slice1.shape(0), 3);  // unchanged
    EXPECT_EQ(slice1.shape(1), 2);  // sliced dimension
    EXPECT_EQ(slice1.shape(2), 10); // unchanged
    
    // Invalid slice operations should throw
    EXPECT_THROW(original.slice(3, 0, 1), std::invalid_argument); // dim out of range
    EXPECT_THROW(original.slice(0, 5, 1), std::invalid_argument); // end before start
    EXPECT_THROW(original.slice(0, 0, 10), std::invalid_argument); // out of bounds
}

// Test GGML context tensor operations
TEST_F(GGMLTensorTest, ContextOperations) {
    // Create input tensors
    Tensor a = context->zeros(vector_shape, default_dtype);
    Tensor b = context->zeros(vector_shape, default_dtype);
    
    // Fill with data
    float* a_data = static_cast<float*>(a.data());
    float* b_data = static_cast<float*>(b.data());
    
    for (size_t i = 0; i < a.size(); i++) {
        a_data[i] = static_cast<float>(i);
        b_data[i] = 1.0f;
    }
    
    // Test add
    Tensor add_result = context->add(a, b);
    EXPECT_TRUE(add_result.is_valid());
    EXPECT_EQ(add_result.shape(), vector_shape);
    
    // Test subtract
    Tensor sub_result = context->subtract(a, b);
    EXPECT_TRUE(sub_result.is_valid());
    EXPECT_EQ(sub_result.shape(), vector_shape);
    
    // Test multiply
    Tensor mul_result = context->multiply(a, b);
    EXPECT_TRUE(mul_result.is_valid());
    EXPECT_EQ(mul_result.shape(), vector_shape);
    
    // Test divide
    Tensor div_result = context->divide(a, b);
    EXPECT_TRUE(div_result.is_valid());
    EXPECT_EQ(div_result.shape(), vector_shape);
    
    // Test matrix multiplication with compatible dimensions
    Tensor c = context->zeros({10, 5}, default_dtype);
    Tensor d = context->zeros({5, 3}, default_dtype);
    
    float* c_data = static_cast<float*>(c.data());
    float* d_data = static_cast<float*>(d.data());
    
    for (size_t i = 0; i < c.size(); i++) {
        c_data[i] = 1.0f;
    }
    
    for (size_t i = 0; i < d.size(); i++) {
        d_data[i] = 1.0f;
    }
    
    Tensor matmul_result = context->matmul(c, d);
    EXPECT_TRUE(matmul_result.is_valid());
    EXPECT_EQ(matmul_result.ndim(), 2);
    EXPECT_EQ(matmul_result.shape(0), 10);
    EXPECT_EQ(matmul_result.shape(1), 3);
    
    // Test activations
    Tensor relu_result = context->relu(a);
    EXPECT_TRUE(relu_result.is_valid());
    EXPECT_EQ(relu_result.shape(), a.shape());
    
    Tensor gelu_result = context->gelu(a);
    EXPECT_TRUE(gelu_result.is_valid());
    EXPECT_EQ(gelu_result.shape(), a.shape());
    
    Tensor silu_result = context->silu(a);
    EXPECT_TRUE(silu_result.is_valid());
    EXPECT_EQ(silu_result.shape(), a.shape());
    
    Tensor softmax_result = context->softmax(a, 0);
    EXPECT_TRUE(softmax_result.is_valid());
    EXPECT_EQ(softmax_result.shape(), a.shape());
    
    // Check backend name
    EXPECT_EQ(context->backend(), "ggml");
}

// Test error handling for GGML operations
TEST_F(GGMLTensorTest, ErrorHandling) {
    // Create tensors with mismatched shapes
    Tensor a = context->zeros(vector_shape, default_dtype);
    Tensor b = context->zeros(matrix_shape, default_dtype);
    
    // Operations between incompatible tensors should fail
    EXPECT_THROW(context->add(a, b), std::runtime_error);
    EXPECT_THROW(context->subtract(a, b), std::runtime_error);
    EXPECT_THROW(context->multiply(a, b), std::runtime_error);
    EXPECT_THROW(context->divide(a, b), std::runtime_error);
    
    // Matrix multiplication with incompatible dimensions should fail
    Tensor c = context->zeros({3, 4}, default_dtype);
    Tensor d = context->zeros({5, 2}, default_dtype);
    EXPECT_THROW(context->matmul(c, d), std::runtime_error);
    
    // Invalid tensor operations
    EXPECT_THROW(a.shape(5), std::out_of_range); // Dimension out of range
}

// Test ggml_alloc_tensor and ggml_compute functions
TEST_F(GGMLTensorTest, AllocationAndComputation) {
    // Test tensor allocation
    int64_t dims[3] = {2, 3, 4};
    struct ggml_tensor* tensor = context->alloc_tensor(GGML_TYPE_F32, 3, dims);
    EXPECT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->type, GGML_TYPE_F32);
    EXPECT_EQ(tensor->ne[0], 2);
    EXPECT_EQ(tensor->ne[1], 3);
    EXPECT_EQ(tensor->ne[2], 4);
    
    // Create a graph (manually since we're not exposing ggml_cgraph directly)
    struct ggml_cgraph* graph = ggml_new_graph(context->ggml_ctx());
    EXPECT_NE(graph, nullptr);
    
    // Add some operation nodes
    Tensor a = context->zeros({5}, default_dtype);
    Tensor b = context->ones({5}, default_dtype);
    Tensor c = context->add(a, b);  // This should add to the graph
    
    // Compute the graph (this should not throw)
    EXPECT_NO_THROW(context->compute(graph));
    
    // Free graph (GGML context will handle tensor cleanup)
    ggml_free_graph(graph);
}

// New: Test performance and thread pool integration
TEST_F(GGMLTensorTest, PerformanceAndThreading) {
    // Create larger tensors for performance testing
    const std::vector<size_t> large_matrix1 = {256, 256};
    const std::vector<size_t> large_matrix2 = {256, 256};
    
    Tensor a = context->ones(large_matrix1, default_dtype);
    Tensor b = context->ones(large_matrix2, default_dtype);
    
    // Measure time for matrix multiplication
    auto start_time = std::chrono::high_resolution_clock::now();
    Tensor result = context->matmul(a, b);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Just verify the operation completed, no specific time threshold
    EXPECT_TRUE(result.is_valid());
    EXPECT_EQ(result.shape(0), 256);
    EXPECT_EQ(result.shape(1), 256);
    
    // Log the time (for information, not assertion)
    std::cout << "Matrix multiplication time: " << duration.count() << "ms" << std::endl;
}

// Main function for running tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}