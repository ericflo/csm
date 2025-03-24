#include <gtest/gtest.h>
#include <ccsm/mlx/mlx_tensor.h>
#include <ccsm/tensor.h>
#include <vector>
#include <memory>

namespace ccsm {
namespace testing {

class MLXTensorTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test if MLX is available
TEST_F(MLXTensorTest, TestMLXAvailability) {
    bool available = MLXContext::is_available();
    
    #ifdef CCSM_WITH_MLX
    // If compiled with MLX, availability depends on the system
    // Skip further tests if MLX is not available
    if (!available) {
        GTEST_SKIP() << "MLX not available, skipping tests";
    }
    #else
    // If not compiled with MLX, it should definitely not be available
    EXPECT_FALSE(available);
    #endif
}

#ifdef CCSM_WITH_MLX
// Only run these tests if MLX is available
TEST_F(MLXTensorTest, TestMLXTensorCreation) {
    // Skip if MLX is not available on this system
    if (!MLXContext::is_available()) {
        GTEST_SKIP() << "MLX not available, skipping test";
    }
    
    // Create an MLX tensor
    std::vector<size_t> shape = {2, 3};
    Tensor tensor = MLXTensorFactory::zeros(shape, DataType::F32);
    
    // Verify tensor properties
    EXPECT_EQ(tensor.ndim(), 2);
    EXPECT_EQ(tensor.shape(0), 2);
    EXPECT_EQ(tensor.shape(1), 3);
    EXPECT_EQ(tensor.size(), 6);
    EXPECT_EQ(tensor.dtype(), DataType::F32);
    
    // Verify tensor implementation type
    EXPECT_EQ(tensor.context()->backend(), "mlx");
}

TEST_F(MLXTensorTest, TestMLXTensorOperations) {
    // Skip if MLX is not available on this system
    if (!MLXContext::is_available()) {
        GTEST_SKIP() << "MLX not available, skipping test";
    }
    
    // Create MLX context
    Context* context = ContextRegistry::get("mlx");
    ASSERT_NE(context, nullptr);
    
    // Create tensor with context
    std::vector<size_t> shape = {2, 2};
    Tensor a = context->ones(shape, DataType::F32);
    Tensor b = context->ones(shape, DataType::F32);
    
    // Basic operations
    Tensor c = context->add(a, b);
    EXPECT_EQ(c.shape(0), 2);
    EXPECT_EQ(c.shape(1), 2);
    
    // Access data and verify
    const float* data = static_cast<const float*>(c.data());
    for (size_t i = 0; i < c.size(); ++i) {
        EXPECT_FLOAT_EQ(data[i], 2.0f);
    }
    
    // Matrix multiplication
    Tensor d = context->matmul(a, b);
    EXPECT_EQ(d.shape(0), 2);
    EXPECT_EQ(d.shape(1), 2);
    
    data = static_cast<const float*>(d.data());
    for (size_t i = 0; i < d.size(); ++i) {
        EXPECT_FLOAT_EQ(data[i], 2.0f);
    }
}

TEST_F(MLXTensorTest, TestMLXTensorReshape) {
    // Skip if MLX is not available on this system
    if (!MLXContext::is_available()) {
        GTEST_SKIP() << "MLX not available, skipping test";
    }
    
    // Create an MLX tensor
    std::vector<size_t> shape = {2, 3};
    Tensor tensor = MLXTensorFactory::zeros(shape, DataType::F32);
    
    // Reshape tensor
    std::vector<size_t> new_shape = {3, 2};
    Tensor reshaped = tensor.reshape(new_shape);
    
    // Verify reshaped tensor properties
    EXPECT_EQ(reshaped.ndim(), 2);
    EXPECT_EQ(reshaped.shape(0), 3);
    EXPECT_EQ(reshaped.shape(1), 2);
    EXPECT_EQ(reshaped.size(), 6);
    EXPECT_EQ(reshaped.dtype(), DataType::F32);
}

TEST_F(MLXTensorTest, TestMLXTensorView) {
    // Skip if MLX is not available on this system
    if (!MLXContext::is_available()) {
        GTEST_SKIP() << "MLX not available, skipping test";
    }
    
    // Create an MLX tensor
    std::vector<size_t> shape = {2, 3};
    Tensor tensor = MLXTensorFactory::zeros(shape, DataType::F32);
    
    // Create a view of the tensor
    std::vector<size_t> new_shape = {3, 2};
    Tensor viewed = tensor.view(new_shape);
    
    // Verify viewed tensor properties
    EXPECT_EQ(viewed.ndim(), 2);
    EXPECT_EQ(viewed.shape(0), 3);
    EXPECT_EQ(viewed.shape(1), 2);
    EXPECT_EQ(viewed.size(), 6);
    EXPECT_EQ(viewed.dtype(), DataType::F32);
    
    // Modify the view and verify that the original tensor is also modified
    float* view_data = static_cast<float*>(viewed.data());
    view_data[0] = 1.0f;
    
    float* tensor_data = static_cast<float*>(tensor.data());
    EXPECT_FLOAT_EQ(tensor_data[0], 1.0f);
}

TEST_F(MLXTensorTest, TestMLXTensorSlice) {
    // Skip if MLX is not available on this system
    if (!MLXContext::is_available()) {
        GTEST_SKIP() << "MLX not available, skipping test";
    }
    
    // Create an MLX tensor with some data
    std::vector<size_t> shape = {4, 3};
    Tensor tensor = MLXTensorFactory::zeros(shape, DataType::F32);
    
    // Fill with sequential data
    float* data = static_cast<float*>(tensor.data());
    for (size_t i = 0; i < tensor.size(); ++i) {
        data[i] = static_cast<float>(i);
    }
    
    // Slice the tensor along the first dimension
    Tensor sliced = tensor.slice(0, 1, 3);
    
    // Verify sliced tensor properties
    EXPECT_EQ(sliced.ndim(), 2);
    EXPECT_EQ(sliced.shape(0), 2);
    EXPECT_EQ(sliced.shape(1), 3);
    EXPECT_EQ(sliced.size(), 6);
    EXPECT_EQ(sliced.dtype(), DataType::F32);
    
    // Verify slice data
    const float* slice_data = static_cast<const float*>(sliced.data());
    for (size_t i = 0; i < sliced.size(); ++i) {
        EXPECT_FLOAT_EQ(slice_data[i], static_cast<float>(i + 3));
    }
}

TEST_F(MLXTensorTest, TestMLXActivations) {
    // Skip if MLX is not available on this system
    if (!MLXContext::is_available()) {
        GTEST_SKIP() << "MLX not available, skipping test";
    }
    
    // Create MLX context
    Context* context = ContextRegistry::get("mlx");
    ASSERT_NE(context, nullptr);
    
    // Create tensor with context
    std::vector<size_t> shape = {2, 2};
    Tensor tensor = context->zeros(shape, DataType::F32);
    
    // Fill with test data
    float* data = static_cast<float*>(tensor.data());
    data[0] = -1.0f;
    data[1] = 0.0f;
    data[2] = 1.0f;
    data[3] = 2.0f;
    
    // Test ReLU activation
    Tensor relu_result = context->relu(tensor);
    const float* relu_data = static_cast<const float*>(relu_result.data());
    EXPECT_FLOAT_EQ(relu_data[0], 0.0f);
    EXPECT_FLOAT_EQ(relu_data[1], 0.0f);
    EXPECT_FLOAT_EQ(relu_data[2], 1.0f);
    EXPECT_FLOAT_EQ(relu_data[3], 2.0f);
    
    // Test SiLU (Swish) activation
    Tensor silu_result = context->silu(tensor);
    const float* silu_data = static_cast<const float*>(silu_result.data());
    EXPECT_NEAR(silu_data[0], -0.268f, 0.001f);
    EXPECT_NEAR(silu_data[1], 0.0f, 0.001f);
    EXPECT_NEAR(silu_data[2], 0.731f, 0.001f);
    EXPECT_NEAR(silu_data[3], 1.762f, 0.001f);
    
    // Test GELU activation
    Tensor gelu_result = context->gelu(tensor);
    const float* gelu_data = static_cast<const float*>(gelu_result.data());
    EXPECT_NEAR(gelu_data[0], -0.158f, 0.001f);
    EXPECT_NEAR(gelu_data[1], 0.0f, 0.001f);
    EXPECT_NEAR(gelu_data[2], 0.841f, 0.001f);
    EXPECT_NEAR(gelu_data[3], 1.954f, 0.001f);
}

TEST_F(MLXTensorTest, TestMLXTensorConversion) {
    // Skip if MLX is not available on this system
    if (!MLXContext::is_available()) {
        GTEST_SKIP() << "MLX not available, skipping test";
    }
    
    // Create a CPU tensor
    std::vector<size_t> shape = {2, 3};
    Tensor cpu_tensor = TensorFactory::ones(shape, DataType::F32);
    
    // Convert to MLX tensor
    Tensor mlx_tensor = MLXTensorFactory::from_tensor(cpu_tensor);
    
    // Verify MLX tensor properties
    EXPECT_EQ(mlx_tensor.ndim(), 2);
    EXPECT_EQ(mlx_tensor.shape(0), 2);
    EXPECT_EQ(mlx_tensor.shape(1), 3);
    EXPECT_EQ(mlx_tensor.size(), 6);
    EXPECT_EQ(mlx_tensor.dtype(), DataType::F32);
    EXPECT_EQ(mlx_tensor.context()->backend(), "mlx");
    
    // Convert back to CPU tensor
    Tensor reconverted = TensorFactory::from_tensor(mlx_tensor);
    
    // Verify reconverted tensor properties
    EXPECT_EQ(reconverted.ndim(), 2);
    EXPECT_EQ(reconverted.shape(0), 2);
    EXPECT_EQ(reconverted.shape(1), 3);
    EXPECT_EQ(reconverted.size(), 6);
    EXPECT_EQ(reconverted.dtype(), DataType::F32);
    EXPECT_NE(reconverted.context()->backend(), "mlx");
    
    // Verify data consistency
    const float* cpu_data = static_cast<const float*>(cpu_tensor.data());
    const float* reconverted_data = static_cast<const float*>(reconverted.data());
    for (size_t i = 0; i < cpu_tensor.size(); ++i) {
        EXPECT_FLOAT_EQ(cpu_data[i], reconverted_data[i]);
    }
}

TEST_F(MLXTensorTest, TestMLXDeviceManager) {
    // Skip if MLX is not available on this system
    if (!MLXContext::is_available()) {
        GTEST_SKIP() << "MLX not available, skipping test";
    }
    
    // Check device availability
    EXPECT_TRUE(MLXDevice::is_available());
    
    // Get default device
    MLXDevice default_device = MLXDevice::default_device();
    
    // Get device properties
    mlx_device_type device_type = default_device.type();
    int device_index = default_device.index();
    std::string device_name = default_device.name();
    
    // Verify device properties
    EXPECT_NE(device_name, "");
    EXPECT_GE(device_index, 0);
    
    // Device synchronization
    MLXDevice::synchronize();
}
#endif // CCSM_WITH_MLX

} // namespace testing
} // namespace ccsm