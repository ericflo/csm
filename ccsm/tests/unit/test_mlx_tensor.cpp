#include <gtest/gtest.h>
#include <ccsm/mlx/mlx_tensor.h>
#include <ccsm/tensor.h>
#include <vector>
#include <memory>
#include <string>
#include <sstream>

namespace ccsm {
namespace testing {

class MLXTensorTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
    
    // Helper to create a tensor with specific values
    template<typename T>
    Tensor create_tensor(const std::vector<size_t>& shape, std::vector<T> data) {
        Tensor tensor = TensorFactory::empty(shape, get_data_type<T>());
        T* ptr = static_cast<T*>(tensor.data());
        if (ptr) {
            std::copy(data.begin(), data.end(), ptr);
        }
        return tensor;
    }
    
    // Helper to create a random tensor
    template<typename T>
    Tensor create_random_tensor(const std::vector<size_t>& shape) {
        size_t size = 1;
        for (auto dim : shape) {
            size *= dim;
        }
        
        std::vector<T> data(size);
        for (size_t i = 0; i < size; ++i) {
            data[i] = static_cast<T>(i % 100) / 10.0;
        }
        
        return create_tensor<T>(shape, data);
    }
    
    // Helper to get data type
    template<typename T>
    DataType get_data_type() {
        if (std::is_same<T, float>::value) {
            return DataType::F32;
        } else if (std::is_same<T, int32_t>::value) {
            return DataType::I32;
        } else if (std::is_same<T, int64_t>::value) {
            return DataType::I64;
        } else {
            throw std::runtime_error("Unsupported type");
        }
    }
};

// Test MLX availability
TEST_F(MLXTensorTest, TestMLXAvailability) {
    bool available = MLXContext::is_available();
    
    #ifdef CCSM_WITH_MLX
    // Skip further tests if MLX is not available
    if (!available) {
        GTEST_SKIP() << "MLX not available, skipping tests";
    }
    #else
    EXPECT_FALSE(available);
    GTEST_SKIP() << "MLX not compiled in, skipping tests";
    #endif
}

#ifdef CCSM_WITH_MLX
// Test creating MLX tensors
TEST_F(MLXTensorTest, TestMLXTensorCreation) {
    // Skip if MLX is not available
    if (!MLXContext::is_available()) {
        GTEST_SKIP() << "MLX not available, skipping test";
    }
    
    // Create a context
    MLXContext context;
    
    // Create tensors with different shapes and dtypes
    Tensor zeros = context.zeros({2, 3}, DataType::F32);
    Tensor ones = context.ones({3, 4}, DataType::F32);
    
    // Check shapes
    EXPECT_EQ(zeros.shape(0), 2);
    EXPECT_EQ(zeros.shape(1), 3);
    EXPECT_EQ(ones.shape(0), 3);
    EXPECT_EQ(ones.shape(1), 4);
    
    // Check dtypes
    EXPECT_EQ(zeros.dtype(), DataType::F32);
    EXPECT_EQ(ones.dtype(), DataType::F32);
    
    // Check total sizes
    EXPECT_EQ(zeros.size(), 6);
    EXPECT_EQ(ones.size(), 12);
}

// Test basic MLX tensor operations
TEST_F(MLXTensorTest, TestMLXTensorOperations) {
    // Skip if MLX is not available
    if (!MLXContext::is_available()) {
        GTEST_SKIP() << "MLX not available, skipping test";
    }
    
    // Create a context
    MLXContext context;
    
    // Create input tensors
    Tensor a = create_tensor<float>({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor b = create_tensor<float>({2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});
    
    // Test addition
    Tensor c = context.add(a, b);
    EXPECT_EQ(c.shape(0), 2);
    EXPECT_EQ(c.shape(1), 2);
    
    // Test subtraction
    Tensor d = context.subtract(b, a);
    EXPECT_EQ(d.shape(0), 2);
    EXPECT_EQ(d.shape(1), 2);
    
    // Test multiplication
    Tensor e = context.multiply(a, b);
    EXPECT_EQ(e.shape(0), 2);
    EXPECT_EQ(e.shape(1), 2);
    
    // Test division
    Tensor f = context.divide(b, a);
    EXPECT_EQ(f.shape(0), 2);
    EXPECT_EQ(f.shape(1), 2);
    
    // Test matrix multiplication
    Tensor g = context.matmul(a, b);
    EXPECT_EQ(g.shape(0), 2);
    EXPECT_EQ(g.shape(1), 2);
}

// Test MLX tensor reshaping and slicing
TEST_F(MLXTensorTest, TestMLXTensorReshapeSlice) {
    // Skip if MLX is not available
    if (!MLXContext::is_available()) {
        GTEST_SKIP() << "MLX not available, skipping test";
    }
    
    // Create some test tensors using MLX Context
    MLXContext context;
    Tensor tensor = context.zeros({2, 3, 4}, DataType::F32);
    
    // Test reshaping
    Tensor reshaped = tensor.reshape({4, 6});
    EXPECT_EQ(reshaped.shape(0), 4);
    EXPECT_EQ(reshaped.shape(1), 6);
    EXPECT_EQ(reshaped.size(), tensor.size());
    
    // Test view (which is also a reshape in MLX)
    Tensor viewed = tensor.view({6, 4});
    EXPECT_EQ(viewed.shape(0), 6);
    EXPECT_EQ(viewed.shape(1), 4);
    EXPECT_EQ(viewed.size(), tensor.size());
    
    // Test slicing
    Tensor sliced = tensor.slice(1, 0, 2);
    EXPECT_EQ(sliced.shape(0), 2);
    EXPECT_EQ(sliced.shape(1), 2);
    EXPECT_EQ(sliced.shape(2), 4);
}

// Test MLX activations
TEST_F(MLXTensorTest, TestMLXActivations) {
    // Skip if MLX is not available
    if (!MLXContext::is_available()) {
        GTEST_SKIP() << "MLX not available, skipping test";
    }
    
    // Create a context
    MLXContext context;
    
    // Create a test tensor
    Tensor tensor = create_tensor<float>({2, 3}, {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f});
    
    // Test ReLU
    Tensor relu_result = context.relu(tensor);
    EXPECT_EQ(relu_result.shape(0), 2);
    EXPECT_EQ(relu_result.shape(1), 3);
    
    // Test GELU
    Tensor gelu_result = context.gelu(tensor);
    EXPECT_EQ(gelu_result.shape(0), 2);
    EXPECT_EQ(gelu_result.shape(1), 3);
    
    // Test SiLU
    Tensor silu_result = context.silu(tensor);
    EXPECT_EQ(silu_result.shape(0), 2);
    EXPECT_EQ(silu_result.shape(1), 3);
    
    // Test Softmax
    Tensor softmax_result = context.softmax(tensor, 1);
    EXPECT_EQ(softmax_result.shape(0), 2);
    EXPECT_EQ(softmax_result.shape(1), 3);
}

// Test MLX reductions
TEST_F(MLXTensorTest, TestMLXReductions) {
    // Skip if MLX is not available
    if (!MLXContext::is_available()) {
        GTEST_SKIP() << "MLX not available, skipping test";
    }
    
    // Create a context
    MLXContext context;
    
    // Create a test tensor
    Tensor tensor = create_tensor<float>({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    
    // Test sum
    Tensor sum_result = context.sum(tensor, 1);
    EXPECT_EQ(sum_result.shape(0), 2);
    EXPECT_EQ(sum_result.shape(1), 1);
    
    // Test mean
    Tensor mean_result = context.mean(tensor, 0);
    EXPECT_EQ(mean_result.shape(0), 1);
    EXPECT_EQ(mean_result.shape(1), 3);
}

// Test tensor creation with different dtypes
TEST_F(MLXTensorTest, TestMLXDifferentDtypes) {
    // Skip if MLX is not available
    if (!MLXContext::is_available()) {
        GTEST_SKIP() << "MLX not available, skipping test";
    }
    
    // Create a context
    MLXContext context;
    
    // Create tensors with different dtypes
    Tensor f32_tensor = context.zeros({2, 3}, DataType::F32);
    Tensor f16_tensor = context.zeros({2, 3}, DataType::F16);
    Tensor bf16_tensor = context.zeros({2, 3}, DataType::BF16);
    Tensor i32_tensor = context.zeros({2, 3}, DataType::I32);
    Tensor i16_tensor = context.zeros({2, 3}, DataType::I16);
    Tensor i8_tensor = context.zeros({2, 3}, DataType::I8);
    
    // Check dtypes
    EXPECT_EQ(f32_tensor.dtype(), DataType::F32);
    EXPECT_EQ(f16_tensor.dtype(), DataType::F16);
    EXPECT_EQ(bf16_tensor.dtype(), DataType::BF16);
    EXPECT_EQ(i32_tensor.dtype(), DataType::I32);
    EXPECT_EQ(i16_tensor.dtype(), DataType::I16);
    EXPECT_EQ(i8_tensor.dtype(), DataType::I8);
}

// Test MLX device operations
TEST_F(MLXTensorTest, TestMLXDevice) {
    // Skip if MLX is not available
    if (!MLXContext::is_available()) {
        GTEST_SKIP() << "MLX not available, skipping test";
    }
    
    // Get default device
    MLXDevice device = MLXDevice::default_device();
    
    // Check device properties
    EXPECT_TRUE(device.name().size() > 0);
    
    // Test setting default device
    MLXDevice::set_default_device(device);
    
    // Test synchronization
    device.synchronize();
}
#endif // CCSM_WITH_MLX

} // namespace testing
} // namespace ccsm