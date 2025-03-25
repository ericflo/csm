#include <gtest/gtest.h>
#include <ccsm/mlx/mlx_tensor.h>
#include <ccsm/mlx/mlx_transformer.h>
#include <ccsm/tensor.h>
#include <vector>
#include <random>
#include <cmath>

namespace ccsm {
namespace testing {

class MLXTensorOpsTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
    
    // Helper to create a random tensor with given shape
    Tensor create_random_tensor(const std::vector<size_t>& shape, DataType dtype = DataType::F32) {
        Tensor tensor = TensorFactory::zeros(shape, dtype);
        
        // Calculate total number of elements
        size_t total_elements = 1;
        for (auto dim : shape) {
            total_elements *= dim;
        }
        
        // Fill with random values
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        
        if (dtype == DataType::F32) {
            float* data = static_cast<float*>(tensor.data());
            for (size_t i = 0; i < total_elements; ++i) {
                data[i] = dist(gen);
            }
        } else if (dtype == DataType::F16 || dtype == DataType::BF16) {
            // Approximation of half precision
            float* data = static_cast<float*>(tensor.data());
            for (size_t i = 0; i < total_elements; ++i) {
                data[i] = dist(gen);
            }
        } else if (dtype == DataType::I32) {
            int32_t* data = static_cast<int32_t*>(tensor.data());
            for (size_t i = 0; i < total_elements; ++i) {
                data[i] = static_cast<int32_t>(dist(gen) * 100.0f);
            }
        }
        
        return tensor;
    }
};

#ifdef CCSM_WITH_MLX
// Test MLX tensor creation
TEST_F(MLXTensorOpsTest, TestMLXTensorCreation) {
    // Create an MLX tensor with various shapes and types
    Tensor tensor1 = MLXTensorFactory::zeros({10, 10}, DataType::F32);
    Tensor tensor2 = MLXTensorFactory::zeros({5, 5, 5}, DataType::F16);
    Tensor tensor3 = MLXTensorFactory::zeros({100}, DataType::I32);
    
    // Verify tensor properties
    EXPECT_EQ(tensor1.shape(), std::vector<size_t>({10, 10}));
    EXPECT_EQ(tensor1.dtype(), DataType::F32);
    
    EXPECT_EQ(tensor2.shape(), std::vector<size_t>({5, 5, 5}));
    EXPECT_EQ(tensor2.dtype(), DataType::F16);
    
    EXPECT_EQ(tensor3.shape(), std::vector<size_t>({100}));
    EXPECT_EQ(tensor3.dtype(), DataType::I32);
    
    // Verify dynamic cast to MLXTensorImpl works
    EXPECT_NE(std::dynamic_pointer_cast<MLXTensorImpl>(tensor1.impl()), nullptr);
    EXPECT_NE(std::dynamic_pointer_cast<MLXTensorImpl>(tensor2.impl()), nullptr);
    EXPECT_NE(std::dynamic_pointer_cast<MLXTensorImpl>(tensor3.impl()), nullptr);
}

// Test basic MLX tensor operations
TEST_F(MLXTensorOpsTest, TestBasicMLXOperations) {
    // Create MLX tensors
    Tensor a = MLXTensorFactory::zeros({2, 3}, DataType::F32);
    Tensor b = MLXTensorFactory::zeros({2, 3}, DataType::F32);
    
    // Fill with data
    float* a_data = static_cast<float*>(a.data());
    float* b_data = static_cast<float*>(b.data());
    
    for (int i = 0; i < 6; ++i) {
        a_data[i] = static_cast<float>(i);
        b_data[i] = static_cast<float>(i + 6);
    }
    
    // Test addition
    Tensor c = a + b;
    EXPECT_EQ(c.shape(), std::vector<size_t>({2, 3}));
    EXPECT_EQ(c.dtype(), DataType::F32);
    
    float* c_data = static_cast<float*>(c.data());
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(c_data[i], a_data[i] + b_data[i]);
    }
    
    // Test multiplication
    Tensor d = a * b;
    EXPECT_EQ(d.shape(), std::vector<size_t>({2, 3}));
    EXPECT_EQ(d.dtype(), DataType::F32);
    
    float* d_data = static_cast<float*>(d.data());
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(d_data[i], a_data[i] * b_data[i]);
    }
    
    // Test subtraction
    Tensor e = b - a;
    EXPECT_EQ(e.shape(), std::vector<size_t>({2, 3}));
    EXPECT_EQ(e.dtype(), DataType::F32);
    
    float* e_data = static_cast<float*>(e.data());
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(e_data[i], b_data[i] - a_data[i]);
    }
    
    // Test division
    Tensor f = b / (a + 1.0f); // Avoid division by zero
    EXPECT_EQ(f.shape(), std::vector<size_t>({2, 3}));
    EXPECT_EQ(f.dtype(), DataType::F32);
    
    float* f_data = static_cast<float*>(f.data());
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(f_data[i], b_data[i] / (a_data[i] + 1.0f));
    }
}

// Test matrix multiplication for MLX tensors
TEST_F(MLXTensorOpsTest, TestMatrixMultiplication) {
    // Create MLX tensors
    Tensor a = MLXTensorFactory::zeros({2, 3}, DataType::F32);
    Tensor b = MLXTensorFactory::zeros({3, 2}, DataType::F32);
    
    // Fill with data
    float* a_data = static_cast<float*>(a.data());
    float* b_data = static_cast<float*>(b.data());
    
    for (int i = 0; i < 6; ++i) {
        a_data[i] = static_cast<float>(i + 1);
        b_data[i] = static_cast<float>(i + 1);
    }
    
    // Test matmul
    Tensor c = a.matmul(b);
    EXPECT_EQ(c.shape(), std::vector<size_t>({2, 2}));
    EXPECT_EQ(c.dtype(), DataType::F32);
    
    float* c_data = static_cast<float*>(c.data());
    
    // Manually compute expected result
    float expected[4] = {
        a_data[0] * b_data[0] + a_data[1] * b_data[2] + a_data[2] * b_data[4],
        a_data[0] * b_data[1] + a_data[1] * b_data[3] + a_data[2] * b_data[5],
        a_data[3] * b_data[0] + a_data[4] * b_data[2] + a_data[5] * b_data[4],
        a_data[3] * b_data[1] + a_data[4] * b_data[3] + a_data[5] * b_data[5]
    };
    
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(c_data[i], expected[i]);
    }
}

// Test rotary embedding for MLX tensors
TEST_F(MLXTensorOpsTest, TestRotaryEmbedding) {
    // Create tensor for testing
    std::vector<size_t> shape = {1, 4, 8}; // [batch, seq_len, dim]
    Tensor tensor = create_random_tensor(shape);
    
    // Convert to MLX tensor
    auto mlx_impl = std::dynamic_pointer_cast<MLXTensorImpl>(tensor.impl());
    ASSERT_NE(mlx_impl, nullptr);
    
    // Position indices
    std::vector<int> positions = {0, 1, 2, 3};
    
    // Create stream
    mlx_stream stream = mlx_default_cpu_stream_new();
    
    // Apply rotary embeddings
    mlx_array result = mlx_rotary_embedding(
        mlx_impl->mlx_array_handle(),
        positions,
        10000.0f,
        stream);
    
    // Verify result
    uint32_t ndim;
    mlx_array_ndim(result, &ndim);
    EXPECT_EQ(ndim, 3);
    
    const int* result_shape = mlx_array_shape(result);
    EXPECT_EQ(result_shape[0], 1);
    EXPECT_EQ(result_shape[1], 4);
    EXPECT_EQ(result_shape[2], 8);
    
    // Free resources
    mlx_array_free(result);
}

// Test attention mechanism for MLX tensors
TEST_F(MLXTensorOpsTest, TestAttention) {
    // Create query, key, value tensors
    std::vector<size_t> q_shape = {1, 4, 8}; // [batch, seq_len, dim]
    std::vector<size_t> k_shape = {1, 4, 8}; // [batch, seq_len, dim]
    std::vector<size_t> v_shape = {1, 4, 8}; // [batch, seq_len, dim]
    
    Tensor q_tensor = create_random_tensor(q_shape);
    Tensor k_tensor = create_random_tensor(k_shape);
    Tensor v_tensor = create_random_tensor(v_shape);
    
    // Convert to MLX tensors
    auto q_impl = std::dynamic_pointer_cast<MLXTensorImpl>(q_tensor.impl());
    auto k_impl = std::dynamic_pointer_cast<MLXTensorImpl>(k_tensor.impl());
    auto v_impl = std::dynamic_pointer_cast<MLXTensorImpl>(v_tensor.impl());
    
    ASSERT_NE(q_impl, nullptr);
    ASSERT_NE(k_impl, nullptr);
    ASSERT_NE(v_impl, nullptr);
    
    // Position indices
    std::vector<int> positions = {0, 1, 2, 3};
    
    // Create stream
    mlx_stream stream = mlx_default_cpu_stream_new();
    
    // Apply attention
    mlx_array result = mlx_attention(
        q_impl->mlx_array_handle(),
        k_impl->mlx_array_handle(),
        v_impl->mlx_array_handle(),
        positions,
        1.0f / std::sqrt(8.0f),
        stream);
    
    // Verify result
    uint32_t ndim;
    mlx_array_ndim(result, &ndim);
    EXPECT_EQ(ndim, 3);
    
    const int* result_shape = mlx_array_shape(result);
    EXPECT_EQ(result_shape[0], 1);
    EXPECT_EQ(result_shape[1], 4);
    EXPECT_EQ(result_shape[2], 8);
    
    // Free resources
    mlx_array_free(result);
}

// Test feed-forward network for MLX tensors
TEST_F(MLXTensorOpsTest, TestFeedForward) {
    // Create input tensor
    std::vector<size_t> x_shape = {1, 4, 32}; // [batch, seq_len, dim]
    Tensor x_tensor = create_random_tensor(x_shape);
    
    // Create weight tensors
    std::vector<size_t> w1_shape = {32, 128}; // [dim, hidden]
    std::vector<size_t> w2_shape = {128, 32}; // [hidden, dim]
    std::vector<size_t> w3_shape = {32, 128}; // [dim, hidden]
    
    Tensor w1_tensor = create_random_tensor(w1_shape);
    Tensor w2_tensor = create_random_tensor(w2_shape);
    Tensor w3_tensor = create_random_tensor(w3_shape);
    
    // Convert to MLX tensors
    auto x_impl = std::dynamic_pointer_cast<MLXTensorImpl>(x_tensor.impl());
    auto w1_impl = std::dynamic_pointer_cast<MLXTensorImpl>(w1_tensor.impl());
    auto w2_impl = std::dynamic_pointer_cast<MLXTensorImpl>(w2_tensor.impl());
    auto w3_impl = std::dynamic_pointer_cast<MLXTensorImpl>(w3_tensor.impl());
    
    ASSERT_NE(x_impl, nullptr);
    ASSERT_NE(w1_impl, nullptr);
    ASSERT_NE(w2_impl, nullptr);
    ASSERT_NE(w3_impl, nullptr);
    
    // Create stream
    mlx_stream stream = mlx_default_cpu_stream_new();
    
    // Apply feed-forward
    mlx_array result = mlx_feed_forward(
        x_impl->mlx_array_handle(),
        w1_impl->mlx_array_handle(),
        w2_impl->mlx_array_handle(),
        w3_impl->mlx_array_handle(),
        stream);
    
    // Verify result
    uint32_t ndim;
    mlx_array_ndim(result, &ndim);
    EXPECT_EQ(ndim, 3);
    
    const int* result_shape = mlx_array_shape(result);
    EXPECT_EQ(result_shape[0], 1);
    EXPECT_EQ(result_shape[1], 4);
    EXPECT_EQ(result_shape[2], 32);
    
    // Free resources
    mlx_array_free(result);
}

// Test transformer layer for MLX tensors
TEST_F(MLXTensorOpsTest, TestTransformerLayer) {
    // Create weights map
    std::unordered_map<std::string, mlx_array> weights;
    
    // Model dimensions
    int d_model = 32;
    int n_heads = 4;
    int n_kv_heads = 4;
    int head_dim = d_model / n_heads;
    
    // Create weights for the transformer layer
    // Attention norm
    {
        std::vector<float> data(d_model, 1.0f);
        int shape[] = {d_model};
        weights["layers.0.attention_norm.weight"] = mlx_array_new_data(
            data.data(), shape, 1, MLX_FLOAT32);
    }
    
    // Query weight
    {
        std::vector<float> data(d_model * d_model, 0.01f);
        int shape[] = {d_model, d_model};
        weights["layers.0.attention.wq.weight"] = mlx_array_new_data(
            data.data(), shape, 2, MLX_FLOAT32);
    }
    
    // Key weight
    {
        std::vector<float> data(d_model * d_model, 0.01f);
        int shape[] = {d_model, d_model};
        weights["layers.0.attention.wk.weight"] = mlx_array_new_data(
            data.data(), shape, 2, MLX_FLOAT32);
    }
    
    // Value weight
    {
        std::vector<float> data(d_model * d_model, 0.01f);
        int shape[] = {d_model, d_model};
        weights["layers.0.attention.wv.weight"] = mlx_array_new_data(
            data.data(), shape, 2, MLX_FLOAT32);
    }
    
    // Output weight
    {
        std::vector<float> data(d_model * d_model, 0.01f);
        int shape[] = {d_model, d_model};
        weights["layers.0.attention.wo.weight"] = mlx_array_new_data(
            data.data(), shape, 2, MLX_FLOAT32);
    }
    
    // FFN norm
    {
        std::vector<float> data(d_model, 1.0f);
        int shape[] = {d_model};
        weights["layers.0.ffn_norm.weight"] = mlx_array_new_data(
            data.data(), shape, 1, MLX_FLOAT32);
    }
    
    // FFN weights
    {
        int hidden_dim = 4 * d_model;
        std::vector<float> data(d_model * hidden_dim, 0.01f);
        int shape[] = {d_model, hidden_dim};
        weights["layers.0.feed_forward.w1.weight"] = mlx_array_new_data(
            data.data(), shape, 2, MLX_FLOAT32);
    }
    
    {
        int hidden_dim = 4 * d_model;
        std::vector<float> data(hidden_dim * d_model, 0.01f);
        int shape[] = {hidden_dim, d_model};
        weights["layers.0.feed_forward.w2.weight"] = mlx_array_new_data(
            data.data(), shape, 2, MLX_FLOAT32);
    }
    
    {
        int hidden_dim = 4 * d_model;
        std::vector<float> data(d_model * hidden_dim, 0.01f);
        int shape[] = {d_model, hidden_dim};
        weights["layers.0.feed_forward.w3.weight"] = mlx_array_new_data(
            data.data(), shape, 2, MLX_FLOAT32);
    }
    
    // Create a transformer layer
    MLXTransformerLayer layer("layers.0", weights, d_model, n_heads, n_kv_heads);
    
    // Create input tensor
    std::vector<size_t> x_shape = {1, 4, d_model}; // [batch, seq_len, dim]
    Tensor x_tensor = create_random_tensor(x_shape);
    auto x_impl = std::dynamic_pointer_cast<MLXTensorImpl>(x_tensor.impl());
    ASSERT_NE(x_impl, nullptr);
    
    // Create KV cache
    MLXKVCache kv_cache(1, n_kv_heads, head_dim, 1024);
    
    // Positions
    std::vector<int> positions = {0, 1, 2, 3};
    
    // Create stream
    mlx_stream stream = mlx_default_cpu_stream_new();
    
    // Forward pass
    mlx_array result = layer.forward(
        x_impl->mlx_array_handle(),
        &kv_cache,
        positions,
        stream);
    
    // Verify result
    uint32_t ndim;
    mlx_array_ndim(result, &ndim);
    EXPECT_EQ(ndim, 3);
    
    const int* result_shape = mlx_array_shape(result);
    EXPECT_EQ(result_shape[0], 1);
    EXPECT_EQ(result_shape[1], 4);
    EXPECT_EQ(result_shape[2], d_model);
    
    // Test KV cache was updated
    EXPECT_EQ(kv_cache.current_seq_len(), 4);
    
    // Clean up
    mlx_array_free(result);
    for (auto& [name, array] : weights) {
        mlx_array_free(array);
    }
}

// Test complete transformer for MLX
TEST_F(MLXTensorOpsTest, TestCompleteTransformer) {
    // Create weights map
    std::unordered_map<std::string, mlx_array> weights;
    
    // Model dimensions
    int d_model = 32;
    int n_layers = 2;
    int n_heads = 4;
    int n_kv_heads = 4;
    int vocab_size = 100;
    int head_dim = d_model / n_heads;
    
    // Create embedding weights
    {
        std::vector<float> data(vocab_size * d_model, 0.01f);
        int shape[] = {vocab_size, d_model};
        weights["token_embedding.weight"] = mlx_array_new_data(
            data.data(), shape, 2, MLX_FLOAT32);
    }
    
    // Create norm weights
    {
        std::vector<float> data(d_model, 1.0f);
        int shape[] = {d_model};
        weights["norm.weight"] = mlx_array_new_data(
            data.data(), shape, 1, MLX_FLOAT32);
    }
    
    // Create output weights
    {
        std::vector<float> data(d_model * vocab_size, 0.01f);
        int shape[] = {d_model, vocab_size};
        weights["output.weight"] = mlx_array_new_data(
            data.data(), shape, 2, MLX_FLOAT32);
    }
    
    // Create transformer layer weights (for both layers)
    for (int layer = 0; layer < n_layers; ++layer) {
        std::string prefix = "layers." + std::to_string(layer) + ".";
        
        // Attention norm
        {
            std::vector<float> data(d_model, 1.0f);
            int shape[] = {d_model};
            weights[prefix + "attention_norm.weight"] = mlx_array_new_data(
                data.data(), shape, 1, MLX_FLOAT32);
        }
        
        // Query weight
        {
            std::vector<float> data(d_model * d_model, 0.01f);
            int shape[] = {d_model, d_model};
            weights[prefix + "attention.wq.weight"] = mlx_array_new_data(
                data.data(), shape, 2, MLX_FLOAT32);
        }
        
        // Key weight
        {
            std::vector<float> data(d_model * d_model, 0.01f);
            int shape[] = {d_model, d_model};
            weights[prefix + "attention.wk.weight"] = mlx_array_new_data(
                data.data(), shape, 2, MLX_FLOAT32);
        }
        
        // Value weight
        {
            std::vector<float> data(d_model * d_model, 0.01f);
            int shape[] = {d_model, d_model};
            weights[prefix + "attention.wv.weight"] = mlx_array_new_data(
                data.data(), shape, 2, MLX_FLOAT32);
        }
        
        // Output weight
        {
            std::vector<float> data(d_model * d_model, 0.01f);
            int shape[] = {d_model, d_model};
            weights[prefix + "attention.wo.weight"] = mlx_array_new_data(
                data.data(), shape, 2, MLX_FLOAT32);
        }
        
        // FFN norm
        {
            std::vector<float> data(d_model, 1.0f);
            int shape[] = {d_model};
            weights[prefix + "ffn_norm.weight"] = mlx_array_new_data(
                data.data(), shape, 1, MLX_FLOAT32);
        }
        
        // FFN weights
        {
            int hidden_dim = 4 * d_model;
            std::vector<float> data(d_model * hidden_dim, 0.01f);
            int shape[] = {d_model, hidden_dim};
            weights[prefix + "feed_forward.w1.weight"] = mlx_array_new_data(
                data.data(), shape, 2, MLX_FLOAT32);
        }
        
        {
            int hidden_dim = 4 * d_model;
            std::vector<float> data(hidden_dim * d_model, 0.01f);
            int shape[] = {hidden_dim, d_model};
            weights[prefix + "feed_forward.w2.weight"] = mlx_array_new_data(
                data.data(), shape, 2, MLX_FLOAT32);
        }
        
        {
            int hidden_dim = 4 * d_model;
            std::vector<float> data(d_model * hidden_dim, 0.01f);
            int shape[] = {d_model, hidden_dim};
            weights[prefix + "feed_forward.w3.weight"] = mlx_array_new_data(
                data.data(), shape, 2, MLX_FLOAT32);
        }
    }
    
    // Create the transformer
    MLXTransformer transformer(weights, d_model, n_layers, n_heads, n_kv_heads, vocab_size);
    
    // Create input tokens and positions
    std::vector<int> tokens = {1, 2, 3, 4};
    std::vector<int> positions = {0, 1, 2, 3};
    
    // Create KV cache
    MLXKVCache kv_cache(n_layers, n_kv_heads, head_dim, 1024);
    
    // Create stream
    mlx_stream stream = mlx_default_cpu_stream_new();
    
    // Forward pass
    mlx_array result = transformer.forward(tokens, positions, &kv_cache, stream);
    
    // Verify result
    uint32_t ndim;
    mlx_array_ndim(result, &ndim);
    EXPECT_EQ(ndim, 2);
    
    const int* result_shape = mlx_array_shape(result);
    EXPECT_EQ(result_shape[0], 4); // seq_len
    EXPECT_EQ(result_shape[1], vocab_size); // vocab_size
    
    // Test KV cache was updated
    EXPECT_EQ(kv_cache.current_seq_len(), 4);
    
    // Test cache pruning
    bool prune_result = transformer.prune_kv_cache(2);
    EXPECT_TRUE(prune_result);
    EXPECT_EQ(kv_cache.current_seq_len(), 2);
    
    // Test cache reset
    transformer.reset_caches();
    EXPECT_EQ(kv_cache.current_seq_len(), 0);
    
    // Clean up
    mlx_array_free(result);
    for (auto& [name, array] : weights) {
        mlx_array_free(array);
    }
}

// Test MLX tensor serialization
TEST_F(MLXTensorOpsTest, TestMLXTensorSerialization) {
    // Create an MLX tensor
    Tensor tensor = MLXTensorFactory::zeros({2, 3}, DataType::F32);
    
    // Fill with data
    float* data = static_cast<float*>(tensor.data());
    for (int i = 0; i < 6; ++i) {
        data[i] = static_cast<float>(i);
    }
    
    // Get serialized data
    std::vector<uint8_t> serialized = tensor.serialize();
    EXPECT_FALSE(serialized.empty());
    
    // Deserialize
    Tensor deserialized = Tensor::deserialize(serialized);
    
    // Verify tensor properties
    EXPECT_EQ(deserialized.shape(), std::vector<size_t>({2, 3}));
    EXPECT_EQ(deserialized.dtype(), DataType::F32);
    
    // Verify data
    float* deserialized_data = static_cast<float*>(deserialized.data());
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(deserialized_data[i], static_cast<float>(i));
    }
}

// Test MLX memory optimization
TEST_F(MLXTensorOpsTest, TestMLXMemoryOptimization) {
    // Create MLX device
    MLXDevice device;
    
    // Test memory usage API
    size_t initial_memory = device.get_memory_used();
    
    // Allocate some tensors
    std::vector<Tensor> tensors;
    for (int i = 0; i < 10; ++i) {
        tensors.push_back(MLXTensorFactory::zeros({100, 100}, DataType::F32));
    }
    
    // Memory should have increased
    size_t after_alloc_memory = device.get_memory_used();
    EXPECT_GT(after_alloc_memory, initial_memory);
    
    // Clear tensors
    tensors.clear();
    
    // Memory should eventually decrease after garbage collection
    // Note: MLX may not immediately release memory, so we need to give it some time
    device.synchronize();
    size_t final_memory = device.get_memory_used();
    
    // We can't guarantee exact memory release timing due to garbage collection,
    // but memory shouldn't keep increasing
    EXPECT_LE(final_memory, after_alloc_memory * 1.1); // Allow for some overhead
}
#endif // CCSM_WITH_MLX

// Test MLXDevice utility functions (these should work regardless of MLX availability)
TEST_F(MLXTensorOpsTest, TestMLXDeviceUtilities) {
    // This test should work with or without MLX
    MLXDevice device;
    
    // Test device API
    bool is_available = device.is_available();
    bool is_metal_supported = device.is_metal_supported();
    
    // Even if MLX is not available, the API should not crash
    // We can't assert specific values because they depend on system configuration
    // and whether MLX is compiled in
    
    // Test device info
    std::string device_info = device.get_device_info();
    EXPECT_FALSE(device_info.empty());
    
    // Test device name
    std::string device_name = device.get_device_name();
    EXPECT_FALSE(device_name.empty());
}

} // namespace testing
} // namespace ccsm