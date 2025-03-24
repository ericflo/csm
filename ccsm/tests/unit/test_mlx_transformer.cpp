#include <gtest/gtest.h>
#include <ccsm/mlx/mlx_transformer.h>
#include <ccsm/mlx/mlx_tensor.h>
#include <vector>
#include <memory>
#include <random>

namespace ccsm {
namespace testing {

// Create a mock weight map for testing MLX transformer
#ifdef CCSM_WITH_MLX
std::unordered_map<std::string, mlx_array> create_mock_mlx_weights(int d_model, int n_heads, int n_kv_heads, int vocab_size) {
    std::unordered_map<std::string, mlx_array> weights;
    std::vector<int64_t> shape;
    
    // Random number generator for test data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    
    // Create embedding weight
    shape = {vocab_size, d_model};
    size_t size = vocab_size * d_model;
    std::vector<float> data(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = dist(gen);
    }
    
    mlx_array tok_embeddings;
    mlx_array_from_data(data.data(), MLX_FLOAT32, shape.data(), shape.size(), 0, &tok_embeddings);
    weights["tok_embeddings.weight"] = tok_embeddings;
    
    // Create norm weights
    shape = {d_model};
    size = d_model;
    data.resize(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = dist(gen);
    }
    
    mlx_array norm_weight;
    mlx_array_from_data(data.data(), MLX_FLOAT32, shape.data(), shape.size(), 0, &norm_weight);
    weights["norm.weight"] = norm_weight;
    
    // Create norm bias
    for (size_t i = 0; i < size; ++i) {
        data[i] = dist(gen);
    }
    
    mlx_array norm_bias;
    mlx_array_from_data(data.data(), MLX_FLOAT32, shape.data(), shape.size(), 0, &norm_bias);
    weights["norm.bias"] = norm_bias;
    
    // Create output weight (same as embedding for tied weights)
    weights["output.weight"] = tok_embeddings;
    
    // Create layer weights
    for (int layer = 0; layer < 2; ++layer) {
        std::string prefix = "layers." + std::to_string(layer) + ".";
        
        // Attention norm weights
        mlx_array attention_norm_weight;
        mlx_array_from_data(data.data(), MLX_FLOAT32, shape.data(), shape.size(), 0, &attention_norm_weight);
        weights[prefix + "attention_norm.weight"] = attention_norm_weight;
        
        // Attention norm bias
        mlx_array attention_norm_bias;
        mlx_array_from_data(data.data(), MLX_FLOAT32, shape.data(), shape.size(), 0, &attention_norm_bias);
        weights[prefix + "attention_norm.bias"] = attention_norm_bias;
        
        // Query weight
        shape = {d_model, n_heads * (d_model / n_heads)};
        size = d_model * n_heads * (d_model / n_heads);
        data.resize(size);
        for (size_t i = 0; i < size; ++i) {
            data[i] = dist(gen);
        }
        
        mlx_array wq_weight;
        mlx_array_from_data(data.data(), MLX_FLOAT32, shape.data(), shape.size(), 0, &wq_weight);
        weights[prefix + "attention.wq.weight"] = wq_weight;
        
        // Key weight
        shape = {d_model, n_kv_heads * (d_model / n_heads)};
        size = d_model * n_kv_heads * (d_model / n_heads);
        data.resize(size);
        for (size_t i = 0; i < size; ++i) {
            data[i] = dist(gen);
        }
        
        mlx_array wk_weight;
        mlx_array_from_data(data.data(), MLX_FLOAT32, shape.data(), shape.size(), 0, &wk_weight);
        weights[prefix + "attention.wk.weight"] = wk_weight;
        
        // Value weight
        mlx_array wv_weight;
        mlx_array_from_data(data.data(), MLX_FLOAT32, shape.data(), shape.size(), 0, &wv_weight);
        weights[prefix + "attention.wv.weight"] = wv_weight;
        
        // Output projection weight
        shape = {n_heads * (d_model / n_heads), d_model};
        size = n_heads * (d_model / n_heads) * d_model;
        data.resize(size);
        for (size_t i = 0; i < size; ++i) {
            data[i] = dist(gen);
        }
        
        mlx_array wo_weight;
        mlx_array_from_data(data.data(), MLX_FLOAT32, shape.data(), shape.size(), 0, &wo_weight);
        weights[prefix + "attention.wo.weight"] = wo_weight;
        
        // FFN norm weights
        shape = {d_model};
        size = d_model;
        data.resize(size);
        for (size_t i = 0; i < size; ++i) {
            data[i] = dist(gen);
        }
        
        mlx_array ffn_norm_weight;
        mlx_array_from_data(data.data(), MLX_FLOAT32, shape.data(), shape.size(), 0, &ffn_norm_weight);
        weights[prefix + "ffn_norm.weight"] = ffn_norm_weight;
        
        // FFN norm bias
        mlx_array ffn_norm_bias;
        mlx_array_from_data(data.data(), MLX_FLOAT32, shape.data(), shape.size(), 0, &ffn_norm_bias);
        weights[prefix + "ffn_norm.bias"] = ffn_norm_bias;
        
        // FFN weights (SwiGLU architecture)
        int ffn_dim = 4 * d_model;
        
        // W1 weight
        shape = {d_model, ffn_dim};
        size = d_model * ffn_dim;
        data.resize(size);
        for (size_t i = 0; i < size; ++i) {
            data[i] = dist(gen);
        }
        
        mlx_array w1_weight;
        mlx_array_from_data(data.data(), MLX_FLOAT32, shape.data(), shape.size(), 0, &w1_weight);
        weights[prefix + "feed_forward.w1.weight"] = w1_weight;
        
        // W2 weight
        shape = {ffn_dim, d_model};
        size = ffn_dim * d_model;
        data.resize(size);
        for (size_t i = 0; i < size; ++i) {
            data[i] = dist(gen);
        }
        
        mlx_array w2_weight;
        mlx_array_from_data(data.data(), MLX_FLOAT32, shape.data(), shape.size(), 0, &w2_weight);
        weights[prefix + "feed_forward.w2.weight"] = w2_weight;
        
        // W3 weight (for SwiGLU)
        shape = {d_model, ffn_dim};
        size = d_model * ffn_dim;
        data.resize(size);
        for (size_t i = 0; i < size; ++i) {
            data[i] = dist(gen);
        }
        
        mlx_array w3_weight;
        mlx_array_from_data(data.data(), MLX_FLOAT32, shape.data(), shape.size(), 0, &w3_weight);
        weights[prefix + "feed_forward.w3.weight"] = w3_weight;
    }
    
    return weights;
}

// Free the memory for MLX arrays in a weight map
void free_mock_mlx_weights(std::unordered_map<std::string, mlx_array>& weights) {
    for (auto& [name, array] : weights) {
        mlx_array_free(array);
    }
}
#endif

class MLXTransformerTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test MLX availability
TEST_F(MLXTransformerTest, TestMLXAvailability) {
    #ifdef CCSM_WITH_MLX
    bool available = MLXDevice::is_available();
    
    // Skip further tests if MLX is not available
    if (!available) {
        GTEST_SKIP() << "MLX not available, skipping tests";
    }
    #else
    GTEST_SKIP() << "MLX not compiled in, skipping tests";
    #endif
}

#ifdef CCSM_WITH_MLX
// Test KV cache creation and management
TEST_F(MLXTransformerTest, TestKVCache) {
    // Skip if MLX is not available
    if (!MLXDevice::is_available()) {
        GTEST_SKIP() << "MLX not available, skipping test";
    }
    
    // Create KV cache
    int n_layers = 2;
    int n_heads = 8;
    int head_dim = 64;
    int max_seq_len = 16;
    
    MLXKVCache kv_cache(n_layers, n_heads, head_dim, max_seq_len);
    
    // Check initial state
    EXPECT_EQ(kv_cache.size(), 0);
    EXPECT_EQ(kv_cache.max_seq_len(), max_seq_len);
    EXPECT_EQ(kv_cache.current_seq_len(), 0);
    
    // Create stream for operations
    mlx_stream stream;
    mlx_stream_create(&stream);
    
    // Create some mock key and value tensors for layer 0
    std::vector<int64_t> shape = {2, n_heads, head_dim};
    std::vector<float> data(2 * n_heads * head_dim, 1.0f);
    
    mlx_array k, v;
    mlx_array_from_data(data.data(), MLX_FLOAT32, shape.data(), shape.size(), 0, &k);
    mlx_array_from_data(data.data(), MLX_FLOAT32, shape.data(), shape.size(), 0, &v);
    
    // Update cache with positions [0, 1]
    std::vector<int> positions = {0, 1};
    kv_cache.update(0, k, v, positions, stream);
    
    // Check updated state
    EXPECT_EQ(kv_cache.current_seq_len(), 2);
    
    // Get cached values for layer 0
    mlx_array k_cache = kv_cache.k_cache(0);
    mlx_array v_cache = kv_cache.v_cache(0);
    
    // Verify cache shapes
    uint32_t ndim;
    mlx_array_ndim(k_cache, &ndim);
    EXPECT_EQ(ndim, 3);
    
    std::vector<int64_t> cache_shape(ndim);
    mlx_array_shape(k_cache, cache_shape.data());
    EXPECT_EQ(cache_shape[0], max_seq_len);
    EXPECT_EQ(cache_shape[1], n_heads);
    EXPECT_EQ(cache_shape[2], head_dim);
    
    // Clear cache
    kv_cache.clear();
    EXPECT_EQ(kv_cache.current_seq_len(), 0);
    
    // Free arrays
    mlx_array_free(k);
    mlx_array_free(v);
    mlx_stream_free(stream);
}

// Test transformer layer creation and forward pass
TEST_F(MLXTransformerTest, TestTransformerLayer) {
    // Skip if MLX is not available
    if (!MLXDevice::is_available()) {
        GTEST_SKIP() << "MLX not available, skipping test";
    }
    
    // Model dimensions
    int d_model = 128;
    int n_heads = 4;
    int n_kv_heads = 4;
    int vocab_size = 100;
    float rope_theta = 10000.0f;
    
    // Create mock weights
    auto weights = create_mock_mlx_weights(d_model, n_heads, n_kv_heads, vocab_size);
    
    // Create transformer layer
    MLXTransformerLayer layer("layers.0.", weights, d_model, n_heads, n_kv_heads, rope_theta);
    
    // Create KV cache
    MLXKVCache kv_cache(1, n_kv_heads, d_model / n_heads, 16);
    
    // Create input tensor
    std::vector<int64_t> shape = {2, d_model};
    std::vector<float> data(2 * d_model, 0.1f);
    
    mlx_array x;
    mlx_array_from_data(data.data(), MLX_FLOAT32, shape.data(), shape.size(), 0, &x);
    
    // Create stream for operations
    mlx_stream stream;
    mlx_stream_create(&stream);
    
    // Forward pass
    std::vector<int> positions = {0, 1};
    mlx_array output = layer.forward(x, &kv_cache, positions, stream);
    
    // Verify output shape
    uint32_t ndim;
    mlx_array_ndim(output, &ndim);
    EXPECT_EQ(ndim, 2);
    
    std::vector<int64_t> output_shape(ndim);
    mlx_array_shape(output, output_shape.data());
    EXPECT_EQ(output_shape[0], 2);
    EXPECT_EQ(output_shape[1], d_model);
    
    // Free arrays
    mlx_array_free(x);
    mlx_array_free(output);
    mlx_stream_free(stream);
    free_mock_mlx_weights(weights);
}

// Test full transformer model
TEST_F(MLXTransformerTest, TestTransformerModel) {
    // Skip if MLX is not available
    if (!MLXDevice::is_available()) {
        GTEST_SKIP() << "MLX not available, skipping test";
    }
    
    // Model dimensions
    int d_model = 128;
    int n_layers = 2;
    int n_heads = 4;
    int n_kv_heads = 4;
    int vocab_size = 100;
    float rope_theta = 10000.0f;
    
    // Create mock weights
    auto weights = create_mock_mlx_weights(d_model, n_heads, n_kv_heads, vocab_size);
    
    // Create transformer model
    MLXTransformer transformer(weights, d_model, n_layers, n_heads, n_kv_heads, vocab_size, rope_theta);
    
    // Create KV cache
    MLXKVCache kv_cache(n_layers, n_kv_heads, d_model / n_heads, 16);
    
    // Create stream for operations
    mlx_stream stream;
    mlx_stream_create(&stream);
    
    // Input tokens and positions
    std::vector<int> tokens = {1, 2, 3};
    std::vector<int> positions = {0, 1, 2};
    
    // Forward pass
    mlx_array output = transformer.forward(tokens, positions, &kv_cache, stream);
    
    // Verify output shape
    uint32_t ndim;
    mlx_array_ndim(output, &ndim);
    EXPECT_EQ(ndim, 2);
    
    std::vector<int64_t> output_shape(ndim);
    mlx_array_shape(output, output_shape.data());
    EXPECT_EQ(output_shape[0], 3);
    EXPECT_EQ(output_shape[1], vocab_size);
    
    // Reset caches
    transformer.reset_caches();
    
    // Free arrays
    mlx_array_free(output);
    mlx_stream_free(stream);
    free_mock_mlx_weights(weights);
}

// Test helper functions for MLX transformer operations
TEST_F(MLXTransformerTest, TestMLXTransformerHelpers) {
    // Skip if MLX is not available
    if (!MLXDevice::is_available()) {
        GTEST_SKIP() << "MLX not available, skipping test";
    }
    
    // Create stream for operations
    mlx_stream stream;
    mlx_stream_create(&stream);
    
    // Test rotary position embeddings
    {
        int head_dim = 64;
        std::vector<int64_t> shape = {2, 4, head_dim};
        std::vector<float> data(2 * 4 * head_dim, 0.1f);
        
        mlx_array x;
        mlx_array_from_data(data.data(), MLX_FLOAT32, shape.data(), shape.size(), 0, &x);
        
        std::vector<int> positions = {0, 1};
        float theta = 10000.0f;
        
        mlx_array rotated = mlx_rotary_embedding(x, positions, theta, stream);
        
        // Verify output shape
        uint32_t ndim;
        mlx_array_ndim(rotated, &ndim);
        EXPECT_EQ(ndim, 3);
        
        std::vector<int64_t> output_shape(ndim);
        mlx_array_shape(rotated, output_shape.data());
        EXPECT_EQ(output_shape[0], 2);
        EXPECT_EQ(output_shape[1], 4);
        EXPECT_EQ(output_shape[2], head_dim);
        
        mlx_array_free(x);
        mlx_array_free(rotated);
    }
    
    // Test attention mechanism
    {
        int head_dim = 64;
        std::vector<int64_t> q_shape = {2, 4, head_dim};
        std::vector<int64_t> k_shape = {2, 4, head_dim};
        std::vector<int64_t> v_shape = {2, 4, head_dim};
        
        std::vector<float> q_data(2 * 4 * head_dim, 0.1f);
        std::vector<float> k_data(2 * 4 * head_dim, 0.1f);
        std::vector<float> v_data(2 * 4 * head_dim, 0.1f);
        
        mlx_array q, k, v;
        mlx_array_from_data(q_data.data(), MLX_FLOAT32, q_shape.data(), q_shape.size(), 0, &q);
        mlx_array_from_data(k_data.data(), MLX_FLOAT32, k_shape.data(), k_shape.size(), 0, &k);
        mlx_array_from_data(v_data.data(), MLX_FLOAT32, v_shape.data(), v_shape.size(), 0, &v);
        
        std::vector<int> positions = {0, 1};
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        
        mlx_array attn_output = mlx_attention(q, k, v, positions, scale, stream);
        
        // Verify output shape
        uint32_t ndim;
        mlx_array_ndim(attn_output, &ndim);
        EXPECT_EQ(ndim, 3);
        
        std::vector<int64_t> output_shape(ndim);
        mlx_array_shape(attn_output, output_shape.data());
        EXPECT_EQ(output_shape[0], 2);
        EXPECT_EQ(output_shape[1], 4);
        EXPECT_EQ(output_shape[2], head_dim);
        
        mlx_array_free(q);
        mlx_array_free(k);
        mlx_array_free(v);
        mlx_array_free(attn_output);
    }
    
    // Test feed-forward network
    {
        int d_model = 128;
        int ffn_dim = 512;
        
        std::vector<int64_t> x_shape = {2, d_model};
        std::vector<int64_t> w1_shape = {d_model, ffn_dim};
        std::vector<int64_t> w2_shape = {ffn_dim, d_model};
        std::vector<int64_t> w3_shape = {d_model, ffn_dim};
        
        std::vector<float> x_data(2 * d_model, 0.1f);
        std::vector<float> w1_data(d_model * ffn_dim, 0.01f);
        std::vector<float> w2_data(ffn_dim * d_model, 0.01f);
        std::vector<float> w3_data(d_model * ffn_dim, 0.01f);
        
        mlx_array x, w1, w2, w3;
        mlx_array_from_data(x_data.data(), MLX_FLOAT32, x_shape.data(), x_shape.size(), 0, &x);
        mlx_array_from_data(w1_data.data(), MLX_FLOAT32, w1_shape.data(), w1_shape.size(), 0, &w1);
        mlx_array_from_data(w2_data.data(), MLX_FLOAT32, w2_shape.data(), w2_shape.size(), 0, &w2);
        mlx_array_from_data(w3_data.data(), MLX_FLOAT32, w3_shape.data(), w3_shape.size(), 0, &w3);
        
        mlx_array ffn_output = mlx_feed_forward(x, w1, w2, w3, stream);
        
        // Verify output shape
        uint32_t ndim;
        mlx_array_ndim(ffn_output, &ndim);
        EXPECT_EQ(ndim, 2);
        
        std::vector<int64_t> output_shape(ndim);
        mlx_array_shape(ffn_output, output_shape.data());
        EXPECT_EQ(output_shape[0], 2);
        EXPECT_EQ(output_shape[1], d_model);
        
        mlx_array_free(x);
        mlx_array_free(w1);
        mlx_array_free(w2);
        mlx_array_free(w3);
        mlx_array_free(ffn_output);
    }
    
    mlx_stream_free(stream);
}
#endif // CCSM_WITH_MLX

} // namespace testing
} // namespace ccsm