#include <gtest/gtest.h>
#include <ccsm/mlx/mlx_transformer.h>
#include <ccsm/mlx/mlx_tensor.h>
#include <ccsm/utils.h>
#include <unordered_map>
#include <memory>
#include <vector>
#include <random>

namespace ccsm {
namespace testing {

class MLXTransformerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup test weights for the transformer
        std::unordered_map<std::string, mlx_array> weights;
        
        // Create random weight initialization function
        auto create_random_weight = [](const std::vector<int>& shape) {
            // Calculate total number of elements
            int total_elements = 1;
            for (int dim : shape) {
                total_elements *= dim;
            }
            
            // Initialize with random values
            std::vector<float> data(total_elements);
            std::mt19937 rng(42); // Fixed seed for reproducibility
            std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
            
            for (int i = 0; i < total_elements; ++i) {
                data[i] = dist(rng);
            }
            
            // Create MLX array
            return mlx_array_new_data(data.data(), shape.data(), shape.size(), MLX_FLOAT32);
        };
        
        // Dimensions
        const int vocab_size = 128;
        const int d_model = 64;
        const int n_heads = 4;
        const int n_kv_heads = 2;
        const int head_dim = d_model / n_heads;
        const int intermediate_size = d_model * 4;
        
        // Create embedding weights
        weights["token_embedding.weight"] = create_random_weight({vocab_size, d_model});
        weights["norm.weight"] = create_random_weight({d_model});
        weights["norm.bias"] = create_random_weight({d_model});
        weights["output.weight"] = create_random_weight({vocab_size, d_model});
        
        // Create transformer layer weights
        for (int layer = 0; layer < 2; ++layer) {
            std::string prefix = "layers." + std::to_string(layer);
            
            // Attention weights
            weights[prefix + ".attention_norm.weight"] = create_random_weight({d_model});
            weights[prefix + ".attention_norm.bias"] = create_random_weight({d_model});
            weights[prefix + ".attention.wq.weight"] = create_random_weight({d_model, d_model});
            weights[prefix + ".attention.wk.weight"] = create_random_weight({d_model, head_dim * n_kv_heads});
            weights[prefix + ".attention.wv.weight"] = create_random_weight({d_model, head_dim * n_kv_heads});
            weights[prefix + ".attention.wo.weight"] = create_random_weight({d_model, d_model});
            
            // Feed-forward weights
            weights[prefix + ".ffn_norm.weight"] = create_random_weight({d_model});
            weights[prefix + ".ffn_norm.bias"] = create_random_weight({d_model});
            weights[prefix + ".ffn.w1.weight"] = create_random_weight({d_model, intermediate_size});
            weights[prefix + ".ffn.w2.weight"] = create_random_weight({intermediate_size, d_model});
            weights[prefix + ".ffn.w3.weight"] = create_random_weight({d_model, intermediate_size});
        }
        
        // Store weights and dimensions for tests
        test_weights_ = weights;
        d_model_ = d_model;
        n_heads_ = n_heads;
        n_kv_heads_ = n_kv_heads;
        vocab_size_ = vocab_size;
    }
    
    void TearDown() override {
        // Clean up all MLX arrays
        for (auto& pair : test_weights_) {
            if (pair.second.ctx) {
                mlx_array_free(pair.second);
            }
        }
        test_weights_.clear();
    }
    
    // Test weight map
    std::unordered_map<std::string, mlx_array> test_weights_;
    
    // Model dimensions
    int d_model_ = 0;
    int n_heads_ = 0;
    int n_kv_heads_ = 0;
    int vocab_size_ = 0;
    
    // Fixed example input
    std::vector<int> example_tokens = {1, 2, 3, 4, 5};
    std::vector<int> example_positions = {0, 1, 2, 3, 4};
};

#ifdef CCSM_WITH_MLX
// Test creating a transformer layer
TEST_F(MLXTransformerTest, CreateTransformerLayer) {
    CCSM_INFO("Testing MLXTransformerLayer creation");
    
    // Create a transformer layer
    MLXTransformerLayer layer("layers.0", test_weights_, d_model_, n_heads_, n_kv_heads_);
    
    // If creation doesn't throw, test passes
    EXPECT_EQ(layer.d_model(), d_model_);
    EXPECT_EQ(layer.n_heads(), n_heads_);
    EXPECT_EQ(layer.n_kv_heads(), n_kv_heads_);
    EXPECT_EQ(layer.head_dim(), d_model_ / n_heads_);
}

// Test creating a KV cache
TEST_F(MLXTransformerTest, CreateKVCache) {
    CCSM_INFO("Testing MLXKVCache creation");
    
    // Create a KV cache
    int n_layers = 2;
    int head_dim = d_model_ / n_heads_;
    int max_seq_len = 10;
    MLXKVCache cache(n_layers, n_kv_heads_, head_dim, max_seq_len);
    
    // Check cache properties
    EXPECT_EQ(cache.size(), n_layers);
    EXPECT_EQ(cache.max_seq_len(), max_seq_len);
    EXPECT_EQ(cache.current_seq_len(), 0);
}

// Test creating a transformer
TEST_F(MLXTransformerTest, CreateTransformer) {
    CCSM_INFO("Testing MLXTransformer creation");
    
    // Create a transformer
    int n_layers = 2;
    MLXTransformer transformer(
        test_weights_,
        d_model_,
        n_layers,
        n_heads_,
        n_kv_heads_,
        vocab_size_
    );
    
    // Check transformer properties
    EXPECT_EQ(transformer.d_model(), d_model_);
    EXPECT_EQ(transformer.n_layers(), n_layers);
    EXPECT_EQ(transformer.n_heads(), n_heads_);
    EXPECT_EQ(transformer.n_kv_heads(), n_kv_heads_);
    EXPECT_EQ(transformer.vocab_size(), vocab_size_);
}

// Test rotary position embeddings
TEST_F(MLXTransformerTest, RotaryEmbeddings) {
    CCSM_INFO("Testing rotary position embeddings");
    
    // Create input tensor with shape [batch=1, seq_len=3, head_dim=4]
    std::vector<float> input_data = {
        0.1f, 0.2f, 0.3f, 0.4f,  // token 1
        0.5f, 0.6f, 0.7f, 0.8f,  // token 2
        0.9f, 1.0f, 1.1f, 1.2f   // token 3
    };
    int input_shape[] = {1, 3, 4};
    mlx_array input = mlx_array_new_data(input_data.data(), input_shape, 3, MLX_FLOAT32);
    
    // Apply rotary embeddings
    mlx_stream stream = mlx_default_cpu_stream_new();
    std::vector<int> positions = {0, 1, 2};
    mlx_array output = mlx_rotary_embedding(input, positions, 10000.0f, stream);
    
    // Output should have the same shape as the input
    uint32_t ndim;
    mlx_array_ndim(output, &ndim);
    EXPECT_EQ(ndim, 3);
    
    const int* shape = mlx_array_shape(output);
    EXPECT_EQ(shape[0], 1);
    EXPECT_EQ(shape[1], 3);
    EXPECT_EQ(shape[2], 4);
    
    // Clean up
    mlx_array_free(input);
    mlx_array_free(output);
    mlx_stream_free(stream);
}

// Test attention mechanism
TEST_F(MLXTransformerTest, Attention) {
    CCSM_INFO("Testing attention mechanism");
    
    // Create query, key, value tensors with shape [batch=1, seq_len=3, head_dim=4]
    std::vector<float> q_data(12, 0.1f);
    std::vector<float> k_data(12, 0.2f);
    std::vector<float> v_data(12, 0.3f);
    
    int tensor_shape[] = {1, 3, 4};
    mlx_array query = mlx_array_new_data(q_data.data(), tensor_shape, 3, MLX_FLOAT32);
    mlx_array key = mlx_array_new_data(k_data.data(), tensor_shape, 3, MLX_FLOAT32);
    mlx_array value = mlx_array_new_data(v_data.data(), tensor_shape, 3, MLX_FLOAT32);
    
    // Apply attention
    mlx_stream stream = mlx_default_cpu_stream_new();
    std::vector<int> positions = {0, 1, 2};
    mlx_array output = mlx_attention(query, key, value, positions, 0.5f, stream);
    
    // Output should have the same shape as the query
    uint32_t ndim;
    mlx_array_ndim(output, &ndim);
    EXPECT_EQ(ndim, 3);
    
    const int* shape = mlx_array_shape(output);
    EXPECT_EQ(shape[0], 1);
    EXPECT_EQ(shape[1], 3);
    EXPECT_EQ(shape[2], 4);
    
    // Clean up
    mlx_array_free(query);
    mlx_array_free(key);
    mlx_array_free(value);
    mlx_array_free(output);
    mlx_stream_free(stream);
}

// Test feed-forward network
TEST_F(MLXTransformerTest, FeedForward) {
    CCSM_INFO("Testing feed-forward network");
    
    // Create input tensor with shape [batch=1, seq_len=2, d_model=4]
    std::vector<float> input_data = {
        0.1f, 0.2f, 0.3f, 0.4f,  // token 1
        0.5f, 0.6f, 0.7f, 0.8f   // token 2
    };
    int input_shape[] = {1, 2, 4};
    mlx_array input = mlx_array_new_data(input_data.data(), input_shape, 3, MLX_FLOAT32);
    
    // Create weight tensors
    int w1_shape[] = {4, 8};  // [d_model, intermediate]
    int w2_shape[] = {8, 4};  // [intermediate, d_model]
    int w3_shape[] = {4, 8};  // [d_model, intermediate]
    
    std::vector<float> w1_data(32, 0.01f);
    std::vector<float> w2_data(32, 0.02f);
    std::vector<float> w3_data(32, 0.03f);
    
    mlx_array w1 = mlx_array_new_data(w1_data.data(), w1_shape, 2, MLX_FLOAT32);
    mlx_array w2 = mlx_array_new_data(w2_data.data(), w2_shape, 2, MLX_FLOAT32);
    mlx_array w3 = mlx_array_new_data(w3_data.data(), w3_shape, 2, MLX_FLOAT32);
    
    // Apply feed-forward network
    mlx_stream stream = mlx_default_cpu_stream_new();
    mlx_array output = mlx_feed_forward(input, w1, w2, w3, stream);
    
    // Output should have the same shape as the input
    uint32_t ndim;
    mlx_array_ndim(output, &ndim);
    EXPECT_EQ(ndim, 3);
    
    const int* shape = mlx_array_shape(output);
    EXPECT_EQ(shape[0], 1);
    EXPECT_EQ(shape[1], 2);
    EXPECT_EQ(shape[2], 4);
    
    // Clean up
    mlx_array_free(input);
    mlx_array_free(w1);
    mlx_array_free(w2);
    mlx_array_free(w3);
    mlx_array_free(output);
    mlx_stream_free(stream);
}

// Test transformer layer forward pass
TEST_F(MLXTransformerTest, TransformerLayerForward) {
    CCSM_INFO("Testing transformer layer forward pass");
    
    // Create a transformer layer
    MLXTransformerLayer layer("layers.0", test_weights_, d_model_, n_heads_, n_kv_heads_);
    
    // Create input tensor with shape [batch=1, seq_len=3, d_model]
    std::vector<float> input_data(3 * d_model_, 0.1f);
    int input_shape[] = {1, 3, d_model_};
    mlx_array input = mlx_array_new_data(input_data.data(), input_shape, 3, MLX_FLOAT32);
    
    // Create a KV cache
    MLXKVCache cache(1, n_kv_heads_, d_model_ / n_heads_, 10);
    
    // Apply forward pass
    mlx_stream stream = mlx_default_cpu_stream_new();
    std::vector<int> positions = {0, 1, 2};
    mlx_array output = layer.forward(input, &cache, positions, stream);
    
    // Output should have the same shape as the input
    uint32_t ndim;
    mlx_array_ndim(output, &ndim);
    EXPECT_EQ(ndim, 3);
    
    const int* shape = mlx_array_shape(output);
    EXPECT_EQ(shape[0], 1);
    EXPECT_EQ(shape[1], 3);
    EXPECT_EQ(shape[2], d_model_);
    
    // Clean up
    mlx_array_free(input);
    mlx_array_free(output);
    mlx_stream_free(stream);
}

// Test KV cache operations
TEST_F(MLXTransformerTest, KVCacheOperations) {
    CCSM_INFO("Testing KV cache operations");
    
    // Create a KV cache
    int n_layers = 2;
    int head_dim = d_model_ / n_heads_;
    int max_seq_len = 10;
    MLXKVCache cache(n_layers, n_kv_heads_, head_dim, max_seq_len);
    
    // Check initial state
    EXPECT_EQ(cache.current_seq_len(), 0);
    
    // Create key and value tensors for updating
    int tensor_shape[] = {1, 3, n_kv_heads_, head_dim};
    std::vector<float> k_data(3 * n_kv_heads_ * head_dim, 0.1f);
    std::vector<float> v_data(3 * n_kv_heads_ * head_dim, 0.2f);
    
    mlx_array key = mlx_array_new_data(k_data.data(), tensor_shape, 4, MLX_FLOAT32);
    mlx_array value = mlx_array_new_data(v_data.data(), tensor_shape, 4, MLX_FLOAT32);
    
    // Update cache
    mlx_stream stream = mlx_default_cpu_stream_new();
    std::vector<int> positions = {0, 1, 2};
    cache.update(0, key, value, positions, stream);
    
    // Check updated state
    EXPECT_EQ(cache.current_seq_len(), 3);
    
    // Get cached values
    mlx_array cached_k = cache.k_cache(0);
    mlx_array cached_v = cache.v_cache(0);
    
    // Cached arrays should be valid
    EXPECT_TRUE(cached_k.ctx != nullptr);
    EXPECT_TRUE(cached_v.ctx != nullptr);
    
    // Clear cache
    cache.clear();
    EXPECT_EQ(cache.current_seq_len(), 0);
    
    // Resize cache
    cache.resize(5);
    EXPECT_EQ(cache.current_seq_len(), 5);
    
    // Clean up
    mlx_array_free(key);
    mlx_array_free(value);
    mlx_stream_free(stream);
}

// Test transformer forward pass
TEST_F(MLXTransformerTest, TransformerForward) {
    CCSM_INFO("Testing transformer forward pass");
    
    // Create a transformer
    int n_layers = 2;
    MLXTransformer transformer(
        test_weights_,
        d_model_,
        n_layers,
        n_heads_,
        n_kv_heads_,
        vocab_size_
    );
    
    // Apply forward pass
    mlx_stream stream = mlx_default_cpu_stream_new();
    mlx_array logits = transformer.forward(example_tokens, example_positions, nullptr, stream);
    
    // Logits should have a shape related to vocab_size
    EXPECT_TRUE(logits.ctx != nullptr);
    
    uint32_t ndim;
    mlx_array_ndim(logits, &ndim);
    
    // Output shape should be [1, vocab_size]
    if (ndim > 0) {
        const int* shape = mlx_array_shape(logits);
        if (ndim == 1) {
            EXPECT_EQ(shape[0], vocab_size_);
        } else if (ndim == 2) {
            EXPECT_EQ(shape[1], vocab_size_);
        }
    }
    
    // Reset caches
    transformer.reset_caches();
    
    // Clean up
    mlx_array_free(logits);
    mlx_stream_free(stream);
}
#endif // CCSM_WITH_MLX

} // namespace testing
} // namespace ccsm