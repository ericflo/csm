#include <gtest/gtest.h>
#include <ccsm/cpu/ggml_model.h>
#include <ccsm/cpu/ggml_tensor.h>
#include <ccsm/model.h>
#include <ccsm/tensor.h>
#include <vector>
#include <memory>

using namespace ccsm;

// Test fixture for KVCache tests
class KVCacheTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize KV cache
        n_layers = 12;
        n_heads = 32;
        n_kv_heads = 4;
        head_dim = 128;
        max_seq_len = 2048;
        
        kv_cache = std::make_shared<KVCache>(n_layers, n_heads, n_kv_heads, head_dim, max_seq_len);
    }
    
    size_t n_layers;
    size_t n_heads;
    size_t n_kv_heads;
    size_t head_dim;
    size_t max_seq_len;
    std::shared_ptr<KVCache> kv_cache;
};

// Test KVCache initialization
TEST_F(KVCacheTest, InitializeCache) {
    // Check dimensions
    EXPECT_EQ(kv_cache->size(), n_layers * 2);
    EXPECT_EQ(kv_cache->max_seq_len(), max_seq_len);
    EXPECT_EQ(kv_cache->current_seq_len(), 0);
    
    // Check cache access
    for (size_t i = 0; i < n_layers; i++) {
        // Get K cache tensor for layer i
        struct ggml_tensor* k = kv_cache->k_cache(i);
        EXPECT_NE(k, nullptr);
        EXPECT_EQ(k->ne[0], head_dim);
        EXPECT_EQ(k->ne[1], n_kv_heads);
        EXPECT_EQ(k->ne[2], max_seq_len);
        
        // Get V cache tensor for layer i
        struct ggml_tensor* v = kv_cache->v_cache(i);
        EXPECT_NE(v, nullptr);
        EXPECT_EQ(v->ne[0], head_dim);
        EXPECT_EQ(v->ne[1], n_kv_heads);
        EXPECT_EQ(v->ne[2], max_seq_len);
    }
}

// Test KVCache resize and clear operations
TEST_F(KVCacheTest, ResizeAndClearCache) {
    // Resize
    size_t new_seq_len = 100;
    kv_cache->resize(new_seq_len);
    EXPECT_EQ(kv_cache->current_seq_len(), new_seq_len);
    
    // Clear
    kv_cache->clear();
    EXPECT_EQ(kv_cache->current_seq_len(), 0);
    
    // Try to resize beyond max - should throw
    EXPECT_THROW(kv_cache->resize(max_seq_len + 1), std::runtime_error);
    
    // Test out of bounds access
    EXPECT_THROW(kv_cache->k_cache(-1), std::out_of_range);
    EXPECT_THROW(kv_cache->k_cache(n_layers), std::out_of_range);
    EXPECT_THROW(kv_cache->v_cache(-1), std::out_of_range);
    EXPECT_THROW(kv_cache->v_cache(n_layers), std::out_of_range);
}

// Test fixture for GGMLModel tests
class GGMLModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create model configuration
        config.vocab_size = 32000;
        config.audio_vocab_size = 2051;
        config.d_model = 4096;
        config.n_heads = 32;
        config.n_kv_heads = 4;
        config.n_layers = 32;
        config.n_audio_layers = 12;
        config.max_seq_len = 2048;
        config.num_codebooks = 32;
        config.name = "test-model";
        
        // Create model
        model = std::make_shared<GGMLModel>(config);
    }
    
    ModelConfig config;
    std::shared_ptr<GGMLModel> model;
};

// Test model initialization
TEST_F(GGMLModelTest, InitializeModel) {
    // Just verify that the model was created without crashing
    EXPECT_EQ(model->config().name, "test-model");
    EXPECT_EQ(model->config().vocab_size, 32000);
    EXPECT_EQ(model->config().audio_vocab_size, 2051);
}

// Test frame generation
TEST_F(GGMLModelTest, GenerateFrame) {
    // Create dummy input
    std::vector<int> tokens = {1, 2, 3, 4, 5};
    std::vector<int> positions = {0, 1, 2, 3, 4};
    
    // Generate a frame with different parameters
    std::vector<int> frame1 = model->generate_frame(tokens, positions, 0.8f, 50);
    EXPECT_EQ(frame1.size(), config.num_codebooks);
    
    // Greedy sampling (temperature = 0)
    std::vector<int> frame2 = model->generate_frame(tokens, positions, 0.0f, 50);
    EXPECT_EQ(frame2.size(), config.num_codebooks);
    
    // Different top-k
    std::vector<int> frame3 = model->generate_frame(tokens, positions, 0.8f, 10);
    EXPECT_EQ(frame3.size(), config.num_codebooks);
    
    // Test cache reset
    model->reset_caches();
}

// Test logits generation
TEST_F(GGMLModelTest, GetLogits) {
    // Create dummy input
    std::vector<int> tokens = {1, 2, 3, 4, 5};
    std::vector<int> positions = {0, 1, 2, 3, 4};
    
    // Get backbone logits
    std::vector<float> backbone_logits = model->get_backbone_logits(tokens, positions);
    EXPECT_EQ(backbone_logits.size(), config.audio_vocab_size);
    
    // Get decoder logits for various codebooks
    for (int codebook = 1; codebook < 5; codebook++) {
        std::vector<float> decoder_logits = model->get_decoder_logits(tokens, positions, codebook);
        EXPECT_EQ(decoder_logits.size(), config.audio_vocab_size);
    }
}

