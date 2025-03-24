#include <gtest/gtest.h>
#include <ccsm/model.h>
#include <ccsm/cpu/ggml_model.h>
#ifdef CCSM_WITH_MLX
#include <ccsm/mlx/mlx_model.h>
#endif
#include <vector>
#include <string>
#include <memory>

namespace ccsm {
namespace testing {

// Test fixture for unified model interface
class UnifiedModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a model config for testing
        config.name = "test_model";
        config.d_model = 512;
        config.n_heads = 8;
        config.n_layers = 6;
        config.max_seq_len = 512;
        config.vocab_size = 32000;
        config.audio_vocab_size = 1024;
        config.n_audio_layers = 4;
        config.num_codebooks = 8;
    }
    
    void TearDown() override {
        // Clean up
    }
    
    // Create test model instances
    std::shared_ptr<Model> create_ggml_model() {
        return std::make_shared<GGMLModel>(config);
    }
    
    #ifdef CCSM_WITH_MLX
    std::shared_ptr<Model> create_mlx_model() {
        return std::make_shared<MLXModel>(config);
    }
    #endif
    
    // Common test data
    ModelConfig config;
    std::vector<int> test_tokens = {1, 2, 3, 4, 5};
    std::vector<int> test_positions = {0, 1, 2, 3, 4};
};

// Test that model configurations are correctly stored
TEST_F(UnifiedModelTest, ModelConfigTest) {
    auto model = create_ggml_model();
    
    EXPECT_EQ(model->config().name, "test_model");
    EXPECT_EQ(model->config().d_model, 512);
    EXPECT_EQ(model->config().n_heads, 8);
    EXPECT_EQ(model->config().n_layers, 6);
    EXPECT_EQ(model->config().max_seq_len, 512);
    EXPECT_EQ(model->config().vocab_size, 32000);
    EXPECT_EQ(model->config().audio_vocab_size, 1024);
    EXPECT_EQ(model->config().n_audio_layers, 4);
    EXPECT_EQ(model->config().num_codebooks, 8);
}

// Test both backends to ensure consistent interface behavior
TEST_F(UnifiedModelTest, BothBackendsTest) {
    // Create models for different backends
    auto ggml_model = create_ggml_model();
    
    // Common operations to test interface consistency
    ggml_model->reset_caches();
    ggml_model->optimize_memory(128);
    ggml_model->prune_caches(0.5f);
    
    #ifdef CCSM_WITH_MLX
    auto mlx_model = create_mlx_model();
    mlx_model->reset_caches();
    mlx_model->optimize_memory(128);
    mlx_model->prune_caches(0.5f);
    
    // Both models should have the same config values
    EXPECT_EQ(ggml_model->config().d_model, mlx_model->config().d_model);
    EXPECT_EQ(ggml_model->config().n_heads, mlx_model->config().n_heads);
    EXPECT_EQ(ggml_model->config().n_layers, mlx_model->config().n_layers);
    #endif
}

// Test that generate_frame produces the expected number of tokens
TEST_F(UnifiedModelTest, GenerateFrameTest) {
    auto model = create_ggml_model();
    
    auto frame = model->generate_frame(test_tokens, test_positions);
    
    // Should produce one token per codebook
    EXPECT_EQ(frame.size(), config.num_codebooks);
}

// Test that model logits have the expected size
TEST_F(UnifiedModelTest, LogitsTest) {
    auto model = create_ggml_model();
    
    // Backbone logits should match vocab size
    auto backbone_logits = model->get_backbone_logits(test_tokens, test_positions);
    EXPECT_EQ(backbone_logits.size(), config.vocab_size);
    
    // Decoder logits should match audio vocab size
    auto decoder_logits = model->get_decoder_logits(test_tokens, test_positions, 0);
    EXPECT_EQ(decoder_logits.size(), config.audio_vocab_size);
}

// Test MLX-specific functionality when available
#ifdef CCSM_WITH_MLX
TEST_F(UnifiedModelTest, MLXModelTest) {
    auto mlx_model = create_mlx_model();
    
    // Test MLX-specific interface
    auto mlx_model_ptr = std::dynamic_pointer_cast<MLXModel>(mlx_model);
    ASSERT_TRUE(mlx_model_ptr != nullptr);
    
    // Test MLX-specific methods
    EXPECT_FALSE(mlx_model_ptr->has_weight("nonexistent_weight"));
    
    // Create transformer context
    auto transformer_ctx = mlx_model_ptr->create_transformer_context();
    EXPECT_TRUE(transformer_ctx != nullptr);
}
#endif

} // namespace testing
} // namespace ccsm