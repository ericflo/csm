#include <gtest/gtest.h>
#include <ccsm/mlx/mlx_model.h>
#include <ccsm/mlx/mlx_tensor.h>
#include <ccsm/model_loader.h>
#include <memory>
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>

namespace ccsm {
namespace testing {

// Mock MLX model loader for testing
class MockMLXModelLoader : public ModelLoader {
public:
    MockMLXModelLoader(const std::string& path) : ModelLoader(path) {}
    
    bool load(WeightMap& weights) override {
        // Add mock weights
        weights["tok_embeddings.weight"] = TensorFactory::zeros({100, 128}, DataType::F32);
        weights["norm.weight"] = TensorFactory::ones({128}, DataType::F32);
        weights["norm.bias"] = TensorFactory::zeros({128}, DataType::F32);
        weights["output.weight"] = TensorFactory::zeros({100, 128}, DataType::F32);
        
        // Layer weights
        for (int i = 0; i < 2; ++i) {
            std::string prefix = "layers." + std::to_string(i) + ".";
            weights[prefix + "attention_norm.weight"] = TensorFactory::ones({128}, DataType::F32);
            weights[prefix + "attention_norm.bias"] = TensorFactory::zeros({128}, DataType::F32);
            weights[prefix + "attention.wq.weight"] = TensorFactory::zeros({128, 128}, DataType::F32);
            weights[prefix + "attention.wk.weight"] = TensorFactory::zeros({128, 128}, DataType::F32);
            weights[prefix + "attention.wv.weight"] = TensorFactory::zeros({128, 128}, DataType::F32);
            weights[prefix + "attention.wo.weight"] = TensorFactory::zeros({128, 128}, DataType::F32);
            weights[prefix + "ffn_norm.weight"] = TensorFactory::ones({128}, DataType::F32);
            weights[prefix + "ffn_norm.bias"] = TensorFactory::zeros({128}, DataType::F32);
            weights[prefix + "feed_forward.w1.weight"] = TensorFactory::zeros({128, 512}, DataType::F32);
            weights[prefix + "feed_forward.w2.weight"] = TensorFactory::zeros({512, 128}, DataType::F32);
            weights[prefix + "feed_forward.w3.weight"] = TensorFactory::zeros({128, 512}, DataType::F32);
        }
        
        return true;
    }
};

class MLXModelTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test MLX availability
TEST_F(MLXModelTest, TestMLXAvailability) {
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
// Test MLX model creation
TEST_F(MLXModelTest, TestMLXModelCreation) {
    // Skip if MLX is not available
    if (!MLXContext::is_available()) {
        GTEST_SKIP() << "MLX not available, skipping test";
    }
    
    // Create model configuration
    ModelConfig config;
    config.d_model = 128;
    config.n_layers = 2;
    config.n_heads = 4;
    config.n_kv_heads = 4;
    config.vocab_size = 100;
    config.max_seq_len = 16;
    config.rope_theta = 10000.0f;
    
    // Create MLX model
    MLXModel model(config);
    
    // Load model weights from a mock loader
    std::shared_ptr<ModelLoader> loader = std::make_shared<MockMLXModelLoader>("/path/to/mock/model");
    bool load_success = model.load_weights(loader);
    
    // Verify load succeeded
    EXPECT_TRUE(load_success);
}

// Test MLX model forward pass
TEST_F(MLXModelTest, TestMLXModelForward) {
    // Skip if MLX is not available
    if (!MLXContext::is_available()) {
        GTEST_SKIP() << "MLX not available, skipping test";
    }
    
    // Create model configuration
    ModelConfig config;
    config.d_model = 128;
    config.n_layers = 2;
    config.n_heads = 4;
    config.n_kv_heads = 4;
    config.vocab_size = 100;
    config.max_seq_len = 16;
    config.rope_theta = 10000.0f;
    
    // Create MLX model
    MLXModel model(config);
    
    // Load model weights from a mock loader
    std::shared_ptr<ModelLoader> loader = std::make_shared<MockMLXModelLoader>("/path/to/mock/model");
    bool load_success = model.load_weights(loader);
    EXPECT_TRUE(load_success);
    
    // Input tokens and positions
    std::vector<int> tokens = {1, 2, 3};
    std::vector<int> positions = {0, 1, 2};
    
    // Generate a frame
    std::vector<int> frame = model.generate_frame(tokens, positions, 0.9f, 50);
    
    // Verify frame was generated
    EXPECT_FALSE(frame.empty());
    
    // Get backbone logits
    std::vector<float> backbone_logits = model.get_backbone_logits(tokens, positions);
    
    // Verify backbone logits
    EXPECT_EQ(backbone_logits.size(), config.vocab_size);
    
    // Get decoder logits for codebook 0
    std::vector<float> decoder_logits = model.get_decoder_logits(tokens, positions, 0);
    
    // Verify decoder logits
    EXPECT_EQ(decoder_logits.size(), config.vocab_size);
    
    // Reset caches
    model.reset_caches();
    
    // Test memory optimization
    model.optimize_memory(1024);
    model.prune_caches(0.5f);
}

// Test MLX model weight access
TEST_F(MLXModelTest, TestMLXModelWeightAccess) {
    // Skip if MLX is not available
    if (!MLXContext::is_available()) {
        GTEST_SKIP() << "MLX not available, skipping test";
    }
    
    // Create model configuration
    ModelConfig config;
    config.d_model = 128;
    config.n_layers = 2;
    config.n_heads = 4;
    config.n_kv_heads = 4;
    config.vocab_size = 100;
    config.max_seq_len = 16;
    config.rope_theta = 10000.0f;
    
    // Create MLX model
    MLXModel model(config);
    
    // Load model weights from a mock loader
    std::shared_ptr<ModelLoader> loader = std::make_shared<MockMLXModelLoader>("/path/to/mock/model");
    bool load_success = model.load_weights(loader);
    EXPECT_TRUE(load_success);
    
    // Check weight existence
    EXPECT_TRUE(model.has_weight("tok_embeddings.weight"));
    EXPECT_TRUE(model.has_weight("norm.weight"));
    EXPECT_TRUE(model.has_weight("layers.0.attention_norm.weight"));
    EXPECT_FALSE(model.has_weight("nonexistent.weight"));
    
    // Get weight array
    mlx_array tok_embeddings = model.get_weight_array("tok_embeddings.weight");
    
    // Verify array
    uint32_t ndim;
    mlx_array_ndim(tok_embeddings, &ndim);
    EXPECT_EQ(ndim, 2);
    
    std::vector<int64_t> shape(ndim);
    mlx_array_shape(tok_embeddings, shape.data());
    EXPECT_EQ(shape[0], 100);
    EXPECT_EQ(shape[1], 128);
    
    // Free arrays
    mlx_array_free(tok_embeddings);
}

// Test transformer context creation
TEST_F(MLXModelTest, TestTransformerContextCreation) {
    // Skip if MLX is not available
    if (!MLXContext::is_available()) {
        GTEST_SKIP() << "MLX not available, skipping test";
    }
    
    // Create model configuration
    ModelConfig config;
    config.d_model = 128;
    config.n_layers = 2;
    config.n_heads = 4;
    config.n_kv_heads = 4;
    config.vocab_size = 100;
    config.max_seq_len = 16;
    config.rope_theta = 10000.0f;
    
    // Create MLX model
    MLXModel model(config);
    
    // Load model weights from a mock loader
    std::shared_ptr<ModelLoader> loader = std::make_shared<MockMLXModelLoader>("/path/to/mock/model");
    bool load_success = model.load_weights(loader);
    EXPECT_TRUE(load_success);
    
    // Create transformer context
    std::shared_ptr<MLXTransformer> transformer = model.create_transformer_context();
    
    // Verify transformer context was created
    EXPECT_NE(transformer.get(), nullptr);
    
    // Verify transformer parameters
    EXPECT_EQ(transformer->d_model(), config.d_model);
    EXPECT_EQ(transformer->n_layers(), config.n_layers);
    EXPECT_EQ(transformer->n_heads(), config.n_heads);
    EXPECT_EQ(transformer->n_kv_heads(), config.n_kv_heads);
    EXPECT_EQ(transformer->vocab_size(), config.vocab_size);
}

// Test weight loading from different sources
TEST_F(MLXModelTest, TestWeightLoading) {
    // Skip if MLX is not available
    if (!MLXContext::is_available()) {
        GTEST_SKIP() << "MLX not available, skipping test";
    }
    
    // Create model configuration
    ModelConfig config;
    config.d_model = 128;
    config.n_layers = 2;
    config.n_heads = 4;
    config.n_kv_heads = 4;
    config.vocab_size = 100;
    config.max_seq_len = 16;
    config.rope_theta = 10000.0f;
    
    // Create MLX model
    MLXModel model(config);
    
    // Create weight map
    WeightMap weights;
    weights["tok_embeddings.weight"] = TensorFactory::zeros({100, 128}, DataType::F32);
    weights["norm.weight"] = TensorFactory::ones({128}, DataType::F32);
    weights["norm.bias"] = TensorFactory::zeros({128}, DataType::F32);
    weights["output.weight"] = TensorFactory::zeros({100, 128}, DataType::F32);
    
    // Layer weights
    for (int i = 0; i < 2; ++i) {
        std::string prefix = "layers." + std::to_string(i) + ".";
        weights[prefix + "attention_norm.weight"] = TensorFactory::ones({128}, DataType::F32);
        weights[prefix + "attention_norm.bias"] = TensorFactory::zeros({128}, DataType::F32);
        weights[prefix + "attention.wq.weight"] = TensorFactory::zeros({128, 128}, DataType::F32);
        weights[prefix + "attention.wk.weight"] = TensorFactory::zeros({128, 128}, DataType::F32);
        weights[prefix + "attention.wv.weight"] = TensorFactory::zeros({128, 128}, DataType::F32);
        weights[prefix + "attention.wo.weight"] = TensorFactory::zeros({128, 128}, DataType::F32);
        weights[prefix + "ffn_norm.weight"] = TensorFactory::ones({128}, DataType::F32);
        weights[prefix + "ffn_norm.bias"] = TensorFactory::zeros({128}, DataType::F32);
        weights[prefix + "feed_forward.w1.weight"] = TensorFactory::zeros({128, 512}, DataType::F32);
        weights[prefix + "feed_forward.w2.weight"] = TensorFactory::zeros({512, 128}, DataType::F32);
        weights[prefix + "feed_forward.w3.weight"] = TensorFactory::zeros({128, 512}, DataType::F32);
    }
    
    // Load weights directly
    bool load_success = model.load_weights(weights);
    EXPECT_TRUE(load_success);
    
    // Create temporary file for testing
    std::string temp_dir = std::filesystem::temp_directory_path().string();
    std::string model_path = temp_dir + "/test_mlx_model.bin";
    
    {
        std::ofstream file(model_path, std::ios::binary);
        EXPECT_TRUE(file.is_open());
        file << "MOCK_MODEL_DATA";
        file.close();
    }
    
    // Register a mock loader for this file extension
    ModelLoaderRegistry::register_loader(".bin", [](const std::string& path) {
        return std::make_shared<MockMLXModelLoader>(path);
    });
    
    // Create a new model
    MLXModel model2(config);
    
    // Load weights from file
    load_success = model2.load_weights(model_path);
    EXPECT_TRUE(load_success);
    
    // Clean up temporary file
    std::filesystem::remove(model_path);
}
#endif // CCSM_WITH_MLX

} // namespace testing
} // namespace ccsm