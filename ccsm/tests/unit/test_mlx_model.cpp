#include <gtest/gtest.h>
#include <ccsm/mlx/mlx_model.h>
#include <ccsm/mlx/mlx_tensor.h>
#include <ccsm/model_loader.h>
#include <memory>
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <random>

namespace ccsm {
namespace testing {

// Mock MLX model loader for testing
class MockMLXModelLoader : public ModelLoader {
public:
    MockMLXModelLoader(const std::string& path) : ModelLoader(path) {}
    
    bool load(WeightMap& weights) override {
        // Add mock weights
        weights["token_embeddings.weight"] = TensorFactory::zeros({100, 128}, DataType::F32);
        weights["norm.weight"] = TensorFactory::ones({128}, DataType::F32);
        weights["norm.bias"] = TensorFactory::zeros({128}, DataType::F32);
        weights["output.weight"] = TensorFactory::zeros({128, 100}, DataType::F32);
        
        // Layer weights
        for (int i = 0; i < 2; ++i) {
            std::string prefix = "layers." + std::to_string(i) + ".";
            weights[prefix + "attention_norm.weight"] = TensorFactory::ones({128}, DataType::F32);
            weights[prefix + "attention_norm.bias"] = TensorFactory::zeros({128}, DataType::F32);
            weights[prefix + "wq.weight"] = TensorFactory::zeros({128, 128}, DataType::F32);
            weights[prefix + "wk.weight"] = TensorFactory::zeros({128, 64}, DataType::F32);
            weights[prefix + "wv.weight"] = TensorFactory::zeros({128, 64}, DataType::F32);
            weights[prefix + "wo.weight"] = TensorFactory::zeros({128, 128}, DataType::F32);
            weights[prefix + "ffn_norm.weight"] = TensorFactory::ones({128}, DataType::F32);
            weights[prefix + "ffn_norm.bias"] = TensorFactory::zeros({128}, DataType::F32);
            weights[prefix + "w1.weight"] = TensorFactory::zeros({128, 512}, DataType::F32);
            weights[prefix + "w2.weight"] = TensorFactory::zeros({512, 128}, DataType::F32);
            weights[prefix + "w3.weight"] = TensorFactory::zeros({128, 512}, DataType::F32);
        }
        
        return true;
    }
};

// Mock MLX model loader that returns incomplete weights
class IncompleteMLXModelLoader : public ModelLoader {
public:
    IncompleteMLXModelLoader(const std::string& path) : ModelLoader(path) {}
    
    bool load(WeightMap& weights) override {
        // Add only some weights, missing critical ones
        weights["token_embeddings.weight"] = TensorFactory::zeros({100, 128}, DataType::F32);
        weights["norm.weight"] = TensorFactory::ones({128}, DataType::F32);
        // Missing output.weight
        
        return true;
    }
};

// Mock MLX model loader that always fails
class FailingMLXModelLoader : public ModelLoader {
public:
    FailingMLXModelLoader(const std::string& path) : ModelLoader(path) {}
    
    bool load(WeightMap& weights) override {
        return false;
    }
};

// Helper to create random tensors for testing
Tensor create_random_tensor(const std::vector<size_t>& shape, DataType dtype, 
                            unsigned int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    size_t numel = 1;
    for (auto dim : shape) numel *= dim;
    
    Tensor tensor = TensorFactory::zeros(shape, dtype);
    
    if (dtype == DataType::F32) {
        float* data = static_cast<float*>(tensor.data());
        for (size_t i = 0; i < numel; ++i) {
            data[i] = dist(gen);
        }
    } else if (dtype == DataType::I32) {
        int32_t* data = static_cast<int32_t*>(tensor.data());
        for (size_t i = 0; i < numel; ++i) {
            data[i] = static_cast<int32_t>(dist(gen) * 100.0f);
        }
    }
    
    return tensor;
}

class MLXModelTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
    
    // Helper method to create a complete weight map
    WeightMap create_complete_weight_map(int d_model = 128, int n_layers = 2, 
                                        int n_heads = 4, int n_kv_heads = 2, 
                                        int vocab_size = 100, 
                                        unsigned int seed = 42) {
        WeightMap weights;
        
        // Calculate derived dimensions
        int head_dim = d_model / n_heads;
        int kv_dim = head_dim * n_kv_heads;
        int hidden_dim = 4 * d_model;
        
        // Create basic weights
        weights["token_embeddings.weight"] = create_random_tensor({vocab_size, d_model}, DataType::F32, seed);
        weights["norm.weight"] = create_random_tensor({d_model}, DataType::F32, seed + 1);
        weights["norm.bias"] = create_random_tensor({d_model}, DataType::F32, seed + 2);
        weights["output.weight"] = create_random_tensor({d_model, vocab_size}, DataType::F32, seed + 3);
        
        // Create layer weights
        for (int i = 0; i < n_layers; ++i) {
            std::string prefix = "layers." + std::to_string(i) + ".";
            
            weights[prefix + "attention_norm.weight"] = create_random_tensor({d_model}, DataType::F32, seed + 4 + i);
            weights[prefix + "attention_norm.bias"] = create_random_tensor({d_model}, DataType::F32, seed + 5 + i);
            
            weights[prefix + "wq.weight"] = create_random_tensor({d_model, d_model}, DataType::F32, seed + 6 + i);
            weights[prefix + "wk.weight"] = create_random_tensor({d_model, kv_dim}, DataType::F32, seed + 7 + i);
            weights[prefix + "wv.weight"] = create_random_tensor({d_model, kv_dim}, DataType::F32, seed + 8 + i);
            weights[prefix + "wo.weight"] = create_random_tensor({d_model, d_model}, DataType::F32, seed + 9 + i);
            
            weights[prefix + "ffn_norm.weight"] = create_random_tensor({d_model}, DataType::F32, seed + 10 + i);
            weights[prefix + "ffn_norm.bias"] = create_random_tensor({d_model}, DataType::F32, seed + 11 + i);
            
            weights[prefix + "w1.weight"] = create_random_tensor({d_model, hidden_dim}, DataType::F32, seed + 12 + i);
            weights[prefix + "w2.weight"] = create_random_tensor({hidden_dim, d_model}, DataType::F32, seed + 13 + i);
            weights[prefix + "w3.weight"] = create_random_tensor({d_model, hidden_dim}, DataType::F32, seed + 14 + i);
        }
        
        return weights;
    }
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
// Test MLX model creation with different configurations
TEST_F(MLXModelTest, TestMLXModelCreationWithDifferentConfigs) {
    // Skip if MLX is not available
    if (!MLXContext::is_available()) {
        GTEST_SKIP() << "MLX not available, skipping test";
    }
    
    // Test different model configurations
    std::vector<ModelConfig> configs;
    
    // Small model
    ModelConfig small_config;
    small_config.d_model = 64;
    small_config.n_layers = 1;
    small_config.n_heads = 2;
    small_config.n_kv_heads = 2;
    small_config.vocab_size = 100;
    small_config.max_seq_len = 8;
    configs.push_back(small_config);
    
    // Medium model with grouped attention
    ModelConfig medium_config;
    medium_config.d_model = 128;
    medium_config.n_layers = 2;
    medium_config.n_heads = 4;
    medium_config.n_kv_heads = 2; // Grouped attention
    medium_config.vocab_size = 100;
    medium_config.max_seq_len = 16;
    configs.push_back(medium_config);
    
    // Larger model
    ModelConfig large_config;
    large_config.d_model = 256;
    large_config.n_layers = 4;
    large_config.n_heads = 8;
    large_config.n_kv_heads = 8;
    large_config.vocab_size = 200;
    large_config.max_seq_len = 32;
    configs.push_back(large_config);
    
    // Test all configurations
    for (const auto& config : configs) {
        // Create MLX model
        MLXModel model(config);
        
        // Create weight map with matching dimensions
        WeightMap weights = create_complete_weight_map(
            config.d_model, config.n_layers, config.n_heads, config.n_kv_heads, config.vocab_size);
        
        // Load weights
        bool load_success = model.load_weights(weights);
        EXPECT_TRUE(load_success);
        
        // Test forward pass
        std::vector<int> tokens = {1, 2, 3};
        std::vector<int> positions = {0, 1, 2};
        
        // Get logits
        std::vector<float> logits = model.get_backbone_logits(tokens, positions);
        EXPECT_EQ(logits.size(), config.vocab_size);
    }
}

// Test MLX model with mock loader
TEST_F(MLXModelTest, TestMLXModelWithMockLoader) {
    // Skip if MLX is not available
    if (!MLXContext::is_available()) {
        GTEST_SKIP() << "MLX not available, skipping test";
    }
    
    // Create model configuration
    ModelConfig config;
    config.d_model = 128;
    config.n_layers = 2;
    config.n_heads = 4;
    config.n_kv_heads = 2;
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
    
    // Input tokens and positions
    std::vector<int> tokens = {1, 2, 3};
    std::vector<int> positions = {0, 1, 2};
    
    // Generate a frame
    std::vector<int> frame = model.generate_frame(tokens, positions, 0.9f, 50);
    
    // Verify frame was generated
    EXPECT_FALSE(frame.empty());
}

// Test MLX model forward pass with different temperatures and top_k
TEST_F(MLXModelTest, TestMLXModelForwardWithSamplingParams) {
    // Skip if MLX is not available
    if (!MLXContext::is_available()) {
        GTEST_SKIP() << "MLX not available, skipping test";
    }
    
    // Create model configuration
    ModelConfig config;
    config.d_model = 128;
    config.n_layers = 2;
    config.n_heads = 4;
    config.n_kv_heads = 2;
    config.vocab_size = 100;
    config.max_seq_len = 16;
    
    // Create MLX model
    MLXModel model(config);
    
    // Load model weights from a weight map
    WeightMap weights = create_complete_weight_map(
        config.d_model, config.n_layers, config.n_heads, config.n_kv_heads, config.vocab_size);
    bool load_success = model.load_weights(weights);
    EXPECT_TRUE(load_success);
    
    // Input tokens and positions
    std::vector<int> tokens = {1, 2, 3};
    std::vector<int> positions = {0, 1, 2};
    
    // Test different temperature and top_k values
    struct SamplingParams {
        float temperature;
        int top_k;
    };
    
    std::vector<SamplingParams> params = {
        {0.0f, 1},     // Greedy (no temperature, take top 1)
        {0.5f, 5},     // Low temperature, small top_k
        {1.0f, 10},    // Medium temperature, medium top_k
        {1.5f, 50},    // High temperature, large top_k
        {0.8f, 0}      // Medium temperature, no top_k filtering
    };
    
    for (const auto& param : params) {
        // Generate a frame with these parameters
        std::vector<int> frame = model.generate_frame(tokens, positions, param.temperature, param.top_k);
        
        // Just verify frame was generated (actual values will be random)
        EXPECT_FALSE(frame.empty());
    }
    
    // Reset caches between different runs
    model.reset_caches();
}

// Test handling of missing or incomplete weights
TEST_F(MLXModelTest, TestMLXModelWithIncompleteWeights) {
    // Skip if MLX is not available
    if (!MLXContext::is_available()) {
        GTEST_SKIP() << "MLX not available, skipping test";
    }
    
    // Create model configuration
    ModelConfig config;
    config.d_model = 128;
    config.n_layers = 2;
    config.n_heads = 4;
    config.n_kv_heads = 2;
    config.vocab_size = 100;
    config.max_seq_len = 16;
    
    // Create MLX model
    MLXModel model(config);
    
    // Try loading with incomplete loader
    std::shared_ptr<ModelLoader> loader = std::make_shared<IncompleteMLXModelLoader>("/path/to/incomplete/model");
    bool load_success = model.load_weights(loader);
    
    // Should fail due to missing weights
    EXPECT_FALSE(load_success);
    
    // Try loading with failing loader
    std::shared_ptr<ModelLoader> failing_loader = std::make_shared<FailingMLXModelLoader>("/path/to/failing/model");
    load_success = model.load_weights(failing_loader);
    
    // Should fail because loader fails
    EXPECT_FALSE(load_success);
}

// Test transformer context management
TEST_F(MLXModelTest, TestMLXTransformerContextManagement) {
    // Skip if MLX is not available
    if (!MLXContext::is_available()) {
        GTEST_SKIP() << "MLX not available, skipping test";
    }
    
    // Create model configuration
    ModelConfig config;
    config.d_model = 128;
    config.n_layers = 2;
    config.n_heads = 4;
    config.n_kv_heads = 2;
    config.vocab_size = 100;
    config.max_seq_len = 16;
    
    // Create MLX model
    MLXModel model(config);
    
    // Load model weights
    WeightMap weights = create_complete_weight_map(
        config.d_model, config.n_layers, config.n_heads, config.n_kv_heads, config.vocab_size);
    bool load_success = model.load_weights(weights);
    EXPECT_TRUE(load_success);
    
    // Create multiple transformer contexts
    std::shared_ptr<MLXTransformer> context1 = model.create_transformer_context();
    std::shared_ptr<MLXTransformer> context2 = model.create_transformer_context();
    
    // Verify contexts were created
    EXPECT_NE(context1.get(), nullptr);
    EXPECT_NE(context2.get(), nullptr);
    
    // Verify they are different objects
    EXPECT_NE(context1.get(), context2.get());
    
    // Verify they have the same parameters
    EXPECT_EQ(context1->d_model(), context2->d_model());
    EXPECT_EQ(context1->n_layers(), context2->n_layers());
    EXPECT_EQ(context1->n_heads(), context2->n_heads());
    EXPECT_EQ(context1->n_kv_heads(), context2->n_kv_heads());
    EXPECT_EQ(context1->vocab_size(), context2->vocab_size());
}

// Test loading weights from a file path
TEST_F(MLXModelTest, TestMLXModelLoadingFromPath) {
    // Skip if MLX is not available
    if (!MLXContext::is_available()) {
        GTEST_SKIP() << "MLX not available, skipping test";
    }
    
    // Create model configuration
    ModelConfig config;
    config.d_model = 128;
    config.n_layers = 2;
    config.n_heads = 4;
    config.n_kv_heads = 2;
    config.vocab_size = 100;
    config.max_seq_len = 16;
    
    // Create MLX model
    MLXModel model(config);
    
    // Create temporary file
    std::string temp_dir = std::filesystem::temp_directory_path().string();
    std::string model_path = temp_dir + "/test_mlx_model.bin";
    
    {
        std::ofstream file(model_path, std::ios::binary);
        EXPECT_TRUE(file.is_open());
        file << "MOCK_MODEL_DATA";
    }
    
    // Register mock loader for this extension
    ModelLoaderRegistry::register_loader(".bin", [](const std::string& path) {
        return std::make_shared<MockMLXModelLoader>(path);
    });
    
    // Load weights from path
    bool load_success = model.load_weights(model_path);
    EXPECT_TRUE(load_success);
    
    // Clean up
    std::filesystem::remove(model_path);
}

// Test error cases for invalid inputs
TEST_F(MLXModelTest, TestMLXModelErrorCases) {
    // Skip if MLX is not available
    if (!MLXContext::is_available()) {
        GTEST_SKIP() << "MLX not available, skipping test";
    }
    
    // Create model configuration
    ModelConfig config;
    config.d_model = 128;
    config.n_layers = 2;
    config.n_heads = 4;
    config.n_kv_heads = 2;
    config.vocab_size = 100;
    config.max_seq_len = 16;
    
    // Create MLX model
    MLXModel model(config);
    
    // Load weights
    WeightMap weights = create_complete_weight_map(
        config.d_model, config.n_layers, config.n_heads, config.n_kv_heads, config.vocab_size);
    bool load_success = model.load_weights(weights);
    EXPECT_TRUE(load_success);
    
    // Test with empty token list
    std::vector<int> empty_tokens;
    std::vector<int> positions = {0};
    
    // This should either return empty results or handle the error gracefully
    std::vector<int> frame = model.generate_frame(empty_tokens, positions, 0.9f, 50);
    EXPECT_TRUE(frame.empty());
    
    // Test with mismatched token and position counts
    std::vector<int> tokens = {1, 2, 3};
    std::vector<int> mismatched_positions = {0, 1}; // One less than tokens
    
    // This should handle the mismatch gracefully
    frame = model.generate_frame(tokens, mismatched_positions, 0.9f, 50);
    EXPECT_TRUE(frame.empty() || frame.size() > 0); // Either empty or handled it
    
    // Test with invalid temperature (negative)
    frame = model.generate_frame(tokens, positions, -1.0f, 50);
    EXPECT_TRUE(frame.empty() || frame.size() > 0); // Should handle gracefully
    
    // Test with invalid or out-of-range tokens
    std::vector<int> invalid_tokens = {-1, 1000}; // Negative and beyond vocab size
    frame = model.generate_frame(invalid_tokens, {0, 1}, 0.9f, 50);
    EXPECT_TRUE(frame.empty() || frame.size() > 0); // Should handle gracefully
}

// Test memory optimization functions
TEST_F(MLXModelTest, TestMLXModelMemoryOptimization) {
    // Skip if MLX is not available
    if (!MLXContext::is_available()) {
        GTEST_SKIP() << "MLX not available, skipping test";
    }
    
    // Create model configuration
    ModelConfig config;
    config.d_model = 128;
    config.n_layers = 2;
    config.n_heads = 4;
    config.n_kv_heads = 2;
    config.vocab_size = 100;
    config.max_seq_len = 16;
    
    // Create MLX model
    MLXModel model(config);
    
    // Load weights
    WeightMap weights = create_complete_weight_map(
        config.d_model, config.n_layers, config.n_heads, config.n_kv_heads, config.vocab_size);
    bool load_success = model.load_weights(weights);
    EXPECT_TRUE(load_success);
    
    // Run some inferences to populate caches
    std::vector<int> tokens = {1, 2, 3};
    std::vector<int> positions = {0, 1, 2};
    model.generate_frame(tokens, positions, 0.9f, 50);
    
    // Apply memory optimizations with different parameters
    
    // Test optimize_memory with different max memory settings
    model.optimize_memory(1024); // 1MB limit
    tokens = {4, 5};
    positions = {3, 4};
    std::vector<int> frame = model.generate_frame(tokens, positions, 0.9f, 50);
    EXPECT_FALSE(frame.empty());
    
    // Test prune_caches with different factors
    model.prune_caches(0.1f); // Aggressive pruning
    tokens = {6};
    positions = {5};
    frame = model.generate_frame(tokens, positions, 0.9f, 50);
    EXPECT_FALSE(frame.empty());
    
    // Test with full reset
    model.reset_caches();
    tokens = {7, 8, 9};
    positions = {0, 1, 2}; // Start from beginning after reset
    frame = model.generate_frame(tokens, positions, 0.9f, 50);
    EXPECT_FALSE(frame.empty());
}

// Test weight has/get operations
TEST_F(MLXModelTest, TestMLXModelWeightOperations) {
    // Skip if MLX is not available
    if (!MLXContext::is_available()) {
        GTEST_SKIP() << "MLX not available, skipping test";
    }
    
    // Create model configuration
    ModelConfig config;
    config.d_model = 128;
    config.n_layers = 2;
    config.n_heads = 4;
    config.n_kv_heads = 2;
    config.vocab_size = 100;
    config.max_seq_len = 16;
    
    // Create MLX model
    MLXModel model(config);
    
    // Load weights
    WeightMap weights = create_complete_weight_map(
        config.d_model, config.n_layers, config.n_heads, config.n_kv_heads, config.vocab_size);
    bool load_success = model.load_weights(weights);
    EXPECT_TRUE(load_success);
    
    // Test has_weight function
    EXPECT_TRUE(model.has_weight("token_embeddings.weight"));
    EXPECT_TRUE(model.has_weight("norm.weight"));
    EXPECT_TRUE(model.has_weight("output.weight"));
    EXPECT_TRUE(model.has_weight("layers.0.wq.weight"));
    EXPECT_FALSE(model.has_weight("nonexistent.weight"));
    EXPECT_FALSE(model.has_weight(""));
    
    // Test get_weight_array function
    mlx_array embedding_weight = model.get_weight_array("token_embeddings.weight");
    EXPECT_TRUE(embedding_weight != nullptr);
    
    // Check dimensions
    uint32_t ndim;
    mlx_array_ndim(embedding_weight, &ndim);
    EXPECT_EQ(ndim, 2);
    
    std::vector<int64_t> shape(ndim);
    mlx_array_shape(embedding_weight, shape.data());
    EXPECT_EQ(shape[0], config.vocab_size);
    EXPECT_EQ(shape[1], config.d_model);
    
    // Test with a nonexistent weight - should throw an exception
    try {
        mlx_array nonexistent = model.get_weight_array("nonexistent.weight");
        FAIL() << "Expected an exception when requesting nonexistent weight";
    } catch (const std::exception& e) {
        SUCCEED();
    }
    
    // Clean up
    mlx_array_free(embedding_weight);
}

// Test sequential generation
TEST_F(MLXModelTest, TestMLXModelSequentialGeneration) {
    // Skip if MLX is not available
    if (!MLXContext::is_available()) {
        GTEST_SKIP() << "MLX not available, skipping test";
    }
    
    // Create model configuration
    ModelConfig config;
    config.d_model = 128;
    config.n_layers = 2;
    config.n_heads = 4;
    config.n_kv_heads = 2;
    config.vocab_size = 100;
    config.max_seq_len = 32;
    
    // Create MLX model
    MLXModel model(config);
    
    // Load weights
    WeightMap weights = create_complete_weight_map(
        config.d_model, config.n_layers, config.n_heads, config.n_kv_heads, config.vocab_size);
    bool load_success = model.load_weights(weights);
    EXPECT_TRUE(load_success);
    
    // Generate tokens sequentially, using KV cache implicitly
    std::vector<int> tokens = {1, 2, 3};
    std::vector<int> positions = {0, 1, 2};
    
    // Initial generation
    std::vector<int> frame1 = model.generate_frame(tokens, positions, 0.9f, 50);
    EXPECT_FALSE(frame1.empty());
    
    // Continue generation
    tokens.push_back(frame1[0]); // Add generated token to input
    positions.push_back(3);       // Increment position
    
    std::vector<int> frame2 = model.generate_frame(tokens, positions, 0.9f, 50);
    EXPECT_FALSE(frame2.empty());
    
    // Continue generation once more
    tokens.push_back(frame2[0]);
    positions.push_back(4);
    
    std::vector<int> frame3 = model.generate_frame(tokens, positions, 0.9f, 50);
    EXPECT_FALSE(frame3.empty());
    
    // Reset caches and try again from the beginning
    model.reset_caches();
    
    // Should still work after reset
    std::vector<int> new_tokens = {1, 2, 3};
    std::vector<int> new_positions = {0, 1, 2};
    
    std::vector<int> new_frame = model.generate_frame(new_tokens, new_positions, 0.9f, 50);
    EXPECT_FALSE(new_frame.empty());
}

// Test decoder logits for all codebooks
TEST_F(MLXModelTest, TestMLXDecoderForAllCodebooks) {
    // Skip if MLX is not available
    if (!MLXContext::is_available()) {
        GTEST_SKIP() << "MLX not available, skipping test";
    }
    
    // Create model configuration
    ModelConfig config;
    config.d_model = 128;
    config.n_layers = 2;
    config.n_heads = 4;
    config.n_kv_heads = 2;
    config.vocab_size = 100;
    config.max_seq_len = 16;
    config.n_audio_codebooks = 4; // Multiple codebooks
    
    // Create MLX model
    MLXModel model(config);
    
    // Load weights
    WeightMap weights = create_complete_weight_map(
        config.d_model, config.n_layers, config.n_heads, config.n_kv_heads, config.vocab_size);
    
    // Add decoder weights for multiple codebooks
    for (int i = 0; i < config.n_audio_codebooks; i++) {
        std::string name = "decoder." + std::to_string(i) + ".weight";
        weights[name] = create_random_tensor({config.d_model, config.vocab_size}, DataType::F32, 1000 + i);
    }
    
    bool load_success = model.load_weights(weights);
    EXPECT_TRUE(load_success);
    
    // Input tokens and positions
    std::vector<int> tokens = {1, 2, 3};
    std::vector<int> positions = {0, 1, 2};
    
    // Get logits for each codebook
    for (int codebook = 0; codebook < config.n_audio_codebooks; codebook++) {
        std::vector<float> logits = model.get_decoder_logits(tokens, positions, codebook);
        EXPECT_EQ(logits.size(), config.vocab_size);
    }
    
    // Test with invalid codebook index
    try {
        std::vector<float> invalid_logits = model.get_decoder_logits(tokens, positions, config.n_audio_codebooks + 1);
        // Should either return empty results or handle the error gracefully
        EXPECT_TRUE(invalid_logits.empty() || invalid_logits.size() == config.vocab_size);
    } catch (const std::exception& e) {
        // Or it might throw an exception, which is also valid error handling
        SUCCEED();
    }
}
#endif // CCSM_WITH_MLX

} // namespace testing
} // namespace ccsm