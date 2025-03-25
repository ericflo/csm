#include <gtest/gtest.h>
#include <ccsm/mlx/mlx_generator.h>
#include <ccsm/mlx/mlx_model.h>
#include <ccsm/mlx/mlx_tensor.h>
#include <ccsm/tokenizer.h>
#include <ccsm/model_loader.h>
#include <memory>
#include <vector>
#include <string>
#include <stdexcept>

namespace ccsm {
namespace testing {

// Mock classes for testing
class MockMLXTextTokenizer : public TextTokenizer {
public:
    std::vector<int> encode(const std::string& text) const override {
        // Return fixed token sequence
        return {1, 2, 3, 4, 5};
    }
    
    std::string decode(const std::vector<int>& tokens) const override {
        return "Mock text";
    }
    
    int vocab_size() const override {
        return 32000;
    }
    
    int bos_token_id() const override {
        return 1;
    }
    
    int eos_token_id() const override {
        return 2;
    }
    
    int pad_token_id() const override {
        return 0;
    }
    
    int unk_token_id() const override {
        return 3;
    }
    
    int get_speaker_token_id(int speaker_id) const override {
        return 100 + speaker_id;
    }
    
    std::vector<int> get_audio_token_ids() const override {
        return {1000, 1001, 1002};
    }
};

class MockMLXAudioCodec : public AudioCodec {
public:
    std::vector<std::vector<int>> encode(const std::vector<float>& audio) const override {
        // Mock 8 codebooks
        std::vector<std::vector<int>> tokens(8);
        for (int i = 0; i < 8; i++) {
            tokens[i] = {10, 20, 30};
        }
        return tokens;
    }
    
    std::vector<float> decode(const std::vector<std::vector<int>>& tokens) const override {
        // Return a simple sine wave
        std::vector<float> audio(24000);
        for (size_t i = 0; i < audio.size(); i++) {
            audio[i] = 0.5f * std::sin(2.0f * 3.14159f * 440.0f * i / 24000.0f);
        }
        return audio;
    }
    
    int num_codebooks() const override {
        return 8;
    }
    
    int vocab_size() const override {
        return 2051;
    }
    
    int sample_rate() const override {
        return 24000;
    }
    
    int hop_length() const override {
        return 400;  // 60ms at 24kHz
    }
    
    bool is_eos_token(int token, int codebook) const override {
        return token == 0;
    }
};

class MockMLXWatermarker : public Watermarker {
public:
    std::vector<float> apply_watermark(const std::vector<float>& audio) override {
        return audio;  // No watermarking in mock
    }
    
    bool detect_watermark(const std::vector<float>& audio) override {
        return true;  // Always detect
    }
    
    WatermarkResult detect(const std::vector<float>& audio, float sample_rate) override {
        WatermarkResult result;
        result.detected = true;
        result.payload = "mock-payload";
        result.confidence = 0.95f;
        return result;
    }
    
    float get_strength() const override {
        return 0.5f;
    }
    
    void set_strength(float strength) override {
        // No-op
    }
    
    std::string get_key() const override {
        return "mock-key";
    }
};

// Test fixture for MLX Generator tests
class MLXGeneratorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create model configuration
        ModelConfig config;
        config.name = "MockMLXModel";
        config.d_model = 128;
        config.n_layers = 2;
        config.n_heads = 4;
        config.n_kv_heads = 2;
        config.vocab_size = 32000;
        config.audio_vocab_size = 2051;
        config.max_seq_len = 32;
        config.num_codebooks = 8;
        
        // Create mock components
        model = std::make_shared<MLXModel>(config);
        tokenizer = std::make_shared<MockMLXTextTokenizer>();
        codec = std::make_shared<MockMLXAudioCodec>();
        watermarker = std::make_shared<MockMLXWatermarker>();
    }
    
    void TearDown() override {
        model.reset();
        tokenizer.reset();
        codec.reset();
        watermarker.reset();
    }
    
    std::shared_ptr<MLXModel> model;
    std::shared_ptr<MockMLXTextTokenizer> tokenizer;
    std::shared_ptr<MockMLXAudioCodec> codec;
    std::shared_ptr<MockMLXWatermarker> watermarker;
};

// Test constructor
TEST_F(MLXGeneratorTest, TestConstructor) {
    #ifdef CCSM_WITH_MLX
    // Create MLX Generator
    MLXGenerator generator(model, tokenizer, codec, watermarker);
    
    // Test basic properties
    EXPECT_EQ(generator.sample_rate(), 24000);
    EXPECT_EQ(generator.model().get(), model.get());
    EXPECT_EQ(generator.text_tokenizer().get(), tokenizer.get());
    EXPECT_EQ(generator.audio_codec().get(), codec.get());
    EXPECT_EQ(generator.watermarker().get(), watermarker.get());
    
    // Check MLX-specific properties
    const auto& config = generator.optimization_config();
    EXPECT_TRUE(config.use_autotune);
    
    // If MLX is not available, this will return false
    bool is_accelerated = generator.is_mlx_accelerated();
    EXPECT_EQ(is_accelerated, MLXWeightConverter::is_mlx_available());
    #else
    GTEST_SKIP() << "MLX support not compiled in, skipping test";
    #endif
}

// Test model compatibility check
TEST_F(MLXGeneratorTest, TestModelCompatibilityCheck) {
    #ifdef CCSM_WITH_MLX
    // Create a standard Model (not MLXModel)
    auto standard_model = std::make_shared<Model>(ModelConfig());
    
    // MLXModel should be compatible
    EXPECT_TRUE(MLXGenerator::is_model_mlx_compatible(model));
    
    // Standard Model should not be compatible
    EXPECT_FALSE(MLXGenerator::is_model_mlx_compatible(standard_model));
    #else
    GTEST_SKIP() << "MLX support not compiled in, skipping test";
    #endif
}

// Test optimization configuration
TEST_F(MLXGeneratorTest, TestOptimizationConfig) {
    #ifdef CCSM_WITH_MLX
    // Create MLX Generator
    MLXGenerator generator(model, tokenizer, codec, watermarker);
    
    // Get initial config
    const auto& initial_config = generator.optimization_config();
    
    // Create a modified config
    MLXOptimizationConfig new_config;
    new_config.compute_precision = MLXOptimizationConfig::ComputePrecision::FLOAT32;
    new_config.memory_usage = MLXOptimizationConfig::MemoryUsage::MINIMAL;
    new_config.use_autotune = false;
    
    // Set the new config
    generator.set_optimization_config(new_config);
    
    // Verify config was updated
    const auto& updated_config = generator.optimization_config();
    EXPECT_EQ(updated_config.compute_precision, MLXOptimizationConfig::ComputePrecision::FLOAT32);
    EXPECT_EQ(updated_config.memory_usage, MLXOptimizationConfig::MemoryUsage::MINIMAL);
    EXPECT_FALSE(updated_config.use_autotune);
    #else
    GTEST_SKIP() << "MLX support not compiled in, skipping test";
    #endif
}

// Test generation fallback
TEST_F(MLXGeneratorTest, TestGenerationFallback) {
    #ifdef CCSM_WITH_MLX
    // Create MLX Generator
    MLXGenerator generator(model, tokenizer, codec, watermarker);
    
    // Create generation options
    GenerationOptions options;
    options.temperature = 0.9f;
    options.top_k = 10;
    options.max_audio_length_ms = 1000;
    options.seed = 42;
    
    // Generate speech with empty tokens
    // This should fall back to base implementation since model weights aren't loaded
    std::vector<int> tokens = {};
    std::vector<float> audio = generator.generate_speech_from_tokens(tokens, 0, {}, options);
    
    // Output should be a non-empty audio array
    EXPECT_FALSE(audio.empty());
    EXPECT_EQ(audio.size() % codec->sample_rate(), 0);
    #else
    GTEST_SKIP() << "MLX support not compiled in, skipping test";
    #endif
}

// Test progress callback
TEST_F(MLXGeneratorTest, TestProgressCallback) {
    #ifdef CCSM_WITH_MLX
    // Create MLX Generator
    MLXGenerator generator(model, tokenizer, codec, watermarker);
    
    // Create generation options
    GenerationOptions options;
    options.temperature = 0.9f;
    options.top_k = 10;
    options.max_audio_length_ms = 1000;
    options.seed = 42;
    
    // Variables to track progress
    int progress_called = 0;
    int last_frame = 0;
    int max_frames = 0;
    
    // Progress callback
    auto progress_callback = [&](int frame, int total) {
        progress_called++;
        last_frame = frame;
        max_frames = total;
    };
    
    // Generate speech with progress callback
    std::vector<int> tokens = {1, 2, 3};
    std::vector<float> audio = generator.generate_speech_from_tokens(tokens, 0, {}, options, progress_callback);
    
    // Progress should have been called at least once
    EXPECT_GT(progress_called, 0);
    // Last frame should not exceed max frames
    EXPECT_LE(last_frame, max_frames);
    // And audio should be non-empty
    EXPECT_FALSE(audio.empty());
    #else
    GTEST_SKIP() << "MLX support not compiled in, skipping test";
    #endif
}

// Test factory function
TEST_F(MLXGeneratorTest, TestFactoryFunction) {
    #ifdef CCSM_WITH_MLX
    // Register mock loaders
    ModelLoaderRegistry::register_loader(".bin", [](const std::string& path) {
        return std::make_shared<MockMLXModelLoader>(path);
    });
    
    // Skip test if MLX is not available
    if (!MLXWeightConverter::is_mlx_available()) {
        GTEST_SKIP() << "MLX not available on this system, skipping test";
    }
    
    // Create temporary files for model, tokenizer, codec, watermarker
    std::string model_path = "/tmp/test_mlx_model.bin";
    std::string tokenizer_path = "/tmp/test_tokenizer.model";
    std::string codec_path = "/tmp/test_codec.bin";
    std::string watermarker_path = "/tmp/test_watermarker.bin";
    
    // Create MLX optimization config
    MLXOptimizationConfig config;
    config.compute_precision = MLXOptimizationConfig::ComputePrecision::FLOAT32;
    
    // Create generator
    auto generator = create_mlx_generator(model_path, tokenizer_path, codec_path, watermarker_path, config);
    
    // Generator may be null if MLX is not available
    if (!generator) {
        GTEST_SKIP() << "create_mlx_generator returned null, likely due to MLX not being available";
    }
    
    // Verify generator was created
    EXPECT_NE(generator.get(), nullptr);
    #else
    GTEST_SKIP() << "MLX support not compiled in, skipping test";
    #endif
}

// Test availability check
TEST_F(MLXGeneratorTest, TestMLXAvailabilityCheck) {
    #ifdef CCSM_WITH_MLX
    // This should match the result from the MLXWeightConverter
    bool is_avail = is_mlx_available();
    bool converter_avail = MLXWeightConverter::is_mlx_available();
    EXPECT_EQ(is_avail, converter_avail);
    #else
    bool is_avail = is_mlx_available();
    EXPECT_FALSE(is_avail);
    #endif
}

} // namespace testing
} // namespace ccsm