#include <gtest/gtest.h>
#include <ccsm/generator.h>
#include <ccsm/model.h>
#include <ccsm/tokenizer.h>
#include <ccsm/watermarking.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <random>
#include <algorithm>

using namespace ccsm;

// Mock model implementation that can be configured for testing
class ConfigurableMockModel : public Model {
public:
    ConfigurableMockModel(const ModelConfig& config) : Model(config) {}
    
    bool load_weights(const std::string& path) override {
        return true;
    }
    
    bool load_weights(std::shared_ptr<ModelLoader> loader) override {
        return true;
    }
    
    bool load_weights(const WeightMap& weights) override {
        return true;
    }
    
    void reset_caches() override {
        // Track calls to reset_caches
        reset_caches_called = true;
    }
    
    void optimize_memory(size_t max_memory_mb) override {
        // Track calls to optimize_memory
        memory_optimized = true;
        memory_limit = max_memory_mb;
    }
    
    void prune_caches(float prune_factor) override {
        // Track calls to prune_caches
        caches_pruned = true;
        pruning_factor = prune_factor;
    }
    
    std::vector<int> generate_frame(
        const std::vector<int>& tokens,
        const std::vector<int>& positions,
        float temperature,
        int top_k
    ) override {
        // Store input parameters for later verification
        last_tokens = tokens;
        last_positions = positions;
        last_temperature = temperature;
        last_top_k = top_k;
        frame_generation_count++;
        
        // Configure frame generation based on test settings
        std::vector<int> result;
        
        if (return_empty_frame) {
            return result; // Empty frame
        }
        
        if (use_fixed_frame) {
            return fixed_frame; // Return pre-configured frame
        }
        
        // Default behavior - generate a frame with configured size
        result.resize(config_.num_codebooks);
        for (size_t i = 0; i < result.size(); i++) {
            // Use the random generator for more realistic behavior
            result[i] = 1 + (random_dist(random_gen) % (config_.audio_vocab_size - 1));
        }
        
        return result;
    }
    
    std::vector<float> get_backbone_logits(
        const std::vector<int>& tokens,
        const std::vector<int>& positions
    ) override {
        // Track backbone invocation
        backbone_logits_called = true;
        backbone_tokens = tokens;
        backbone_positions = positions;
        
        // Return mock logits
        if (use_fixed_backbone_logits) {
            return fixed_backbone_logits;
        }
        
        std::vector<float> result(config_.audio_vocab_size);
        for (size_t i = 0; i < result.size(); i++) {
            result[i] = random_dist(random_gen) * 2.0f - 1.0f; // Range [-1, 1]
        }
        
        // Make token 1 have highest logit for predictable sampling
        result[1] = 10.0f;
        
        return result;
    }
    
    std::vector<float> get_decoder_logits(
        const std::vector<int>& tokens,
        const std::vector<int>& positions,
        int codebook
    ) override {
        // Track decoder invocation
        decoder_logits_called = true;
        decoder_tokens = tokens;
        decoder_positions = positions;
        last_codebook = codebook;
        
        // Return mock logits
        if (use_fixed_decoder_logits) {
            return fixed_decoder_logits;
        }
        
        std::vector<float> result(config_.audio_vocab_size);
        for (size_t i = 0; i < result.size(); i++) {
            result[i] = random_dist(random_gen) * 2.0f - 1.0f; // Range [-1, 1]
        }
        
        // Make token (codebook + 1) have highest logit for predictable sampling
        result[(codebook + 1) % config_.audio_vocab_size] = 10.0f;
        
        return result;
    }
    
    // Configuration methods for tests
    void set_return_empty_frame(bool value) {
        return_empty_frame = value;
    }
    
    void set_fixed_frame(const std::vector<int>& frame) {
        fixed_frame = frame;
        use_fixed_frame = true;
    }
    
    void set_fixed_backbone_logits(const std::vector<float>& logits) {
        fixed_backbone_logits = logits;
        use_fixed_backbone_logits = true;
    }
    
    void set_fixed_decoder_logits(const std::vector<float>& logits) {
        fixed_decoder_logits = logits;
        use_fixed_decoder_logits = true;
    }
    
    void set_random_seed(unsigned int seed) {
        random_gen.seed(seed);
    }
    
    // State tracking for tests
    bool reset_caches_called = false;
    bool memory_optimized = false;
    bool caches_pruned = false;
    size_t memory_limit = 0;
    float pruning_factor = 0.0f;
    int frame_generation_count = 0;
    
    std::vector<int> last_tokens;
    std::vector<int> last_positions;
    float last_temperature = 0.0f;
    int last_top_k = 0;
    
    bool backbone_logits_called = false;
    std::vector<int> backbone_tokens;
    std::vector<int> backbone_positions;
    
    bool decoder_logits_called = false;
    std::vector<int> decoder_tokens;
    std::vector<int> decoder_positions;
    int last_codebook = -1;
    
private:
    // Test configuration
    bool return_empty_frame = false;
    bool use_fixed_frame = false;
    std::vector<int> fixed_frame;
    
    bool use_fixed_backbone_logits = false;
    std::vector<float> fixed_backbone_logits;
    
    bool use_fixed_decoder_logits = false;
    std::vector<float> fixed_decoder_logits;
    
    // Random generator for more realistic mock behavior
    std::mt19937 random_gen{42}; // Fixed seed for reproducibility
    std::uniform_real_distribution<float> random_dist{0.0f, 1.0f};
};

// Configurable mock tokenizer for testing
class ConfigurableMockTokenizer : public TextTokenizer {
public:
    // Core Tokenizer interface
    std::vector<int> encode(const std::string& text) const override {
        if (throw_on_encode) {
            throw std::runtime_error("Simulated tokenizer error");
        }
        
        if (fixed_encoded_tokens.empty()) {
            // Default behavior
            std::vector<int> tokens;
            for (char c : text) {
                tokens.push_back(static_cast<int>(c));
            }
            return tokens;
        } else {
            return fixed_encoded_tokens;
        }
    }
    
    std::string decode(const std::vector<int>& tokens) const override {
        if (throw_on_decode) {
            throw std::runtime_error("Simulated tokenizer error");
        }
        
        if (fixed_decoded_text.empty()) {
            // Default behavior
            std::string result;
            for (int token : tokens) {
                result += static_cast<char>(token % 128);
            }
            return result;
        } else {
            return fixed_decoded_text;
        }
    }
    
    int vocab_size() const override {
        return fixed_vocab_size;
    }
    
    // TextTokenizer interface
    int bos_token_id() const override {
        return fixed_bos_token;
    }
    
    int eos_token_id() const override {
        return fixed_eos_token;
    }
    
    int pad_token_id() const override {
        return fixed_pad_token;
    }
    
    int unk_token_id() const override {
        return fixed_unk_token;
    }
    
    int get_speaker_token_id(int speaker_id) const override {
        return fixed_speaker_prefix + speaker_id;
    }
    
    std::vector<int> get_audio_token_ids() const override {
        return fixed_audio_tokens;
    }
    
    // Configuration methods for tests
    void set_throw_on_encode(bool value) { throw_on_encode = value; }
    void set_throw_on_decode(bool value) { throw_on_decode = value; }
    void set_fixed_encoded_tokens(const std::vector<int>& tokens) { fixed_encoded_tokens = tokens; }
    void set_fixed_decoded_text(const std::string& text) { fixed_decoded_text = text; }
    void set_fixed_vocab_size(int size) { fixed_vocab_size = size; }
    void set_fixed_special_tokens(int bos, int eos, int pad, int unk) {
        fixed_bos_token = bos;
        fixed_eos_token = eos;
        fixed_pad_token = pad;
        fixed_unk_token = unk;
    }
    void set_fixed_speaker_prefix(int prefix) { fixed_speaker_prefix = prefix; }
    void set_fixed_audio_tokens(const std::vector<int>& tokens) { fixed_audio_tokens = tokens; }
    
private:
    // Configuration
    bool throw_on_encode = false;
    bool throw_on_decode = false;
    std::vector<int> fixed_encoded_tokens;
    std::string fixed_decoded_text;
    int fixed_vocab_size = 32000;
    int fixed_bos_token = 1;
    int fixed_eos_token = 2;
    int fixed_pad_token = 0;
    int fixed_unk_token = 3;
    int fixed_speaker_prefix = 1000;
    std::vector<int> fixed_audio_tokens = {1, 2, 3, 4};
};

// Configurable mock watermarker for testing
class ConfigurableMockWatermarker : public Watermarker {
public:
    bool watermark(std::vector<float>& audio) override {
        // Track calls and parameters
        watermark_called = true;
        last_audio = audio;
        
        // Modify the audio if configured
        if (apply_modification) {
            for (auto& sample : audio) {
                sample *= 0.9f; // Reduce amplitude to simulate watermarking
            }
        }
        
        return !simulate_failure;
    }
    
    bool verify(const std::vector<float>& audio) override {
        verify_called = true;
        last_verify_audio = audio;
        
        return !simulate_verify_failure;
    }
    
    WatermarkResult extract(const std::vector<float>& audio) override {
        extract_called = true;
        last_extract_audio = audio;
        
        WatermarkResult result;
        result.success = !simulate_extract_failure;
        result.message = simulate_extract_failure ? "Simulated extraction failure" : "Mock watermark";
        result.confidence = simulate_extract_failure ? 0.0f : 0.95f;
        
        return result;
    }
    
    // Configuration methods
    void set_simulate_failure(bool value) { simulate_failure = value; }
    void set_simulate_verify_failure(bool value) { simulate_verify_failure = value; }
    void set_simulate_extract_failure(bool value) { simulate_extract_failure = value; }
    void set_apply_modification(bool value) { apply_modification = value; }
    
    // State tracking
    bool watermark_called = false;
    bool verify_called = false;
    bool extract_called = false;
    std::vector<float> last_audio;
    std::vector<float> last_verify_audio;
    std::vector<float> last_extract_audio;
    
private:
    bool simulate_failure = false;
    bool simulate_verify_failure = false;
    bool simulate_extract_failure = false;
    bool apply_modification = true;
};

// Test fixtures for different Generator test scenarios
class GeneratorConfigurationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a basic model config
        model_config.vocab_size = 32000;
        model_config.audio_vocab_size = 2051;
        model_config.d_model = 1024;
        model_config.n_heads = 16;
        model_config.n_layers = 12;
        model_config.n_audio_layers = 4;
        model_config.num_codebooks = 8;
        
        // Create mock components
        model = std::make_shared<ConfigurableMockModel>(model_config);
        tokenizer = std::make_shared<ConfigurableMockTokenizer>();
        watermarker = std::make_shared<ConfigurableMockWatermarker>();
        
        // Set fixed tokens for reproducible tests
        tokenizer->set_fixed_special_tokens(1, 2, 0, 3);
        
        // Create generator with mocks
        generator = std::make_shared<Generator>(model, tokenizer, watermarker);
    }
    
    // Test components
    ModelConfig model_config;
    std::shared_ptr<ConfigurableMockModel> model;
    std::shared_ptr<ConfigurableMockTokenizer> tokenizer;
    std::shared_ptr<ConfigurableMockWatermarker> watermarker;
    std::shared_ptr<Generator> generator;
    
    // Helper methods
    std::vector<float> decode_frame_to_audio(const std::vector<int>& frame, int sample_rate = 24000) {
        // Simple mock audio decoding - just return a vector of the right size
        // In a real system, this would actually decode the audio tokens
        size_t audio_size = sample_rate * frame.size() / 50; // Simulate ~20ms per token
        std::vector<float> audio(audio_size);
        
        // Fill with a simple pattern based on the frame content
        for (size_t i = 0; i < audio_size; i++) {
            // Use frame values to generate audio pattern (just for testing)
            int frame_idx = i % frame.size();
            float phase = static_cast<float>(i) / audio_size * 2.0f * 3.14159f;
            audio[i] = 0.1f * frame[frame_idx] * std::sin(phase * (frame[frame_idx] % 10 + 1));
        }
        
        return audio;
    }
};

// Test basic generator configuration
TEST_F(GeneratorConfigurationTest, BasicConfiguration) {
    // Default configuration should be valid
    EXPECT_EQ(generator->model(), model);
    EXPECT_EQ(generator->tokenizer(), tokenizer);
    EXPECT_EQ(generator->watermarker(), watermarker);
    
    // Default parameters should be reasonable
    EXPECT_NEAR(generator->default_temperature(), 0.9f, 0.001f);
    EXPECT_EQ(generator->default_top_k(), 50);
}

// Test generator with different temperature values
TEST_F(GeneratorConfigurationTest, TemperatureSettings) {
    // Set up model to track parameters
    std::string test_text = "Generate speech with different temperatures";
    
    // Generate with default temperature
    generator->generate_speech(test_text);
    EXPECT_NEAR(model->last_temperature, 0.9f, 0.001f);
    
    // Generate with custom temperature
    generator->generate_speech(test_text, -1, 0.5f);
    EXPECT_NEAR(model->last_temperature, 0.5f, 0.001f);
    
    // Generate with very low temperature (should use minimum)
    generator->generate_speech(test_text, -1, 0.01f);
    EXPECT_GE(model->last_temperature, 0.05f); // Should be clamped to minimum
    
    // Generate with very high temperature (should use maximum)
    generator->generate_speech(test_text, -1, 2.0f);
    EXPECT_LE(model->last_temperature, 1.5f); // Should be clamped to maximum
    
    // Set custom default temperature
    generator->set_default_temperature(0.7f);
    EXPECT_NEAR(generator->default_temperature(), 0.7f, 0.001f);
    
    // Generate with new default
    generator->generate_speech(test_text);
    EXPECT_NEAR(model->last_temperature, 0.7f, 0.001f);
}

// Test generator with different top_k values
TEST_F(GeneratorConfigurationTest, TopKSettings) {
    // Set up model to track parameters
    std::string test_text = "Generate speech with different top_k values";
    
    // Generate with default top_k
    generator->generate_speech(test_text);
    EXPECT_EQ(model->last_top_k, 50);
    
    // Generate with custom top_k
    generator->generate_speech(test_text, -1, 0.9f, 10);
    EXPECT_EQ(model->last_top_k, 10);
    
    // Generate with very low top_k (should use minimum)
    generator->generate_speech(test_text, -1, 0.9f, 0);
    EXPECT_GE(model->last_top_k, 1); // Should be clamped to minimum
    
    // Generate with very high top_k
    generator->generate_speech(test_text, -1, 0.9f, 1000);
    EXPECT_EQ(model->last_top_k, 1000); // High values are allowed
    
    // Set custom default top_k
    generator->set_default_top_k(20);
    EXPECT_EQ(generator->default_top_k(), 20);
    
    // Generate with new default
    generator->generate_speech(test_text);
    EXPECT_EQ(model->last_top_k, 20);
}

// Test speaker ID handling
TEST_F(GeneratorConfigurationTest, SpeakerIDHandling) {
    std::string test_text = "Test with different speakers";
    
    // Generate with default speaker (should be -1)
    generator->generate_speech(test_text);
    
    // Check that no speaker token was added
    bool found_speaker_token = false;
    for (int token : model->last_tokens) {
        if (token >= tokenizer->get_speaker_token_id(0)) {
            found_speaker_token = true;
            break;
        }
    }
    EXPECT_FALSE(found_speaker_token);
    
    // Generate with specific speaker ID
    int test_speaker_id = 5;
    generator->generate_speech(test_text, test_speaker_id);
    
    // Check that the speaker token was added
    int expected_speaker_token = tokenizer->get_speaker_token_id(test_speaker_id);
    bool found_correct_speaker = false;
    for (int token : model->last_tokens) {
        if (token == expected_speaker_token) {
            found_correct_speaker = true;
            break;
        }
    }
    EXPECT_TRUE(found_correct_speaker);
}

// Test watermarking configuration
TEST_F(GeneratorConfigurationTest, WatermarkingConfiguration) {
    // Generate speech with watermarking enabled (default)
    std::string test_text = "Test watermarking";
    auto result = generator->generate_speech(test_text);
    
    // Check that watermarking was applied
    EXPECT_TRUE(watermarker->watermark_called);
    
    // Reset state
    watermarker->watermark_called = false;
    
    // Disable watermarking
    generator->set_enable_watermarking(false);
    result = generator->generate_speech(test_text);
    
    // Check that watermarking was not applied
    EXPECT_FALSE(watermarker->watermark_called);
    
    // Re-enable watermarking
    generator->set_enable_watermarking(true);
    result = generator->generate_speech(test_text);
    
    // Check that watermarking was applied again
    EXPECT_TRUE(watermarker->watermark_called);
}

// Test generation with watermarking failures
TEST_F(GeneratorConfigurationTest, WatermarkingFailures) {
    std::string test_text = "Test watermarking failures";
    
    // Configure watermarker to fail
    watermarker->set_simulate_failure(true);
    
    // Generate speech - should still succeed but with warning
    auto result = generator->generate_speech(test_text);
    
    // Audio should still be returned
    EXPECT_FALSE(result.audio.empty());
    
    // Watermarking should have been attempted
    EXPECT_TRUE(watermarker->watermark_called);
}

// Test token length limiting
TEST_F(GeneratorConfigurationTest, TokenLengthLimiting) {
    // Set up a text that would produce many tokens
    std::string long_text(10000, 'a');
    
    // Set tokenizer to produce 1 token per character (for testing)
    std::vector<int> long_tokens(long_text.size());
    std::iota(long_tokens.begin(), long_tokens.end(), 10); // Start at token 10
    tokenizer->set_fixed_encoded_tokens(long_tokens);
    
    // Generate with default max tokens (should be limited)
    generator->generate_speech(long_text);
    
    // Check that tokens were limited
    EXPECT_LE(model->last_tokens.size(), 2048); // Default max length
    
    // Set custom max tokens
    generator->set_max_text_tokens(512);
    
    // Generate again
    generator->generate_speech(long_text);
    
    // Check that tokens were limited to new value
    EXPECT_LE(model->last_tokens.size(), 512);
}

// Test error handling with problematic inputs
TEST_F(GeneratorConfigurationTest, ErrorHandling) {
    // Configure tokenizer to throw on encode
    tokenizer->set_throw_on_encode(true);
    
    // Generate with tokenizer that will fail
    try {
        auto result = generator->generate_speech("Test error handling");
        FAIL() << "Expected exception was not thrown";
    } catch (const std::exception& e) {
        // Exception should be thrown
        EXPECT_STREQ(e.what(), "Simulated tokenizer error");
    }
    
    // Reset tokenizer
    tokenizer->set_throw_on_encode(false);
    
    // Configure model to return empty frame
    model->set_return_empty_frame(true);
    
    // Generate with model that returns empty frame
    try {
        auto result = generator->generate_speech("Test empty frame");
        FAIL() << "Expected exception was not thrown";
    } catch (const std::exception& e) {
        // Exception should be thrown for empty frame
        EXPECT_TRUE(std::string(e.what()).find("Empty frame") != std::string::npos);
    }
}

// Test frame decoding
TEST_F(GeneratorConfigurationTest, FrameDecoding) {
    // Create a fixed test frame
    std::vector<int> test_frame = {10, 20, 30, 40, 50, 60, 70, 80};
    model->set_fixed_frame(test_frame);
    
    // Generate speech
    auto result = generator->generate_speech("Test frame decoding");
    
    // Audio should not be empty
    EXPECT_FALSE(result.audio.empty());
    
    // Frame should be stored
    EXPECT_EQ(result.frame, test_frame);
}

// Test progress callback
TEST_F(GeneratorConfigurationTest, ProgressCallback) {
    std::string test_text = "Test progress callback";
    bool callback_called = false;
    float last_progress = 0.0f;
    
    // Set up callback
    auto progress_callback = [&](float progress) {
        callback_called = true;
        last_progress = progress;
        return true; // Continue generation
    };
    
    // Generate with callback
    generator->generate_speech(test_text, -1, 0.9f, 50, progress_callback);
    
    // Callback should have been called
    EXPECT_TRUE(callback_called);
    EXPECT_FLOAT_EQ(last_progress, 1.0f); // Final progress should be 1.0
    
    // Test cancellation
    callback_called = false;
    last_progress = 0.0f;
    bool generation_cancelled = false;
    
    // Set up cancellation callback
    auto cancel_callback = [&](float progress) {
        callback_called = true;
        last_progress = progress;
        return false; // Cancel generation after first call
    };
    
    try {
        generator->generate_speech(test_text, -1, 0.9f, 50, cancel_callback);
    } catch (const std::exception& e) {
        // Exception should indicate cancellation
        generation_cancelled = std::string(e.what()).find("cancelled") != std::string::npos;
    }
    
    // Verify cancellation
    EXPECT_TRUE(callback_called);
    EXPECT_LT(last_progress, 1.0f); // Progress should be less than 1.0
    EXPECT_TRUE(generation_cancelled);
}

// Test memory optimization
TEST_F(GeneratorConfigurationTest, MemoryOptimization) {
    // Generate with default settings (no memory optimization)
    generator->generate_speech("Test without memory optimization");
    EXPECT_FALSE(model->memory_optimized);
    
    // Enable memory optimization
    generator->set_memory_optimization(true, 1024); // 1GB limit
    
    // Generate with memory optimization
    generator->generate_speech("Test with memory optimization");
    EXPECT_TRUE(model->memory_optimized);
    EXPECT_EQ(model->memory_limit, 1024);
    
    // Reset model state
    model->memory_optimized = false;
    
    // Add some hooks to test aggressive memory management
    generator->set_memory_optimization(true, 1024, 512, 0.7f);
    
    // Generate with aggressive memory optimization
    generator->generate_speech("Test with aggressive memory optimization");
    EXPECT_TRUE(model->memory_optimized);
    EXPECT_TRUE(model->caches_pruned);
    EXPECT_NEAR(model->pruning_factor, 0.7f, 0.001f);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}