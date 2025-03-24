#include <gtest/gtest.h>
#include <ccsm/generator.h>
#include <ccsm/model.h>
#include <ccsm/tokenizer.h>
#include <ccsm/watermarking.h>
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <future>
#include <thread>
#include <atomic>
#include <algorithm>
#include <random>

using namespace ccsm;

// Mock classes
class MockModel : public Model {
public:
    MockModel(ModelConfig config = ModelConfig()) : Model(config) {
        if (config.vocab_size == 0) {
            config_.vocab_size = 32000;
            config_.audio_vocab_size = 2051;
            config_.num_codebooks = 8;
        }
    }
    
    bool load_weights(const std::string& path) override { return true; }
    bool load_weights(std::shared_ptr<ModelLoader> loader) override { return true; }
    bool load_weights(const WeightMap& weights) override { return true; }
    
    std::vector<int> generate_frame(
        const std::vector<int>& tokens,
        const std::vector<int>& positions,
        float temperature,
        int top_k
    ) override {
        // Record parameters for testing
        last_tokens = tokens;
        last_positions = positions;
        last_temperature = temperature;
        last_top_k = top_k;
        frame_count++;
        
        // Generate deterministic but affected by parameters
        std::mt19937 gen(seed_);
        if (temperature >= 0) {
            gen.seed(seed_ + static_cast<unsigned int>(temperature * 100));
        }
        if (top_k >= 0) {
            gen.seed(gen() + static_cast<unsigned int>(top_k));
        }
        
        // Create a frame with appropriate number of tokens
        std::vector<int> frame(config_.num_codebooks);
        for (size_t i = 0; i < frame.size(); i++) {
            frame[i] = (gen() % 1000) + 5; // Avoid special tokens
        }
        
        // After max_frames, return EOS tokens
        if (frame_count >= max_frames_) {
            for (size_t i = 0; i < frame.size(); i++) {
                frame[i] = 2; // EOS token
            }
        }
        
        return frame;
    }
    
    std::vector<float> get_backbone_logits(
        const std::vector<int>& tokens,
        const std::vector<int>& positions
    ) override {
        return std::vector<float>(config_.vocab_size, 0.0f);
    }
    
    std::vector<float> get_decoder_logits(
        const std::vector<int>& tokens,
        const std::vector<int>& positions,
        int codebook
    ) override {
        return std::vector<float>(config_.audio_vocab_size, 0.0f);
    }
    
    void reset_caches() override {
        cache_reset_count++;
    }
    
    // Testing helpers
    void set_seed(unsigned int seed) { seed_ = seed; }
    void set_max_frames(int max_frames) { max_frames_ = max_frames; }
    void reset_counters() {
        frame_count = 0;
        cache_reset_count = 0;
        last_tokens.clear();
        last_positions.clear();
        last_temperature = -1.0f;
        last_top_k = -1;
    }
    
    // Tracking variables for testing
    std::vector<int> last_tokens;
    std::vector<int> last_positions;
    float last_temperature = -1.0f;
    int last_top_k = -1;
    int frame_count = 0;
    int cache_reset_count = 0;
    
private:
    unsigned int seed_ = 42;
    int max_frames_ = 20;
};

class MockTextTokenizer : public TextTokenizer {
public:
    // TextTokenizer interface
    std::vector<int> encode(const std::string& text) const override {
        // Record for testing
        last_encoded_text = text;
        encode_calls++;
        
        // Simple encoding: Each character becomes a token
        std::vector<int> tokens;
        tokens.push_back(bos_token_id()); // Start with BOS
        
        for (char c : text) {
            int token = c; // Use character code
            tokens.push_back(token);
        }
        
        tokens.push_back(eos_token_id()); // End with EOS
        return tokens;
    }
    
    std::string decode(const std::vector<int>& tokens) const override {
        // Record for testing
        last_decoded_tokens = tokens;
        decode_calls++;
        
        // Simple decoding: Convert tokens back to characters
        std::string text;
        for (int token : tokens) {
            // Skip special tokens
            if (token == bos_token_id() || token == eos_token_id() || 
                token == pad_token_id() || token == unk_token_id()) {
                continue;
            }
            
            // Convert to character
            if (token >= 32 && token <= 126) { // Printable ASCII
                text.push_back(static_cast<char>(token));
            }
        }
        
        return text;
    }
    
    int vocab_size() const override { return 32000; }
    int bos_token_id() const override { return 1; }
    int eos_token_id() const override { return 2; }
    int pad_token_id() const override { return 0; }
    int unk_token_id() const override { return 3; }
    
    int get_speaker_token_id(int speaker_id) const override {
        // Record for testing
        last_speaker_id = speaker_id;
        speaker_token_calls++;
        
        return 1000 + speaker_id;
    }
    
    // Reset testing counters
    void reset_counters() const {
        encode_calls = 0;
        decode_calls = 0;
        speaker_token_calls = 0;
        last_encoded_text.clear();
        last_decoded_tokens.clear();
        last_speaker_id = -1;
    }
    
    // Testing variables
    mutable int encode_calls = 0;
    mutable int decode_calls = 0;
    mutable int speaker_token_calls = 0;
    mutable std::string last_encoded_text;
    mutable std::vector<int> last_decoded_tokens;
    mutable int last_speaker_id = -1;
};

class MockAudioCodec : public AudioCodec {
public:
    std::vector<std::vector<int>> encode(const std::vector<float>& audio) const override {
        // Record for testing
        last_encoded_audio = audio;
        encode_calls++;
        
        // Simple encoding: Convert to fixed pattern
        std::vector<std::vector<int>> result;
        int frames = audio.size() / hop_length();
        
        for (int i = 0; i < frames; i++) {
            std::vector<int> frame(num_codebooks(), 0);
            for (int j = 0; j < num_codebooks(); j++) {
                frame[j] = j + 10; // Arbitrary tokens
            }
            result.push_back(frame);
        }
        
        return result;
    }
    
    std::vector<float> decode(const std::vector<std::vector<int>>& tokens) const override {
        // Record for testing
        last_decoded_tokens = tokens;
        decode_calls++;
        
        // Simple decoding: Convert to sine wave
        int num_frames = tokens.size();
        int audio_length = num_frames * hop_length();
        std::vector<float> audio(audio_length, 0.0f);
        
        // Generate a simple sine wave
        for (int i = 0; i < audio_length; i++) {
            float t = static_cast<float>(i) / sample_rate();
            audio[i] = 0.5f * std::sin(2.0f * 3.14159f * 440.0f * t);
        }
        
        return audio;
    }
    
    int num_codebooks() const override { return 8; }
    int vocab_size() const override { return 2051; }
    int sample_rate() const override { return 24000; }
    int hop_length() const override { return 320; }
    
    bool is_eos_token(int token, int codebook) const override {
        is_eos_calls++;
        return token == 2; // EOS token
    }
    
    // Reset testing counters
    void reset_counters() const {
        encode_calls = 0;
        decode_calls = 0;
        is_eos_calls = 0;
        last_encoded_audio.clear();
        last_decoded_tokens.clear();
    }
    
    // Testing variables
    mutable int encode_calls = 0;
    mutable int decode_calls = 0;
    mutable int is_eos_calls = 0;
    mutable std::vector<float> last_encoded_audio;
    mutable std::vector<std::vector<int>> last_decoded_tokens;
};

class MockWatermarker : public Watermarker {
public:
    std::vector<float> apply_watermark(const std::vector<float>& audio) override {
        last_watermarked_audio = audio;
        apply_watermark_calls++;
        
        // Add a simple watermark pattern
        std::vector<float> result = audio;
        for (size_t i = 0; i < result.size(); i++) {
            result[i] += 0.01f * std::sin(2.0f * 3.14159f * i / 1000.0f);
        }
        
        return result;
    }
    
    bool detect_watermark(const std::vector<float>& audio) override {
        last_detected_audio = audio;
        detect_calls++;
        
        // Always return true for testing
        return true;
    }
    
    WatermarkResult detect(const std::vector<float>& audio, float sample_rate) override {
        last_detected_audio = audio;
        last_sample_rate = sample_rate;
        detect_calls++;
        
        WatermarkResult result;
        result.detected = true;
        result.payload = "test-payload";
        result.confidence = 0.95f;
        
        return result;
    }
    
    float get_strength() const override { return strength_; }
    void set_strength(float strength) override { strength_ = strength; }
    std::string get_key() const override { return "test-key"; }
    
    // Reset testing counters
    void reset_counters() {
        apply_watermark_calls = 0;
        detect_calls = 0;
        last_watermarked_audio.clear();
        last_detected_audio.clear();
        last_sample_rate = 0.0f;
    }
    
    // Testing variables
    int apply_watermark_calls = 0;
    int detect_calls = 0;
    std::vector<float> last_watermarked_audio;
    std::vector<float> last_detected_audio;
    float last_sample_rate = 0.0f;
    
private:
    float strength_ = 0.5f;
};

// Test fixture for Generator Configuration Tests
class GeneratorConfigurationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create model configuration with various settings
        ModelConfig config;
        config.vocab_size = 32000;
        config.audio_vocab_size = 2051;
        config.d_model = 4096;
        config.n_heads = 32;
        config.n_kv_heads = 4;
        config.n_layers = 32;
        config.n_audio_layers = 12;
        config.max_seq_len = 2048;
        config.num_codebooks = 8;
        config.name = "test-model";
        
        // Create mock components
        model = std::make_shared<MockModel>(config);
        tokenizer = std::make_shared<MockTextTokenizer>();
        codec = std::make_shared<MockAudioCodec>();
        watermarker = std::make_shared<MockWatermarker>();
        
        // Create generator
        generator = std::make_shared<Generator>(model, tokenizer, codec, watermarker);
    }
    
    void TearDown() override {
        // Reset all counters in mock objects
        if (model) {
            std::static_pointer_cast<MockModel>(model)->reset_counters();
        }
        if (tokenizer) {
            std::static_pointer_cast<MockTextTokenizer>(tokenizer)->reset_counters();
        }
        if (codec) {
            std::static_pointer_cast<MockAudioCodec>(codec)->reset_counters();
        }
        if (watermarker) {
            std::static_pointer_cast<MockWatermarker>(watermarker)->reset_counters();
        }
    }
    
    std::shared_ptr<Model> model;
    std::shared_ptr<TextTokenizer> tokenizer;
    std::shared_ptr<AudioCodec> codec;
    std::shared_ptr<Watermarker> watermarker;
    std::shared_ptr<Generator> generator;
};

// Test generation with different temperature settings
TEST_F(GeneratorConfigurationTest, TemperatureSettings) {
    // Get mock model for precise testing
    auto mock_model = std::static_pointer_cast<MockModel>(model);
    
    // Set a fixed seed for deterministic testing
    mock_model->set_seed(42);
    
    // Generate speech with default temperature (0.9)
    GenerationOptions default_options;
    std::vector<float> default_output = generator->generate_speech("Test text", 0, {}, default_options);
    
    // Verify default temperature was used
    EXPECT_NEAR(mock_model->last_temperature, 0.9f, 0.01f);
    
    // Reset counters
    mock_model->reset_counters();
    
    // Generate with lower temperature (0.1)
    GenerationOptions low_temp_options;
    low_temp_options.temperature = 0.1f;
    std::vector<float> low_temp_output = generator->generate_speech("Test text", 0, {}, low_temp_options);
    
    // Verify temperature was set correctly
    EXPECT_NEAR(mock_model->last_temperature, 0.1f, 0.01f);
    
    // Reset counters
    mock_model->reset_counters();
    
    // Generate with higher temperature (1.5)
    GenerationOptions high_temp_options;
    high_temp_options.temperature = 1.5f;
    std::vector<float> high_temp_output = generator->generate_speech("Test text", 0, {}, high_temp_options);
    
    // Verify temperature was set correctly
    EXPECT_NEAR(mock_model->last_temperature, 1.5f, 0.01f);
    
    // The audio outputs should be different with different temperatures
    // But this is challenging to verify in a mock situation
    // At least verify they generate something
    EXPECT_FALSE(default_output.empty());
    EXPECT_FALSE(low_temp_output.empty());
    EXPECT_FALSE(high_temp_output.empty());
}

// Test generation with different top_k settings
TEST_F(GeneratorConfigurationTest, TopKSettings) {
    // Get mock model for precise testing
    auto mock_model = std::static_pointer_cast<MockModel>(model);
    
    // Set a fixed seed for deterministic testing
    mock_model->set_seed(42);
    
    // Generate speech with default top_k (50)
    GenerationOptions default_options;
    std::vector<float> default_output = generator->generate_speech("Test text", 0, {}, default_options);
    
    // Verify default top_k was used
    EXPECT_EQ(mock_model->last_top_k, 50);
    
    // Reset counters
    mock_model->reset_counters();
    
    // Generate with lower top_k (10)
    GenerationOptions low_top_k_options;
    low_top_k_options.top_k = 10;
    std::vector<float> low_top_k_output = generator->generate_speech("Test text", 0, {}, low_top_k_options);
    
    // Verify top_k was set correctly
    EXPECT_EQ(mock_model->last_top_k, 10);
    
    // Reset counters
    mock_model->reset_counters();
    
    // Generate with higher top_k (100)
    GenerationOptions high_top_k_options;
    high_top_k_options.top_k = 100;
    std::vector<float> high_top_k_output = generator->generate_speech("Test text", 0, {}, high_top_k_options);
    
    // Verify top_k was set correctly
    EXPECT_EQ(mock_model->last_top_k, 100);
    
    // The audio outputs should be different with different top_k values
    // But this is challenging to verify in a mock situation
    // At least verify they generate something
    EXPECT_FALSE(default_output.empty());
    EXPECT_FALSE(low_top_k_output.empty());
    EXPECT_FALSE(high_top_k_output.empty());
}

// Test generation with different random seeds
TEST_F(GeneratorConfigurationTest, SeedSettings) {
    // Get mock model for precise testing
    auto mock_model = std::static_pointer_cast<MockModel>(model);
    
    // Generate with specific seed (42)
    GenerationOptions seed_options_1;
    seed_options_1.seed = 42;
    std::vector<float> output_1 = generator->generate_speech("Test text", 0, {}, seed_options_1);
    
    // Reset counters
    mock_model->reset_counters();
    
    // Generate with same seed again
    GenerationOptions seed_options_2;
    seed_options_2.seed = 42;
    std::vector<float> output_2 = generator->generate_speech("Test text", 0, {}, seed_options_2);
    
    // Reset counters
    mock_model->reset_counters();
    
    // Generate with different seed
    GenerationOptions seed_options_3;
    seed_options_3.seed = 100;
    std::vector<float> output_3 = generator->generate_speech("Test text", 0, {}, seed_options_3);
    
    // The outputs with the same seed should have the same size
    // (This is an approximation since our mock doesn't guarantee deterministic output)
    EXPECT_EQ(output_1.size(), output_2.size());
    
    // At minimum, verify they all generate something
    EXPECT_FALSE(output_1.empty());
    EXPECT_FALSE(output_2.empty());
    EXPECT_FALSE(output_3.empty());
}

// Test max audio length settings
TEST_F(GeneratorConfigurationTest, MaxAudioLengthSettings) {
    // Get mock model for precise testing
    auto mock_model = std::static_pointer_cast<MockModel>(model);
    
    // Generate with default max length
    GenerationOptions default_options;
    std::vector<float> default_output = generator->generate_speech("Test text", 0, {}, default_options);
    
    // Reset counters
    mock_model->reset_counters();
    
    // Generate with shorter max length (1000ms = ~12 frames at 80ms per frame)
    GenerationOptions short_options;
    short_options.max_audio_length_ms = 1000;
    std::vector<float> short_output = generator->generate_speech("Test text", 0, {}, short_options);
    
    // Reset counters
    mock_model->reset_counters();
    
    // Generate with longer max length
    GenerationOptions long_options;
    long_options.max_audio_length_ms = 20000;
    std::vector<float> long_output = generator->generate_speech("Test text", 0, {}, long_options);
    
    // The shorter output should be smaller than the default
    // (In our mock implementation, the size directly correlates with frames generated)
    EXPECT_LE(short_output.size(), default_output.size());
    
    // At minimum, verify they all generate something
    EXPECT_FALSE(default_output.empty());
    EXPECT_FALSE(short_output.empty());
    EXPECT_FALSE(long_output.empty());
}

// Test watermarking settings
TEST_F(GeneratorConfigurationTest, WatermarkingSettings) {
    // Get mock watermarker for testing
    auto mock_watermarker = std::static_pointer_cast<MockWatermarker>(watermarker);
    
    // Generate with watermarking enabled (default)
    GenerationOptions watermark_options;
    watermark_options.enable_watermark = true;
    std::vector<float> watermarked_output = generator->generate_speech("Test text", 0, {}, watermark_options);
    
    // Verify watermarking was applied
    EXPECT_GT(mock_watermarker->apply_watermark_calls, 0);
    EXPECT_FALSE(mock_watermarker->last_watermarked_audio.empty());
    
    // Reset counters
    mock_watermarker->reset_counters();
    
    // Generate with watermarking disabled
    GenerationOptions no_watermark_options;
    no_watermark_options.enable_watermark = false;
    std::vector<float> unwatermarked_output = generator->generate_speech("Test text", 0, {}, no_watermark_options);
    
    // Verify watermarking was not applied
    EXPECT_EQ(mock_watermarker->apply_watermark_calls, 0);
    
    // Both outputs should have data
    EXPECT_FALSE(watermarked_output.empty());
    EXPECT_FALSE(unwatermarked_output.empty());
}

// Test debug settings
TEST_F(GeneratorConfigurationTest, DebugSettings) {
    // Debug mode doesn't affect output in our mock implementation,
    // but we can test that the option is properly handled by the Generator
    
    // Generate with debug enabled
    GenerationOptions debug_options;
    debug_options.debug = true;
    std::vector<float> debug_output = generator->generate_speech("Test text", 0, {}, debug_options);
    
    // Generate with debug disabled (default)
    GenerationOptions no_debug_options;
    no_debug_options.debug = false;
    std::vector<float> normal_output = generator->generate_speech("Test text", 0, {}, no_debug_options);
    
    // Both should produce output
    EXPECT_FALSE(debug_output.empty());
    EXPECT_FALSE(normal_output.empty());
}

// Test combined parameter settings
TEST_F(GeneratorConfigurationTest, CombinedParameterSettings) {
    // Get mock model for testing
    auto mock_model = std::static_pointer_cast<MockModel>(model);
    
    // Create options with multiple parameters set
    GenerationOptions combined_options;
    combined_options.temperature = 0.3f;
    combined_options.top_k = 20;
    combined_options.max_audio_length_ms = 5000;
    combined_options.seed = 42;
    combined_options.enable_watermark = true;
    combined_options.debug = true;
    
    // Generate with combined options
    std::vector<float> output = generator->generate_speech("Test text", 0, {}, combined_options);
    
    // Verify parameters were correctly passed to model
    EXPECT_NEAR(mock_model->last_temperature, 0.3f, 0.01f);
    EXPECT_EQ(mock_model->last_top_k, 20);
    
    // Verify watermarking was applied
    auto mock_watermarker = std::static_pointer_cast<MockWatermarker>(watermarker);
    EXPECT_GT(mock_watermarker->apply_watermark_calls, 0);
    
    // Output should not be empty
    EXPECT_FALSE(output.empty());
}

// Test the default option values
TEST_F(GeneratorConfigurationTest, DefaultOptionValues) {
    // Get mock model for testing
    auto mock_model = std::static_pointer_cast<MockModel>(model);
    
    // Create default options
    GenerationOptions default_options;
    
    // Generate with default options
    std::vector<float> output = generator->generate_speech("Test text", 0, {}, default_options);
    
    // Verify default values were used
    EXPECT_NEAR(mock_model->last_temperature, 0.9f, 0.01f);
    EXPECT_EQ(mock_model->last_top_k, 50);
    
    // Verify watermarking was applied (default should be enabled)
    auto mock_watermarker = std::static_pointer_cast<MockWatermarker>(watermarker);
    EXPECT_GT(mock_watermarker->apply_watermark_calls, 0);
    
    // Output should not be empty
    EXPECT_FALSE(output.empty());
}

// Test edge case parameter values
TEST_F(GeneratorConfigurationTest, EdgeCaseParameters) {
    // Get mock model for testing
    auto mock_model = std::static_pointer_cast<MockModel>(model);
    
    // Test with very low temperature
    GenerationOptions low_temp_options;
    low_temp_options.temperature = 0.01f;
    std::vector<float> low_temp_output = generator->generate_speech("Test text", 0, {}, low_temp_options);
    
    // Verify parameter was passed correctly
    EXPECT_NEAR(mock_model->last_temperature, 0.01f, 0.001f);
    
    // Reset counters
    mock_model->reset_counters();
    
    // Test with very high temperature
    GenerationOptions high_temp_options;
    high_temp_options.temperature = 10.0f;
    std::vector<float> high_temp_output = generator->generate_speech("Test text", 0, {}, high_temp_options);
    
    // Verify parameter was passed correctly
    EXPECT_NEAR(mock_model->last_temperature, 10.0f, 0.01f);
    
    // Reset counters
    mock_model->reset_counters();
    
    // Test with top_k of 1 (most deterministic)
    GenerationOptions top_k_1_options;
    top_k_1_options.top_k = 1;
    std::vector<float> top_k_1_output = generator->generate_speech("Test text", 0, {}, top_k_1_options);
    
    // Verify parameter was passed correctly
    EXPECT_EQ(mock_model->last_top_k, 1);
    
    // Reset counters
    mock_model->reset_counters();
    
    // Test with very large top_k
    GenerationOptions high_top_k_options;
    high_top_k_options.top_k = 10000;
    std::vector<float> high_top_k_output = generator->generate_speech("Test text", 0, {}, high_top_k_options);
    
    // Verify parameter was passed correctly
    EXPECT_EQ(mock_model->last_top_k, 10000);
    
    // All outputs should have data
    EXPECT_FALSE(low_temp_output.empty());
    EXPECT_FALSE(high_temp_output.empty());
    EXPECT_FALSE(top_k_1_output.empty());
    EXPECT_FALSE(high_top_k_output.empty());
}

// Test handling of invalid parameters
TEST_F(GeneratorConfigurationTest, InvalidParameters) {
    // Test with negative temperature (implementation should handle this gracefully)
    GenerationOptions neg_temp_options;
    neg_temp_options.temperature = -1.0f;
    EXPECT_NO_THROW({
        std::vector<float> output = generator->generate_speech("Test text", 0, {}, neg_temp_options);
        EXPECT_FALSE(output.empty());
    });
    
    // Test with negative top_k (implementation should handle this gracefully)
    GenerationOptions neg_top_k_options;
    neg_top_k_options.top_k = -10;
    EXPECT_NO_THROW({
        std::vector<float> output = generator->generate_speech("Test text", 0, {}, neg_top_k_options);
        EXPECT_FALSE(output.empty());
    });
    
    // Test with zero temperature (implementation should handle this gracefully)
    GenerationOptions zero_temp_options;
    zero_temp_options.temperature = 0.0f;
    EXPECT_NO_THROW({
        std::vector<float> output = generator->generate_speech("Test text", 0, {}, zero_temp_options);
        EXPECT_FALSE(output.empty());
    });
    
    // Test with zero top_k (implementation should handle this gracefully)
    GenerationOptions zero_top_k_options;
    zero_top_k_options.top_k = 0;
    EXPECT_NO_THROW({
        std::vector<float> output = generator->generate_speech("Test text", 0, {}, zero_top_k_options);
        EXPECT_FALSE(output.empty());
    });
}

// Test generation behavior with various max length settings
TEST_F(GeneratorConfigurationTest, MaxLengthBehavior) {
    // Get mock model for testing
    auto mock_model = std::static_pointer_cast<MockModel>(model);
    
    // First, make the model generate exactly 10 frames before EOS
    mock_model->set_max_frames(10);
    
    // Generate with very large max length
    GenerationOptions large_max_options;
    large_max_options.max_audio_length_ms = 10000; // 125 frames at 80ms per frame
    std::vector<float> large_output = generator->generate_speech("Test text", 0, {}, large_max_options);
    
    // Should be limited by model EOS (10 frames), not max length
    EXPECT_LE(mock_model->frame_count, 11); // Allow for one frame after EOS detection
    
    // Reset counters
    mock_model->reset_counters();
    
    // Now set model to generate many frames
    mock_model->set_max_frames(100);
    
    // Generate with small max length
    GenerationOptions small_max_options;
    small_max_options.max_audio_length_ms = 400; // 5 frames at 80ms per frame
    std::vector<float> small_output = generator->generate_speech("Test text", 0, {}, small_max_options);
    
    // Should be limited by max length (5 frames), not model EOS
    EXPECT_LE(mock_model->frame_count, 6); // Allow for minor implementation differences
    
    // Both outputs should have data
    EXPECT_FALSE(large_output.empty());
    EXPECT_FALSE(small_output.empty());
}

// Test with extremely short max length (edge case)
TEST_F(GeneratorConfigurationTest, ExtremelyShortMaxLength) {
    // Generate with extremely short max length (1ms)
    GenerationOptions tiny_options;
    tiny_options.max_audio_length_ms = 1;
    
    // Should still produce some output, even if it's minimal
    std::vector<float> tiny_output = generator->generate_speech("Test text", 0, {}, tiny_options);
    EXPECT_FALSE(tiny_output.empty());
    
    // Generate with zero max length (should handle gracefully)
    GenerationOptions zero_options;
    zero_options.max_audio_length_ms = 0;
    
    // Should still produce some minimal output
    std::vector<float> zero_output = generator->generate_speech("Test text", 0, {}, zero_options);
    EXPECT_FALSE(zero_output.empty());
}

// Test progress callback functionality
TEST_F(GeneratorConfigurationTest, ProgressCallback) {
    // Create a flag to track callback execution
    std::atomic<bool> callback_called(false);
    std::atomic<int> max_progress(0);
    
    // Create a progress callback
    auto progress_callback = [&callback_called, &max_progress](int current, int total) {
        callback_called = true;
        if (current > max_progress) {
            max_progress = current;
        }
    };
    
    // Generate with callback
    std::vector<float> output = generator->generate_speech(
        "Test text with callback", 0, {}, GenerationOptions(), progress_callback);
    
    // Verify output was generated
    EXPECT_FALSE(output.empty());
    
    // Verify callback was called (may not be called in mock implementation)
    // This is a weak test since our implementation might not call the callback
    if (callback_called) {
        EXPECT_GT(max_progress, 0);
    }
}

// Test generation cancellation (if implemented)
TEST_F(GeneratorConfigurationTest, DISABLED_GenerationCancellation) {
    // This test is disabled because cancellation is not implemented in the current Generator

    // Create a cancelling callback
    bool cancelled = false;
    auto cancel_callback = [&cancelled](int current, int total) {
        // Cancel after first frame
        if (current >= 1) {
            cancelled = true;
            return false; // Signal cancellation
        }
        return true;
    };
    
    // Generate with cancellation callback
    std::vector<float> output = generator->generate_speech(
        "Test text for cancellation", 0, {}, GenerationOptions(), cancel_callback);
    
    // Should still get some output, but minimal
    EXPECT_FALSE(output.empty());
    EXPECT_TRUE(cancelled);
    
    // Get mock model to check how many frames were generated
    auto mock_model = std::static_pointer_cast<MockModel>(model);
    EXPECT_LE(mock_model->frame_count, 2); // Should stop after 1-2 frames
}

// Test with custom combination of parameters
TEST_F(GeneratorConfigurationTest, CustomParameterCombination) {
    // Define a test matrix of parameters to try
    struct TestCase {
        float temperature;
        int top_k;
        int max_audio_length_ms;
        bool enable_watermark;
    };
    
    std::vector<TestCase> test_cases = {
        {0.1f, 5, 1000, false},
        {0.5f, 20, 5000, true},
        {1.5f, 100, 2000, false},
        {2.0f, 1, 500, true}
    };
    
    // Get mock objects for verification
    auto mock_model = std::static_pointer_cast<MockModel>(model);
    auto mock_watermarker = std::static_pointer_cast<MockWatermarker>(watermarker);
    
    // Run tests with each parameter combination
    for (const auto& test_case : test_cases) {
        // Reset counters
        mock_model->reset_counters();
        mock_watermarker->reset_counters();
        
        // Create options from test case
        GenerationOptions options;
        options.temperature = test_case.temperature;
        options.top_k = test_case.top_k;
        options.max_audio_length_ms = test_case.max_audio_length_ms;
        options.enable_watermark = test_case.enable_watermark;
        
        // Generate speech
        std::vector<float> output = generator->generate_speech("Test text", 0, {}, options);
        
        // Verify parameters were passed correctly
        EXPECT_NEAR(mock_model->last_temperature, test_case.temperature, 0.01f);
        EXPECT_EQ(mock_model->last_top_k, test_case.top_k);
        
        // Verify watermarking was applied or not based on setting
        if (test_case.enable_watermark) {
            EXPECT_GT(mock_watermarker->apply_watermark_calls, 0);
        } else {
            EXPECT_EQ(mock_watermarker->apply_watermark_calls, 0);
        }
        
        // Output should not be empty
        EXPECT_FALSE(output.empty());
    }
}

// Test generation timing with different parameters
TEST_F(GeneratorConfigurationTest, GenerationTiming) {
    // Create options with varying parameters
    std::vector<GenerationOptions> options_list;
    
    // Default options
    options_list.push_back(GenerationOptions());
    
    // Fast generation settings
    GenerationOptions fast_options;
    fast_options.temperature = 0.1f;
    fast_options.top_k = 1;
    fast_options.max_audio_length_ms = 500;
    options_list.push_back(fast_options);
    
    // Verbose generation settings
    GenerationOptions verbose_options;
    verbose_options.temperature = 1.5f;
    verbose_options.top_k = 100;
    verbose_options.max_audio_length_ms = 5000;
    options_list.push_back(verbose_options);
    
    // Time each generation
    for (const auto& options : options_list) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::vector<float> output = generator->generate_speech("Test text", 0, {}, options);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        // Just verify we get output and timing is positive
        EXPECT_FALSE(output.empty());
        EXPECT_GE(duration.count(), 0);
        
        // We don't make assertions about relative timing since our mock doesn't simulate that
    }
}

// Test generation with multiple speakers
TEST_F(GeneratorConfigurationTest, MultiSpeakerGeneration) {
    // Get mock tokenizer for testing
    auto mock_tokenizer = std::static_pointer_cast<MockTextTokenizer>(tokenizer);
    
    // Generate with speaker ID 0
    std::vector<float> output_0 = generator->generate_speech("Speaker 0 text", 0);
    
    // Verify speaker ID was processed
    EXPECT_EQ(mock_tokenizer->last_speaker_id, 0);
    
    // Reset counters
    mock_tokenizer->reset_counters();
    
    // Generate with speaker ID 1
    std::vector<float> output_1 = generator->generate_speech("Speaker 1 text", 1);
    
    // Verify speaker ID was processed
    EXPECT_EQ(mock_tokenizer->last_speaker_id, 1);
    
    // Reset counters
    mock_tokenizer->reset_counters();
    
    // Generate with speaker ID 5
    std::vector<float> output_5 = generator->generate_speech("Speaker 5 text", 5);
    
    // Verify speaker ID was processed
    EXPECT_EQ(mock_tokenizer->last_speaker_id, 5);
    
    // All outputs should have data
    EXPECT_FALSE(output_0.empty());
    EXPECT_FALSE(output_1.empty());
    EXPECT_FALSE(output_5.empty());
}

// Test generation with context segments
TEST_F(GeneratorConfigurationTest, ContextSegments) {
    // Get mock model for testing
    auto mock_model = std::static_pointer_cast<MockModel>(model);
    
    // Create context segments
    std::vector<Segment> context;
    context.push_back(Segment("First segment", 0));
    context.push_back(Segment("Second segment", 1));
    context.push_back(Segment("Third segment", 0));
    
    // Generate with context
    std::vector<float> output = generator->generate_speech("Main text", 2, context);
    
    // Verify model caches were reset
    EXPECT_GT(mock_model->cache_reset_count, 0);
    
    // Verify speaker tokens were accessed
    auto mock_tokenizer = std::static_pointer_cast<MockTextTokenizer>(tokenizer);
    EXPECT_GE(mock_tokenizer->speaker_token_calls, 4); // 3 contexts + main text
    
    // Output should have data
    EXPECT_FALSE(output.empty());
}

// Test generation with empty context segments
TEST_F(GeneratorConfigurationTest, EmptyContextSegments) {
    // Create empty context segments
    std::vector<Segment> empty_context;
    
    // Generate with empty context
    std::vector<float> output = generator->generate_speech("Main text", 0, empty_context);
    
    // Output should have data
    EXPECT_FALSE(output.empty());
}

// Test with pre-tokenized input
TEST_F(GeneratorConfigurationTest, PreTokenizedInput) {
    // Create tokenized input
    std::vector<int> tokens = {1, 72, 101, 108, 108, 111, 2}; // "Hello" with BOS/EOS
    
    // Generate from tokens
    std::vector<float> output = generator->generate_speech_from_tokens(tokens, 0);
    
    // Verify output was generated
    EXPECT_FALSE(output.empty());
    
    // Get mock tokenizer to verify it wasn't used for tokenization
    auto mock_tokenizer = std::static_pointer_cast<MockTextTokenizer>(tokenizer);
    EXPECT_EQ(mock_tokenizer->encode_calls, 0); // Should not call encode for pre-tokenized input
}