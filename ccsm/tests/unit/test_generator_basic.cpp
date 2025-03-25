#include <gtest/gtest.h>
#include <ccsm/generator.h>
#include <ccsm/model.h>
#include <ccsm/tokenizer.h>
#include <ccsm/watermarking.h>
#include <memory>
#include <vector>
#include <string>
#include <functional>

using namespace ccsm;

// Basic mock classes for simple generator tests
class BasicMockModel : public Model {
public:
    BasicMockModel() : Model(createConfig()) {}
    
    static ModelConfig createConfig() {
        ModelConfig config;
        config.name = "Basic Mock Model";
        config.vocab_size = 32000;
        config.audio_vocab_size = 2051;
        config.d_model = 1024;
        config.n_heads = 16;
        config.n_kv_heads = 8;
        config.n_layers = 32;
        config.n_audio_layers = 12;
        config.max_seq_len = 2048;
        config.num_codebooks = 8;
        return config;
    }
    
    bool load_weights(const std::string& path) override { return true; }
    bool load_weights(std::shared_ptr<ModelLoader> loader) override { return true; }
    bool load_weights(const WeightMap& weights) override { return true; }
    
    std::vector<int> generate_frame(
        const std::vector<int>& tokens,
        const std::vector<int>& positions,
        float temperature = 0.9f,
        int top_k = 50) override {
        
        // Track calls
        frame_count++;
        last_tokens = tokens;
        last_positions = positions;
        last_temperature = temperature;
        last_top_k = top_k;
        
        // Generate a consistent frame based on the input parameters
        std::vector<int> frame(config_.num_codebooks);
        unsigned int seed = static_cast<unsigned int>(temperature * 1000 + top_k);
        std::mt19937 rng(seed);
        
        // Fill the frame with deterministic values
        for (int i = 0; i < config_.num_codebooks; i++) {
            int value = (rng() % (config_.audio_vocab_size - 1)) + 1; // Avoid 0 which is often EOS
            frame[i] = value;
        }
        
        return frame;
    }
    
    void reset_caches() override {
        reset_called = true;
    }
    
    void optimize_memory(size_t max_memory_mb = 0) override {
        memory_optimized = true;
    }
    
    void prune_caches(float prune_factor = 0.5f) override {
        caches_pruned = true;
    }
    
    std::vector<float> get_backbone_logits(
        const std::vector<int>& tokens,
        const std::vector<int>& positions) override {
        return std::vector<float>(config_.vocab_size, 0.0f);
    }
    
    std::vector<float> get_decoder_logits(
        const std::vector<int>& tokens,
        const std::vector<int>& positions,
        int codebook) override {
        return std::vector<float>(config_.audio_vocab_size, 0.0f);
    }
    
    // Test helpers
    int frame_count = 0;
    bool reset_called = false;
    bool memory_optimized = false;
    bool caches_pruned = false;
    std::vector<int> last_tokens;
    std::vector<int> last_positions;
    float last_temperature = 0.0f;
    int last_top_k = 0;
    
    void reset_stats() {
        frame_count = 0;
        reset_called = false;
        memory_optimized = false;
        caches_pruned = false;
        last_tokens.clear();
        last_positions.clear();
        last_temperature = 0.0f;
        last_top_k = 0;
    }
};

class BasicMockTokenizer : public TextTokenizer {
public:
    std::vector<int> encode(const std::string& text) const override {
        // Simple mock that breaks text into tokens of single characters
        std::vector<int> tokens;
        for (char c : text) {
            tokens.push_back(static_cast<int>(c) % 32000);
        }
        return tokens;
    }
    
    std::string decode(const std::vector<int>& tokens) const override {
        // Simple mock that just converts tokens to characters
        std::string text;
        for (int token : tokens) {
            text += static_cast<char>(token % 128);
        }
        return text;
    }
    
    int vocab_size() const override { return 32000; }
    int bos_token_id() const override { return 1; }
    int eos_token_id() const override { return 2; }
    int pad_token_id() const override { return 0; }
    int unk_token_id() const override { return 3; }
    
    int get_speaker_token_id(int speaker_id) const override {
        return 1000 + speaker_id;
    }
    
    std::vector<int> get_audio_token_ids() const override {
        return {1000, 1001, 1002, 1003};
    }
};

class BasicMockAudioCodec : public AudioCodec {
public:
    std::vector<std::vector<int>> encode(const std::vector<float>& audio) const override {
        // Create a mock encoding with 8 codebooks
        std::vector<std::vector<int>> result(8);
        for (int i = 0; i < 8; i++) {
            result[i].push_back(i + 1);
        }
        return result;
    }
    
    std::vector<float> decode(const std::vector<std::vector<int>>& tokens) const override {
        // Create a simple audio waveform
        // Find the maximum number of tokens in any codebook
        size_t max_tokens = 0;
        for (const auto& codebook : tokens) {
            max_tokens = std::max(max_tokens, codebook.size());
        }
        
        // Create audio samples (approximately 24000 samples per second)
        size_t samples_per_token = 1920; // 80ms at 24kHz
        std::vector<float> audio(max_tokens * samples_per_token, 0.0f);
        
        // Create a simple sine wave
        for (size_t i = 0; i < audio.size(); i++) {
            float t = static_cast<float>(i) / 24000.0f;
            audio[i] = 0.5f * sin(2.0f * M_PI * 440.0f * t);
        }
        
        return audio;
    }
    
    int num_codebooks() const override { return 8; }
    int vocab_size() const override { return 2051; }
    int sample_rate() const override { return 24000; }
    int hop_length() const override { return 320; }
    
    bool is_eos_token(int token, int codebook) const override {
        return token == 0;
    }
};

class BasicMockWatermarker : public Watermarker {
public:
    std::vector<float> apply_watermark(const std::vector<float>& audio) override {
        was_applied = true;
        return audio; // Just return the same audio
    }
    
    bool detect_watermark(const std::vector<float>& audio) override {
        return was_applied;
    }
    
    WatermarkResult detect(const std::vector<float>& audio, float sample_rate) override {
        WatermarkResult result;
        result.detected = was_applied;
        result.payload = "test-watermark";
        result.confidence = 0.95f;
        return result;
    }
    
    float get_strength() const override { return 0.5f; }
    void set_strength(float strength) override { /* no-op */ }
    std::string get_key() const override { return "test-key"; }
    
    bool was_applied = false;
};

// Test fixture
class GeneratorBasicTest : public ::testing::Test {
protected:
    void SetUp() override {
        model = std::make_shared<BasicMockModel>();
        tokenizer = std::make_shared<BasicMockTokenizer>();
        audio_codec = std::make_shared<BasicMockAudioCodec>();
        watermarker = std::make_shared<BasicMockWatermarker>();
        
        generator = std::make_shared<Generator>(model, tokenizer, audio_codec, watermarker);
    }
    
    // Helper methods to access mock implementations
    std::shared_ptr<BasicMockModel> get_model() {
        return std::static_pointer_cast<BasicMockModel>(model);
    }
    
    std::shared_ptr<BasicMockWatermarker> get_watermarker() {
        return std::static_pointer_cast<BasicMockWatermarker>(watermarker);
    }
    
    std::shared_ptr<Model> model;
    std::shared_ptr<TextTokenizer> tokenizer;
    std::shared_ptr<AudioCodec> audio_codec;
    std::shared_ptr<Watermarker> watermarker;
    std::shared_ptr<Generator> generator;
};

// Test basic generation functionality
TEST_F(GeneratorBasicTest, BasicGeneration) {
    // Generate speech with default parameters
    std::string text = "Hello, world!";
    auto audio = generator->generate_speech(text, 0);
    
    // Verify output is not empty
    EXPECT_FALSE(audio.empty());
    
    // Verify model was called
    EXPECT_GT(get_model()->frame_count, 0);
    
    // Verify watermarking was applied
    EXPECT_TRUE(get_watermarker()->was_applied);
}

// Test basic parameter variation
TEST_F(GeneratorBasicTest, ParameterVariation) {
    // Reset stats
    get_model()->reset_stats();
    get_watermarker()->was_applied = false;
    
    // Generate with specific temperature and top_k
    std::string text = "Test parameter variation";
    float temperature = 0.5f;
    int top_k = 10;
    
    auto audio = generator->generate_speech(text, 0, temperature, top_k);
    
    // Verify output is not empty
    EXPECT_FALSE(audio.empty());
    
    // Verify parameters were passed correctly
    EXPECT_NEAR(get_model()->last_temperature, temperature, 0.001f);
    EXPECT_EQ(get_model()->last_top_k, top_k);
}

// Test generation with context
TEST_F(GeneratorBasicTest, GenerationWithContext) {
    // Reset stats
    get_model()->reset_stats();
    
    // Create context
    std::vector<Segment> context = {
        Segment("First message", 1),
        Segment("Second message", 2)
    };
    
    // Generate with context
    std::string text = "Reply to the context";
    auto audio = generator->generate_speech(text, 0, context);
    
    // Verify model was called
    EXPECT_GT(get_model()->frame_count, 0);
    
    // Verify context was included
    auto& tokens = get_model()->last_tokens;
    
    // Look for speaker tokens
    bool found_speaker1 = false;
    bool found_speaker2 = false;
    
    for (int token : tokens) {
        if (token == tokenizer->get_speaker_token_id(1)) {
            found_speaker1 = true;
        }
        if (token == tokenizer->get_speaker_token_id(2)) {
            found_speaker2 = true;
        }
    }
    
    EXPECT_TRUE(found_speaker1);
    EXPECT_TRUE(found_speaker2);
}

// Test generation options
TEST_F(GeneratorBasicTest, GenerationOptions) {
    // Reset stats
    get_model()->reset_stats();
    get_watermarker()->was_applied = false;
    
    // Create options
    GenerationOptions options;
    options.temperature = 0.7f;
    options.top_k = 20;
    options.max_audio_length_ms = 5000;
    options.seed = 42;
    options.enable_watermark = true;
    
    // Generate with options
    std::string text = "Test generation options";
    auto audio = generator->generate_speech(text, 0, {}, options);
    
    // Verify parameters were used
    EXPECT_NEAR(get_model()->last_temperature, options.temperature, 0.001f);
    EXPECT_EQ(get_model()->last_top_k, options.top_k);
    EXPECT_TRUE(get_watermarker()->was_applied);
    
    // Test with watermarking disabled
    get_watermarker()->was_applied = false;
    options.enable_watermark = false;
    
    audio = generator->generate_speech(text, 0, {}, options);
    
    // Verify watermarking was not applied
    EXPECT_FALSE(get_watermarker()->was_applied);
}

// Test progress callback
TEST_F(GeneratorBasicTest, ProgressCallback) {
    // Reset stats
    get_model()->reset_stats();
    
    // Create tracking variables
    int callback_count = 0;
    int last_current = 0;
    int last_total = 0;
    
    // Create callback
    auto progress_callback = [&](int current, int total) {
        callback_count++;
        last_current = current;
        last_total = total;
    };
    
    // Generate with callback
    std::string text = "Test progress callback";
    auto audio = generator->generate_speech(text, 0, {}, GenerationOptions(), progress_callback);
    
    // Verify callback was called
    EXPECT_GT(callback_count, 0);
    EXPECT_GT(last_total, 0);
    EXPECT_EQ(last_current, get_model()->frame_count); // Should match the number of frames
}

// Test pre-tokenized input
TEST_F(GeneratorBasicTest, PreTokenizedInput) {
    // Reset stats
    get_model()->reset_stats();
    
    // Create pre-tokenized input
    std::vector<int> tokens = {100, 200, 300, 400, 500};
    
    // Generate from tokens
    auto audio = generator->generate_speech_from_tokens(tokens, 0);
    
    // Verify model was called with these tokens
    EXPECT_GT(get_model()->frame_count, 0);
    
    // The tokens should be in the input to the model
    bool found_tokens = false;
    for (size_t i = 0; i <= get_model()->last_tokens.size() - tokens.size(); i++) {
        bool match = true;
        for (size_t j = 0; j < tokens.size(); j++) {
            if (get_model()->last_tokens[i + j] != tokens[j]) {
                match = false;
                break;
            }
        }
        if (match) {
            found_tokens = true;
            break;
        }
    }
    
    EXPECT_TRUE(found_tokens);
}

// Test generation settings limits
TEST_F(GeneratorBasicTest, GenerationLimits) {
    // Test with extreme temperature values
    GenerationOptions cold_options;
    cold_options.temperature = 0.01f; // Very cold, should be clamped
    
    auto cold_audio = generator->generate_speech("Test cold temperature", 0, {}, cold_options);
    EXPECT_GE(get_model()->last_temperature, 0.05f); // Should be clamped to minimum
    
    // Test with hot temperature
    GenerationOptions hot_options;
    hot_options.temperature = 2.0f; // Very hot, should be clamped
    
    auto hot_audio = generator->generate_speech("Test hot temperature", 0, {}, hot_options);
    EXPECT_LE(get_model()->last_temperature, 1.5f); // Should be clamped to maximum
    
    // Test with invalid top_k
    GenerationOptions invalid_top_k;
    invalid_top_k.top_k = 0; // Invalid, should be clamped
    
    auto top_k_audio = generator->generate_speech("Test invalid top_k", 0, {}, invalid_top_k);
    EXPECT_GE(get_model()->last_top_k, 1); // Should be clamped to minimum
}

// Test with very long input
TEST_F(GeneratorBasicTest, LongInput) {
    // Create a very long input text
    std::string long_text(10000, 'a');
    
    // Generate with long text
    auto audio = generator->generate_speech(long_text, 0);
    
    // Verify token length was limited
    EXPECT_LE(get_model()->last_tokens.size(), generator->max_text_tokens());
}

// Run test
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}