#include <gtest/gtest.h>
#include <ccsm/generator.h>
#include <ccsm/model.h>
#include <ccsm/tokenizer.h>
#include <ccsm/watermarking.h>
#include <memory>
#include <string>
#include <vector>
#include <stdexcept>
#include <random>
#include <chrono>

using namespace ccsm;

// Mock model for integration tests
class IntegrationMockModel : public Model {
public:
    IntegrationMockModel(const ModelConfig& config) : Model(config) {}
    
    bool load_weights(const std::string& path) override {
        load_weights_called = true;
        return true;
    }
    
    bool load_weights(std::shared_ptr<ModelLoader> loader) override {
        load_weights_called = true;
        return true;
    }
    
    bool load_weights(const WeightMap& weights) override {
        load_weights_called = true;
        return true;
    }
    
    std::vector<int> generate_frame(
        const std::vector<int>& tokens,
        const std::vector<int>& positions,
        float temperature,
        int top_k
    ) override {
        // Store parameters for testing
        last_tokens = tokens;
        last_positions = positions;
        last_temperature = temperature;
        last_top_k = top_k;
        call_count++;
        
        // Use deterministic seed if provided, otherwise use call count
        unsigned int seed_to_use = static_cast<unsigned int>(seed >= 0 ? seed : call_count);
        std::mt19937 rng(seed_to_use);
        std::uniform_int_distribution<int> dist(1, config_.audio_vocab_size - 1);
        
        // Return output based on seed and temperature
        std::vector<int> result(config_.num_codebooks);
        for (size_t i = 0; i < result.size(); i++) {
            // Vary output based on temperature to simulate real model behavior
            if (temperature < 0.1f) {
                // Low temperature = consistent output
                result[i] = (call_count + static_cast<int>(i)) % (config_.audio_vocab_size - 1) + 1;
            } else {
                // Higher temperature = more randomness
                result[i] = dist(rng);
            }
            
            // Inject EOS token if configured
            if (should_generate_eos && call_count >= eos_after_frames && i == 0) {
                result[i] = 0; // EOS token
            }
        }
        return result;
    }
    
    void reset_caches() override {
        // Track reset calls
        reset_called = true;
    }
    
    void optimize_memory(size_t max_memory_mb = 0) override {
        // Track memory optimization
        memory_optimized = true;
        memory_limit = max_memory_mb;
    }
    
    void prune_caches(float prune_factor = 0.5f) override {
        // Track cache pruning
        cache_pruned = true;
        pruning_factor = prune_factor;
    }
    
    std::vector<float> get_backbone_logits(
        const std::vector<int>& tokens,
        const std::vector<int>& positions
    ) override {
        // Return mock logits
        std::vector<float> result(config_.audio_vocab_size);
        for (size_t i = 0; i < result.size(); i++) {
            result[i] = static_cast<float>(i) / static_cast<float>(result.size());
        }
        return result;
    }
    
    std::vector<float> get_decoder_logits(
        const std::vector<int>& tokens,
        const std::vector<int>& positions,
        int codebook
    ) override {
        // Return mock logits
        std::vector<float> result(config_.audio_vocab_size);
        for (size_t i = 0; i < result.size(); i++) {
            result[i] = static_cast<float>(i) / static_cast<float>(result.size());
        }
        return result;
    }
    
    // Testing controls and accessors
    const std::vector<int>& get_last_tokens() const { return last_tokens; }
    const std::vector<int>& get_last_positions() const { return last_positions; }
    float get_last_temperature() const { return last_temperature; }
    int get_last_top_k() const { return last_top_k; }
    int get_call_count() const { return call_count; }
    bool was_reset_called() const { return reset_called; }
    bool was_memory_optimized() const { return memory_optimized; }
    bool was_cache_pruned() const { return cache_pruned; }
    size_t get_memory_limit() const { return memory_limit; }
    float get_pruning_factor() const { return pruning_factor; }
    bool was_load_weights_called() const { return load_weights_called; }
    
    // Control EOS token generation
    void set_eos_generation(bool should_generate, int after_frames) {
        should_generate_eos = should_generate;
        eos_after_frames = after_frames;
    }
    
    // Control generation seed
    void set_seed(int new_seed) {
        seed = new_seed;
    }
    
    // Reset state for tests
    void reset_state() {
        call_count = 0;
        last_tokens.clear();
        last_positions.clear();
        last_temperature = 0.0f;
        last_top_k = 0;
        reset_called = false;
        memory_optimized = false;
        cache_pruned = false;
        memory_limit = 0;
        pruning_factor = 0.0f;
        load_weights_called = false;
    }
    
private:
    std::vector<int> last_tokens;
    std::vector<int> last_positions;
    float last_temperature = 0.0f;
    int last_top_k = 0;
    int call_count = 0;
    bool reset_called = false;
    bool memory_optimized = false;
    bool cache_pruned = false;
    size_t memory_limit = 0;
    float pruning_factor = 0.0f;
    bool load_weights_called = false;
    
    // EOS generation controls
    bool should_generate_eos = false;
    int eos_after_frames = 10;
    
    // Random seed
    int seed = -1;
};

// Enhanced mock text tokenizer for integration tests
class IntegrationMockTextTokenizer : public TextTokenizer {
public:
    IntegrationMockTextTokenizer() = default;
    
    std::vector<int> encode(const std::string& text) const override {
        // Store for testing
        const_cast<IntegrationMockTextTokenizer*>(this)->last_text = text;
        const_cast<IntegrationMockTextTokenizer*>(this)->encode_call_count++;
        
        if (fixed_tokens.empty()) {
            // Return tokens based on text length
            std::vector<int> tokens;
            for (size_t i = 0; i < text.size(); i++) {
                tokens.push_back(static_cast<int>(text[i]) % 1000 + 10);
            }
            return tokens;
        } else {
            return fixed_tokens;
        }
    }
    
    std::string decode(const std::vector<int>& tokens) const override {
        // Store for testing
        const_cast<IntegrationMockTextTokenizer*>(this)->last_tokens = tokens;
        const_cast<IntegrationMockTextTokenizer*>(this)->decode_call_count++;
        
        // Create a simple text from tokens
        std::string text;
        for (const int token : tokens) {
            char c = static_cast<char>((token % 26) + 'a');
            text.push_back(c);
        }
        return text;
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
        return {5000, 5001, 5002, 5003};
    }
    
    // Testing accessors
    const std::string& get_last_text() const { return last_text; }
    const std::vector<int>& get_last_tokens() const { return last_tokens; }
    int get_encode_call_count() const { return encode_call_count; }
    int get_decode_call_count() const { return decode_call_count; }
    
    // For test control
    void set_fixed_tokens(const std::vector<int>& tokens) {
        fixed_tokens = tokens;
    }
    
    void reset_state() {
        last_text.clear();
        last_tokens.clear();
        encode_call_count = 0;
        decode_call_count = 0;
        fixed_tokens.clear();
    }
    
private:
    std::string last_text;
    std::vector<int> last_tokens;
    int encode_call_count = 0;
    int decode_call_count = 0;
    std::vector<int> fixed_tokens;
};

// Enhanced mock audio codec for integration tests
class IntegrationMockAudioCodec : public AudioCodec {
public:
    IntegrationMockAudioCodec() = default;
    
    std::vector<std::vector<int>> encode(const std::vector<float>& audio) const override {
        // Store for testing
        const_cast<IntegrationMockAudioCodec*>(this)->last_audio = audio;
        const_cast<IntegrationMockAudioCodec*>(this)->encode_call_count++;
        
        // Create token frames
        std::vector<std::vector<int>> tokens(num_codebooks_);
        size_t num_frames = (audio.size() / samples_per_frame_) + 1;
        
        for (int cb = 0; cb < num_codebooks_; cb++) {
            for (size_t i = 0; i < num_frames; i++) {
                tokens[cb].push_back((i + cb) % (vocab_size_ - 1) + 1);
            }
        }
        
        return tokens;
    }
    
    std::vector<float> decode(const std::vector<std::vector<int>>& tokens) const override {
        // Store for testing
        const_cast<IntegrationMockAudioCodec*>(this)->last_tokens = tokens;
        const_cast<IntegrationMockAudioCodec*>(this)->decode_call_count++;
        
        if (!fixed_audio.empty()) {
            return fixed_audio;
        }
        
        // Find max frames
        size_t max_frames = 0;
        for (const auto& codebook : tokens) {
            max_frames = std::max(max_frames, codebook.size());
        }
        
        // Create audio samples
        std::vector<float> audio(max_frames * samples_per_frame_, 0.0f);
        for (size_t i = 0; i < audio.size(); i++) {
            audio[i] = 0.1f * std::sin(2.0f * 3.14159f * 440.0f * i / sample_rate_);
        }
        
        return audio;
    }
    
    int num_codebooks() const override {
        return num_codebooks_;
    }
    
    int vocab_size() const override {
        return vocab_size_;
    }
    
    int sample_rate() const override {
        return sample_rate_;
    }
    
    int hop_length() const override {
        return hop_length_;
    }
    
    bool is_eos_token(int token, int codebook) const override {
        return token == 0;
    }
    
    // Testing accessors
    const std::vector<float>& get_last_audio() const { return last_audio; }
    const std::vector<std::vector<int>>& get_last_tokens() const { return last_tokens; }
    int get_encode_call_count() const { return encode_call_count; }
    int get_decode_call_count() const { return decode_call_count; }
    
    // For test control
    void set_fixed_audio(const std::vector<float>& audio) {
        fixed_audio = audio;
    }
    
    void reset_state() {
        last_audio.clear();
        last_tokens.clear();
        encode_call_count = 0;
        decode_call_count = 0;
        fixed_audio.clear();
    }
    
private:
    std::vector<float> last_audio;
    std::vector<std::vector<int>> last_tokens;
    int encode_call_count = 0;
    int decode_call_count = 0;
    std::vector<float> fixed_audio;
    
    const int num_codebooks_ = 8;
    const int vocab_size_ = 2051;
    const int sample_rate_ = 24000;
    const int hop_length_ = 320;
    const int samples_per_frame_ = 320;
};

// Enhanced mock watermarker for integration tests
class IntegrationMockWatermarker : public Watermarker {
public:
    IntegrationMockWatermarker() = default;
    
    std::vector<float> apply_watermark(const std::vector<float>& audio) override {
        // Store for testing
        last_audio = audio;
        watermark_called = true;
        
        // Return slightly modified audio
        std::vector<float> result = audio;
        for (size_t i = 0; i < result.size(); i++) {
            result[i] *= (1.0f + 0.01f * watermark_strength); // Small modification
        }
        return result;
    }
    
    bool detect_watermark(const std::vector<float>& audio) override {
        // Store for testing
        last_detect_audio = audio;
        detect_called = true;
        
        return force_detection_result;
    }
    
    WatermarkResult detect(const std::vector<float>& audio, float sample_rate) override {
        // Store for testing
        last_detect_audio = audio;
        detect_result_called = true;
        
        WatermarkResult result;
        result.detected = force_detection_result;
        result.payload = "integration-test-payload";
        result.confidence = force_detection_result ? 0.95f : 0.05f;
        
        return result;
    }
    
    float get_strength() const override {
        return watermark_strength;
    }
    
    void set_strength(float strength) override {
        watermark_strength = strength;
    }
    
    std::string get_key() const override {
        return "integration-test-key";
    }
    
    // Testing accessors
    const std::vector<float>& get_last_audio() const { return last_audio; }
    const std::vector<float>& get_last_detect_audio() const { return last_detect_audio; }
    bool was_watermark_called() const { return watermark_called; }
    bool was_detect_called() const { return detect_called; }
    bool was_detect_result_called() const { return detect_result_called; }
    
    // For test control
    void set_detection_result(bool should_detect) {
        force_detection_result = should_detect;
    }
    
    void reset_state() {
        last_audio.clear();
        last_detect_audio.clear();
        watermark_called = false;
        detect_called = false;
        detect_result_called = false;
        force_detection_result = true;
    }
    
private:
    std::vector<float> last_audio;
    std::vector<float> last_detect_audio;
    float watermark_strength = 0.5f;
    bool watermark_called = false;
    bool detect_called = false;
    bool detect_result_called = false;
    bool force_detection_result = true;
};

// Test fixture for generation workflow tests
class GenerationWorkflowTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create model config
        config.vocab_size = 32000;
        config.audio_vocab_size = 2051;
        config.d_model = 4096;
        config.n_heads = 32;
        config.n_kv_heads = 4;
        config.n_layers = 32;
        config.n_audio_layers = 12;
        config.max_seq_len = 2048;
        config.num_codebooks = 8;
        config.name = "integration-test-model";
        
        // Create components
        model = std::make_shared<IntegrationMockModel>(config);
        text_tokenizer = std::make_shared<IntegrationMockTextTokenizer>();
        audio_codec = std::make_shared<IntegrationMockAudioCodec>();
        watermarker = std::make_shared<IntegrationMockWatermarker>();
        
        // Create generator
        generator = std::make_shared<Generator>(model, text_tokenizer, audio_codec, watermarker);
    }
    
    void TearDown() override {
        // Reset state for next test
        model->reset_state();
        text_tokenizer->reset_state();
        audio_codec->reset_state();
        watermarker->reset_state();
    }
    
    ModelConfig config;
    std::shared_ptr<IntegrationMockModel> model;
    std::shared_ptr<IntegrationMockTextTokenizer> text_tokenizer;
    std::shared_ptr<IntegrationMockAudioCodec> audio_codec;
    std::shared_ptr<IntegrationMockWatermarker> watermarker;
    std::shared_ptr<Generator> generator;
};

// Test basic constructor
TEST_F(GenerationWorkflowTest, ConstructorTest) {
    // Test constructor with all components
    EXPECT_NO_THROW({
        auto gen = std::make_shared<Generator>(model, text_tokenizer, audio_codec, watermarker);
        EXPECT_NE(gen, nullptr);
    });
    
    // Test constructor without watermarker
    EXPECT_NO_THROW({
        auto gen = std::make_shared<Generator>(model, text_tokenizer, audio_codec);
        EXPECT_NE(gen, nullptr);
    });
    
    // Test constructor with null components (should throw)
    EXPECT_ANY_THROW({
        auto gen = std::make_shared<Generator>(nullptr, text_tokenizer, audio_codec);
    });
    
    EXPECT_ANY_THROW({
        auto gen = std::make_shared<Generator>(model, nullptr, audio_codec);
    });
    
    EXPECT_ANY_THROW({
        auto gen = std::make_shared<Generator>(model, text_tokenizer, nullptr);
    });
}

// Test basic speech generation
TEST_F(GenerationWorkflowTest, BasicSpeechGeneration) {
    // Generate speech
    auto audio = generator->generate_speech("Hello, world!", 0);
    
    // Verify output
    EXPECT_FALSE(audio.empty());
    EXPECT_GT(audio.size(), 0);
    
    // Verify component interactions
    EXPECT_GT(text_tokenizer->get_encode_call_count(), 0);
    EXPECT_TRUE(model->was_reset_called());
    EXPECT_GT(model->get_call_count(), 0);
    EXPECT_GT(audio_codec->get_decode_call_count(), 0);
    EXPECT_TRUE(watermarker->was_watermark_called());
    
    // Verify generation parameters
    EXPECT_NEAR(model->get_last_temperature(), 0.9f, 0.001f); // Default temperature
    EXPECT_EQ(model->get_last_top_k(), 50); // Default top_k
}

// Test different temperature settings
TEST_F(GenerationWorkflowTest, TemperatureSettings) {
    // Test with default temperature (0.9)
    auto audio1 = generator->generate_speech("Test with default temperature", 0);
    float temp1 = model->get_last_temperature();
    int frames1 = model->get_call_count();
    
    // Reset
    model->reset_state();
    
    // Test with low temperature (0.1)
    GenerationOptions options_low;
    options_low.temperature = 0.1f;
    
    auto audio2 = generator->generate_speech("Test with low temperature", 0, {}, options_low);
    float temp2 = model->get_last_temperature();
    int frames2 = model->get_call_count();
    
    // Verify temperatures were applied
    EXPECT_NEAR(temp1, 0.9f, 0.001f);
    EXPECT_NEAR(temp2, 0.1f, 0.001f);
    
    // Both should have generated output
    EXPECT_FALSE(audio1.empty());
    EXPECT_FALSE(audio2.empty());
}

// Test different top_k settings
TEST_F(GenerationWorkflowTest, TopKSettings) {
    // Test with default top_k (50)
    auto audio1 = generator->generate_speech("Test with default top_k", 0);
    int topk1 = model->get_last_top_k();
    
    // Reset
    model->reset_state();
    
    // Test with custom top_k (10)
    GenerationOptions options_custom;
    options_custom.top_k = 10;
    
    auto audio2 = generator->generate_speech("Test with custom top_k", 0, {}, options_custom);
    int topk2 = model->get_last_top_k();
    
    // Verify top_k values were applied
    EXPECT_EQ(topk1, 50);
    EXPECT_EQ(topk2, 10);
    
    // Both should have generated output
    EXPECT_FALSE(audio1.empty());
    EXPECT_FALSE(audio2.empty());
}

// Test speaker identity
TEST_F(GenerationWorkflowTest, SpeakerIdentity) {
    // Generate with speaker 1
    auto audio1 = generator->generate_speech("Test speaker 1", 1);
    
    // Check tokens for speaker 1 token
    int speaker1_token = text_tokenizer->get_speaker_token_id(1);
    bool found_speaker1 = false;
    
    for (int token : model->get_last_tokens()) {
        if (token == speaker1_token) {
            found_speaker1 = true;
            break;
        }
    }
    
    EXPECT_TRUE(found_speaker1);
    
    // Reset
    model->reset_state();
    
    // Generate with speaker 2
    auto audio2 = generator->generate_speech("Test speaker 2", 2);
    
    // Check tokens for speaker 2 token
    int speaker2_token = text_tokenizer->get_speaker_token_id(2);
    bool found_speaker2 = false;
    
    for (int token : model->get_last_tokens()) {
        if (token == speaker2_token) {
            found_speaker2 = true;
            break;
        }
    }
    
    EXPECT_TRUE(found_speaker2);
    
    // Both should have generated output
    EXPECT_FALSE(audio1.empty());
    EXPECT_FALSE(audio2.empty());
}

// Test conversation context
TEST_F(GenerationWorkflowTest, ConversationContext) {
    // Create conversation context
    std::vector<Segment> context = {
        Segment("First message", 1),
        Segment("Second message", 2),
        Segment("Third message", 1)
    };
    
    // Generate with context
    auto audio = generator->generate_speech("Response", 2, context);
    
    // Verify output
    EXPECT_FALSE(audio.empty());
    
    // Verify all speakers are included in tokens
    int speaker1_token = text_tokenizer->get_speaker_token_id(1);
    int speaker2_token = text_tokenizer->get_speaker_token_id(2);
    
    bool found_speaker1 = false;
    bool found_speaker2 = false;
    
    for (int token : model->get_last_tokens()) {
        if (token == speaker1_token) {
            found_speaker1 = true;
        } else if (token == speaker2_token) {
            found_speaker2 = true;
        }
    }
    
    EXPECT_TRUE(found_speaker1);
    EXPECT_TRUE(found_speaker2);
}

// Test watermarking
TEST_F(GenerationWorkflowTest, Watermarking) {
    // Test with watermarking enabled (default)
    auto audio_with_watermark = generator->generate_speech("Test with watermark", 0);
    EXPECT_TRUE(watermarker->was_watermark_called());
    
    // Reset
    model->reset_state();
    watermarker->reset_state();
    
    // Test with watermarking disabled
    GenerationOptions options;
    options.enable_watermark = false;
    
    auto audio_without_watermark = generator->generate_speech("Test without watermark", 0, {}, options);
    EXPECT_FALSE(watermarker->was_watermark_called());
    
    // Both should generate output
    EXPECT_FALSE(audio_with_watermark.empty());
    EXPECT_FALSE(audio_without_watermark.empty());
}

// Test progress callback
TEST_F(GenerationWorkflowTest, ProgressCallback) {
    // Track progress
    int progress_calls = 0;
    int last_current = 0;
    int last_total = 0;
    
    // Progress callback
    auto progress_callback = [&](int current, int total) {
        progress_calls++;
        last_current = current;
        last_total = total;
    };
    
    // Generate with callback
    auto audio = generator->generate_speech("Test progress callback", 0, {}, GenerationOptions(), progress_callback);
    
    // Verify callback was called
    EXPECT_GT(progress_calls, 0);
    EXPECT_GT(last_current, 0);
    EXPECT_GT(last_total, 0);
    EXPECT_LE(last_current, last_total);
    
    // Output should be generated
    EXPECT_FALSE(audio.empty());
}

// Test EOS token handling
TEST_F(GenerationWorkflowTest, EOSTokenHandling) {
    // Configure model to emit EOS after 5 frames
    model->set_eos_generation(true, 5);
    
    // Generate audio
    auto audio = generator->generate_speech("Test EOS handling", 0);
    
    // Should have stopped after EOS
    EXPECT_GT(model->get_call_count(), 0);
    EXPECT_LE(model->get_call_count(), 6); // At most 6 frames (5th has EOS)
    
    // Output should not be empty
    EXPECT_FALSE(audio.empty());
}

// Test max audio length
TEST_F(GenerationWorkflowTest, MaxAudioLength) {
    // Configure model to not generate EOS
    model->set_eos_generation(false, 0);
    
    // Test with default max audio length
    auto audio_default = generator->generate_speech("Test default length", 0);
    int frames_default = model->get_call_count();
    
    // Reset
    model->reset_state();
    
    // Test with short max audio length
    GenerationOptions options_short;
    options_short.max_audio_length_ms = 500; // Very short
    
    auto audio_short = generator->generate_speech("Test short length", 0, {}, options_short);
    int frames_short = model->get_call_count();
    
    // Reset
    model->reset_state();
    
    // Test with long max audio length
    GenerationOptions options_long;
    options_long.max_audio_length_ms = 10000; // Long
    
    auto audio_long = generator->generate_speech("Test long length", 0, {}, options_long);
    int frames_long = model->get_call_count();
    
    // Verify frame counts
    EXPECT_GT(frames_long, frames_short);
    
    // All should have generated output
    EXPECT_FALSE(audio_default.empty());
    EXPECT_FALSE(audio_short.empty());
    EXPECT_FALSE(audio_long.empty());
}

// Test memory optimization
TEST_F(GenerationWorkflowTest, MemoryOptimization) {
    // Memory optimization not enabled by default
    EXPECT_FALSE(model->was_memory_optimized());
    
    // Enable memory optimization
    generator->set_memory_optimization(true, 1024);
    
    // Should apply immediately
    EXPECT_TRUE(model->was_memory_optimized());
    EXPECT_EQ(model->get_memory_limit(), 1024);
    
    // Reset
    model->reset_state();
    
    // Generate with optimization enabled
    auto audio = generator->generate_speech("Test with memory optimization", 0);
    
    // Output should be generated
    EXPECT_FALSE(audio.empty());
}

// Test with fixed seed for reproducibility
TEST_F(GenerationWorkflowTest, SeedReproducibility) {
    // Configure model with fixed seed
    model->set_seed(42);
    
    // Generate with fixed seed
    GenerationOptions options1;
    options1.seed = 42;
    
    auto audio1 = generator->generate_speech("Test with seed 42", 0, {}, options1);
    int frames1 = model->get_call_count();
    
    // Reset
    model->reset_state();
    model->set_seed(42);
    
    // Generate again with same seed
    GenerationOptions options2;
    options2.seed = 42;
    
    auto audio2 = generator->generate_speech("Test with seed 42", 0, {}, options2);
    int frames2 = model->get_call_count();
    
    // Should generate same number of frames
    EXPECT_EQ(frames1, frames2);
    
    // Both should have generated output
    EXPECT_FALSE(audio1.empty());
    EXPECT_FALSE(audio2.empty());
    
    // Reset
    model->reset_state();
    model->set_seed(43);
    
    // Generate with different seed
    GenerationOptions options3;
    options3.seed = 43;
    
    auto audio3 = generator->generate_speech("Test with seed 43", 0, {}, options3);
    
    // All should have output of similar length (but not necessarily identical)
    EXPECT_FALSE(audio3.empty());
}

// Comprehensive test of all features
TEST_F(GenerationWorkflowTest, ComprehensiveTest) {
    // Create context
    std::vector<Segment> context = {
        Segment("First message in conversation", 1),
        Segment("Second message in conversation", 2),
        Segment("Third message in conversation", 1)
    };
    
    // Custom options
    GenerationOptions options;
    options.temperature = 0.7f;
    options.top_k = 40;
    options.max_audio_length_ms = 2000;
    options.seed = 12345;
    options.enable_watermark = true;
    options.debug = true;
    
    // Progress tracking
    int progress_calls = 0;
    auto progress_callback = [&](int current, int total) {
        progress_calls++;
    };
    
    // Memory optimization
    generator->set_memory_optimization(true, 1024, 512, 0.7f);
    
    // Generate with everything
    auto audio = generator->generate_speech(
        "Comprehensive test response",
        2,
        context,
        options,
        progress_callback
    );
    
    // Verify full workflow
    EXPECT_FALSE(audio.empty());
    EXPECT_GT(progress_calls, 0);
    EXPECT_GT(text_tokenizer->get_encode_call_count(), 0);
    EXPECT_TRUE(model->was_reset_called());
    EXPECT_GT(model->get_call_count(), 0);
    EXPECT_GT(audio_codec->get_decode_call_count(), 0);
    EXPECT_TRUE(watermarker->was_watermark_called());
    EXPECT_NEAR(model->get_last_temperature(), 0.7f, 0.001f);
    EXPECT_EQ(model->get_last_top_k(), 40);
    
    // Verify context was included
    int speaker1_token = text_tokenizer->get_speaker_token_id(1);
    int speaker2_token = text_tokenizer->get_speaker_token_id(2);
    
    bool found_speaker1 = false;
    bool found_speaker2 = false;
    
    for (int token : model->get_last_tokens()) {
        if (token == speaker1_token) {
            found_speaker1 = true;
        } else if (token == speaker2_token) {
            found_speaker2 = true;
        }
    }
    
    EXPECT_TRUE(found_speaker1);
    EXPECT_TRUE(found_speaker2);
}

// Main function for running tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}