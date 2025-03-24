#include <ccsm/generator.h>
#include <ccsm/model.h>
#include <ccsm/tokenizer.h>
#include <ccsm/watermarking.h>
#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <random>
#include <chrono>

namespace ccsm {
namespace {

// Mock classes for testing Generator functionality
class MockTextTokenizer : public TextTokenizer {
public:
    MockTextTokenizer(int vocab_size = 32000) : vocab_size_(vocab_size) {}
    
    std::vector<int> encode(const std::string& text) const override {
        // Return a deterministic sequence based on the input text length
        std::vector<int> tokens;
        for (size_t i = 0; i < text.length(); i++) {
            tokens.push_back(static_cast<int>(text[i]) % vocab_size_);
        }
        return tokens;
    }
    
    std::string decode(const std::vector<int>& tokens) const override {
        // Simple decoding strategy
        std::string result;
        for (auto token : tokens) {
            result += static_cast<char>((token % 26) + 'a');
        }
        return result;
    }
    
    int vocab_size() const override { return vocab_size_; }
    int bos_token_id() const override { return 1; }
    int eos_token_id() const override { return 2; }
    int pad_token_id() const override { return 0; }
    int unk_token_id() const override { return 3; }
    
    int get_speaker_token_id(int speaker_id) const override {
        return 10000 + speaker_id;
    }
    
    std::vector<int> get_audio_token_ids() const override {
        return {5000, 5001, 5002, 5003};
    }

private:
    int vocab_size_;
};

class MockAudioCodec : public AudioCodec {
public:
    MockAudioCodec(int num_codebooks = 8, int vocab_size = 2051, int sample_rate = 24000) 
        : num_codebooks_(num_codebooks), vocab_size_(vocab_size), sample_rate_(sample_rate) {}
    
    std::vector<std::vector<int>> encode(const std::vector<float>& audio) const override {
        // Create mock encoded audio with multiple codebooks
        std::vector<std::vector<int>> result(num_codebooks_);
        size_t frames = audio.size() / (sample_rate_ / 10); // Approx 100ms frames
        
        for (int cb = 0; cb < num_codebooks_; cb++) {
            for (size_t i = 0; i < frames; i++) {
                result[cb].push_back((i + cb) % (vocab_size_ - 1) + 1); // Avoid 0 (EOS)
            }
        }
        
        return result;
    }
    
    std::vector<float> decode(const std::vector<std::vector<int>>& tokens) const override {
        // Generate simple audio based on tokens
        size_t max_frames = 0;
        for (const auto& codebook : tokens) {
            max_frames = std::max(max_frames, codebook.size());
        }
        
        // Create audio: 10ms per frame at sample_rate_
        size_t samples_per_frame = sample_rate_ / 100;
        std::vector<float> result(max_frames * samples_per_frame, 0.0f);
        
        // Fill with simple patterns
        for (size_t i = 0; i < result.size(); i++) {
            result[i] = 0.1f * sin(2.0f * 3.14159f * 440.0f * i / sample_rate_);
        }
        
        return result;
    }
    
    int num_codebooks() const override { return num_codebooks_; }
    int vocab_size() const override { return vocab_size_; }
    int sample_rate() const override { return sample_rate_; }
    int hop_length() const override { return sample_rate_ / 100; }
    
    bool is_eos_token(int token, int codebook) const override {
        return token == 0;
    }

private:
    int num_codebooks_;
    int vocab_size_;
    int sample_rate_;
};

class MockWatermarker : public Watermarker {
public:
    MockWatermarker(float strength = 0.5f) : strength_(strength) {}
    
    std::vector<float> apply_watermark(const std::vector<float>& audio) override {
        // Copy the audio and add a slight offset to simulate watermarking
        std::vector<float> result = audio;
        for (size_t i = 0; i < result.size(); i++) {
            result[i] += strength_ * 0.01f * sin(i * 0.1f);
        }
        was_applied_ = true;
        return result;
    }
    
    bool detect_watermark(const std::vector<float>& audio) override {
        return was_applied_;
    }
    
    WatermarkResult detect(const std::vector<float>& audio, float sample_rate) override {
        WatermarkResult result;
        result.detected = was_applied_;
        result.payload = "test-payload";
        result.confidence = was_applied_ ? 0.95f : 0.05f;
        return result;
    }
    
    float get_strength() const override { return strength_; }
    void set_strength(float strength) override { strength_ = strength; }
    std::string get_key() const override { return "test-key"; }
    
    // For testing whether watermarking was applied
    bool was_applied() const { return was_applied_; }
    void reset_applied() { was_applied_ = false; }

private:
    float strength_;
    bool was_applied_ = false;
};

class MockModel : public Model {
public:
    MockModel() : Model(ModelConfig()) {
        config_.name = "Mock Model";
        config_.vocab_size = 32000;
        config_.audio_vocab_size = 2048;
        config_.d_model = 1024;
        config_.n_heads = 16;
        config_.n_kv_heads = 16;
        config_.n_layers = 32;
        config_.max_seq_len = 2048;
    }
    
    bool load_weights(const std::string& path) override { return true; }
    bool load_weights(std::shared_ptr<ModelLoader> loader) override { return true; }
    bool load_weights(const WeightMap& weights) override { return true; }
    
    void reset_caches() override {
        // Verify the reset was called
        reset_called_++;
    }
    
    void optimize_memory(size_t max_memory_mb = 0) override {
        // Keep track of optimization calls
        optimize_called_ = true;
        last_max_memory_mb_ = max_memory_mb;
    }
    
    void prune_caches(float prune_factor = 0.5f) override {
        // Keep track of pruning calls
        prune_called_ = true;
        last_prune_factor_ = prune_factor;
    }
    
    std::vector<int> generate_frame(
        const std::vector<int>& tokens,
        const std::vector<int>& positions,
        float temperature = 0.9f,
        int top_k = 50) override {
        // Keep track of parameters used
        last_temperature_ = temperature;
        last_top_k_ = top_k;
        frame_count_++;
        
        // Use random number generator with seed controlled by temperature 
        // This allows tests to verify different temperatures produce different results
        std::mt19937 rng(static_cast<unsigned int>(temperature * 100));
        std::uniform_int_distribution<int> dist(1, config_.audio_vocab_size - 1);
        
        // Generate tokens based on config
        std::vector<int> result(config_.num_codebooks);
        for (int i = 0; i < config_.num_codebooks; i++) {
            result[i] = dist(rng);
        }
        
        // Return mock result
        return result;
    }
    
    std::vector<float> get_backbone_logits(
        const std::vector<int>& tokens,
        const std::vector<int>& positions) override {
        // Return dummy logits
        return std::vector<float>(config_.vocab_size, 0.0f);
    }
    
    std::vector<float> get_decoder_logits(
        const std::vector<int>& tokens,
        const std::vector<int>& positions,
        int codebook) override {
        // Return dummy logits
        return std::vector<float>(config_.audio_vocab_size, 0.0f);
    }
    
    // Mock helper methods for testing
    int frame_count() const { return frame_count_; }
    float last_temperature() const { return last_temperature_; }
    int last_top_k() const { return last_top_k_; }
    int reset_called() const { return reset_called_; }
    void reset_metrics() {
        frame_count_ = 0;
        last_temperature_ = 0.0f;
        last_top_k_ = 0;
        reset_called_ = 0;
        optimize_called_ = false;
        last_max_memory_mb_ = 0;
        prune_called_ = false;
        last_prune_factor_ = 0.0f;
    }
    
    bool optimize_called() const { return optimize_called_; }
    size_t last_max_memory_mb() const { return last_max_memory_mb_; }
    bool prune_called() const { return prune_called_; }
    float last_prune_factor() const { return last_prune_factor_; }

private:
    int frame_count_ = 0;
    float last_temperature_ = 0.0f;
    int last_top_k_ = 0;
    int reset_called_ = 0;
    bool optimize_called_ = false;
    size_t last_max_memory_mb_ = 0;
    bool prune_called_ = false;
    float last_prune_factor_ = 0.0f;
};

// Test fixture
class GeneratorAdvancedTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create mock components
        model_ = std::make_shared<MockModel>();
        tokenizer_ = std::make_shared<MockTextTokenizer>();
        codec_ = std::make_shared<MockAudioCodec>();
        watermarker_ = std::make_shared<MockWatermarker>();
        
        // Create generator with mock components
        generator_ = std::make_shared<Generator>(model_, tokenizer_, codec_, watermarker_);
    }
    
    // Get mock components with proper types for testing
    std::shared_ptr<MockModel> model() {
        return std::static_pointer_cast<MockModel>(model_);
    }
    
    std::shared_ptr<MockWatermarker> watermarker() {
        return std::static_pointer_cast<MockWatermarker>(watermarker_);
    }
    
    // Mock components
    std::shared_ptr<Model> model_;
    std::shared_ptr<TextTokenizer> tokenizer_;
    std::shared_ptr<AudioCodec> codec_;
    std::shared_ptr<Watermarker> watermarker_;
    
    // Generator
    std::shared_ptr<Generator> generator_;
};

// Test different generation parameters
TEST_F(GeneratorAdvancedTest, GenerationParameters) {
    // Setup
    model()->reset_metrics();
    watermarker()->reset_applied();
    
    // Test with default parameters
    GenerationOptions default_options;
    auto result1 = generator_->generate_speech("Test text", 0, {}, default_options);
    
    // Verify default parameters were used
    EXPECT_NEAR(model()->last_temperature(), 0.9f, 1e-6);
    EXPECT_EQ(model()->last_top_k(), 50);
    
    // Test with custom parameters
    GenerationOptions custom_options;
    custom_options.temperature = 0.5f;
    custom_options.top_k = 25;
    auto result2 = generator_->generate_speech("Test text", 0, {}, custom_options);
    
    // Verify custom parameters were used
    EXPECT_NEAR(model()->last_temperature(), 0.5f, 1e-6);
    EXPECT_EQ(model()->last_top_k(), 25);
    
    // Verify different parameters produce different results
    EXPECT_NE(result1.size(), 0);
    EXPECT_NE(result2.size(), 0);
    
    // Different temperatures should lead to different tokens
    bool results_differ = false;
    if (result1.size() == result2.size()) {
        for (size_t i = 0; i < result1.size(); i++) {
            if (std::abs(result1[i] - result2[i]) > 1e-6) {
                results_differ = true;
                break;
            }
        }
    } else {
        results_differ = true;
    }
    EXPECT_TRUE(results_differ) << "Different temperatures should produce different results";
}

// Test seed reproducibility
TEST_F(GeneratorAdvancedTest, SeedReproducibility) {
    // Setup
    model()->reset_metrics();
    
    // Generate with specific seed
    GenerationOptions options1;
    options1.seed = 42;
    auto result1 = generator_->generate_speech("Test reproducibility", 0, {}, options1);
    
    // Reset
    model()->reset_metrics();
    
    // Generate again with the same seed
    GenerationOptions options2;
    options2.seed = 42;
    auto result2 = generator_->generate_speech("Test reproducibility", 0, {}, options2);
    
    // Different seed should produce different results
    GenerationOptions options3;
    options3.seed = 43;
    auto result3 = generator_->generate_speech("Test reproducibility", 0, {}, options3);
    
    // Random seed should give different results
    GenerationOptions options_random;
    options_random.seed = -1; // random seed
    auto result_random1 = generator_->generate_speech("Test reproducibility", 0, {}, options_random);
    
    model()->reset_metrics();
    auto result_random2 = generator_->generate_speech("Test reproducibility", 0, {}, options_random);
    
    // Verify results: same seed should have same length
    EXPECT_EQ(result1.size(), result2.size());
    
    // Check seeds produce consistent output vs different seeds
    bool same_seed_differs = false;
    bool diff_seed_differs = false;
    bool random_seed_differs = false;
    
    // Check if same seed results are identical (should be)
    if (result1.size() == result2.size()) {
        for (size_t i = 0; i < result1.size(); i++) {
            if (std::abs(result1[i] - result2[i]) > 1e-6) {
                same_seed_differs = true;
                break;
            }
        }
    }
    
    // Check if different seed results differ (should differ)
    if (result1.size() == result3.size()) {
        for (size_t i = 0; i < result1.size(); i++) {
            if (std::abs(result1[i] - result3[i]) > 1e-6) {
                diff_seed_differs = true;
                break;
            }
        }
    } else {
        diff_seed_differs = true;
    }
    
    // Check if random seed results differ (should differ)
    if (result_random1.size() == result_random2.size()) {
        for (size_t i = 0; i < result_random1.size(); i++) {
            if (std::abs(result_random1[i] - result_random2[i]) > 1e-6) {
                random_seed_differs = true;
                break;
            }
        }
    } else {
        random_seed_differs = true;
    }
    
    // Verify seed behavior
    EXPECT_FALSE(same_seed_differs) << "Same seed should produce identical results";
    EXPECT_TRUE(diff_seed_differs) << "Different seeds should produce different results";
    EXPECT_TRUE(random_seed_differs) << "Random seeds should produce different results";
}

// Test watermarking
TEST_F(GeneratorAdvancedTest, Watermarking) {
    // Reset
    watermarker()->reset_applied();
    
    // Test with watermarking enabled (default)
    GenerationOptions options_with_watermark;
    options_with_watermark.enable_watermark = true;
    
    auto result_with_watermark = generator_->generate_speech("Test watermarking", 0, {}, options_with_watermark);
    EXPECT_TRUE(watermarker()->was_applied()) << "Watermark should be applied when enabled";
    
    // Reset
    watermarker()->reset_applied();
    
    // Test with watermarking disabled
    GenerationOptions options_without_watermark;
    options_without_watermark.enable_watermark = false;
    
    auto result_without_watermark = generator_->generate_speech("Test watermarking", 0, {}, options_without_watermark);
    EXPECT_FALSE(watermarker()->was_applied()) << "Watermark should not be applied when disabled";
}

// Test context processing
TEST_F(GeneratorAdvancedTest, ContextProcessing) {
    // Setup
    model()->reset_metrics();
    
    // Create context segments
    std::vector<Segment> context = {
        Segment("First message", 1),
        Segment("Second message", 2),
        Segment("Third message", 1)
    };
    
    // Generate with context
    auto result = generator_->generate_speech("Test with context", 3, context);
    
    // Verify model was reset
    EXPECT_GE(model()->reset_called(), 1) << "Model caches should be reset before generation";
    
    // Generate without context
    model()->reset_metrics();
    auto result_no_context = generator_->generate_speech("Test without context", 3);
    
    // Different context should produce different length outputs
    EXPECT_NE(result.size(), result_no_context.size()) << "Context should affect generation length";
}

// Test progress callback
TEST_F(GeneratorAdvancedTest, ProgressCallback) {
    // Setup
    int progress_calls = 0;
    int last_current = 0;
    int last_total = 0;
    
    // Create progress callback
    auto progress_callback = [&](int current, int total) {
        progress_calls++;
        last_current = current;
        last_total = total;
    };
    
    // Generate with callback
    GenerationOptions options;
    options.max_audio_length_ms = 1000; // Keep it short for test
    
    auto result = generator_->generate_speech("Test progress", 0, {}, options, progress_callback);
    
    // Verify callback was called at least once
    EXPECT_GT(progress_calls, 0) << "Progress callback should be called";
    
    // Verify final values
    EXPECT_GT(last_current, 0) << "Current progress should be positive";
    EXPECT_EQ(last_current, last_total) << "Final progress should equal total";
}

// Test max audio length constraint
TEST_F(GeneratorAdvancedTest, MaxAudioLength) {
    // Setup
    model()->reset_metrics();
    
    // Generate with short max length
    GenerationOptions short_options;
    short_options.max_audio_length_ms = 500; // Very short
    
    auto short_result = generator_->generate_speech("Test max length", 0, {}, short_options);
    int short_frames = model()->frame_count();
    
    // Reset
    model()->reset_metrics();
    
    // Generate with longer max length
    GenerationOptions long_options;
    long_options.max_audio_length_ms = 2000; // Longer
    
    auto long_result = generator_->generate_speech("Test max length", 0, {}, long_options);
    int long_frames = model()->frame_count();
    
    // Verify frame counts
    EXPECT_GT(long_frames, short_frames) << "Longer max length should generate more frames";
    
    // Verify audio lengths correspond to the requested constraint
    float samples_per_ms = generator_->audio_codec()->sample_rate() / 1000.0f;
    
    // Allow some buffer in the comparison since frame boundaries won't exactly match ms boundaries
    EXPECT_LE(short_result.size(), 1.2f * short_options.max_audio_length_ms * samples_per_ms)
        << "Short audio should respect max length constraint";
}

// Test generation with pre-tokenized input
TEST_F(GeneratorAdvancedTest, PreTokenizedInput) {
    // Setup
    model()->reset_metrics();
    
    // Create tokenized input
    std::vector<int> tokens = {100, 200, 300, 400, 500};
    
    // Generate with tokenized input
    auto result = generator_->generate_speech_from_tokens(tokens, 0);
    
    // Verify generation happened
    EXPECT_GT(model()->frame_count(), 0) << "Generation should produce frames";
    EXPECT_GT(result.size(), 0) << "Generated audio should not be empty";
}

// Comprehensive test combining multiple features
TEST_F(GeneratorAdvancedTest, ComprehensiveGeneration) {
    // Setup
    model()->reset_metrics();
    watermarker()->reset_applied();
    
    // Create complex context
    std::vector<Segment> context = {
        Segment("First message with some context", 1),
        Segment("Second message with a response", 2)
    };
    
    // Create complex options
    GenerationOptions options;
    options.temperature = 0.7f;
    options.top_k = 30;
    options.max_audio_length_ms = 1500;
    options.seed = 123;
    options.enable_watermark = true;
    
    // Track progress
    int progress_calls = 0;
    auto progress_callback = [&](int current, int total) {
        progress_calls++;
    };
    
    // Perform generation
    auto result = generator_->generate_speech(
        "This is a test of comprehensive generation with multiple features",
        3,
        context,
        options,
        progress_callback
    );
    
    // Verify all aspects
    EXPECT_GT(model()->frame_count(), 0) << "Generation should produce frames";
    EXPECT_NEAR(model()->last_temperature(), options.temperature, 1e-6) << "Temperature should be applied";
    EXPECT_EQ(model()->last_top_k(), options.top_k) << "Top-k should be applied";
    EXPECT_TRUE(watermarker()->was_applied()) << "Watermark should be applied";
    EXPECT_GT(progress_calls, 0) << "Progress should be reported";
    EXPECT_GT(result.size(), 0) << "Generated audio should not be empty";
    
    // Verify audio length constraint
    float samples_per_ms = generator_->audio_codec()->sample_rate() / 1000.0f;
    EXPECT_LE(result.size(), 1.2f * options.max_audio_length_ms * samples_per_ms)
        << "Audio should respect max length constraint";
}

} // namespace
} // namespace ccsm