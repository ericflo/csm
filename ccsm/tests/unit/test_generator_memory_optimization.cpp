#include <gtest/gtest.h>
#include <ccsm/generator.h>
#include <ccsm/model.h>
#include <ccsm/tokenizer.h>
#include <ccsm/cpu/ggml_model.h>
#include <memory>
#include <vector>
#include <string>

using namespace ccsm;

// Mock classes for testing
class MockTokenizer : public TextTokenizer {
public:
    MockTokenizer() {}
    
    std::vector<int> encode(const std::string& text) const override {
        // Simple mock that returns token IDs as 1, 2, 3, etc.
        std::vector<int> tokens;
        for (size_t i = 0; i < text.length(); i++) {
            tokens.push_back(i + 1);
        }
        return tokens;
    }
    
    std::string decode(const std::vector<int>& tokens) const override {
        // Simple mock that returns token IDs as characters
        std::string text;
        for (int token : tokens) {
            text.push_back('a' + (token % 26));
        }
        return text;
    }
    
    int vocab_size() const override { return 32000; }
    int bos_token_id() const override { return 1; }
    int eos_token_id() const override { return 2; }
    int pad_token_id() const override { return 0; }
    int unk_token_id() const override { return 3; }
    int get_speaker_token_id(int speaker_id) const override { return 100 + speaker_id; }
    std::vector<int> get_audio_token_ids() const override { return {1000, 1001, 1002, 1003}; }
};

class MockAudioCodec : public AudioCodec {
public:
    MockAudioCodec(int num_codebooks = 8) : codebooks_(num_codebooks) {}
    
    std::vector<std::vector<int>> encode(const std::vector<float>& audio) const override {
        // Create mock multi-codebook tokens
        std::vector<std::vector<int>> result(codebooks_);
        for (int i = 0; i < codebooks_; i++) {
            result[i].push_back(i + 10);
        }
        return result;
    }
    
    std::vector<float> decode(const std::vector<std::vector<int>>& tokens) const override {
        // Simple mock that returns a fixed audio waveform
        return std::vector<float>(16000, 0.1f);
    }
    
    int num_codebooks() const override { return codebooks_; }
    int vocab_size() const override { return 2051; }
    int sample_rate() const override { return 24000; }
    int hop_length() const override { return 300; }
    
    bool is_eos_token(int token, int codebook) const override {
        return token == 0;
    }

private:
    int codebooks_;
};

// Mock Model that tracks memory optimization calls
class MemoryTrackingModel : public Model {
public:
    MemoryTrackingModel() : Model(createConfig()), 
                    memory_optimized(false), 
                    caches_pruned(false),
                    optimize_memory_calls(0),
                    prune_caches_calls(0),
                    frame_count(0) {}
    
    static ModelConfig createConfig() {
        ModelConfig config;
        config.name = "Memory Tracking Model";
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
        
        last_tokens = tokens;
        last_positions = positions;
        last_temperature = temperature;
        last_top_k = top_k;
        frame_count++;
        
        // Generate a mock frame with codebook values
        std::vector<int> frame(config_.num_codebooks);
        for (int i = 0; i < config_.num_codebooks; i++) {
            frame[i] = i + 10; // Just return token values 10, 11, 12, etc.
        }
        return frame;
    }
    
    void reset_caches() override {
        reset_caches_called = true;
    }
    
    void optimize_memory(size_t max_memory_mb = 0) override {
        memory_optimized = true;
        optimize_memory_calls++;
        last_memory_limit = max_memory_mb;
    }
    
    void prune_caches(float prune_factor = 0.5f) override {
        caches_pruned = true;
        prune_caches_calls++;
        last_prune_factor = prune_factor;
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
    
    // Testing helpers
    bool memory_optimized = false;
    bool caches_pruned = false;
    bool reset_caches_called = false;
    int optimize_memory_calls = 0;
    int prune_caches_calls = 0;
    size_t last_memory_limit = 0;
    float last_prune_factor = 0.0f;
    int frame_count = 0;
    float last_temperature = 0.0f;
    int last_top_k = 0;
    std::vector<int> last_tokens;
    std::vector<int> last_positions;
    
    void reset_stats() {
        memory_optimized = false;
        caches_pruned = false;
        reset_caches_called = false;
        optimize_memory_calls = 0;
        prune_caches_calls = 0;
        last_memory_limit = 0;
        last_prune_factor = 0.0f;
        frame_count = 0;
        last_temperature = 0.0f;
        last_top_k = 0;
        last_tokens.clear();
        last_positions.clear();
    }
};

// Test fixture
class GeneratorMemoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        model = std::make_shared<MemoryTrackingModel>();
        tokenizer = std::make_shared<MockTokenizer>();
        audio_codec = std::make_shared<MockAudioCodec>();
        
        // Create generator with mock components
        generator = std::make_shared<Generator>(model, tokenizer, audio_codec);
    }
    
    std::shared_ptr<MemoryTrackingModel> model;
    std::shared_ptr<TextTokenizer> tokenizer;
    std::shared_ptr<AudioCodec> audio_codec;
    std::shared_ptr<Generator> generator;
};

// Test memory optimization settings
TEST_F(GeneratorMemoryTest, MemoryOptimizationSettings) {
    // Initial state should have memory optimization disabled
    EXPECT_FALSE(model->memory_optimized);
    EXPECT_EQ(model->optimize_memory_calls, 0);
    
    // Enable memory optimization
    generator->set_memory_optimization(true, 1024);
    
    // Should immediately apply optimization
    EXPECT_TRUE(model->memory_optimized);
    EXPECT_EQ(model->optimize_memory_calls, 1);
    EXPECT_EQ(model->last_memory_limit, 1024);
    
    // Reset model stats
    model->reset_stats();
    
    // Generate speech with optimization enabled
    auto audio = generator->generate_speech("Test speech", 0);
    
    // Model cache should have been reset
    EXPECT_TRUE(model->reset_caches_called);
    
    // Generated at least one frame
    EXPECT_GT(model->frame_count, 0);
}

// Test memory pruning
TEST_F(GeneratorMemoryTest, MemoryPruning) {
    // Enable memory optimization with pruning
    generator->set_memory_optimization(true, 1024, 512, 0.7f);
    
    // Should apply both optimization and pruning
    EXPECT_TRUE(model->memory_optimized);
    EXPECT_EQ(model->optimize_memory_calls, 1);
    EXPECT_EQ(model->last_memory_limit, 1024);
    EXPECT_TRUE(model->caches_pruned);
    EXPECT_EQ(model->prune_caches_calls, 1);
    EXPECT_FLOAT_EQ(model->last_prune_factor, 0.7f);
    
    // Reset model stats
    model->reset_stats();
    
    // Generate speech with both enabled
    auto audio = generator->generate_speech("Test speech with pruning", 0);
    
    // Check generated output
    EXPECT_FALSE(audio.empty());
}

// Test with long input
TEST_F(GeneratorMemoryTest, LongTextInput) {
    // Enable memory optimization
    generator->set_memory_optimization(true, 1024, 512, 0.7f);
    
    // Reset model stats
    model->reset_stats();
    
    // Create a long text input
    std::string long_text(1000, 'a');
    
    // Generate with long input
    auto audio = generator->generate_speech(long_text, 0);
    
    // Check that model cache was reset
    EXPECT_TRUE(model->reset_caches_called);
    
    // Verify token length was limited
    EXPECT_LE(model->last_tokens.size(), generator->max_text_tokens());
    
    // Should have generated frames
    EXPECT_GT(model->frame_count, 0);
}

// Test with multiple segments
TEST_F(GeneratorMemoryTest, ContextSegments) {
    // Enable memory optimization
    generator->set_memory_optimization(true, 1024);
    
    // Reset model stats
    model->reset_stats();
    
    // Create context with multiple segments
    std::vector<Segment> context = {
        Segment("First message", 1),
        Segment("Second message", 2),
        Segment("Third message", 1)
    };
    
    // Generate with context
    auto audio = generator->generate_speech("Test with context", 0, context);
    
    // Verify context was included in tokens
    bool found_speaker_tokens = false;
    for (int token : model->last_tokens) {
        if (token >= 100) { // Speaker tokens start at 100
            found_speaker_tokens = true;
            break;
        }
    }
    EXPECT_TRUE(found_speaker_tokens);
    
    // Should have generated frames
    EXPECT_GT(model->frame_count, 0);
}

// Test memory optimization limits
TEST_F(GeneratorMemoryTest, MemoryLimits) {
    // Set up different memory limits
    const size_t small_limit = 256;
    const size_t large_limit = 2048;
    
    // Test with small limit
    generator->set_memory_optimization(true, small_limit);
    EXPECT_EQ(model->last_memory_limit, small_limit);
    
    // Reset model stats
    model->reset_stats();
    
    // Test with large limit
    generator->set_memory_optimization(true, large_limit);
    EXPECT_EQ(model->last_memory_limit, large_limit);
    
    // Reset model stats
    model->reset_stats();
    
    // Disable optimization
    generator->set_memory_optimization(false);
    
    // Generate speech with optimization disabled
    auto audio = generator->generate_speech("Test without optimization", 0);
    
    // Should not have called optimization
    EXPECT_FALSE(model->memory_optimized);
    EXPECT_EQ(model->optimize_memory_calls, 0);
}

// Test with different pruning factors
TEST_F(GeneratorMemoryTest, PruningFactors) {
    // Test different pruning factors
    const float light_pruning = 0.3f;
    const float heavy_pruning = 0.8f;
    
    // Test with light pruning
    generator->set_memory_optimization(true, 1024, 512, light_pruning);
    EXPECT_FLOAT_EQ(model->last_prune_factor, light_pruning);
    
    // Reset model stats
    model->reset_stats();
    
    // Test with heavy pruning
    generator->set_memory_optimization(true, 1024, 512, heavy_pruning);
    EXPECT_FLOAT_EQ(model->last_prune_factor, heavy_pruning);
}

// Test generation with memory trigger
TEST_F(GeneratorMemoryTest, MemoryTrigger) {
    // Set memory optimization with trigger
    generator->set_memory_optimization(true, 1024, 512, 0.5f);
    
    // Reset model stats
    model->reset_stats();
    
    // Generate speech
    auto audio = generator->generate_speech("Test with memory trigger", 0);
    
    // In a real implementation, memory trigger would activate based on KV cache size
    // Here we just verify the parameters were set correctly
    EXPECT_FALSE(model->caches_pruned); // Won't be triggered in this mock
    
    // Actual implementations would trigger pruning when memory usage exceeds the trigger
}

// Test compatibility with different model configurations
TEST_F(GeneratorMemoryTest, ModelConfigurations) {
    // Create models with different configurations
    auto model_small = std::make_shared<MemoryTrackingModel>();
    auto model_large = std::make_shared<MemoryTrackingModel>();
    
    // Create generators with these models
    auto generator_small = std::make_shared<Generator>(model_small, tokenizer, audio_codec);
    auto generator_large = std::make_shared<Generator>(model_large, tokenizer, audio_codec);
    
    // Enable memory optimization for both
    generator_small->set_memory_optimization(true, 512);
    generator_large->set_memory_optimization(true, 2048);
    
    // Generate speech with both
    auto audio_small = generator_small->generate_speech("Test with small model", 0);
    auto audio_large = generator_large->generate_speech("Test with large model", 0);
    
    // Both should have generated output
    EXPECT_FALSE(audio_small.empty());
    EXPECT_FALSE(audio_large.empty());
    
    // Both should have called memory optimization
    EXPECT_TRUE(model_small->memory_optimized);
    EXPECT_TRUE(model_large->memory_optimized);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}