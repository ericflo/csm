#include <gtest/gtest.h>
#include <ccsm/generator.h>
#include <ccsm/model.h>
#include <ccsm/tokenizer.h>
#include <memory>
#include <vector>
#include <string>
#include <stdexcept>

namespace ccsm {
namespace testing {

// Mock classes for testing
class MockContextTokenizer : public TextTokenizer {
public:
    std::vector<int> encode(const std::string& text) const override {
        // Simple encoding - just use character values
        std::vector<int> tokens;
        for (char c : text) {
            tokens.push_back(static_cast<int>(c));
        }
        return tokens;
    }
    
    std::string decode(const std::vector<int>& tokens) const override {
        std::string text;
        for (int token : tokens) {
            if (token < 256) { // Only decode if in ASCII range
                text.push_back(static_cast<char>(token));
            }
        }
        return text;
    }
    
    int vocab_size() const override {
        return 10000;
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
        return 5000 + speaker_id; // Speaker tokens are 5000 + speaker_id
    }
    
    std::vector<int> get_audio_token_ids() const override {
        return {6000, 6001, 6002}; // Audio tokens start from 6000
    }
};

class MockContextModel : public Model {
public:
    explicit MockContextModel(const ModelConfig& config) : Model(config) {}
    
    bool load_weights(const std::string& path) override {
        // Mock implementation, always returns true
        loaded_ = true;
        return true;
    }
    
    void reset_caches() override {
        cache_resets++;
    }
    
    void prune_caches(float factor) override {
        cache_prunes++;
        prune_factor = factor;
    }
    
    std::vector<int> generate_frame(
        const std::vector<int>& tokens,
        const std::vector<int>& positions,
        float temperature,
        int top_k) override {
        
        // Record the tokens and positions for testing
        last_tokens = tokens;
        last_positions = positions;
        last_temperature = temperature;
        last_top_k = top_k;
        frames_generated++;
        
        // Return a mock frame (8 codebooks)
        return {101, 102, 103, 104, 105, 106, 107, 108};
    }
    
    // Variables for testing
    bool loaded_ = false;
    int cache_resets = 0;
    int cache_prunes = 0;
    float prune_factor = 0.0f;
    std::vector<int> last_tokens;
    std::vector<int> last_positions;
    float last_temperature = 0.0f;
    int last_top_k = 0;
    int frames_generated = 0;
};

class MockContextAudioCodec : public AudioCodec {
public:
    std::vector<std::vector<int>> encode(const std::vector<float>& audio) const override {
        // Mock 8 codebooks with 3 tokens each
        std::vector<std::vector<int>> tokens(8);
        for (int i = 0; i < 8; i++) {
            tokens[i] = {10, 20, 30};
        }
        return tokens;
    }
    
    std::vector<float> decode(const std::vector<std::vector<int>>& tokens) const override {
        // Return a simple sine wave
        int num_frames = tokens.empty() ? 0 : tokens[0].size();
        std::vector<float> audio(num_frames * 400); // 400 samples per frame
        for (size_t i = 0; i < audio.size(); i++) {
            audio[i] = 0.5f * std::sin(2.0f * 3.14159f * 440.0f * i / 24000.0f);
        }
        return audio;
    }
    
    int num_codebooks() const override {
        return 8;
    }
    
    int vocab_size() const override {
        return 1024;
    }
    
    int sample_rate() const override {
        return 24000;
    }
    
    int hop_length() const override {
        return 400;  // 400 samples per frame at 24kHz
    }
    
    bool is_eos_token(int token, int codebook) const override {
        return token == 0;
    }
};

// Test fixture for Context Management tests
class ContextManagementTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create model configuration
        ModelConfig config;
        config.name = "MockContextModel";
        config.d_model = 128;
        config.n_layers = 2;
        config.n_heads = 4;
        config.n_kv_heads = 2;
        config.vocab_size = 10000;
        config.audio_vocab_size = 1024;
        config.max_seq_len = 2048;
        config.num_codebooks = 8;
        
        // Create mock components
        model = std::make_shared<MockContextModel>(config);
        tokenizer = std::make_shared<MockContextTokenizer>();
        codec = std::make_shared<MockContextAudioCodec>();
        
        // Create generator
        generator = std::make_shared<Generator>(model, tokenizer, codec);
        
        // Enable memory optimization
        generator->set_memory_optimization(true, 0.5f);
    }
    
    void TearDown() override {
        generator.reset();
        model.reset();
        tokenizer.reset();
        codec.reset();
    }
    
    std::shared_ptr<MockContextModel> model;
    std::shared_ptr<MockContextTokenizer> tokenizer;
    std::shared_ptr<MockContextAudioCodec> codec;
    std::shared_ptr<Generator> generator;
};

// Test basic context handling
TEST_F(ContextManagementTest, TestBasicContextHandling) {
    // Create a simple context with one segment
    Segment segment;
    segment.text = "Hello world";
    segment.speaker_id = 1;
    
    std::vector<Segment> context = {segment};
    
    // Generate speech with context
    std::vector<int> tokens = tokenizer->encode("How are you?");
    std::vector<float> audio = generator->generate_speech_from_tokens(tokens, 2, context);
    
    // Check model interactions
    EXPECT_GT(model->frames_generated, 0);
    EXPECT_EQ(model->cache_resets, 1); // Cache should be reset once at the start
    
    // Verify tokens passed to model include both context and input tokens
    std::vector<int> expected_speaker1_token = {tokenizer->get_speaker_token_id(1)};
    std::vector<int> expected_context_tokens = tokenizer->encode("Hello world");
    std::vector<int> expected_speaker2_token = {tokenizer->get_speaker_token_id(2)};
    
    // First token should be the context speaker token
    EXPECT_EQ(model->last_tokens[0], expected_speaker1_token[0]);
    
    // Check that context tokens are present
    for (size_t i = 0; i < expected_context_tokens.size(); i++) {
        EXPECT_EQ(model->last_tokens[i + 1], expected_context_tokens[i]);
    }
    
    // Check that the current speaker token is present
    EXPECT_EQ(model->last_tokens[expected_context_tokens.size() + 1], expected_speaker2_token[0]);
    
    // Check that input tokens are present
    for (size_t i = 0; i < tokens.size(); i++) {
        EXPECT_EQ(model->last_tokens[i + expected_context_tokens.size() + 2], tokens[i]);
    }
    
    // Verify positions are sequential
    for (size_t i = 1; i < model->last_positions.size(); i++) {
        EXPECT_EQ(model->last_positions[i], model->last_positions[i - 1] + 1);
    }
}

// Test multiple segments in context
TEST_F(ContextManagementTest, TestMultipleContextSegments) {
    // Create context with multiple segments
    std::vector<Segment> context = {
        {.text = "First message", .speaker_id = 1},
        {.text = "Second reply", .speaker_id = 2},
        {.text = "Third message", .speaker_id = 1}
    };
    
    // Generate speech with multi-segment context
    std::vector<int> tokens = tokenizer->encode("Final reply");
    std::vector<float> audio = generator->generate_speech_from_tokens(tokens, 2, context);
    
    // Check model interactions
    EXPECT_GT(model->frames_generated, 0);
    EXPECT_EQ(model->cache_resets, 1);
    
    // Calculate expected token count
    size_t expected_token_count = 0;
    
    // Speaker 1 token + "First message" tokens
    expected_token_count += 1 + tokenizer->encode("First message").size();
    
    // Speaker 2 token + "Second reply" tokens
    expected_token_count += 1 + tokenizer->encode("Second reply").size();
    
    // Speaker 1 token + "Third message" tokens
    expected_token_count += 1 + tokenizer->encode("Third message").size();
    
    // Speaker 2 token (current speaker) + "Final reply" tokens
    expected_token_count += 1 + tokens.size();
    
    // Verify total token count
    EXPECT_EQ(model->last_tokens.size(), expected_token_count);
    
    // Verify positions count equals token count
    EXPECT_EQ(model->last_positions.size(), expected_token_count);
}

// Test context limitation
TEST_F(ContextManagementTest, TestContextLimitation) {
    // Create a very long context that should be limited
    std::string long_text(5000, 'a'); // 5000 'a' characters
    Segment long_segment;
    long_segment.text = long_text;
    long_segment.speaker_id = 1;
    
    std::vector<Segment> context = {long_segment};
    
    // Set a small max_text_tokens value
    generator->set_max_text_tokens(100);
    
    // Generate speech with long context
    std::vector<int> tokens = tokenizer->encode("Short text");
    std::vector<float> audio = generator->generate_speech_from_tokens(tokens, 2, context);
    
    // Check model interactions
    EXPECT_GT(model->frames_generated, 0);
    
    // Verify tokens were limited to max_text_tokens
    EXPECT_LE(model->last_tokens.size(), 100);
    EXPECT_LE(model->last_positions.size(), 100);
}

// Test memory optimization during generation
TEST_F(ContextManagementTest, TestMemoryOptimization) {
    // Create context
    Segment segment;
    segment.text = "Test memory optimization";
    segment.speaker_id = 1;
    
    std::vector<Segment> context = {segment};
    
    // Generate speech with memory optimization enabled
    // Set max_audio_length_ms high enough to generate multiple frames
    GenerationOptions options;
    options.max_audio_length_ms = 2000; // Should generate about 50 frames (at 24kHz, 400 samples per frame)
    
    std::vector<int> tokens = tokenizer->encode("Memory test");
    std::vector<float> audio = generator->generate_speech_from_tokens(tokens, 2, context, options);
    
    // Check model interactions
    EXPECT_GT(model->frames_generated, 10); // Should have generated many frames
    EXPECT_GT(model->cache_prunes, 0); // Cache should have been pruned at least once
    EXPECT_EQ(model->prune_factor, 0.5f); // Prune factor should match what was set
}

// Test context with different speaker IDs
TEST_F(ContextManagementTest, TestDifferentSpeakerIDs) {
    // Create context with unusual speaker IDs
    std::vector<Segment> context = {
        {.text = "Speaker 100", .speaker_id = 100},
        {.text = "Speaker 200", .speaker_id = 200},
        {.text = "Speaker 300", .speaker_id = 300}
    };
    
    // Generate speech with multi-speaker context
    std::vector<int> tokens = tokenizer->encode("Speaker 400");
    std::vector<float> audio = generator->generate_speech_from_tokens(tokens, 400, context);
    
    // Check model interactions
    EXPECT_GT(model->frames_generated, 0);
    
    // Verify speaker tokens
    EXPECT_EQ(model->last_tokens[0], tokenizer->get_speaker_token_id(100));
    
    // Find index of second speaker token
    size_t idx = 1 + tokenizer->encode("Speaker 100").size();
    EXPECT_EQ(model->last_tokens[idx], tokenizer->get_speaker_token_id(200));
    
    // Find index of third speaker token
    idx += 1 + tokenizer->encode("Speaker 200").size();
    EXPECT_EQ(model->last_tokens[idx], tokenizer->get_speaker_token_id(300));
    
    // Find index of current speaker token
    idx += 1 + tokenizer->encode("Speaker 300").size();
    EXPECT_EQ(model->last_tokens[idx], tokenizer->get_speaker_token_id(400));
}

// Test empty context
TEST_F(ContextManagementTest, TestEmptyContext) {
    // Create empty context
    std::vector<Segment> context = {};
    
    // Generate speech with empty context
    std::vector<int> tokens = tokenizer->encode("No context");
    std::vector<float> audio = generator->generate_speech_from_tokens(tokens, 1, context);
    
    // Check model interactions
    EXPECT_GT(model->frames_generated, 0);
    
    // Verify tokens only include speaker + input
    size_t expected_token_count = 1 + tokens.size(); // Speaker token + input tokens
    EXPECT_EQ(model->last_tokens.size(), expected_token_count);
    
    // First token should be speaker token
    EXPECT_EQ(model->last_tokens[0], tokenizer->get_speaker_token_id(1));
    
    // Followed by input tokens
    for (size_t i = 0; i < tokens.size(); i++) {
        EXPECT_EQ(model->last_tokens[i + 1], tokens[i]);
    }
}

// Test context with no speaker IDs
TEST_F(ContextManagementTest, TestContextWithoutSpeakers) {
    // Create context without speaker IDs
    std::vector<Segment> context = {
        {.text = "No speaker 1", .speaker_id = -1},
        {.text = "No speaker 2", .speaker_id = -1}
    };
    
    // Generate speech with speakerless context
    std::vector<int> tokens = tokenizer->encode("With speaker");
    std::vector<float> audio = generator->generate_speech_from_tokens(tokens, 1, context);
    
    // Check model interactions
    EXPECT_GT(model->frames_generated, 0);
    
    // Calculate expected token count
    size_t expected_token_count = 0;
    
    // "No speaker 1" tokens (no speaker token)
    expected_token_count += tokenizer->encode("No speaker 1").size();
    
    // "No speaker 2" tokens (no speaker token)
    expected_token_count += tokenizer->encode("No speaker 2").size();
    
    // Speaker 1 token + "With speaker" tokens
    expected_token_count += 1 + tokens.size();
    
    // Verify total token count
    EXPECT_EQ(model->last_tokens.size(), expected_token_count);
}

// Test context with mixed speaker presence
TEST_F(ContextManagementTest, TestMixedSpeakerPresence) {
    // Create context with mix of speakers and no speakers
    std::vector<Segment> context = {
        {.text = "With speaker", .speaker_id = 1},
        {.text = "No speaker", .speaker_id = -1},
        {.text = "With another speaker", .speaker_id = 2}
    };
    
    // Generate speech with mixed context
    std::vector<int> tokens = tokenizer->encode("Final text");
    std::vector<float> audio = generator->generate_speech_from_tokens(tokens, 3, context);
    
    // Check model interactions
    EXPECT_GT(model->frames_generated, 0);
    
    // Verify first token is speaker 1
    EXPECT_EQ(model->last_tokens[0], tokenizer->get_speaker_token_id(1));
    
    // Find where speaker 2 should be
    size_t speaker1_text_size = tokenizer->encode("With speaker").size();
    size_t no_speaker_text_size = tokenizer->encode("No speaker").size();
    size_t speaker2_index = 1 + speaker1_text_size + no_speaker_text_size;
    
    // Verify speaker 2 token
    EXPECT_EQ(model->last_tokens[speaker2_index], tokenizer->get_speaker_token_id(2));
}

// Test very large number of context segments
TEST_F(ContextManagementTest, TestLargeNumberOfSegments) {
    // Create many context segments
    std::vector<Segment> context;
    for (int i = 0; i < 100; i++) {
        Segment segment;
        segment.text = "Short text " + std::to_string(i);
        segment.speaker_id = i % 3; // Alternate between 3 speakers
        context.push_back(segment);
    }
    
    // Generate speech with many context segments
    std::vector<int> tokens = tokenizer->encode("Final segment");
    
    // Set reasonable token limit to prevent too large context
    generator->set_max_text_tokens(500);
    
    std::vector<float> audio = generator->generate_speech_from_tokens(tokens, 0, context);
    
    // Check model interactions
    EXPECT_GT(model->frames_generated, 0);
    EXPECT_LE(model->last_tokens.size(), 500);
}

// Test context segment ordering
TEST_F(ContextManagementTest, TestContextSegmentOrdering) {
    // Create context with clear order
    std::vector<Segment> context = {
        {.text = "First", .speaker_id = 1},
        {.text = "Second", .speaker_id = 2},
        {.text = "Third", .speaker_id = 1}
    };
    
    // Generate speech
    std::vector<int> tokens = tokenizer->encode("Fourth");
    std::vector<float> audio = generator->generate_speech_from_tokens(tokens, 2, context);
    
    // Check model interactions
    EXPECT_GT(model->frames_generated, 0);
    
    // Check token sequence order
    std::vector<int> first_tokens = tokenizer->encode("First");
    std::vector<int> second_tokens = tokenizer->encode("Second");
    std::vector<int> third_tokens = tokenizer->encode("Third");
    
    // Token positions (assuming context is processed in order)
    size_t pos = 0;
    
    // Speaker 1
    EXPECT_EQ(model->last_tokens[pos++], tokenizer->get_speaker_token_id(1));
    
    // "First" tokens
    for (int token : first_tokens) {
        EXPECT_EQ(model->last_tokens[pos++], token);
    }
    
    // Speaker 2
    EXPECT_EQ(model->last_tokens[pos++], tokenizer->get_speaker_token_id(2));
    
    // "Second" tokens
    for (int token : second_tokens) {
        EXPECT_EQ(model->last_tokens[pos++], token);
    }
    
    // Speaker 1
    EXPECT_EQ(model->last_tokens[pos++], tokenizer->get_speaker_token_id(1));
    
    // "Third" tokens
    for (int token : third_tokens) {
        EXPECT_EQ(model->last_tokens[pos++], token);
    }
    
    // Speaker 2 (current speaker)
    EXPECT_EQ(model->last_tokens[pos++], tokenizer->get_speaker_token_id(2));
}

// Add advanced context handling method tests here
} // namespace testing
} // namespace ccsm