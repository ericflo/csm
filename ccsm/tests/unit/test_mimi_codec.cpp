#include <gtest/gtest.h>
#include <ccsm/mimi_codec.h>
#include <vector>
#include <cmath>

namespace ccsm {
namespace testing {

// MimiCodec Test Fixture
class MimiCodecTest : public ::testing::Test {
protected:
    MimiCodecConfig config;
    
    void SetUp() override {
        // Configure for testing
        config.sample_rate = 24000;
        config.num_codebooks = 8;
        config.vocab_size = 2051;
        config.hop_length = 1920;
        config.seed = 42;  // Fixed seed for deterministic testing
    }
};

// Test creating a codec with mock implementation
TEST_F(MimiCodecTest, CreateMockCodec) {
    try {
        auto codec = std::make_shared<MimiCodec>("mock_model_path", config);
        ASSERT_NE(codec, nullptr);
        EXPECT_EQ(codec->sample_rate(), 24000);
        EXPECT_EQ(codec->num_codebooks(), 8);
        EXPECT_EQ(codec->vocab_size(), 2051);
    } catch (const std::exception& e) {
        #ifdef CCSM_WITH_MIMI
        FAIL() << "Exception: " << e.what();
        #else
        // Without MIMI support, this is expected to throw
        std::cout << "Expected exception when MIMI support is not enabled: " << e.what() << std::endl;
        #endif
    }
}

// Test encode and decode with mock implementation
TEST_F(MimiCodecTest, EncodeDecodeRoundtrip) {
    // Skip test if Mimi support is not compiled in
    #ifndef CCSM_WITH_MIMI
    GTEST_SKIP() << "Mimi codec support not compiled in";
    #endif
    
    // Create a sample audio waveform (a simple sine wave)
    std::vector<float> audio(24000);  // 1 second at 24kHz
    for (size_t i = 0; i < audio.size(); i++) {
        float t = static_cast<float>(i) / 24000.0f;
        audio[i] = 0.5f * std::sin(2.0f * M_PI * 440.0f * t);  // 440Hz A note
    }
    
    // Create the codec
    auto codec = std::make_shared<MimiCodec>("mock_model_path", config);
    
    // Encode the audio
    auto tokens = codec->encode(audio);
    
    // Check basic properties of encoded tokens
    ASSERT_EQ(tokens.size(), 8);  // 8 codebooks
    ASSERT_GT(tokens[0].size(), 0);  // Should have at least one frame
    
    // Decode the tokens back to audio
    auto decoded = codec->decode(tokens);
    
    // Check basic properties of decoded audio
    ASSERT_GT(decoded.size(), 0);
    
    // Note: In a mock implementation, exact roundtrip is not expected
    // In a real implementation, we would check for similarity metrics
}

// Test MimiAudioTokenizer
TEST_F(MimiCodecTest, MimiAudioTokenizer) {
    // Skip test if Mimi support is not compiled in
    #ifndef CCSM_WITH_MIMI
    GTEST_SKIP() << "Mimi codec support not compiled in";
    #endif
    
    // Create the codec
    auto codec = std::make_shared<MimiCodec>("mock_model_path", config);
    
    // Create the tokenizer
    auto tokenizer = std::make_shared<MimiAudioTokenizer>(codec);
    
    // Check basic properties
    EXPECT_EQ(tokenizer->vocab_size(), 2051);
    EXPECT_EQ(tokenizer->audio_bos_token_id(), 0);
    EXPECT_EQ(tokenizer->audio_eos_token_id(), 0);
    EXPECT_EQ(tokenizer->audio_pad_token_id(), 0);
    
    // Test encoding audio
    std::vector<float> audio(24000);  // 1 second at 24kHz
    auto tokens = tokenizer->encode_audio(audio);
    
    // Check basic properties of encoded tokens
    ASSERT_EQ(tokens.size(), 8);  // 8 codebooks
    ASSERT_GT(tokens[0].size(), 0);  // Should have at least one frame
    
    // Test text representation
    std::string text_repr = tokenizer->tokens_to_text(tokens);
    EXPECT_FALSE(text_repr.empty());
    EXPECT_NE(text_repr.find("[AUDIO_TOKENS:"), std::string::npos);
}

// Test factory methods
TEST_F(MimiCodecTest, FactoryMethods) {
    // Skip test if Mimi support is not compiled in
    #ifndef CCSM_WITH_MIMI
    GTEST_SKIP() << "Mimi codec support not compiled in";
    #endif
    
    // Test from_file factory method
    auto codec = MimiCodec::from_file("mock_model_path", config);
    ASSERT_NE(codec, nullptr);
    
    // Test from AudioCodec::from_file factory
    auto audio_codec = AudioCodec::from_file("mock_model_path");
    ASSERT_NE(audio_codec, nullptr);
    
    // Test from AudioTokenizer::from_file factory
    auto audio_tokenizer = AudioTokenizer::from_file("mock_model_path");
    ASSERT_NE(audio_tokenizer, nullptr);
}

// Test configuration
TEST_F(MimiCodecTest, Configuration) {
    // Skip test if Mimi support is not compiled in
    #ifndef CCSM_WITH_MIMI
    GTEST_SKIP() << "Mimi codec support not compiled in";
    #endif
    
    // Create the codec with initial config
    auto codec = std::make_shared<MimiCodec>("mock_model_path", config);
    
    // Check initial config
    auto initial_config = codec->get_config();
    EXPECT_EQ(initial_config.sample_rate, 24000);
    EXPECT_EQ(initial_config.num_codebooks, 8);
    
    // Update config
    MimiCodecConfig new_config = initial_config;
    new_config.sample_rate = 16000;
    new_config.use_full_precision = true;
    
    codec->set_config(new_config);
    
    // Check updated config
    auto updated_config = codec->get_config();
    EXPECT_EQ(updated_config.sample_rate, 16000);
    EXPECT_TRUE(updated_config.use_full_precision);
}

// Test error handling
TEST_F(MimiCodecTest, ErrorHandling) {
    // Skip test if Mimi support is not compiled in
    #ifndef CCSM_WITH_MIMI
    GTEST_SKIP() << "Mimi codec support not compiled in";
    #endif
    
    // Test with empty audio
    auto codec = std::make_shared<MimiCodec>("mock_model_path", config);
    auto empty_audio = std::vector<float>();
    auto empty_tokens = codec->encode(empty_audio);
    
    // Should return empty tokens but not crash
    EXPECT_TRUE(empty_tokens.empty() || empty_tokens[0].empty());
    
    // Test with empty tokens
    auto empty_decoded = codec->decode(std::vector<std::vector<int>>());
    EXPECT_TRUE(empty_decoded.empty());
}

} // namespace testing
} // namespace ccsm