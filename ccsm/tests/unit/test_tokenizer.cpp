#include <gtest/gtest.h>
#include <ccsm/tokenizer.h>
#include <memory>
#include <string>
#include <vector>

using namespace ccsm;

// Mock TextTokenizer implementation for testing
class MockTextTokenizer : public TextTokenizer {
private:
    std::vector<std::string> vocabulary_;
    int bos_id_;
    int eos_id_;
    int pad_id_;
    int unk_id_;

public:
    MockTextTokenizer() 
        : vocabulary_{"<bos>", "<eos>", "<pad>", "<unk>", "hello", "world", "how", "are", "you", 
                     "i", "am", "fine", "thanks", "<s0>", "<s1>", "<s2>", "<s3>"},
          bos_id_(0),
          eos_id_(1),
          pad_id_(2),
          unk_id_(3) {}

    std::vector<int> encode(const std::string& text) const override {
        std::vector<int> result;
        
        // Simple encoding logic for testing
        if (text == "hello world") {
            result = {4, 5}; // "hello" -> 4, "world" -> 5
        } else if (text == "how are you") {
            result = {6, 7, 8}; // "how" -> 6, "are" -> 7, "you" -> 8
        } else if (text == "i am fine thanks") {
            result = {9, 10, 11, 12}; // "i" -> 9, "am" -> 10, "fine" -> 11, "thanks" -> 12
        } else if (text == "<s0>hello") {
            result = {13, 4}; // "<s0>" -> 13, "hello" -> 4
        } else if (text == "<s1>world") {
            result = {14, 5}; // "<s1>" -> 14, "world" -> 5
        } else {
            // Handle unknown input by inserting the unknown token
            result = {unk_id_};
        }
        
        return result;
    }

    std::string decode(const std::vector<int>& tokens) const override {
        std::string result;
        
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (i > 0) result += " ";
            int token = tokens[i];
            if (token >= 0 && token < static_cast<int>(vocabulary_.size())) {
                result += vocabulary_[token];
            } else {
                result += vocabulary_[unk_id_];
            }
        }
        
        return result;
    }

    int vocab_size() const override {
        return vocabulary_.size();
    }

    int bos_token_id() const override {
        return bos_id_;
    }

    int eos_token_id() const override {
        return eos_id_;
    }

    int pad_token_id() const override {
        return pad_id_;
    }

    int unk_token_id() const override {
        return unk_id_;
    }

    int get_speaker_token_id(int speaker_id) const override {
        if (speaker_id >= 0 && speaker_id <= 3) {
            return 13 + speaker_id; // <s0> -> 13, <s1> -> 14, <s2> -> 15, <s3> -> 16
        }
        return unk_id_;
    }
};

// Mock AudioCodec implementation for testing
class MockAudioCodec : public AudioCodec {
private:
    int num_codebooks_;
    int audio_vocab_size_;
    int sample_rate_;
    int hop_length_;

public:
    MockAudioCodec(int num_codebooks = 8, int vocab_size = 2051, int sample_rate = 24000, int hop_length = 320)
        : num_codebooks_(num_codebooks), 
          audio_vocab_size_(vocab_size),
          sample_rate_(sample_rate),
          hop_length_(hop_length) {}

    std::vector<std::vector<int>> encode(const std::vector<float>& audio) const override {
        // Return mock encoding data
        std::vector<std::vector<int>> result;
        
        // Create a set of audio frames (assuming 5 frames for testing)
        const int num_frames = 5;
        for (int i = 0; i < num_frames; i++) {
            std::vector<int> frame(num_codebooks_);
            for (int j = 0; j < num_codebooks_; j++) {
                // Generate a predictable pattern based on frame and codebook
                frame[j] = (i * 10 + j + 1) % (audio_vocab_size_ - 2) + 1;
            }
            result.push_back(frame);
        }
        
        return result;
    }

    std::vector<float> decode(const std::vector<std::vector<int>>& tokens) const override {
        // Create mock audio data
        // For test purposes, we'll create a simple output that's proportional
        // to the number of frames
        const int samples_per_frame = hop_length_;
        const int num_frames = tokens.size();
        std::vector<float> result(num_frames * samples_per_frame, 0.0f);
        
        // Fill with a simple pattern
        for (int i = 0; i < num_frames; i++) {
            for (int j = 0; j < samples_per_frame; j++) {
                float value = static_cast<float>(i) / num_frames;
                // Add a sine wave pattern
                value += 0.2f * sin(2.0f * 3.14159f * j / samples_per_frame);
                result[i * samples_per_frame + j] = value;
            }
        }
        
        return result;
    }

    int num_codebooks() const override {
        return num_codebooks_;
    }

    int vocab_size() const override {
        return audio_vocab_size_;
    }

    int sample_rate() const override {
        return sample_rate_;
    }

    int hop_length() const override {
        return hop_length_;
    }

    bool is_eos_token(int token, int codebook) const override {
        // Use token 2 as EOS across all codebooks
        return token == 2;
    }
};

// Test fixture for TextTokenizer tests
class TextTokenizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        tokenizer = std::make_shared<MockTextTokenizer>();
    }
    
    std::shared_ptr<TextTokenizer> tokenizer;
};

// Test basic text tokenization functions
TEST_F(TextTokenizerTest, BasicFunctionality) {
    // Test vocabulary size
    EXPECT_EQ(tokenizer->vocab_size(), 17);
    
    // Test special tokens
    EXPECT_EQ(tokenizer->bos_token_id(), 0);
    EXPECT_EQ(tokenizer->eos_token_id(), 1);
    EXPECT_EQ(tokenizer->pad_token_id(), 2);
    EXPECT_EQ(tokenizer->unk_token_id(), 3);
    
    // Test speaker token IDs
    EXPECT_EQ(tokenizer->get_speaker_token_id(0), 13);
    EXPECT_EQ(tokenizer->get_speaker_token_id(1), 14);
    EXPECT_EQ(tokenizer->get_speaker_token_id(2), 15);
    EXPECT_EQ(tokenizer->get_speaker_token_id(3), 16);
    
    // Test invalid speaker ID returns UNK
    EXPECT_EQ(tokenizer->get_speaker_token_id(4), tokenizer->unk_token_id());
    EXPECT_EQ(tokenizer->get_speaker_token_id(-1), tokenizer->unk_token_id());
}

// Test encoding and decoding text
TEST_F(TextTokenizerTest, EncodeDecode) {
    // Test encoding various phrases
    std::vector<int> tokens1 = tokenizer->encode("hello world");
    EXPECT_EQ(tokens1.size(), 2);
    EXPECT_EQ(tokens1[0], 4);
    EXPECT_EQ(tokens1[1], 5);
    
    std::vector<int> tokens2 = tokenizer->encode("how are you");
    EXPECT_EQ(tokens2.size(), 3);
    EXPECT_EQ(tokens2[0], 6);
    EXPECT_EQ(tokens2[1], 7);
    EXPECT_EQ(tokens2[2], 8);
    
    std::vector<int> tokens3 = tokenizer->encode("i am fine thanks");
    EXPECT_EQ(tokens3.size(), 4);
    EXPECT_EQ(tokens3[0], 9);
    EXPECT_EQ(tokens3[1], 10);
    EXPECT_EQ(tokens3[2], 11);
    EXPECT_EQ(tokens3[3], 12);
    
    // Test encoding with speaker tokens
    std::vector<int> tokens4 = tokenizer->encode("<s0>hello");
    EXPECT_EQ(tokens4.size(), 2);
    EXPECT_EQ(tokens4[0], 13); // Speaker 0 token
    EXPECT_EQ(tokens4[1], 4);  // "hello"
    
    // Test decoding
    std::string decoded1 = tokenizer->decode(tokens1);
    EXPECT_EQ(decoded1, "hello world");
    
    std::string decoded2 = tokenizer->decode(tokens2);
    EXPECT_EQ(decoded2, "how are you");
    
    // Test decoding with special tokens
    std::vector<int> special_tokens = {0, 4, 5, 1}; // <bos> hello world <eos>
    std::string decoded_special = tokenizer->decode(special_tokens);
    EXPECT_EQ(decoded_special, "<bos> hello world <eos>");
    
    // Test unknown tokens
    std::vector<int> unknown_tokens = {999, 1000};
    std::string decoded_unknown = tokenizer->decode(unknown_tokens);
    EXPECT_EQ(decoded_unknown, "<unk> <unk>");
}

// Test with empty and edge cases
TEST_F(TextTokenizerTest, EdgeCases) {
    // Test empty input
    std::vector<int> empty_tokens = tokenizer->encode("");
    EXPECT_EQ(empty_tokens.size(), 1);
    EXPECT_EQ(empty_tokens[0], tokenizer->unk_token_id());
    
    // Test decoding empty token list
    std::string empty_decoded = tokenizer->decode({});
    EXPECT_EQ(empty_decoded, "");
    
    // Test encoding something not in our mock vocabulary
    std::vector<int> unknown_text_tokens = tokenizer->encode("this is not in vocabulary");
    EXPECT_EQ(unknown_text_tokens.size(), 1);
    EXPECT_EQ(unknown_text_tokens[0], tokenizer->unk_token_id());
}

// Test fixture for AudioCodec tests
class AudioCodecTest : public ::testing::Test {
protected:
    void SetUp() override {
        codec = std::make_shared<MockAudioCodec>();
    }
    
    std::shared_ptr<AudioCodec> codec;
};

// Test basic audio codec properties
TEST_F(AudioCodecTest, BasicProperties) {
    // Check config values
    EXPECT_EQ(codec->num_codebooks(), 8);
    EXPECT_EQ(codec->vocab_size(), 2051);
    EXPECT_EQ(codec->sample_rate(), 24000);
    EXPECT_EQ(codec->hop_length(), 320);
    
    // Test EOS token detection
    EXPECT_TRUE(codec->is_eos_token(2, 0));
    EXPECT_TRUE(codec->is_eos_token(2, 7));
    EXPECT_FALSE(codec->is_eos_token(1, 0));
    EXPECT_FALSE(codec->is_eos_token(3, 0));
}

// Test audio encoding and decoding
TEST_F(AudioCodecTest, EncodeDecode) {
    // Create mock audio input (1 second of silent audio)
    std::vector<float> audio_input(codec->sample_rate(), 0.0f);
    
    // Test encoding
    std::vector<std::vector<int>> encoded = codec->encode(audio_input);
    
    // Verify encoding structure
    EXPECT_EQ(encoded.size(), 5); // Our mock returns 5 frames
    for (const auto& frame : encoded) {
        EXPECT_EQ(frame.size(), codec->num_codebooks());
        
        // Verify all tokens are in range
        for (int token : frame) {
            EXPECT_GT(token, 0);
            EXPECT_LT(token, codec->vocab_size());
        }
    }
    
    // Test decoding
    std::vector<float> decoded = codec->decode(encoded);
    
    // Verify decoded audio has the expected length
    int expected_length = encoded.size() * codec->hop_length();
    EXPECT_EQ(decoded.size(), expected_length);
    
    // Verify decoded audio has reasonable values
    for (float sample : decoded) {
        EXPECT_GE(sample, -1.0f);
        EXPECT_LE(sample, 1.0f);
    }
}

// Test audio codec with different configurations
TEST_F(AudioCodecTest, DifferentConfigurations) {
    // Test various codec configurations
    auto codec2 = std::make_shared<MockAudioCodec>(4, 1024, 16000, 160);
    EXPECT_EQ(codec2->num_codebooks(), 4);
    EXPECT_EQ(codec2->vocab_size(), 1024);
    EXPECT_EQ(codec2->sample_rate(), 16000);
    EXPECT_EQ(codec2->hop_length(), 160);
    
    // Test encoding with this configuration
    std::vector<float> audio_input(codec2->sample_rate(), 0.0f);
    std::vector<std::vector<int>> encoded = codec2->encode(audio_input);
    
    // Verify encoding structure for this configuration
    EXPECT_EQ(encoded.size(), 5); // Our mock still returns 5 frames
    for (const auto& frame : encoded) {
        EXPECT_EQ(frame.size(), codec2->num_codebooks());
        
        // Verify all tokens are in range for this vocabulary
        for (int token : frame) {
            EXPECT_GT(token, 0);
            EXPECT_LT(token, codec2->vocab_size());
        }
    }
}

// No main function here - using the common main_test.cpp