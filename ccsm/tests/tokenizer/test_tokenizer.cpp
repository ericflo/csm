#include "ccsm/tokenizer.h"
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>

using namespace ccsm;

// Simple mock for testing without an actual model file
class MockTokenizer : public TextTokenizer {
private:
    std::vector<std::string> vocabulary_;
    int bos_id_;
    int eos_id_;
    int pad_id_;
    int unk_id_;

public:
    MockTokenizer() 
        : vocabulary_{"<bos>", "<eos>", "<pad>", "<unk>", "hello", "world", "[0]", "[1]", "[2]"},
          bos_id_(0),
          eos_id_(1),
          pad_id_(2),
          unk_id_(3) {}

    std::vector<int> encode(const std::string& text) const override {
        std::vector<int> result;
        if (text == "hello world") {
            result = {4, 5}; // "hello" -> 4, "world" -> 5
        } else if (text == "[0]hello") {
            result = {6, 4}; // "[0]" -> 6, "hello" -> 4
        } else {
            result = {3}; // Unknown -> 3
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
        if (speaker_id >= 0 && speaker_id <= 2) {
            return 6 + speaker_id; // [0] -> 6, [1] -> 7, [2] -> 8
        }
        return unk_id_;
    }
    
    std::vector<int> get_audio_token_ids() const override {
        // Return a simple range of token IDs for audio tokens
        return {6, 7, 8}; // [0], [1], [2] are our audio token IDs
    }
};

// Test fixture for tokenizer tests
class TokenizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        tokenizer = std::make_shared<MockTokenizer>();
    }
    
    std::shared_ptr<MockTokenizer> tokenizer;
};

// Test basic tokenizer functionality
TEST_F(TokenizerTest, BasicFunctionality) {
    // Test encode/decode
    std::string test_text = "hello world";
    auto tokens = tokenizer->encode(test_text);
    std::string decoded = tokenizer->decode(tokens);
    
    EXPECT_EQ(tokens.size(), 2);
    EXPECT_EQ(tokens[0], 4); // "hello"
    EXPECT_EQ(tokens[1], 5); // "world"
    EXPECT_EQ(decoded, "hello world");
}

// Test special tokens
TEST_F(TokenizerTest, SpecialTokens) {
    EXPECT_EQ(tokenizer->vocab_size(), 9);
    EXPECT_EQ(tokenizer->bos_token_id(), 0);
    EXPECT_EQ(tokenizer->eos_token_id(), 1);
    EXPECT_EQ(tokenizer->pad_token_id(), 2);
    EXPECT_EQ(tokenizer->unk_token_id(), 3);
}

// Test speaker token IDs
TEST_F(TokenizerTest, SpeakerTokens) {
    EXPECT_EQ(tokenizer->get_speaker_token_id(0), 6);
    EXPECT_EQ(tokenizer->get_speaker_token_id(1), 7);
    EXPECT_EQ(tokenizer->get_speaker_token_id(2), 8);
    EXPECT_EQ(tokenizer->get_speaker_token_id(3), 3); // Invalid speaker ID returns UNK
}

// Test unknown token handling
TEST_F(TokenizerTest, UnknownTokens) {
    // Test with unknown text
    std::string unknown_text = "unknown text";
    auto unknown_tokens = tokenizer->encode(unknown_text);
    EXPECT_EQ(unknown_tokens.size(), 1);
    EXPECT_EQ(unknown_tokens[0], 3); // UNK token
    
    // Test with out-of-range token IDs
    std::vector<int> out_of_range = {100, 101}; // Non-existent token IDs
    std::string decoded = tokenizer->decode(out_of_range);
    EXPECT_EQ(decoded, "<unk> <unk>");
}