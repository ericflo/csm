#include "ccsm/tokenizer.h"
#include <cassert>
#include <iostream>
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
};

int main() {
    std::cout << "Testing tokenizer..." << std::endl;

    // Create mock tokenizer
    auto tokenizer = std::make_shared<MockTokenizer>();

    // Test encode/decode
    std::string test_text = "hello world";
    auto tokens = tokenizer->encode(test_text);
    std::string decoded = tokenizer->decode(tokens);
    
    std::cout << "Original: " << test_text << std::endl;
    std::cout << "Encoded: ";
    for (auto token : tokens) {
        std::cout << token << " ";
    }
    std::cout << std::endl;
    std::cout << "Decoded: " << decoded << std::endl;

    // Test special tokens
    std::cout << "BOS token ID: " << tokenizer->bos_token_id() << std::endl;
    std::cout << "EOS token ID: " << tokenizer->eos_token_id() << std::endl;
    std::cout << "PAD token ID: " << tokenizer->pad_token_id() << std::endl;
    std::cout << "UNK token ID: " << tokenizer->unk_token_id() << std::endl;

    // Test speaker token IDs
    for (int i = 0; i <= 3; ++i) {
        std::cout << "Speaker " << i << " token ID: " << tokenizer->get_speaker_token_id(i) << std::endl;
    }

    // Basic assertion tests
    assert(tokenizer->vocab_size() == 9);
    assert(tokenizer->bos_token_id() == 0);
    assert(tokenizer->eos_token_id() == 1);
    assert(tokenizer->pad_token_id() == 2);
    assert(tokenizer->unk_token_id() == 3);
    assert(tokenizer->get_speaker_token_id(0) == 6);
    assert(tokenizer->get_speaker_token_id(1) == 7);
    assert(tokenizer->get_speaker_token_id(2) == 8);
    assert(tokenizer->get_speaker_token_id(3) == 3); // Invalid speaker ID returns UNK

    std::cout << "All tests passed!" << std::endl;
    return 0;
}