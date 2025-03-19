#ifndef CCSM_TOKENIZER_H
#define CCSM_TOKENIZER_H

#include <string>
#include <vector>
#include <memory>

namespace ccsm {

// Base tokenizer interface
class Tokenizer {
public:
    virtual ~Tokenizer() = default;
    
    // Encode text to token IDs
    virtual std::vector<int> encode(const std::string& text) const = 0;
    
    // Decode token IDs to text
    virtual std::string decode(const std::vector<int>& tokens) const = 0;
    
    // Get vocabulary size
    virtual int vocab_size() const = 0;
};

// Text tokenizer interface (SentencePiece-based)
class TextTokenizer : public Tokenizer {
public:
    // Create from model file
    static std::shared_ptr<TextTokenizer> from_file(const std::string& path);
    
    // Create from binary data
    static std::shared_ptr<TextTokenizer> from_binary(const std::vector<uint8_t>& data);
    
    // Get special token IDs
    virtual int bos_token_id() const = 0;
    virtual int eos_token_id() const = 0;
    virtual int pad_token_id() const = 0;
    virtual int unk_token_id() const = 0;
    
    // Get speaker token ID
    virtual int get_speaker_token_id(int speaker_id) const = 0;
};

// Audio tokenizer interface
class AudioTokenizer : public Tokenizer {
public:
    // Create from model file
    static std::shared_ptr<AudioTokenizer> from_file(const std::string& path);
    
    // Additional methods for audio tokenization
    virtual std::vector<std::vector<int>> encode_audio(const std::vector<float>& audio) const = 0;
    
    // Get special token IDs for audio
    virtual int audio_bos_token_id() const = 0;
    virtual int audio_eos_token_id() const = 0;
    virtual int audio_pad_token_id() const = 0;
};

// Audio codec interface for Mimi
class AudioCodec {
public:
    virtual ~AudioCodec() = default;
    
    // Load from model file
    static std::shared_ptr<AudioCodec> from_file(const std::string& path);
    
    // Encode audio to RVQ tokens
    virtual std::vector<std::vector<int>> encode(const std::vector<float>& audio) const = 0;
    
    // Decode RVQ tokens to audio
    virtual std::vector<float> decode(const std::vector<std::vector<int>>& tokens) const = 0;
    
    // Get the number of codebooks
    virtual int num_codebooks() const = 0;
    
    // Get vocabulary size for each codebook
    virtual int vocab_size() const = 0;
    
    // Get sample rate
    virtual int sample_rate() const = 0;
    
    // Get hop length (samples per frame)
    virtual int hop_length() const = 0;
    
    // Check if token is EOS
    virtual bool is_eos_token(int token, int codebook) const = 0;
};

} // namespace ccsm

#endif // CCSM_TOKENIZER_H