#ifndef CCSM_MIMI_CODEC_H
#define CCSM_MIMI_CODEC_H

#include <ccsm/tokenizer.h>
#include <string>
#include <vector>
#include <memory>
#include <array>

namespace ccsm {

// Forward declarations
class MimiCodecImpl;

/**
 * Configuration for the Mimi codec
 */
struct MimiCodecConfig {
    // Sample rate to use for audio processing
    int sample_rate = 24000;
    
    // Number of codebooks in the residual vector quantizer
    int num_codebooks = 32;
    
    // Vocabulary size for each codebook
    int vocab_size = 2051;
    
    // Hop length (frame size) in samples
    int hop_length = 1920; // 80ms at 24kHz
    
    // Whether to enable denormalization for better audio quality
    bool enable_denormalization = true;
    
    // Allow CPU fallback if acceleration is unavailable
    bool allow_cpu_fallback = true;
    
    // Whether to use full precision for audio processing
    bool use_full_precision = false;
    
    // Seed for operations that require randomness
    int seed = 42;
};

/**
 * Mimi codec implementation
 * 
 * This class provides integration with the Mimi codec for audio
 * tokenization and detokenization. It handles the conversion between
 * audio waveforms and the residual vector quantizer (RVQ) tokens
 * used by the CSM model.
 */
class MimiCodec : public AudioCodec {
public:
    // Create a new Mimi codec from a model file
    static std::shared_ptr<MimiCodec> from_file(const std::string& path, const MimiCodecConfig& config = {});
    
    // Create a new Mimi codec from memory
    static std::shared_ptr<MimiCodec> from_binary(const std::vector<uint8_t>& data, const MimiCodecConfig& config = {});
    
    // Constructor
    MimiCodec(const std::string& model_path, const MimiCodecConfig& config = {});
    
    // Constructor from memory
    MimiCodec(const std::vector<uint8_t>& model_data, const MimiCodecConfig& config = {});
    
    // Destructor
    ~MimiCodec();
    
    // Encode audio to RVQ tokens
    std::vector<std::vector<int>> encode(const std::vector<float>& audio) const override;
    
    // Decode RVQ tokens to audio
    std::vector<float> decode(const std::vector<std::vector<int>>& tokens) const override;
    
    // Get the number of codebooks
    int num_codebooks() const override;
    
    // Get the vocabulary size
    int vocab_size() const override;
    
    // Get the sample rate
    int sample_rate() const override;
    
    // Get the hop length in samples
    int hop_length() const override;
    
    // Check if a token is an end-of-sequence token for a given codebook
    bool is_eos_token(int token, int codebook) const override;
    
    // Mimi-specific methods
    
    // Preprocess audio for the codec (normalization, resampling, etc.)
    std::vector<float> preprocess_audio(const std::vector<float>& audio) const;
    
    // Postprocess audio from the codec (denormalization, etc.)
    std::vector<float> postprocess_audio(const std::vector<float>& audio) const;
    
    // Set configuration
    void set_config(const MimiCodecConfig& config);
    
    // Get current configuration
    const MimiCodecConfig& get_config() const;
    
private:
    // Private implementation (PIMPL pattern)
    std::unique_ptr<MimiCodecImpl> impl_;
    
    // Configuration
    MimiCodecConfig config_;
};

/**
 * Specialized Mimi audio tokenizer
 * 
 * This class provides text-based representation of audio tokens
 * and integrates with the Mimi codec.
 */
class MimiAudioTokenizer : public AudioTokenizer {
public:
    // Constructor
    MimiAudioTokenizer(std::shared_ptr<MimiCodec> codec);
    
    // Methods implemented from AudioTokenizer interface
    std::vector<int> encode(const std::string& text) const override;
    std::string decode(const std::vector<int>& tokens) const override;
    int vocab_size() const override;
    std::vector<std::vector<int>> encode_audio(const std::vector<float>& audio) const override;
    int audio_bos_token_id() const override;
    int audio_eos_token_id() const override;
    int audio_pad_token_id() const override;
    
    // Mimi-specific methods
    
    // Convert audio tokens to textual representation
    std::string tokens_to_text(const std::vector<std::vector<int>>& tokens) const;
    
    // Convert textual representation to audio tokens
    std::vector<std::vector<int>> text_to_tokens(const std::string& text) const;
    
private:
    // The Mimi codec implementation
    std::shared_ptr<MimiCodec> codec_;
    
    // Special token IDs
    int audio_bos_id_;
    int audio_eos_id_;
    int audio_pad_id_;
};

} // namespace ccsm

#endif // CCSM_MIMI_CODEC_H