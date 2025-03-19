#ifndef CCSM_GENERATOR_H
#define CCSM_GENERATOR_H

#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace ccsm {

// Forward declarations
class Model;
class TextTokenizer;
class AudioCodec;
class Watermarker;

// Segment representing a chunk of conversation
struct Segment {
    std::string text;
    int speaker_id;
    std::vector<float> audio;
    
    Segment(const std::string& text, int speaker_id, const std::vector<float>& audio = {})
        : text(text), speaker_id(speaker_id), audio(audio) {}
};

// Generation options
struct GenerationOptions {
    float temperature = 0.9f;
    int top_k = 50;
    int max_audio_length_ms = 10000;
    int seed = -1; // -1 means random seed
    bool enable_watermark = true;
    bool debug = false;
};

// Generator interface for text-to-speech
class Generator {
public:
    // Constructor
    Generator(std::shared_ptr<Model> model,
              std::shared_ptr<TextTokenizer> text_tokenizer,
              std::shared_ptr<AudioCodec> audio_codec,
              std::shared_ptr<Watermarker> watermarker = nullptr);
    
    // Virtual destructor
    virtual ~Generator() = default;
    
    // Main generation function
    virtual std::vector<float> generate_speech(
        const std::string& text,
        int speaker_id,
        const std::vector<Segment>& context = {},
        const GenerationOptions& options = {},
        std::function<void(int, int)> progress_callback = nullptr);
    
    // Generate from pre-tokenized text
    virtual std::vector<float> generate_speech_from_tokens(
        const std::vector<int>& tokens,
        int speaker_id,
        const std::vector<Segment>& context = {},
        const GenerationOptions& options = {},
        std::function<void(int, int)> progress_callback = nullptr);
    
    // Getters
    int sample_rate() const;
    std::shared_ptr<Model> model() const;
    std::shared_ptr<TextTokenizer> text_tokenizer() const;
    std::shared_ptr<AudioCodec> audio_codec() const;
    
protected:
    std::shared_ptr<Model> model_;
    std::shared_ptr<TextTokenizer> text_tokenizer_;
    std::shared_ptr<AudioCodec> audio_codec_;
    std::shared_ptr<Watermarker> watermarker_;
    int sample_rate_;
};

// Factory function to create a generator with the CSM-1B model
std::shared_ptr<Generator> load_csm_1b(const std::string& device = "cpu");

// Factory function for MLX-accelerated generator
std::shared_ptr<Generator> load_csm_1b_mlx();

} // namespace ccsm

#endif // CCSM_GENERATOR_H