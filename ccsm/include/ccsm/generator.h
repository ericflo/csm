#ifndef CCSM_GENERATOR_H
#define CCSM_GENERATOR_H

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <random>

namespace ccsm {

// Forward declarations
class Model;
class TextTokenizer;
class AudioCodec;
class Watermarker;
class ContextManager;

// Segment representing a chunk of conversation
struct Segment {
    std::string text;
    int speaker_id;
    std::vector<float> audio;
    
    // Default constructor
    Segment() : text(""), speaker_id(-1), audio() {}
    
    // Constructor with parameters
    Segment(const std::string& text, int speaker_id, const std::vector<float>& audio = {})
        : text(text), speaker_id(speaker_id), audio(audio) {}
};

// Advanced Sampling Options for token generation
struct SamplingOptions {
    // Basic sampling parameters
    float temperature = 0.9f;
    int top_k = 50;
    float top_p = 1.0f;  // 1.0 means no nucleus sampling filtering
    
    // Advanced sampling controls
    float repetition_penalty = 1.0f;  // 1.0 means no penalty
    float frequency_penalty = 0.0f;   // 0.0 means no penalty
    float presence_penalty = 0.0f;    // 0.0 means no penalty
    
    // Token biases (token_id -> bias value)
    std::unordered_map<int, float> logit_bias;
    
    // Sampling control
    int min_tokens = 0;               // Minimum tokens to generate before EOS
    int max_tokens = 4096;            // Maximum tokens to generate
    bool greedy = false;              // If true, uses greedy sampling (overrides temperature)
    
    // Constructor with common parameters
    SamplingOptions(float temp = 0.9f, int topk = 50)
        : temperature(temp), top_k(topk) {}
};

// Generation options
struct GenerationOptions {
    // Sampling parameters
    float temperature = 0.9f;
    int top_k = 50;
    float top_p = 1.0f;  // Added for nucleus sampling
    float repetition_penalty = 1.0f;  // Added for repetition control
    
    // Generation constraints
    int max_audio_length_ms = 10000;
    int min_audio_length_ms = 0;
    
    // System parameters
    int seed = -1; // -1 means random seed
    bool enable_watermark = true;
    bool debug = false;
    
    // Advanced sampling parameters (added in Phase 4.3)
    SamplingOptions sampling;
    
    // Constructor
    GenerationOptions() {
        // Initialize sampling with the same parameters
        sampling.temperature = temperature;
        sampling.top_k = top_k;
        sampling.top_p = top_p;
        sampling.repetition_penalty = repetition_penalty;
    }
    
    // Method to update sampling options from main options
    // This maintains backward compatibility
    void update_sampling() {
        sampling.temperature = temperature;
        sampling.top_k = top_k;
        sampling.top_p = top_p;
        sampling.repetition_penalty = repetition_penalty;
    }
};

// Generation result containing audio and token information
struct GenerationResult {
    std::vector<float> audio;
    std::vector<int> frame;
    std::vector<std::vector<int>> frames;
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
        int speaker_id = -1,
        const std::vector<Segment>& context = {},
        const GenerationOptions& options = {},
        std::function<void(int, int)> progress_callback = nullptr);
    
    // Generate from pre-tokenized text
    virtual std::vector<float> generate_speech_from_tokens(
        const std::vector<int>& tokens,
        int speaker_id = -1,
        const std::vector<Segment>& context = {},
        const GenerationOptions& options = {},
        std::function<void(int, int)> progress_callback = nullptr);
    
    // Advanced sampling generation function
    virtual std::vector<float> generate_speech_with_sampling(
        const std::string& text,
        int speaker_id,
        const std::vector<Segment>& context,
        const SamplingOptions& sampling_options,
        int max_audio_length_ms = 10000,
        bool enable_watermark = true,
        int seed = -1,
        std::function<void(int, int)> progress_callback = nullptr);
    
    // Convenience overloads for simple generation
    virtual std::vector<float> generate_speech(
        const std::string& text, 
        int speaker_id, 
        float temperature,
        int top_k = 50) {
        GenerationOptions options;
        options.temperature = temperature;
        options.top_k = top_k;
        options.update_sampling();
        return generate_speech(text, speaker_id, {}, options);
    }
    
    // Configuration methods
    void set_default_temperature(float temperature);
    float default_temperature() const;
    
    void set_default_top_k(int top_k);
    int default_top_k() const;
    
    // Advanced sampling configuration methods
    void set_repetition_penalty(float penalty);
    float repetition_penalty() const;
    
    void set_frequency_penalty(float penalty);
    float frequency_penalty() const;
    
    void set_presence_penalty(float penalty);
    float presence_penalty() const;
    
    void set_top_p(float p);
    float top_p() const;
    
    void set_logit_bias(std::unordered_map<int, float> bias);
    const std::unordered_map<int, float>& logit_bias() const;
    void add_logit_bias(int token_id, float bias);
    void clear_logit_biases();
    
    // General configuration
    void set_enable_watermarking(bool enable);
    bool is_watermarking_enabled() const;
    
    void set_max_text_tokens(int max_tokens);
    int max_text_tokens() const;
    
    void set_memory_optimization(bool enable, size_t max_memory_mb = 0, 
                                int trigger_mb = 0, float prune_factor = 0.5f);
    
    // Advanced context management methods
    
    // Enable or disable advanced context management
    void set_enable_advanced_context(bool enable);
    bool is_advanced_context_enabled() const;
    
    // Get the context manager (creates one if not exists)
    std::shared_ptr<ContextManager> context_manager();
    
    // Generate speech using the advanced context manager
    virtual std::vector<float> generate_speech_with_context_manager(
        const std::string& text,
        int speaker_id = -1,
        const GenerationOptions& options = {},
        std::function<void(int, int)> progress_callback = nullptr);
    
    // Getters
    int sample_rate() const;
    std::shared_ptr<Model> model() const;
    std::shared_ptr<TextTokenizer> text_tokenizer() const;
    std::shared_ptr<AudioCodec> audio_codec() const;
    std::shared_ptr<Watermarker> watermarker() const;
    
protected:
    // Apply sampling parameters to logits
    std::vector<float> apply_sampling(
        const std::vector<float>& logits,
        const std::vector<int>& tokens,
        const SamplingOptions& options);
    
    // Sample from logits to get next token
    int sample_token(const std::vector<float>& logits, float temperature, int top_k, float top_p, std::mt19937& rng);
    
    std::shared_ptr<Model> model_;
    std::shared_ptr<TextTokenizer> text_tokenizer_;
    std::shared_ptr<AudioCodec> audio_codec_;
    std::shared_ptr<Watermarker> watermarker_;
    
    // Advanced context management
    std::shared_ptr<ContextManager> context_manager_;
    bool enable_advanced_context_ = false;
    
    // Default parameters
    float default_temperature_ = 0.9f;
    int default_top_k_ = 50;
    bool enable_watermarking_ = true;
    int max_text_tokens_ = 2048;
    int sample_rate_ = 24000;
    
    // Advanced sampling parameters
    float repetition_penalty_ = 1.0f;
    float frequency_penalty_ = 0.0f;
    float presence_penalty_ = 0.0f;
    float top_p_ = 1.0f;
    std::unordered_map<int, float> logit_bias_;
    
    // Memory optimization
    bool memory_optimization_enabled_ = false;
    size_t max_memory_mb_ = 0;
    int memory_trigger_mb_ = 0;
    float prune_factor_ = 0.5f;
};

// Factory function to create a generator with the CSM-1B model
std::shared_ptr<Generator> load_csm_1b(const std::string& device = "cpu");

// Factory function for MLX-accelerated generator
std::shared_ptr<Generator> load_csm_1b_mlx();

} // namespace ccsm

#endif // CCSM_GENERATOR_H