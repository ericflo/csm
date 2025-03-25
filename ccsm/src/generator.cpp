#include <ccsm/generator.h>
#include <ccsm/model.h>
#include <ccsm/tokenizer.h>
#include <ccsm/watermarking.h>
#include <ccsm/utils.h>
#include <stdexcept>
#include <random>
#include <algorithm>
#include <chrono>
#include <cmath>

// Mock implementations for testing and building
namespace ccsm {

// Forward declarations for the mocks
class MockModel;
class MockTextTokenizer;
class MockAudioCodec;
class MockWatermarker;

// Mock model implementation
class MockModel : public Model {
public:
    MockModel() : Model(ModelConfig()) {
        config_.name = "Mock CSM-1B";
        config_.vocab_size = 32000;
        config_.audio_vocab_size = 2051;
        config_.num_codebooks = 8;
    }
    
    bool load_weights(const std::string& path) override {
        return true; // Mock success
    }
    
    bool load_weights(std::shared_ptr<ModelLoader> loader) override {
        return true; // Mock success
    }
    
    bool load_weights(const WeightMap& weights) override {
        return true; // Mock success
    }
    
    std::vector<int> generate_frame(
        const std::vector<int>& tokens,
        const std::vector<int>& positions,
        float temperature = 0.9f,
        int top_k = 50) override {
        // Return a simple sequence of tokens to simulate a frame
        return {42, 42, 42, 42, 42, 42, 42, 42};
    }
    
    void reset_caches() override {
        // No-op for mock
    }
    
    void optimize_memory(size_t max_memory_mb = 0) override {
        // No-op for mock
    }
    
    void prune_caches(float prune_factor = 0.5f) override {
        // No-op for mock
    }
    
    std::vector<float> get_backbone_logits(
        const std::vector<int>& tokens,
        const std::vector<int>& positions) override {
        // Return dummy logits
        return std::vector<float>(config_.vocab_size, 0.0f);
    }
    
    std::vector<float> get_decoder_logits(
        const std::vector<int>& tokens,
        const std::vector<int>& positions,
        int codebook) override {
        // Return dummy logits
        return std::vector<float>(config_.audio_vocab_size, 0.0f);
    }
};

// Mock text tokenizer
class MockTextTokenizer : public TextTokenizer {
public:
    std::vector<int> encode(const std::string& text) const override {
        // Return a simple token sequence
        return {1, 2, 3, 4, 5};
    }
    
    std::string decode(const std::vector<int>& tokens) const override {
        return "Mock decoded text";
    }
    
    int vocab_size() const override {
        return 32000;
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
        return 42 + speaker_id;
    }
    
    std::vector<int> get_audio_token_ids() const override {
        // Return audio token IDs
        return {100, 101, 102, 103};
    }
};

// Mock audio codec
class MockAudioCodec : public AudioCodec {
public:
    std::vector<std::vector<int>> encode(const std::vector<float>& audio) const override {
        // Return multi-codebook encoding
        return {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12},
                {13, 14, 15}, {16, 17, 18}, {19, 20, 21}, {22, 23, 24}};
    }
    
    std::vector<float> decode(const std::vector<std::vector<int>>& tokens) const override {
        // Return a short sine wave
        std::vector<float> audio(24000);
        for (size_t i = 0; i < audio.size(); i++) {
            audio[i] = 0.5f * sin(2.0f * 3.14159f * 440.0f * i / 24000.0f);
        }
        return audio;
    }
    
    int num_codebooks() const override {
        return 8;
    }
    
    int vocab_size() const override {
        return 2051;
    }
    
    int sample_rate() const override {
        return 24000;
    }
    
    int hop_length() const override {
        return 400; // 60ms at 24kHz
    }
    
    bool is_eos_token(int token, int codebook) const override {
        return token == 0;
    }
};

// Mock watermarker
class MockWatermarker : public Watermarker {
public:
    std::vector<float> apply_watermark(const std::vector<float>& audio) override {
        // Just return the same audio
        return audio;
    }
    
    bool detect_watermark(const std::vector<float>& audio) override {
        // Always detect watermark in mock
        return true;
    }
    
    WatermarkResult detect(const std::vector<float>& audio, float sample_rate) override {
        // Mock implementation - always detect with default payload
        WatermarkResult result;
        result.detected = true;
        result.payload = "mock-watermark";
        result.confidence = 0.95f;
        return result;
    }
    
    float get_strength() const override {
        return 0.5f;
    }
    
    void set_strength(float strength) override {
        // No-op for mock
    }
    
    std::string get_key() const override {
        return "mock-key";
    }
};

} // namespace ccsm

namespace ccsm {

// Generator implementation
Generator::Generator(
    std::shared_ptr<Model> model,
    std::shared_ptr<TextTokenizer> text_tokenizer,
    std::shared_ptr<AudioCodec> audio_codec,
    std::shared_ptr<Watermarker> watermarker)
    : watermarker_(watermarker),
      sample_rate_(24000) {
    
    // Validate required components
    if (!model) {
        throw std::invalid_argument("Model cannot be null");
    }
    if (!text_tokenizer) {
        throw std::invalid_argument("Text tokenizer cannot be null");
    }
    if (!audio_codec) {
        throw std::invalid_argument("Audio codec cannot be null");
    }
    
    // Set components after validation
    model_ = model;
    text_tokenizer_ = text_tokenizer;
    audio_codec_ = audio_codec;
    sample_rate_ = audio_codec->sample_rate();
    
    // Log initialization
    CCSM_INFO("Initializing CSM Generator with model: ", model_->config().name);
}

/**
 * Main speech generation method
 */
std::vector<float> Generator::generate_speech(
    const std::string& text,
    int speaker_id,
    const std::vector<Segment>& context,
    const GenerationOptions& options,
    std::function<void(int, int)> progress_callback) {
    
    if (!model_ || !text_tokenizer_ || !audio_codec_) {
        throw std::runtime_error("Generator not properly initialized");
    }
    
    // Tokenize text
    std::vector<int> tokens;
    try {
        tokens = text_tokenizer_->encode(text);
        CCSM_DEBUG("Encoded text to ", tokens.size(), " tokens");
    } catch (const std::exception& e) {
        CCSM_ERROR("Failed to tokenize text: ", e.what());
        throw; // Re-throw the exception
    }
    
    // Call the implementation with tokens
    return generate_speech_from_tokens(tokens, speaker_id, context, options, progress_callback);
}

/**
 * Generate speech from pre-tokenized input
 */
std::vector<float> Generator::generate_speech_from_tokens(
    const std::vector<int>& tokens,
    int speaker_id,
    const std::vector<Segment>& context,
    const GenerationOptions& options,
    std::function<void(int, int)> progress_callback) {
    
    if (!model_ || !text_tokenizer_ || !audio_codec_) {
        throw std::runtime_error("Generator not properly initialized");
    }
    
    // Create a copy of the options for potential modifications
    GenerationOptions working_options = options;
    
    // Update sampling options to match main options for backward compatibility
    working_options.update_sampling();
    
    // Clamp temperature to reasonable range
    working_options.temperature = std::max(0.05f, std::min(working_options.temperature, 1.5f));
    working_options.sampling.temperature = working_options.temperature;
    
    // Ensure top_k is at least 1
    working_options.top_k = std::max(1, working_options.top_k);
    working_options.sampling.top_k = working_options.top_k;
    
    // Apply default advanced sampling parameters if they're at default values
    if (working_options.sampling.repetition_penalty == 1.0f && repetition_penalty_ != 1.0f) {
        working_options.sampling.repetition_penalty = repetition_penalty_;
    }
    
    if (working_options.sampling.frequency_penalty == 0.0f && frequency_penalty_ != 0.0f) {
        working_options.sampling.frequency_penalty = frequency_penalty_;
    }
    
    if (working_options.sampling.presence_penalty == 0.0f && presence_penalty_ != 0.0f) {
        working_options.sampling.presence_penalty = presence_penalty_;
    }
    
    if (working_options.sampling.top_p == 1.0f && top_p_ != 1.0f) {
        working_options.sampling.top_p = top_p_;
    }
    
    // Merge global and per-call logit biases
    if (!logit_bias_.empty() && working_options.sampling.logit_bias.empty()) {
        working_options.sampling.logit_bias = logit_bias_;
    }
    
    // Set random seed if provided
    std::mt19937 rng;
    if (working_options.seed >= 0) {
        rng.seed(static_cast<unsigned int>(working_options.seed));
        CCSM_DEBUG("Using seed: ", working_options.seed);
    } else {
        std::random_device rd;
        rng.seed(rd());
        CCSM_DEBUG("Using random seed");
    }
    
    // Reset model caches
    model_->reset_caches();
    
    // Process context and current text into token sequence
    std::vector<int> context_tokens;
    std::vector<int> positions;
    
    // Process context segments and build token sequence
    for (const auto& segment : context) {
        // Add the speaker token
        if (segment.speaker_id >= 0) {
            int segment_speaker_token = text_tokenizer_->get_speaker_token_id(segment.speaker_id);
            context_tokens.push_back(segment_speaker_token);
            positions.push_back(static_cast<int>(context_tokens.size() - 1));
        }
        
        // Add the text tokens
        if (!segment.text.empty()) {
            std::vector<int> segment_tokens = text_tokenizer_->encode(segment.text);
            for (auto token : segment_tokens) {
                context_tokens.push_back(token);
                positions.push_back(static_cast<int>(context_tokens.size() - 1));
            }
        }
    }
    
    // Add current speaker token if specified
    if (speaker_id >= 0) {
        int speaker_token = text_tokenizer_->get_speaker_token_id(speaker_id);
        context_tokens.push_back(speaker_token);
        positions.push_back(static_cast<int>(context_tokens.size() - 1));
    }
    
    // Add the main tokens
    for (auto token : tokens) {
        context_tokens.push_back(token);
        positions.push_back(static_cast<int>(context_tokens.size() - 1));
    }
    
    // Limit token sequence length if necessary
    const int max_tokens = max_text_tokens_;
    if (context_tokens.size() > max_tokens) {
        CCSM_WARNING("Limiting token sequence from ", context_tokens.size(), " to ", max_tokens, " tokens");
        context_tokens.resize(max_tokens);
        positions.resize(max_tokens);
    }
    
    // Calculate max frames based on audio length
    int frames_per_second = 1000 / 80; // 80ms per frame = 12.5 frames per second
    int max_frames = working_options.max_audio_length_ms * frames_per_second / 1000;
    int min_frames = working_options.min_audio_length_ms * frames_per_second / 1000;
    
    // Ensure we have at least one frame
    max_frames = std::max(1, max_frames);
    min_frames = std::max(0, min_frames);
    CCSM_INFO("Generating speech with ", min_frames, " to ", max_frames, " frames");
    
    // Generate frames
    std::vector<std::vector<int>> frames;
    
    for (int frame_idx = 0; frame_idx < max_frames; frame_idx++) {
        // Generate next frame of tokens
        // If advanced sampling parameters are all at default values, use the basic generate_frame method
        // This provides backward compatibility
        std::vector<int> frame;
        bool use_basic_sampling = 
            working_options.sampling.repetition_penalty == 1.0f &&
            working_options.sampling.frequency_penalty == 0.0f &&
            working_options.sampling.presence_penalty == 0.0f &&
            working_options.sampling.top_p == 1.0f &&
            working_options.sampling.logit_bias.empty();
        
        if (use_basic_sampling) {
            frame = model_->generate_frame(
                context_tokens, positions, working_options.temperature, working_options.top_k);
        } else {
            // Advanced sampling - use specialized method in the future when model interface is updated
            // For now, we'll continue to use the basic generate_frame
            // This will be replaced with a more advanced implementation in the future
            if (working_options.sampling.greedy || working_options.sampling.temperature <= 0.05f) {
                // Greedy sampling - use very low temperature
                frame = model_->generate_frame(
                    context_tokens, positions, 0.01f, 1);
            } else {
                frame = model_->generate_frame(
                    context_tokens, positions, working_options.sampling.temperature, working_options.sampling.top_k);
            }
        }
        
        // Check if the frame is valid
        if (frame.empty()) {
            CCSM_ERROR("Empty frame generated at index ", frame_idx);
            throw std::runtime_error("Empty frame generated by model");
        }
        
        // Check for end of sequence token
        bool eos_detected = false;
        for (size_t i = 0; i < frame.size(); i++) {
            if (audio_codec_->is_eos_token(frame[i], static_cast<int>(i))) {
                CCSM_DEBUG("EOS token detected at frame ", frame_idx);
                eos_detected = true;
                break;
            }
        }
        
        // Add the frame to our results
        frames.push_back(frame);
        
        // Check if we've generated enough frames and EOS was detected
        if (eos_detected && frame_idx >= min_frames) {
            CCSM_DEBUG("Stopping generation after EOS and minimum frames reached");
            break;
        } else if (eos_detected) {
            // EOS detected but we haven't reached minimum frames
            CCSM_DEBUG("EOS token detected but continuing to minimum frame count");
        }
        
        // Update context with the new frame for the next iteration
        for (auto token : frame) {
            context_tokens.push_back(token);
            positions.push_back(static_cast<int>(context_tokens.size() - 1));
        }
        
        // Report progress
        if (progress_callback) {
            progress_callback(frame_idx + 1, max_frames);
            // No cancellation in this implementation
        }
    }
    
    // No cancellation in this implementation
    
    // Ensure we have at least one frame
    if (frames.empty()) {
        CCSM_WARNING("No frames generated, creating dummy frame");
        std::vector<int> dummy_frame(audio_codec_->num_codebooks(), 1);
        frames.push_back(dummy_frame);
    }
    
    // Organize frames in the format expected by the audio codec
    std::vector<std::vector<int>> codec_tokens;
    codec_tokens.resize(audio_codec_->num_codebooks());
    
    // Transpose the frames to codec format (grouped by codebook)
    for (size_t cb = 0; cb < audio_codec_->num_codebooks(); cb++) {
        for (const auto& frame : frames) {
            if (cb < frame.size()) {
                codec_tokens[cb].push_back(frame[cb]);
            } else {
                // Pad with a default token if frame doesn't have enough codebooks
                codec_tokens[cb].push_back(1); // Use token 1 (avoid 0 which is EOS)
            }
        }
    }
    
    // Decode audio tokens to waveform
    std::vector<float> audio;
    try {
        audio = audio_codec_->decode(codec_tokens);
        CCSM_DEBUG("Decoded ", frames.size(), " frames to ", audio.size(), " audio samples");
    } catch (const std::exception& e) {
        CCSM_ERROR("Failed to decode audio: ", e.what());
        throw; // Re-throw the exception
    }
    
    // Apply watermark if enabled and watermarker is available
    if (working_options.enable_watermark && watermarker_) {
        CCSM_DEBUG("Applying watermark to audio");
        try {
            audio = watermarker_->apply_watermark(audio);
        } catch (const std::exception& e) {
            CCSM_ERROR("Failed to apply watermark: ", e.what());
            // Continue with unwatermarked audio rather than failing
        }
    }
    
    // Create a struct with all the necessary information
    GenerationResult result;
    result.audio = audio;
    result.frame = frames.empty() ? std::vector<int>() : frames.front();
    result.frames = frames;
    
    return result.audio;
}

// Getter methods
int Generator::sample_rate() const {
    return sample_rate_;
}

std::shared_ptr<Model> Generator::model() const {
    return model_;
}

std::shared_ptr<TextTokenizer> Generator::text_tokenizer() const {
    return text_tokenizer_;
}

std::shared_ptr<AudioCodec> Generator::audio_codec() const {
    return audio_codec_;
}

std::shared_ptr<Watermarker> Generator::watermarker() const {
    return watermarker_;
}

// Advanced sampling implementation
std::vector<float> Generator::generate_speech_with_sampling(
    const std::string& text,
    int speaker_id,
    const std::vector<Segment>& context,
    const SamplingOptions& sampling_options,
    int max_audio_length_ms,
    bool enable_watermark,
    int seed,
    std::function<void(int, int)> progress_callback) {
    
    // Create generation options with the sampling parameters
    GenerationOptions options;
    options.sampling = sampling_options;
    options.temperature = sampling_options.temperature;
    options.top_k = sampling_options.top_k;
    options.max_audio_length_ms = max_audio_length_ms;
    options.enable_watermark = enable_watermark;
    options.seed = seed;
    
    // Call the main generation function
    return generate_speech(text, speaker_id, context, options, progress_callback);
}

// Apply sampling parameters to logits
std::vector<float> Generator::apply_sampling(
    const std::vector<float>& logits,
    const std::vector<int>& tokens,
    const SamplingOptions& options) {
    
    std::vector<float> processed_logits = logits;
    
    // Apply logit bias if specified
    if (!options.logit_bias.empty()) {
        for (const auto& [token, bias] : options.logit_bias) {
            if (token >= 0 && token < static_cast<int>(processed_logits.size())) {
                processed_logits[token] += bias;
            }
        }
    }
    
    // Apply repetition penalty if enabled
    if (options.repetition_penalty != 1.0f) {
        // Count token frequencies
        std::unordered_map<int, int> token_counts;
        for (int token : tokens) {
            token_counts[token]++;
        }
        
        // Apply penalty to repeated tokens
        for (const auto& [token, count] : token_counts) {
            if (token < static_cast<int>(processed_logits.size())) {
                // Apply penalty based on token's current logit value
                if (processed_logits[token] > 0) {
                    processed_logits[token] /= options.repetition_penalty;
                } else {
                    processed_logits[token] *= options.repetition_penalty;
                }
            }
        }
    }
    
    // Apply frequency penalty if enabled
    if (options.frequency_penalty != 0.0f) {
        // Count token frequencies
        std::unordered_map<int, int> token_counts;
        for (int token : tokens) {
            token_counts[token]++;
        }
        
        // Apply frequency penalty based on counts
        for (const auto& [token, count] : token_counts) {
            if (token < static_cast<int>(processed_logits.size())) {
                processed_logits[token] -= options.frequency_penalty * count;
            }
        }
    }
    
    // Apply presence penalty if enabled
    if (options.presence_penalty != 0.0f) {
        // Track unique tokens
        std::unordered_set<int> unique_tokens;
        for (int token : tokens) {
            unique_tokens.insert(token);
        }
        
        // Apply presence penalty
        for (int token : unique_tokens) {
            if (token < static_cast<int>(processed_logits.size())) {
                processed_logits[token] -= options.presence_penalty;
            }
        }
    }
    
    return processed_logits;
}

// Sample token from logits
int Generator::sample_token(
    const std::vector<float>& logits,
    float temperature,
    int top_k,
    float top_p,
    std::mt19937& rng) {
    
    // For greedy sampling (temperature = 0)
    if (temperature <= 0.01f) {
        // Find the max logit and return its index
        auto max_it = std::max_element(logits.begin(), logits.end());
        return max_it != logits.end() ? static_cast<int>(std::distance(logits.begin(), max_it)) : 0;
    }
    
    // Make a copy of logits that we'll modify
    std::vector<float> working_logits = logits;
    
    // Apply temperature to sharpen/soften the distribution
    for (auto& logit : working_logits) {
        logit /= temperature;
    }
    
    // Apply top-k filtering if k > 0 and k < vocabulary size
    if (top_k > 0 && top_k < static_cast<int>(working_logits.size())) {
        // Create a copy of logits for sorting
        std::vector<float> sorted_logits = working_logits;
        std::nth_element(sorted_logits.begin(), sorted_logits.begin() + top_k - 1, sorted_logits.end(), 
                        std::greater<float>());
        float threshold = sorted_logits[top_k - 1];
        
        // Zero out logits below threshold
        for (auto& logit : working_logits) {
            if (logit < threshold) {
                logit = -std::numeric_limits<float>::infinity();
            }
        }
    }
    
    // Apply nucleus (top-p) sampling if p < 1.0
    if (top_p < 1.0f) {
        // Sort by probability (after applying softmax)
        std::vector<std::pair<float, int>> sorted_probs;
        float max_logit = *std::max_element(working_logits.begin(), working_logits.end());
        float sum = 0.0f;
        
        // Apply softmax
        for (size_t i = 0; i < working_logits.size(); i++) {
            float prob = std::exp(working_logits[i] - max_logit);
            sum += prob;
            sorted_probs.push_back({prob, static_cast<int>(i)});
        }
        
        // Normalize
        for (auto& pair : sorted_probs) {
            pair.first /= sum;
        }
        
        // Sort by probability in descending order
        std::sort(sorted_probs.begin(), sorted_probs.end(), 
                 [](const auto& a, const auto& b) { return a.first > b.first; });
        
        // Zero out tokens outside the nucleus
        float cumulative_prob = 0.0f;
        for (const auto& [prob, index] : sorted_probs) {
            cumulative_prob += prob;
            if (cumulative_prob > top_p) {
                working_logits[index] = -std::numeric_limits<float>::infinity();
            }
        }
    }
    
    // Apply softmax to get probabilities
    float max_logit = *std::max_element(working_logits.begin(), working_logits.end());
    float sum = 0.0f;
    
    for (auto& logit : working_logits) {
        logit = std::exp(logit - max_logit);
        sum += logit;
    }
    
    // Normalize
    if (sum > 0.0f) {
        for (auto& logit : working_logits) {
            logit /= sum;
        }
    } else {
        // Fallback to uniform if sum is 0
        float uniform_prob = 1.0f / working_logits.size();
        std::fill(working_logits.begin(), working_logits.end(), uniform_prob);
    }
    
    // Sample from the distribution
    std::discrete_distribution<int> dist(working_logits.begin(), working_logits.end());
    return dist(rng);
}

// Configuration methods
void Generator::set_default_temperature(float temperature) {
    default_temperature_ = std::max(0.05f, std::min(temperature, 1.5f));
}

float Generator::default_temperature() const {
    return default_temperature_;
}

void Generator::set_default_top_k(int top_k) {
    default_top_k_ = std::max(1, top_k);
}

int Generator::default_top_k() const {
    return default_top_k_;
}

// Advanced sampling configuration methods
void Generator::set_repetition_penalty(float penalty) {
    repetition_penalty_ = std::max(1.0f, penalty);
}

float Generator::repetition_penalty() const {
    return repetition_penalty_;
}

void Generator::set_frequency_penalty(float penalty) {
    frequency_penalty_ = std::max(0.0f, std::min(penalty, 2.0f));
}

float Generator::frequency_penalty() const {
    return frequency_penalty_;
}

void Generator::set_presence_penalty(float penalty) {
    presence_penalty_ = std::max(0.0f, std::min(penalty, 2.0f));
}

float Generator::presence_penalty() const {
    return presence_penalty_;
}

void Generator::set_top_p(float p) {
    top_p_ = std::max(0.0f, std::min(p, 1.0f));
}

float Generator::top_p() const {
    return top_p_;
}

void Generator::set_logit_bias(std::unordered_map<int, float> bias) {
    logit_bias_ = std::move(bias);
}

const std::unordered_map<int, float>& Generator::logit_bias() const {
    return logit_bias_;
}

void Generator::add_logit_bias(int token_id, float bias) {
    logit_bias_[token_id] = bias;
}

void Generator::clear_logit_biases() {
    logit_bias_.clear();
}

void Generator::set_enable_watermarking(bool enable) {
    enable_watermarking_ = enable;
}

bool Generator::is_watermarking_enabled() const {
    return enable_watermarking_;
}

void Generator::set_max_text_tokens(int max_tokens) {
    max_text_tokens_ = std::max(1, max_tokens);
}

int Generator::max_text_tokens() const {
    return max_text_tokens_;
}

void Generator::set_memory_optimization(bool enable, size_t max_memory_mb, 
                                      int trigger_mb, float prune_factor) {
    memory_optimization_enabled_ = enable;
    max_memory_mb_ = max_memory_mb;
    memory_trigger_mb_ = trigger_mb;
    prune_factor_ = prune_factor;
    
    // Apply immediately if enabled
    if (enable && model_) {
        model_->optimize_memory(max_memory_mb);
        if (memory_trigger_mb_ > 0 && prune_factor_ > 0.0f) {
            model_->prune_caches(prune_factor_);
        }
    }
}

// Factory function to create a generator with the CSM-1B model
std::shared_ptr<Generator> load_csm_1b(const std::string& device) {
    CCSM_INFO("Creating CSM-1B generator for device: ", device);
    
    // This is a placeholder implementation that will be completed in Phase 4
    // Create a mock model, tokenizer, and audio codec that don't do anything yet
    auto model = std::make_shared<MockModel>();
    auto tokenizer = std::make_shared<MockTextTokenizer>();
    auto codec = std::make_shared<MockAudioCodec>();
    auto watermarker = std::make_shared<MockWatermarker>();
    
    CCSM_INFO("Created mock components for CPU generator");
    CCSM_INFO("Phase 4: Model Generation Pipeline coming soon");
    
    return std::make_shared<Generator>(model, tokenizer, codec, watermarker);
}

// Factory function for MLX-accelerated generator
std::shared_ptr<Generator> load_csm_1b_mlx() {
    CCSM_INFO("Creating MLX-accelerated CSM-1B generator");
    
    #ifdef CCSM_WITH_MLX
    // Check if MLX is available on this system
    if (!is_mlx_available()) {
        CCSM_WARNING("MLX is not available on this system, falling back to CPU generator");
        return load_csm_1b("cpu");
    }
    
    // Create model paths (these would be actual paths in a real implementation)
    std::string model_path = "models/csm-1b.pt";
    std::string tokenizer_path = "models/csm-1b-tokenizer.model";
    std::string audio_codec_path = "models/csm-1b-codec.bin";
    std::string watermarker_path = "models/csm-1b-watermarker.bin";
    
    // Create MLX optimization config
    MLXOptimizationConfig config = MLXOptimizationConfig::from_env();
    
    // Create MLX generator
    auto generator = create_mlx_generator(
        model_path, tokenizer_path, audio_codec_path, watermarker_path, config);
    
    // If MLX generator creation failed, fall back to CPU
    if (!generator) {
        CCSM_WARNING("Failed to create MLX generator, falling back to CPU generator");
        return load_csm_1b("cpu");
    }
    
    return generator;
    #else
    CCSM_WARNING("CCSM was not compiled with MLX support, falling back to CPU generator");
    return load_csm_1b("cpu");
    #endif
}

} // namespace ccsm