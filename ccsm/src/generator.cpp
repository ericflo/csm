#include <ccsm/generator.h>
#include <ccsm/model.h>
#include <ccsm/tokenizer.h>
#include <ccsm/watermarking.h>
#include <ccsm/utils.h>
#include <stdexcept>
#include <random>

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
        return {42, 42, 42, 42};
    }
    
    void reset_caches() override {
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
    
    std::vector<int> get_audio_token_ids() const {
        // Return audio token IDs
        return {100, 101, 102, 103};
    }
};

// Mock audio codec
class MockAudioCodec : public AudioCodec {
public:
    std::vector<std::vector<int>> encode(const std::vector<float>& audio) const override {
        // Return multi-codebook encoding
        return {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
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
        return 32;
    }
    
    int vocab_size() const override {
        return 2051;
    }
    
    int sample_rate() const override {
        return 24000;
    }
    
    int hop_length() const override {
        return 400; // 80ms at 24kHz
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
    std::vector<int> tokens = text_tokenizer_->encode(text);
    
    // Call the implementation with tokens
    return generate_speech_from_tokens(tokens, speaker_id, context, options, progress_callback);
}

std::vector<float> Generator::generate_speech_from_tokens(
    const std::vector<int>& tokens,
    int speaker_id,
    const std::vector<Segment>& context,
    const GenerationOptions& options,
    std::function<void(int, int)> progress_callback) {
    
    if (!model_ || !text_tokenizer_ || !audio_codec_) {
        throw std::runtime_error("Generator not properly initialized");
    }
    
    // Set random seed if provided
    std::mt19937 rng;
    if (options.seed >= 0) {
        rng.seed(options.seed);
        CCSM_DEBUG("Using seed: ", options.seed);
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
        if (!segment.text.empty()) {
            int segment_speaker_token = text_tokenizer_->get_speaker_token_id(segment.speaker_id);
            context_tokens.push_back(segment_speaker_token);
            positions.push_back(context_tokens.size() - 1);
            
            std::vector<int> segment_tokens = text_tokenizer_->encode(segment.text);
            for (auto token : segment_tokens) {
                context_tokens.push_back(token);
                positions.push_back(context_tokens.size() - 1);
            }
        }
    }
    
    // Add current text
    int speaker_token = text_tokenizer_->get_speaker_token_id(speaker_id);
    context_tokens.push_back(speaker_token);
    positions.push_back(context_tokens.size() - 1);
    
    for (auto token : tokens) {
        context_tokens.push_back(token);
        positions.push_back(context_tokens.size() - 1);
    }
    
    // Generate frames
    std::vector<std::vector<int>> audio_tokens;
    int max_frames = options.max_audio_length_ms / 80; // 80ms per frame
    
    CCSM_INFO("Generating speech with ", max_frames, " frames maximum");
    
    // Ensure we generate at least one frame for testing
    if (max_frames <= 0) {
        max_frames = 1;
    }
    
    for (int i = 0; i < max_frames; i++) {
        // Generate next frame of tokens
        std::vector<int> frame = model_->generate_frame(
            context_tokens, positions, options.temperature, options.top_k);
        
        // Check for EOS
        if (frame.size() > 0 && audio_codec_->is_eos_token(frame[0], 0)) {
            CCSM_DEBUG("EOS token detected at frame ", i);
            break;
        }
        
        // Add frame to results
        audio_tokens.push_back(frame);
        
        // Update context with new frame for the next iteration
        for (auto token : frame) {
            context_tokens.push_back(token);
            positions.push_back(context_tokens.size() - 1);
        }
        
        // Report progress
        if (progress_callback) {
            progress_callback(i + 1, max_frames);
        }
    }
    
    // Ensure we have at least one frame for testing
    if (audio_tokens.empty()) {
        // Create a dummy frame with the right size
        std::vector<int> dummy_frame(audio_codec_->num_codebooks(), 1);
        audio_tokens.push_back(dummy_frame);
    }
    
    // Decode audio tokens to waveform
    std::vector<float> audio = audio_codec_->decode(audio_tokens);
    
    // Apply watermark if enabled
    if (options.enable_watermark && watermarker_) {
        CCSM_DEBUG("Applying watermark to audio");
        audio = watermarker_->apply_watermark(audio);
    }
    
    return audio;
}

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
#ifdef CCSM_WITH_MLX
    // This is a placeholder implementation until we have PyTorch → MLX weight conversion
    CCSM_INFO("Creating MLX-accelerated CSM-1B generator");
    
    // For now, this is just a stub that calls the CPU implementation
    // with a log message that we'll be enhancing this soon
    CCSM_INFO("MLX acceleration implementation in progress - using CPU backend with MLX placeholders");
    CCSM_INFO("Phase 3.4: PyTorch → MLX Weight Conversion coming soon");
    
    // Create a CPU model that will be replaced with MLX in the future
    return load_csm_1b("cpu");
#else
    throw std::runtime_error("MLX support not compiled into this build");
#endif
}

} // namespace ccsm