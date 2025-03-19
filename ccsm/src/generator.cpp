#include <ccsm/generator.h>
#include <ccsm/model.h>
#include <ccsm/tokenizer.h>
#include <ccsm/watermarking.h>
#include <ccsm/utils.h>
#include <stdexcept>
#include <random>

namespace ccsm {

// Generator implementation
Generator::Generator(
    std::shared_ptr<Model> model,
    std::shared_ptr<TextTokenizer> text_tokenizer,
    std::shared_ptr<AudioCodec> audio_codec,
    std::shared_ptr<Watermarker> watermarker)
    : model_(model),
      text_tokenizer_(text_tokenizer),
      audio_codec_(audio_codec),
      watermarker_(watermarker),
      sample_rate_(audio_codec ? audio_codec->sample_rate() : 24000) {
          
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
    
    // TODO: Process context segments and build token sequence
    
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
        
        // TODO: Update context with new frame
        
        // Report progress
        if (progress_callback) {
            progress_callback(i + 1, max_frames);
        }
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
    // TODO: Create model and tokenizers, then return a fully initialized generator
    throw std::runtime_error("load_csm_1b not implemented yet");
}

// Factory function for MLX-accelerated generator
std::shared_ptr<Generator> load_csm_1b_mlx() {
#ifdef CCSM_WITH_MLX
    // TODO: Create MLX-specific model and return a fully initialized generator
    throw std::runtime_error("load_csm_1b_mlx not implemented yet");
#else
    throw std::runtime_error("MLX support not compiled into this build");
#endif
}

} // namespace ccsm