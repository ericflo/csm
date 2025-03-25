#include <ccsm/mlx/mlx_generator.h>
#include <ccsm/mlx/mlx_model.h>
#include <ccsm/mlx/mlx_optimizations.h>
#include <ccsm/mlx/mlx_weight_converter.h>
#include <ccsm/model_loader.h>
#include <ccsm/utils.h>
#include <algorithm>
#include <chrono>
#include <random>
#include <stdexcept>

namespace ccsm {

// Constructor
MLXGenerator::MLXGenerator(
    std::shared_ptr<Model> model,
    std::shared_ptr<TextTokenizer> text_tokenizer,
    std::shared_ptr<AudioCodec> audio_codec,
    std::shared_ptr<Watermarker> watermarker,
    const MLXOptimizationConfig& optimization_config)
    : Generator(model, text_tokenizer, audio_codec, watermarker),
      optimization_config_(optimization_config),
      mlx_available_(false) {
    
    CCSM_INFO("Initializing MLX Generator");
    
    // Check if the model is an MLXModel
    mlx_model_ = std::dynamic_pointer_cast<MLXModel>(model);
    if (!mlx_model_) {
        CCSM_WARNING("Model is not an MLXModel, MLX acceleration will not be used");
    } else {
        // Check if MLX is available
        #ifdef CCSM_WITH_MLX
        mlx_available_ = MLXWeightConverter::is_mlx_available();
        CCSM_INFO("MLX acceleration is ", mlx_available_ ? "available" : "not available");
        
        // Configure MLX for the device
        if (mlx_available_) {
            configure_mlx_for_device(optimization_config_);
            CCSM_INFO("Configured MLX for the device with compute precision: ",
                     optimization_config_.compute_precision == MLXOptimizationConfig::ComputePrecision::FLOAT32 ? "FLOAT32" :
                     optimization_config_.compute_precision == MLXOptimizationConfig::ComputePrecision::BFLOAT16 ? "BFLOAT16" : "FLOAT16");
        }
        #else
        CCSM_WARNING("CCSM was not compiled with MLX support");
        #endif
    }
}

// Destructor
MLXGenerator::~MLXGenerator() {
    CCSM_INFO("Destroying MLX Generator");
}

// Override generate_speech_from_tokens to use MLX optimizations
std::vector<float> MLXGenerator::generate_speech_from_tokens(
    const std::vector<int>& tokens,
    int speaker_id,
    const std::vector<Segment>& context,
    const GenerationOptions& options,
    std::function<void(int, int)> progress_callback) {
    
    // If MLX is not available or the model is not MLXModel, fall back to the base implementation
    if (!mlx_available_ || !mlx_model_) {
        CCSM_INFO("Falling back to base Generator implementation");
        return Generator::generate_speech_from_tokens(tokens, speaker_id, context, options, progress_callback);
    }
    
    CCSM_INFO("Generating speech with MLX acceleration");
    
    // Create a copy of the options for potential modifications
    GenerationOptions working_options = options;
    
    // Clamp temperature to reasonable range
    working_options.temperature = std::max(0.05f, std::min(working_options.temperature, 1.5f));
    
    // Ensure top_k is at least 1
    working_options.top_k = std::max(1, working_options.top_k);
    
    // Set random seed if provided
    if (working_options.seed >= 0) {
        CCSM_DEBUG("Using seed: ", working_options.seed);
    } else {
        CCSM_DEBUG("Using random seed");
    }
    
    // Reset model caches
    model_->reset_caches();
    
    // Optimize memory for generation
    optimize_for_generation();
    
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
    if (context_tokens.size() > max_text_tokens_) {
        CCSM_WARNING("Limiting token sequence from ", context_tokens.size(), " to ", max_text_tokens_, " tokens");
        context_tokens.resize(max_text_tokens_);
        positions.resize(max_text_tokens_);
    }
    
    // Calculate max frames based on audio length
    int frames_per_second = 1000 / 80; // 80ms per frame = 12.5 frames per second
    int max_frames = working_options.max_audio_length_ms * frames_per_second / 1000;
    
    // Ensure we have at least one frame
    max_frames = std::max(1, max_frames);
    CCSM_INFO("Generating speech with up to ", max_frames, " frames using MLX acceleration");
    
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Generate frames with MLX optimization
    std::vector<std::vector<int>> frames;
    
    // Record the start time
    auto generation_start = std::chrono::high_resolution_clock::now();
    
    for (int frame_idx = 0; frame_idx < max_frames; frame_idx++) {
        // Generate next frame of tokens using MLX optimizations
        std::vector<int> frame = generate_frame_mlx(
            context_tokens, positions, working_options.temperature, working_options.top_k);
        
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
        
        if (eos_detected) {
            break;
        }
        
        // Add frame to results
        frames.push_back(frame);
        
        // Update context with the new frame for the next iteration
        for (auto token : frame) {
            context_tokens.push_back(token);
            positions.push_back(static_cast<int>(context_tokens.size() - 1));
        }
        
        // Report progress
        if (progress_callback) {
            progress_callback(frame_idx + 1, max_frames);
        }
        
        // Apply memory optimization if enabled
        if (memory_optimization_enabled_ && frame_idx % 5 == 0) {
            model_->prune_caches(prune_factor_);
        }
    }
    
    // Record the end time
    auto generation_end = std::chrono::high_resolution_clock::now();
    auto generation_duration = std::chrono::duration_cast<std::chrono::milliseconds>(generation_end - generation_start).count();
    CCSM_INFO("Generated ", frames.size(), " frames in ", generation_duration, "ms (", 
              static_cast<float>(frames.size()) * 1000.0f / generation_duration, " frames/sec)");
    
    // Ensure we have at least one frame
    if (frames.empty()) {
        CCSM_WARNING("No frames generated, creating dummy frame");
        std::vector<int> dummy_frame(audio_codec_->num_codebooks(), 1);
        frames.push_back(dummy_frame);
    }
    
    // Process frames into audio
    std::vector<float> audio = process_audio_frames(frames, working_options);
    
    // Measure total generation time
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    CCSM_INFO("Total MLX speech generation time: ", total_duration, "ms for ", audio.size(), " samples");
    
    return audio;
}

// Set optimization config for MLX operations
void MLXGenerator::set_optimization_config(const MLXOptimizationConfig& config) {
    optimization_config_ = config;
    
    #ifdef CCSM_WITH_MLX
    if (mlx_available_) {
        configure_mlx_for_device(optimization_config_);
        CCSM_INFO("Updated MLX configuration");
    }
    #endif
}

// Get current optimization config
const MLXOptimizationConfig& MLXGenerator::optimization_config() const {
    return optimization_config_;
}

// Return whether MLX acceleration is available and being used
bool MLXGenerator::is_mlx_accelerated() const {
    return mlx_available_ && mlx_model_ != nullptr;
}

// Check if the model is compatible with MLX acceleration
bool MLXGenerator::is_model_mlx_compatible(const std::shared_ptr<Model>& model) {
    return std::dynamic_pointer_cast<MLXModel>(model) != nullptr;
}

// Private helper methods

// Optimize for MLX-specific memory management
void MLXGenerator::optimize_for_generation() {
    if (!mlx_model_) return;
    
    // Apply memory optimization settings based on configuration
    CCSM_INFO("Optimizing MLX model for generation");
    
    // Set memory limits based on optimization config
    size_t memory_limit_mb = 0;
    switch (optimization_config_.memory_usage) {
        case MLXOptimizationConfig::MemoryUsage::MINIMAL:
            memory_limit_mb = 512; // 512MB for minimal usage
            break;
        case MLXOptimizationConfig::MemoryUsage::BALANCED:
            memory_limit_mb = 1024; // 1GB for balanced usage
            break;
        case MLXOptimizationConfig::MemoryUsage::PERFORMANCE:
            memory_limit_mb = 0; // No limit for performance
            break;
    }
    
    // Apply memory optimization
    mlx_model_->optimize_memory(memory_limit_mb);
}

// Generate a single audio frame with MLX optimizations
std::vector<int> MLXGenerator::generate_frame_mlx(
    const std::vector<int>& tokens,
    const std::vector<int>& positions,
    float temperature,
    int top_k) {
    
    #ifdef CCSM_WITH_MLX
    // Use MLX optimizations if available
    if (mlx_available_ && mlx_model_) {
        // The MLXModel implementation already has optimizations for MLX
        return mlx_model_->generate_frame(tokens, positions, temperature, top_k);
    }
    #endif
    
    // Fall back to the base implementation
    return model_->generate_frame(tokens, positions, temperature, top_k);
}

// Process generation results for audio decoding
std::vector<float> MLXGenerator::process_audio_frames(
    const std::vector<std::vector<int>>& frames,
    const GenerationOptions& options) {
    
    CCSM_INFO("Processing ", frames.size(), " frames with MLX optimizations");
    
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
        // Time the decoding
        auto decode_start = std::chrono::high_resolution_clock::now();
        
        audio = audio_codec_->decode(codec_tokens);
        
        auto decode_end = std::chrono::high_resolution_clock::now();
        auto decode_duration = std::chrono::duration_cast<std::chrono::milliseconds>(decode_end - decode_start).count();
        CCSM_INFO("Decoded ", frames.size(), " frames to ", audio.size(), " audio samples in ", decode_duration, "ms");
    } catch (const std::exception& e) {
        CCSM_ERROR("Failed to decode audio: ", e.what());
        throw; // Re-throw the exception
    }
    
    // Apply watermark if enabled and watermarker is available
    if (options.enable_watermark && watermarker_) {
        CCSM_DEBUG("Applying watermark to audio");
        try {
            auto watermark_start = std::chrono::high_resolution_clock::now();
            
            audio = watermarker_->apply_watermark(audio);
            
            auto watermark_end = std::chrono::high_resolution_clock::now();
            auto watermark_duration = std::chrono::duration_cast<std::chrono::milliseconds>(watermark_end - watermark_start).count();
            CCSM_INFO("Applied watermark in ", watermark_duration, "ms");
        } catch (const std::exception& e) {
            CCSM_ERROR("Failed to apply watermark: ", e.what());
            // Continue with unwatermarked audio rather than failing
        }
    }
    
    return audio;
}

// Factory function to create an MLX generator
std::shared_ptr<Generator> create_mlx_generator(
    const std::string& model_path,
    const std::string& tokenizer_path,
    const std::string& audio_codec_path,
    const std::string& watermarker_path,
    const MLXOptimizationConfig& optimization_config) {
    
    CCSM_INFO("Creating MLX generator with model: ", model_path);
    
    #ifdef CCSM_WITH_MLX
    if (!MLXWeightConverter::is_mlx_available()) {
        CCSM_WARNING("MLX is not available on this system, falling back to CPU generator");
        return nullptr;
    }
    
    // Create MLX model
    auto model_config = ModelConfig(); // Default config, to be loaded from model
    auto mlx_model = std::make_shared<MLXModel>(model_config);
    
    // Load weights
    if (!mlx_model->load_weights(model_path)) {
        CCSM_ERROR("Failed to load MLX model weights");
        return nullptr;
    }
    
    // Load tokenizer
    auto tokenizer = std::make_shared<TextTokenizer>();
    if (!tokenizer->load(tokenizer_path)) {
        CCSM_ERROR("Failed to load tokenizer");
        return nullptr;
    }
    
    // Load audio codec
    auto codec = std::make_shared<AudioCodec>();
    if (!codec->load(audio_codec_path)) {
        CCSM_ERROR("Failed to load audio codec");
        return nullptr;
    }
    
    // Load watermarker if provided
    std::shared_ptr<Watermarker> watermarker = nullptr;
    if (!watermarker_path.empty()) {
        watermarker = std::make_shared<Watermarker>();
        if (!watermarker->load(watermarker_path)) {
            CCSM_WARNING("Failed to load watermarker, continuing without watermarking");
            watermarker = nullptr;
        }
    }
    
    // Create MLX generator
    return std::make_shared<MLXGenerator>(mlx_model, tokenizer, codec, watermarker, optimization_config);
    #else
    CCSM_WARNING("CCSM was not compiled with MLX support");
    return nullptr;
    #endif
}

// Check if MLX acceleration is available on this system
bool is_mlx_available() {
    #ifdef CCSM_WITH_MLX
    return MLXWeightConverter::is_mlx_available();
    #else
    return false;
    #endif
}

} // namespace ccsm