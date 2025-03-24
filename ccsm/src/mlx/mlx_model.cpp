#include <ccsm/mlx/mlx_model.h>
#include <ccsm/mlx/mlx_tensor.h>
#include <ccsm/mlx/mlx_transformer.h>
#include <ccsm/mlx/mlx_weight_converter.h>
#include <ccsm/model_loader.h>
#include <ccsm/utils.h>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <numeric>

namespace ccsm {

// Private implementation
struct MLXModel::Impl {
    #ifdef CCSM_WITH_MLX
    // MLX weight arrays
    std::unordered_map<std::string, mlx_array> weights;
    
    // MLX transformers
    std::shared_ptr<MLXTransformer> backbone_transformer;
    std::shared_ptr<MLXTransformer> decoder_transformer;
    
    // MLX device
    MLXDevice device;
    
    // KV cache state
    bool cache_initialized = false;
    size_t max_cache_len = 0;
    #endif
    
    // Conversion config for PyTorch weights
    MLXWeightConversionConfig conversion_config;
};

// Constructor
MLXModel::MLXModel(const ModelConfig& config) 
    : Model(config), 
      impl_(std::make_unique<Impl>()),
      rng_(std::random_device{}()) {
    
    CCSM_INFO("Creating MLX model with config: " + config_.name);
    
    #ifdef CCSM_WITH_MLX
    // Configure the conversion config
    impl_->conversion_config.use_bfloat16 = true;
    impl_->conversion_config.cache_converted_weights = true;
    
    // Set maximum cache length based on model config
    impl_->max_cache_len = config_.max_seq_len;
    
    // Initialize transformers
    impl_->backbone_transformer = std::make_shared<MLXTransformer>(
        config_.d_model,
        config_.n_heads,
        config_.n_kv_heads,
        config_.n_layers,
        config_.max_seq_len
    );
    
    impl_->decoder_transformer = std::make_shared<MLXTransformer>(
        config_.d_model,
        config_.n_heads,
        config_.n_kv_heads,
        config_.n_audio_layers,
        config_.max_seq_len
    );
    #else
    CCSM_WARN("MLX support not compiled in, creating stub MLX model");
    #endif
}

// Destructor
MLXModel::~MLXModel() {
    CCSM_INFO("Destroying MLX model");
}

// Load model weights from file
bool MLXModel::load_weights(const std::string& path) {
    CCSM_INFO("Loading MLX model weights from file: " + path);
    
    try {
        // Create model loader based on file extension
        auto loader = ModelLoaderFactory::create(path);
        if (!loader) {
            CCSM_ERROR("Failed to create model loader for: " + path);
            return false;
        }
        
        return load_weights(loader);
    } catch (const std::exception& e) {
        CCSM_ERROR("Error loading MLX model weights: " + std::string(e.what()));
        return false;
    }
}

// Load model weights from loader
bool MLXModel::load_weights(std::shared_ptr<ModelLoader> loader) {
    CCSM_INFO("Loading MLX model weights from loader");
    
    try {
        // Load weights to temporary map
        WeightMap weights;
        if (!loader->load(weights)) {
            CCSM_ERROR("Failed to load weights from loader");
            return false;
        }
        
        return load_weights(weights);
    } catch (const std::exception& e) {
        CCSM_ERROR("Error loading MLX model weights: " + std::string(e.what()));
        return false;
    }
}

// Load model weights directly from a weight map
bool MLXModel::load_weights(const WeightMap& weights) {
    CCSM_INFO("Loading MLX model weights from weight map");
    
    #ifdef CCSM_WITH_MLX
    try {
        // Convert PyTorch weights to MLX arrays
        impl_->weights = convert_pytorch_to_mlx(weights, impl_->conversion_config);
        
        // Load backbone weights
        if (!load_backbone_weights(weights)) {
            CCSM_ERROR("Failed to load backbone weights");
            return false;
        }
        
        // Load decoder weights
        if (!load_decoder_weights(weights)) {
            CCSM_ERROR("Failed to load decoder weights");
            return false;
        }
        
        // Set the transformer weights
        impl_->backbone_transformer->load_weights(impl_->weights);
        impl_->decoder_transformer->load_weights(impl_->weights);
        
        return true;
    } catch (const std::exception& e) {
        CCSM_ERROR("Error loading MLX model weights: " + std::string(e.what()));
        return false;
    }
    #else
    CCSM_WARN("MLX support not compiled in, cannot load weights");
    return false;
    #endif
}

// Generate a complete audio frame (all codebooks)
std::vector<int> MLXModel::generate_frame(
    const std::vector<int>& tokens,
    const std::vector<int>& positions,
    float temperature,
    int top_k) {
    
    CCSM_INFO("Generating MLX audio frame");
    
    #ifdef CCSM_WITH_MLX
    try {
        // Initialize result vector with all codebooks
        std::vector<int> result(config_.num_codebooks, 0);
        
        // Get backbone logits
        auto backbone_logits = get_backbone_logits(tokens, positions);
        
        // Sample semantic token from backbone logits (first codebook)
        result[0] = sample_from_logits(backbone_logits, temperature, top_k)[0];
        
        // Loop through remaining codebooks
        for (int codebook = 1; codebook < config_.num_codebooks; ++codebook) {
            // Create a vector with all tokens generated so far for this frame
            std::vector<int> codebook_tokens = tokens;
            for (int i = 0; i < codebook; ++i) {
                codebook_tokens.push_back(result[i]);
            }
            
            // Get decoder logits for this codebook
            auto decoder_logits = get_decoder_logits(codebook_tokens, positions, codebook);
            
            // Sample token for this codebook
            result[codebook] = sample_from_logits(decoder_logits, temperature, top_k)[0];
        }
        
        return result;
    } catch (const std::exception& e) {
        CCSM_ERROR("Error generating MLX audio frame: " + std::string(e.what()));
        return std::vector<int>(config_.num_codebooks, 0);
    }
    #else
    CCSM_WARN("MLX support not compiled in, returning empty frame");
    return std::vector<int>(config_.num_codebooks, 0);
    #endif
}

// Reset KV caches
void MLXModel::reset_caches() {
    CCSM_INFO("Resetting MLX caches");
    
    #ifdef CCSM_WITH_MLX
    impl_->backbone_transformer->reset_caches();
    impl_->decoder_transformer->reset_caches();
    impl_->cache_initialized = false;
    #endif
}

// Memory optimization methods
void MLXModel::optimize_memory(size_t max_memory_mb) {
    CCSM_INFO("Optimizing MLX memory usage");
    
    #ifdef CCSM_WITH_MLX
    // Limit the KV cache size based on available memory
    size_t available_memory = max_memory_mb > 0 ? max_memory_mb * 1024 * 1024 : 0;
    
    // If no memory limit specified, use a reasonable default
    if (available_memory == 0) {
        // Use 80% of available device memory
        available_memory = 1024 * 1024 * 1024; // 1GB default
    }
    
    // Calculate how much memory we can allocate to KV caches
    size_t kv_cache_memory = available_memory / 2; // Use at most half for KV cache
    
    // Adjust the max KV cache length based on available memory
    size_t backbone_memory_per_token = config_.n_layers * config_.n_kv_heads * config_.d_model / config_.n_heads * 2 * sizeof(float);
    size_t decoder_memory_per_token = config_.n_audio_layers * config_.n_kv_heads * config_.d_model / config_.n_heads * 2 * sizeof(float);
    size_t total_memory_per_token = backbone_memory_per_token + decoder_memory_per_token;
    
    size_t max_tokens = kv_cache_memory / total_memory_per_token;
    impl_->max_cache_len = std::min(max_tokens, static_cast<size_t>(config_.max_seq_len));
    
    CCSM_INFO("Optimized MLX KV cache for " + std::to_string(impl_->max_cache_len) + " tokens");
    #endif
}

void MLXModel::prune_caches(float prune_factor) {
    CCSM_INFO("Pruning MLX caches with factor: " + std::to_string(prune_factor));
    
    #ifdef CCSM_WITH_MLX
    // Calculate new target length based on prune factor
    size_t current_len = impl_->max_cache_len;
    size_t target_len = static_cast<size_t>(current_len * prune_factor);
    
    if (target_len == current_len) {
        CCSM_INFO("No pruning needed, cache already at target size");
        return;
    }
    
    // For now we just reset the caches
    // TODO: Implement more sophisticated pruning
    reset_caches();
    impl_->max_cache_len = target_len;
    
    CCSM_INFO("Pruned MLX KV cache to " + std::to_string(impl_->max_cache_len) + " tokens");
    #endif
}

// Get backbone model logits
std::vector<float> MLXModel::get_backbone_logits(
    const std::vector<int>& tokens,
    const std::vector<int>& positions) {
    
    CCSM_INFO("Getting MLX backbone logits");
    
    #ifdef CCSM_WITH_MLX
    try {
        // Create MLX tensors for tokens and positions
        std::vector<int64_t> shape = {static_cast<int64_t>(tokens.size())};
        
        // Create token tensor
        mlx_array token_array;
        std::vector<int32_t> token_data(tokens.begin(), tokens.end());
        mlx_array_from_data(token_data.data(), shape.data(), shape.size(), MLX_INT32, &token_array);
        
        // Create position tensor
        mlx_array position_array;
        std::vector<int32_t> position_data(positions.begin(), positions.end());
        mlx_array_from_data(position_data.data(), shape.data(), shape.size(), MLX_INT32, &position_array);
        
        // Forward pass through backbone transformer
        mlx_array logits_array = impl_->backbone_transformer->forward(token_array, position_array);
        
        // Convert logits back to vector
        size_t logits_size = mlx_array_size(logits_array);
        std::vector<float> logits(logits_size);
        float* logits_data = static_cast<float*>(mlx_array_data(logits_array));
        
        // Copy the data
        if (logits_data) {
            std::copy(logits_data, logits_data + logits_size, logits.begin());
        }
        
        // Clean up
        mlx_array_free(token_array);
        mlx_array_free(position_array);
        mlx_array_free(logits_array);
        
        return logits;
    } catch (const std::exception& e) {
        CCSM_ERROR("Error getting MLX backbone logits: " + std::string(e.what()));
        return std::vector<float>(config_.vocab_size, 0.0f);
    }
    #else
    CCSM_WARN("MLX support not compiled in, returning empty logits");
    return std::vector<float>(config_.vocab_size, 0.0f);
    #endif
}

// Get decoder model logits
std::vector<float> MLXModel::get_decoder_logits(
    const std::vector<int>& tokens,
    const std::vector<int>& positions,
    int codebook) {
    
    CCSM_INFO("Getting MLX decoder logits for codebook " + std::to_string(codebook));
    
    #ifdef CCSM_WITH_MLX
    try {
        // Validate codebook
        if (codebook < 0 || codebook >= config_.num_codebooks) {
            throw std::runtime_error("Invalid codebook index: " + std::to_string(codebook));
        }
        
        // Create MLX tensors for tokens and positions
        std::vector<int64_t> shape = {static_cast<int64_t>(tokens.size())};
        
        // Create token tensor
        mlx_array token_array;
        std::vector<int32_t> token_data(tokens.begin(), tokens.end());
        mlx_array_from_data(token_data.data(), shape.data(), shape.size(), MLX_INT32, &token_array);
        
        // Create position tensor
        mlx_array position_array;
        std::vector<int32_t> position_data(positions.begin(), positions.end());
        mlx_array_from_data(position_data.data(), shape.data(), shape.size(), MLX_INT32, &position_array);
        
        // Forward pass through decoder transformer
        mlx_array logits_array = impl_->decoder_transformer->forward(token_array, position_array, codebook);
        
        // Convert logits back to vector
        size_t logits_size = mlx_array_size(logits_array);
        std::vector<float> logits(logits_size);
        float* logits_data = static_cast<float*>(mlx_array_data(logits_array));
        
        // Copy the data
        if (logits_data) {
            std::copy(logits_data, logits_data + logits_size, logits.begin());
        }
        
        // Clean up
        mlx_array_free(token_array);
        mlx_array_free(position_array);
        mlx_array_free(logits_array);
        
        return logits;
    } catch (const std::exception& e) {
        CCSM_ERROR("Error getting MLX decoder logits: " + std::string(e.what()));
        return std::vector<float>(config_.audio_vocab_size, 0.0f);
    }
    #else
    CCSM_WARN("MLX support not compiled in, returning empty logits");
    return std::vector<float>(config_.audio_vocab_size, 0.0f);
    #endif
}

// Helper methods

// Load backbone weights
bool MLXModel::load_backbone_weights(const WeightMap& weights) {
    CCSM_INFO("Loading MLX backbone weights");
    
    #ifdef CCSM_WITH_MLX
    try {
        // Check for essential backbone weights
        const std::vector<std::string> essential_weights = {
            "backbone.token_embedding.weight",
            "backbone.norm.weight",
            "backbone.output.weight"
        };
        
        for (const auto& name : essential_weights) {
            if (!has_weight(name)) {
                CCSM_ERROR("Missing essential backbone weight: " + name);
                return false;
            }
        }
        
        // Check for transformer layers
        for (int layer = 0; layer < config_.n_layers; ++layer) {
            std::string prefix = "backbone.layers." + std::to_string(layer) + ".";
            std::vector<std::string> layer_weights = {
                prefix + "attention.wq.weight",
                prefix + "attention.wk.weight",
                prefix + "attention.wv.weight",
                prefix + "attention.wo.weight",
                prefix + "attention_norm.weight",
                prefix + "ffn.w1.weight",
                prefix + "ffn.w2.weight",
                prefix + "ffn.w3.weight",
                prefix + "ffn_norm.weight"
            };
            
            for (const auto& name : layer_weights) {
                if (!has_weight(name)) {
                    CCSM_WARN("Missing backbone layer weight: " + name);
                    // Not all models have all weights (e.g., some might use bias)
                }
            }
        }
        
        return true;
    } catch (const std::exception& e) {
        CCSM_ERROR("Error loading MLX backbone weights: " + std::string(e.what()));
        return false;
    }
    #else
    CCSM_WARN("MLX support not compiled in, cannot load backbone weights");
    return false;
    #endif
}

// Load decoder weights
bool MLXModel::load_decoder_weights(const WeightMap& weights) {
    CCSM_INFO("Loading MLX decoder weights");
    
    #ifdef CCSM_WITH_MLX
    try {
        // Check for essential decoder weights
        const std::vector<std::string> essential_weights = {
            "decoder.token_embedding.weight",
            "decoder.norm.weight"
        };
        
        for (const auto& name : essential_weights) {
            if (!has_weight(name)) {
                CCSM_ERROR("Missing essential decoder weight: " + name);
                return false;
            }
        }
        
        // Check for codebook output layers
        for (int codebook = 0; codebook < config_.num_codebooks; ++codebook) {
            std::string output_weight = "decoder.codebook_outputs." + std::to_string(codebook) + ".weight";
            if (!has_weight(output_weight)) {
                CCSM_WARN("Missing decoder codebook output weight: " + output_weight);
                // Not all models have all codebooks
            }
        }
        
        // Check for transformer layers
        for (int layer = 0; layer < config_.n_audio_layers; ++layer) {
            std::string prefix = "decoder.layers." + std::to_string(layer) + ".";
            std::vector<std::string> layer_weights = {
                prefix + "attention.wq.weight",
                prefix + "attention.wk.weight",
                prefix + "attention.wv.weight",
                prefix + "attention.wo.weight",
                prefix + "attention_norm.weight",
                prefix + "ffn.w1.weight",
                prefix + "ffn.w2.weight",
                prefix + "ffn.w3.weight",
                prefix + "ffn_norm.weight"
            };
            
            for (const auto& name : layer_weights) {
                if (!has_weight(name)) {
                    CCSM_WARN("Missing decoder layer weight: " + name);
                    // Not all models have all weights (e.g., some might use bias)
                }
            }
        }
        
        return true;
    } catch (const std::exception& e) {
        CCSM_ERROR("Error loading MLX decoder weights: " + std::string(e.what()));
        return false;
    }
    #else
    CCSM_WARN("MLX support not compiled in, cannot load decoder weights");
    return false;
    #endif
}

// MLX-specific methods

#ifdef CCSM_WITH_MLX
// Get MLX array for a specific weight
mlx_array MLXModel::get_weight_array(const std::string& name) const {
    if (!has_weight(name)) {
        throw std::runtime_error("Weight not found: " + name);
    }
    return impl_->weights.at(name);
}
#endif

// Check if a specific MLX weight exists
bool MLXModel::has_weight(const std::string& name) const {
    #ifdef CCSM_WITH_MLX
    return impl_->weights.find(name) != impl_->weights.end();
    #else
    return false;
    #endif
}

// Create a new transformer context for inference
std::shared_ptr<MLXTransformer> MLXModel::create_transformer_context() const {
    #ifdef CCSM_WITH_MLX
    return std::make_shared<MLXTransformer>(
        config_.d_model,
        config_.n_heads,
        config_.n_kv_heads,
        config_.n_layers,
        config_.max_seq_len
    );
    #else
    return nullptr;
    #endif
}

// Sample tokens from logits
std::vector<int> MLXModel::sample_from_logits(
    const std::vector<float>& logits, 
    float temperature, 
    int top_k) {
    
    // Make a copy of the logits
    std::vector<float> probs = logits;
    size_t vocab_size = probs.size();
    
    // Apply temperature
    if (temperature > 0.0f) {
        for (auto& p : probs) {
            p /= temperature;
        }
    }
    
    // Apply softmax
    float max_logit = *std::max_element(probs.begin(), probs.end());
    float sum_exp = 0.0f;
    for (auto& p : probs) {
        p = std::exp(p - max_logit);
        sum_exp += p;
    }
    for (auto& p : probs) {
        p /= sum_exp;
    }
    
    // Apply top-k
    if (top_k > 0 && top_k < static_cast<int>(vocab_size)) {
        // Find indices of the top-k probabilities
        std::vector<size_t> indices(vocab_size);
        std::iota(indices.begin(), indices.end(), 0);
        
        std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
            [&probs](size_t a, size_t b) { return probs[a] > probs[b]; });
        
        // Set probabilities of non-top-k indices to zero
        std::vector<float> top_k_probs(vocab_size, 0.0f);
        for (int i = 0; i < top_k; ++i) {
            top_k_probs[indices[i]] = probs[indices[i]];
        }
        probs = top_k_probs;
        
        // Renormalize
        sum_exp = std::accumulate(probs.begin(), probs.end(), 0.0f);
        if (sum_exp > 0.0f) {
            for (auto& p : probs) {
                p /= sum_exp;
            }
        }
    }
    
    // Sample from the distribution
    std::discrete_distribution<int> distribution(probs.begin(), probs.end());
    return {distribution(rng_)};
}

} // namespace ccsm