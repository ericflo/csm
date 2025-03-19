#include <ccsm/cpu/ggml_model.h>
#include <ccsm/model_loader.h>
#include <ccsm/utils.h>

#include "ggml.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>
#include <numeric>
#include <chrono>

namespace ccsm {

// KVCache implementation
KVCache::KVCache(size_t n_layers, size_t n_heads, size_t n_kv_heads, size_t head_dim, size_t max_seq_len)
    : n_layers_(n_layers), 
      n_heads_(n_heads), 
      n_kv_heads_(n_kv_heads), 
      head_dim_(head_dim), 
      max_seq_len_(max_seq_len),
      current_seq_len_(0),
      ctx_(nullptr) {
    
    // Calculate memory needed for caches
    size_t total_size_k = n_layers * n_kv_heads * head_dim * max_seq_len * sizeof(float);
    size_t total_size_v = n_layers * n_kv_heads * head_dim * max_seq_len * sizeof(float);
    
    // Add some extra memory for padding and alignment
    size_t total_size = total_size_k + total_size_v + 1024 * 1024;
    
    // Initialize context
    struct ggml_init_params params = {
        .mem_size   = total_size,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    
    ctx_ = ggml_init(params);
    if (!ctx_) {
        throw std::runtime_error("Failed to initialize KV cache context");
    }
    
    // Create cache tensors
    k_caches_.resize(n_layers);
    v_caches_.resize(n_layers);
    
    // GGML dimensions are ordered from smallest to largest stride
    // For KV caches, we want (D, H_kv, S) dimensions where:
    // D = head dimension, H_kv = number of KV heads, S = sequence length
    for (size_t i = 0; i < n_layers; i++) {
        k_caches_[i] = ggml_new_tensor_3d(ctx_, GGML_TYPE_F32, head_dim, n_kv_heads, max_seq_len);
        v_caches_[i] = ggml_new_tensor_3d(ctx_, GGML_TYPE_F32, head_dim, n_kv_heads, max_seq_len);
        
        // Zero initialize
        ggml_set_zero(k_caches_[i]);
        ggml_set_zero(v_caches_[i]);
    }
}

KVCache::~KVCache() {
    if (ctx_) {
        ggml_free(ctx_);
    }
}

void KVCache::clear() {
    // Reset sequence length
    current_seq_len_ = 0;
    
    // Zero out caches
    for (size_t i = 0; i < n_layers_; i++) {
        ggml_set_zero(k_caches_[i]);
        ggml_set_zero(v_caches_[i]);
    }
}

void KVCache::resize(size_t seq_len) {
    if (seq_len > max_seq_len_) {
        throw std::runtime_error("Cannot resize KV cache beyond max sequence length");
    }
    
    current_seq_len_ = seq_len;
}

struct ggml_tensor* KVCache::k_cache(int layer) {
    if (layer < 0 || static_cast<size_t>(layer) >= n_layers_) {
        throw std::out_of_range("Layer index out of range in KV cache");
    }
    return k_caches_[layer];
}

struct ggml_tensor* KVCache::v_cache(int layer) {
    if (layer < 0 || static_cast<size_t>(layer) >= n_layers_) {
        throw std::out_of_range("Layer index out of range in KV cache");
    }
    return v_caches_[layer];
}

size_t KVCache::size() const {
    return n_layers_ * 2; // Number of tensors (K and V for each layer)
}

size_t KVCache::max_seq_len() const {
    return max_seq_len_;
}

size_t KVCache::current_seq_len() const {
    return current_seq_len_;
}

// GGMLModel implementation
GGMLModel::GGMLModel(const ModelConfig& config)
    : Model(config), ctx_(nullptr) {
    
    // Create RNG with random seed
    std::random_device rd;
    rng_.seed(rd());
    
    // Calculate memory needed for model weights
    size_t backbone_params = 
        config.n_layers * (
            // QKV projection
            3 * config.d_model * config.d_model +
            // Output projection
            config.d_model * config.d_model +
            // FFN weights
            2 * config.d_model * (4 * config.d_model) +
            config.d_model * config.d_model +
            // Layer norms
            4 * config.d_model
        ) +
        // Embedding
        config.vocab_size * config.d_model +
        // Final layer norm
        2 * config.d_model +
        // Output projection
        config.d_model * config.audio_vocab_size;
    
    size_t decoder_params =
        config.n_audio_layers * (
            // QKV projection
            3 * config.d_model * config.d_model +
            // Output projection
            config.d_model * config.d_model +
            // FFN weights
            2 * config.d_model * (4 * config.d_model) +
            config.d_model * config.d_model +
            // Layer norms
            4 * config.d_model
        ) +
        // Embedding
        config.audio_vocab_size * config.d_model +
        // Final layer norm
        2 * config.d_model +
        // Output projection (for each codebook beyond the first)
        (config.num_codebooks - 1) * config.d_model * config.audio_vocab_size;
    
    // Total parameters in bytes (assuming float32)
    size_t total_size = (backbone_params + decoder_params) * sizeof(float);
    
    // Add extra memory for context management
    total_size += 64 * 1024 * 1024; // 64 MB extra
    
    // Initialize context
    struct ggml_init_params params = {
        .mem_size   = total_size,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    
    ctx_ = ggml_init(params);
    if (!ctx_) {
        throw std::runtime_error("Failed to initialize model weights context");
    }
    
    // Initialize KV caches
    backbone_kv_cache_ = std::make_shared<KVCache>(
        config.n_layers,
        config.n_heads,
        config.n_kv_heads,
        config.d_model / config.n_heads,
        config.max_seq_len
    );
    
    decoder_kv_cache_ = std::make_shared<KVCache>(
        config.n_audio_layers,
        config.n_heads,
        config.n_kv_heads,
        config.d_model / config.n_heads,
        config.max_seq_len
    );
    
    CCSM_INFO("Initialized GGMLModel with configuration: ", config.name);
}

GGMLModel::~GGMLModel() {
    if (ctx_) {
        ggml_free(ctx_);
    }
}

bool GGMLModel::load_weights(const std::string& path) {
    CCSM_INFO("Loading model weights from: ", path);
    
    // Create model loader
    auto loader = ModelLoaderFactory::create(path);
    return load_weights(loader);
}

bool GGMLModel::load_weights(std::shared_ptr<ModelLoader> loader) {
    CCSM_INFO("Loading model weights from loader");
    
    // Load weights into temporary map
    WeightMap weights;
    if (!loader->load(weights)) {
        CCSM_ERROR("Failed to load weights from loader");
        return false;
    }
    
    return load_weights(weights);
}

bool GGMLModel::load_weights(const WeightMap& tensor_weights) {
    CCSM_INFO("Loading model weights from tensor map");
    
    // First clear any existing weights
    weights_.clear();
    
    // Load backbone weights
    if (!load_backbone_weights(tensor_weights)) {
        CCSM_ERROR("Failed to load backbone weights");
        return false;
    }
    
    // Load decoder weights
    if (!load_decoder_weights(tensor_weights)) {
        CCSM_ERROR("Failed to load decoder weights");
        return false;
    }
    
    CCSM_INFO("Successfully loaded ", weights_.size(), " weight tensors");
    return true;
}

bool GGMLModel::load_backbone_weights(const WeightMap& tensor_weights) {
    // List of required backbone weights
    std::vector<std::string> required_weights = {
        "model.tok_embeddings.weight",
        "model.norm.weight",
        "model.norm.bias",
        "lm_head.weight"
    };
    
    // Add layer-specific weights
    for (int layer = 0; layer < config_.n_layers; layer++) {
        required_weights.push_back("model.layers." + std::to_string(layer) + ".attention.wq.weight");
        required_weights.push_back("model.layers." + std::to_string(layer) + ".attention.wk.weight");
        required_weights.push_back("model.layers." + std::to_string(layer) + ".attention.wv.weight");
        required_weights.push_back("model.layers." + std::to_string(layer) + ".attention.wo.weight");
        required_weights.push_back("model.layers." + std::to_string(layer) + ".attention_norm.weight");
        required_weights.push_back("model.layers." + std::to_string(layer) + ".attention_norm.bias");
        required_weights.push_back("model.layers." + std::to_string(layer) + ".feed_forward.w1.weight");
        required_weights.push_back("model.layers." + std::to_string(layer) + ".feed_forward.w2.weight");
        required_weights.push_back("model.layers." + std::to_string(layer) + ".feed_forward.w3.weight");
        required_weights.push_back("model.layers." + std::to_string(layer) + ".ffn_norm.weight");
        required_weights.push_back("model.layers." + std::to_string(layer) + ".ffn_norm.bias");
    }
    
    // Check if all required weights are present
    for (const auto& name : required_weights) {
        if (tensor_weights.find(name) == tensor_weights.end()) {
            CCSM_WARNING("Missing required backbone weight: ", name);
            // We'll continue even if weights are missing, to support partial loading
        }
    }
    
    // Convert tensor weights to GGML tensors
    for (const auto& [name, tensor] : tensor_weights) {
        // Skip decoder weights
        if (name.find("decoder") != std::string::npos) {
            continue;
        }
        
        // Get tensor shape and data
        std::vector<size_t> shape = tensor.shape();
        const void* data = tensor.data();
        
        // Prepare GGML shape
        int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
        for (size_t i = 0; i < shape.size() && i < GGML_MAX_DIMS; i++) {
            ne[i] = shape[i];
        }
        
        // Create GGML tensor
        struct ggml_tensor* ggml_tensor = ggml_new_tensor(ctx_, 
                                                      GGMLTensorImpl::to_ggml_type(tensor.dtype()), 
                                                      std::min(static_cast<size_t>(GGML_MAX_DIMS), shape.size()), 
                                                      ne);
        if (!ggml_tensor) {
            CCSM_ERROR("Failed to create GGML tensor for weight: ", name);
            continue;
        }
        
        // Copy data
        size_t tensor_size = ggml_nbytes(ggml_tensor);
        std::memcpy(ggml_tensor->data, data, tensor_size);
        
        // Store in weights map
        weights_[name] = ggml_tensor;
    }
    
    return true;
}

bool GGMLModel::load_decoder_weights(const WeightMap& tensor_weights) {
    // List of required decoder weights
    std::vector<std::string> required_weights = {
        "decoder.tok_embeddings.weight",
        "decoder.norm.weight",
        "decoder.norm.bias"
    };
    
    // Add codebook output projections
    for (int cb = 1; cb < config_.num_codebooks; cb++) {
        required_weights.push_back("decoder.output." + std::to_string(cb) + ".weight");
    }
    
    // Add layer-specific weights
    for (int layer = 0; layer < config_.n_audio_layers; layer++) {
        required_weights.push_back("decoder.layers." + std::to_string(layer) + ".attention.wq.weight");
        required_weights.push_back("decoder.layers." + std::to_string(layer) + ".attention.wk.weight");
        required_weights.push_back("decoder.layers." + std::to_string(layer) + ".attention.wv.weight");
        required_weights.push_back("decoder.layers." + std::to_string(layer) + ".attention.wo.weight");
        required_weights.push_back("decoder.layers." + std::to_string(layer) + ".attention_norm.weight");
        required_weights.push_back("decoder.layers." + std::to_string(layer) + ".attention_norm.bias");
        required_weights.push_back("decoder.layers." + std::to_string(layer) + ".feed_forward.w1.weight");
        required_weights.push_back("decoder.layers." + std::to_string(layer) + ".feed_forward.w2.weight");
        required_weights.push_back("decoder.layers." + std::to_string(layer) + ".feed_forward.w3.weight");
        required_weights.push_back("decoder.layers." + std::to_string(layer) + ".ffn_norm.weight");
        required_weights.push_back("decoder.layers." + std::to_string(layer) + ".ffn_norm.bias");
    }
    
    // Check if all required weights are present
    for (const auto& name : required_weights) {
        if (tensor_weights.find(name) == tensor_weights.end()) {
            CCSM_WARNING("Missing required decoder weight: ", name);
            // We'll continue even if weights are missing, to support partial loading
        }
    }
    
    // Convert tensor weights to GGML tensors
    for (const auto& [name, tensor] : tensor_weights) {
        // Skip backbone weights
        if (name.find("decoder") == std::string::npos) {
            continue;
        }
        
        // Get tensor shape and data
        std::vector<size_t> shape = tensor.shape();
        const void* data = tensor.data();
        
        // Prepare GGML shape
        int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
        for (size_t i = 0; i < shape.size() && i < GGML_MAX_DIMS; i++) {
            ne[i] = shape[i];
        }
        
        // Create GGML tensor
        struct ggml_tensor* ggml_tensor = ggml_new_tensor(ctx_, 
                                                      GGMLTensorImpl::to_ggml_type(tensor.dtype()), 
                                                      std::min(static_cast<size_t>(GGML_MAX_DIMS), shape.size()), 
                                                      ne);
        if (!ggml_tensor) {
            CCSM_ERROR("Failed to create GGML tensor for weight: ", name);
            continue;
        }
        
        // Copy data
        size_t tensor_size = ggml_nbytes(ggml_tensor);
        std::memcpy(ggml_tensor->data, data, tensor_size);
        
        // Store in weights map
        weights_[name] = ggml_tensor;
    }
    
    return true;
}

std::vector<int> GGMLModel::generate_frame(
    const std::vector<int>& tokens,
    const std::vector<int>& positions,
    float temperature,
    int top_k) {
    
    CCSM_DEBUG("Generating frame with temperature: ", temperature, ", top_k: ", top_k);
    
    Timer timer;
    
    // Generate semantic token (codebook 0) using the backbone model
    std::vector<float> backbone_logits = get_backbone_logits(tokens, positions);
    
    // Sample token
    int semantic_token = sample_token(backbone_logits.data(), config_.audio_vocab_size, temperature, top_k);
    
    CCSM_DEBUG("Generated semantic token: ", semantic_token, " in ", timer.elapsed_ms(), "ms");
    
    // Initialize frame with semantic token
    std::vector<int> frame = {semantic_token};
    
    // If semantic token is EOS, return early
    if (semantic_token == 0) { // Assuming 0 is EOS
        return frame;
    }
    
    // Create input for decoder
    std::vector<int> decoder_tokens = tokens;
    decoder_tokens.push_back(semantic_token);
    
    std::vector<int> decoder_positions = positions;
    decoder_positions.push_back(positions.back() + 1);
    
    // Generate tokens for each codebook
    for (int codebook = 1; codebook < config_.num_codebooks; codebook++) {
        // Get logits for this codebook
        std::vector<float> decoder_logits = get_decoder_logits(decoder_tokens, decoder_positions, codebook);
        
        // Sample token
        int token = sample_token(decoder_logits.data(), config_.audio_vocab_size, temperature, top_k);
        
        // Add to frame
        frame.push_back(token);
        
        // Add to decoder input for next codebook
        decoder_tokens.push_back(token);
        decoder_positions.push_back(decoder_positions.back() + 1);
    }
    
    CCSM_DEBUG("Generated full frame in ", timer.elapsed_ms(), "ms");
    
    return frame;
}

void GGMLModel::reset_caches() {
    backbone_kv_cache_->clear();
    decoder_kv_cache_->clear();
}

std::vector<float> GGMLModel::get_backbone_logits(
    const std::vector<int>& tokens,
    const std::vector<int>& positions) {
    
    // Create a computation context
    struct ggml_context* comp_ctx = create_computation_context(256 * 1024 * 1024); // 256 MB
    if (!comp_ctx) {
        throw std::runtime_error("Failed to create computation context");
    }
    
    // Build the backbone computation graph
    struct ggml_cgraph* graph = build_backbone_graph(comp_ctx, tokens, positions);
    
    // Allocate memory for computation
    // TODO: Implement proper memory allocation
    //struct ggml_allocr* allocr = ggml_allocr_alloc(graph, false);
    //if (!allocr) {
    //    ggml_free(comp_ctx);
    //    throw std::runtime_error("Failed to allocate memory for computation");
    //}
    
    // Compute the graph
    ggml_graph_compute_with_ctx(comp_ctx, graph, 1); // Use 1 thread for now
    
    // Get logits from the output tensor
    struct ggml_tensor* logits_tensor = graph->nodes[graph->n_nodes - 1];
    
    // Copy logits to vector
    std::vector<float> logits(config_.audio_vocab_size);
    float* logits_data = (float*)logits_tensor->data;
    std::copy(logits_data, logits_data + config_.audio_vocab_size, logits.begin());
    
    // Free resources
    //ggml_allocr_free(allocr);
    ggml_free(comp_ctx);
    
    return logits;
}

std::vector<float> GGMLModel::get_decoder_logits(
    const std::vector<int>& tokens,
    const std::vector<int>& positions,
    int codebook) {
    
    // Create a computation context
    struct ggml_context* comp_ctx = create_computation_context(256 * 1024 * 1024); // 256 MB
    if (!comp_ctx) {
        throw std::runtime_error("Failed to create computation context");
    }
    
    // Build the decoder computation graph
    struct ggml_cgraph* graph = build_decoder_graph(comp_ctx, tokens, positions, codebook);
    
    // Allocate memory for computation
    // TODO: Implement proper memory allocation
    //struct ggml_allocr* allocr = ggml_allocr_alloc(graph, false);
    //if (!allocr) {
    //    ggml_free(comp_ctx);
    //    throw std::runtime_error("Failed to allocate memory for computation");
    //}
    
    // Compute the graph
    ggml_graph_compute_with_ctx(comp_ctx, graph, 1); // Use 1 thread for now
    
    // Get logits from the output tensor
    struct ggml_tensor* logits_tensor = graph->nodes[graph->n_nodes - 1];
    
    // Copy logits to vector
    std::vector<float> logits(config_.audio_vocab_size);
    float* logits_data = (float*)logits_tensor->data;
    std::copy(logits_data, logits_data + config_.audio_vocab_size, logits.begin());
    
    // Free resources
    //ggml_allocr_free(allocr);
    ggml_free(comp_ctx);
    
    return logits;
}

struct ggml_cgraph* GGMLModel::build_backbone_graph(
    struct ggml_context* ctx,
    const std::vector<int>& tokens,
    const std::vector<int>& positions) {
    
    // TODO: Implement the full backbone transformer
    // This is a placeholder that will just return random logits
    
    // Create a computation graph
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    
    // Create a tensor for output logits
    struct ggml_tensor* logits = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, config_.audio_vocab_size);
    
    // For now, just fill with random values
    float* logits_data = (float*)logits->data;
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < config_.audio_vocab_size; i++) {
        logits_data[i] = dist(rng_);
    }
    
    // Add the tensor to the graph
    ggml_build_forward_expand(graph, logits);
    
    return graph;
}

struct ggml_cgraph* GGMLModel::build_decoder_graph(
    struct ggml_context* ctx,
    const std::vector<int>& tokens,
    const std::vector<int>& positions,
    int codebook) {
    
    // TODO: Implement the full decoder transformer
    // This is a placeholder that will just return random logits
    
    // Create a computation graph
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    
    // Create a tensor for output logits
    struct ggml_tensor* logits = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, config_.audio_vocab_size);
    
    // For now, just fill with random values
    float* logits_data = (float*)logits->data;
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < config_.audio_vocab_size; i++) {
        logits_data[i] = dist(rng_);
    }
    
    // Add the tensor to the graph
    ggml_build_forward_expand(graph, logits);
    
    return graph;
}

int GGMLModel::sample_token(const float* logits, int vocab_size, float temperature, int top_k) {
    // If temperature is very close to 0, use greedy sampling
    if (temperature < 1e-6) {
        return std::distance(logits, std::max_element(logits, logits + vocab_size));
    }
    
    // Make a copy of the logits for temperature scaling
    std::vector<float> probs(logits, logits + vocab_size);
    
    // Apply temperature
    for (int i = 0; i < vocab_size; i++) {
        probs[i] /= temperature;
    }
    
    // Determine top-k indices
    std::vector<int> indices(vocab_size);
    std::iota(indices.begin(), indices.end(), 0);
    
    // Sort indices by probability (descending)
    std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
                     [&probs](int a, int b) { return probs[a] > probs[b]; });
    
    // Keep only top-k
    indices.resize(top_k);
    
    // Apply softmax to top-k
    float max_prob = *std::max_element(indices.begin(), indices.end(),
                                      [&probs](int a, int b) { return probs[a] < probs[b]; });
    
    std::vector<float> top_k_probs(top_k);
    float sum = 0.0f;
    
    for (int i = 0; i < top_k; i++) {
        top_k_probs[i] = std::exp(probs[indices[i]] - max_prob);
        sum += top_k_probs[i];
    }
    
    // Normalize
    for (float& p : top_k_probs) {
        p /= sum;
    }
    
    // Prepare for sampling
    std::vector<float> cumulative(top_k);
    std::partial_sum(top_k_probs.begin(), top_k_probs.end(), cumulative.begin());
    
    // Sample
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng_);
    
    // Find the index where r falls
    int idx = std::lower_bound(cumulative.begin(), cumulative.end(), r) - cumulative.begin();
    
    // Return the token
    return indices[idx];
}

bool GGMLModel::has_weight(const std::string& name) const {
    return weights_.find(name) != weights_.end();
}

struct ggml_tensor* GGMLModel::get_weight(const std::string& name) const {
    auto it = weights_.find(name);
    if (it == weights_.end()) {
        throw std::runtime_error("Weight not found: " + name);
    }
    return it->second;
}

struct ggml_context* GGMLModel::create_computation_context(size_t mem_size) {
    struct ggml_init_params params = {
        .mem_size   = mem_size,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    
    return ggml_init(params);
}

} // namespace ccsm