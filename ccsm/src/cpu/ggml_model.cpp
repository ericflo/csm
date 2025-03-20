#include <ccsm/cpu/ggml_model.h>
#include <ccsm/model_loader.h>
#include <ccsm/utils.h>

// Use direct includes for GGML
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"

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
    
    // Build the backbone computation graph and get the logits tensor
    struct ggml_cgraph* graph;
    struct ggml_tensor* logits_tensor;
    
    std::tie(graph, logits_tensor) = build_backbone_graph(comp_ctx, tokens, positions);
    
    // Compute the graph
    ggml_graph_compute_with_ctx(comp_ctx, graph, 1); // Use 1 thread for now
    
    // Copy logits to vector
    std::vector<float> logits(config_.audio_vocab_size);
    float* logits_data = (float*)logits_tensor->data;
    std::copy(logits_data, logits_data + config_.audio_vocab_size, logits.begin());
    
    // Free resources
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
    
    // Build the decoder computation graph and get the logits tensor
    struct ggml_cgraph* graph;
    struct ggml_tensor* logits_tensor;
    
    std::tie(graph, logits_tensor) = build_decoder_graph(comp_ctx, tokens, positions, codebook);
    
    // Compute the graph
    ggml_graph_compute_with_ctx(comp_ctx, graph, 1); // Use 1 thread for now
    
    // Copy logits to vector
    std::vector<float> logits(config_.audio_vocab_size);
    float* logits_data = (float*)logits_tensor->data;
    std::copy(logits_data, logits_data + config_.audio_vocab_size, logits.begin());
    
    // Free resources
    ggml_free(comp_ctx);
    
    return logits;
}

std::pair<struct ggml_cgraph*, struct ggml_tensor*> GGMLModel::build_backbone_graph(
    struct ggml_context* ctx,
    const std::vector<int>& tokens,
    const std::vector<int>& positions) {
    
    // Make sure we have tokens
    if (tokens.empty()) {
        throw std::runtime_error("Empty token sequence provided to backbone model");
    }
    
    // Create computation graph
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    
    // Shorthand for configuration values
    const int n_embd = config_.d_model;
    const int n_layer = config_.n_layers;
    const int n_head = config_.n_heads;
    const int n_kv_head = config_.n_kv_heads;
    const int n_vocab = config_.vocab_size;
    const int n_audio_vocab = config_.audio_vocab_size;
    const int head_dim = n_embd / n_head;
    const int kv_dim = n_embd / n_head;
    
    // Make sure KV cache is properly sized
    backbone_kv_cache_->resize(positions.back() + 1);
    
    // Create input tokens tensor
    struct ggml_tensor* input_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tokens.size());
    std::memcpy(input_tokens->data, tokens.data(), tokens.size() * sizeof(int));
    
    // Create input positions tensor
    struct ggml_tensor* input_positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, positions.size());
    std::memcpy(input_positions->data, positions.data(), positions.size() * sizeof(int));
    
    // Token embedding
    struct ggml_tensor* tok_embeddings = get_weight("model.tok_embeddings.weight");
    struct ggml_tensor* embd = ggml_get_rows(ctx, tok_embeddings, input_tokens);
    
    // Pre-compute sin/cos for RoPE
    struct ggml_tensor* rope_cos;
    struct ggml_tensor* rope_sin;
    float theta = config_.rope_theta;
    
    // Sin and cos tables for rotary embeddings
    rope_cos = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim / 2);
    rope_sin = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim / 2);
    
    for (int i = 0; i < head_dim / 2; i++) {
        float freq = 1.0f / powf(theta, (2.0f * i) / head_dim);
        
        float* cos_data = (float*)rope_cos->data;
        float* sin_data = (float*)rope_sin->data;
        
        cos_data[i] = freq;
        sin_data[i] = freq;
    }
    
    // Transformer layers
    for (int layer_idx = 0; layer_idx < n_layer; layer_idx++) {
        // Layer prefix for weight names
        std::string layer_prefix = "model.layers." + std::to_string(layer_idx) + ".";
        
        // Add residual connection
        struct ggml_tensor* residual = embd;
        
        // First normalization (attention norm)
        struct ggml_tensor* attn_norm_weight = get_weight(layer_prefix + "attention_norm.weight");
        struct ggml_tensor* attn_norm_bias = get_weight(layer_prefix + "attention_norm.bias");
        struct ggml_tensor* cur = ggml_rms_norm(ctx, embd, 1e-5f);
        cur = ggml_mul(ctx, cur, attn_norm_weight);
        cur = ggml_add(ctx, cur, attn_norm_bias);
        
        // QKV projections
        struct ggml_tensor* wq = get_weight(layer_prefix + "attention.wq.weight");
        struct ggml_tensor* wk = get_weight(layer_prefix + "attention.wk.weight");
        struct ggml_tensor* wv = get_weight(layer_prefix + "attention.wv.weight");
        
        // Project to Q, K, V
        struct ggml_tensor* q = ggml_mul_mat(ctx, wq, cur);
        struct ggml_tensor* k = ggml_mul_mat(ctx, wk, cur);
        struct ggml_tensor* v = ggml_mul_mat(ctx, wv, cur);
        
        // Reshape Q for attention
        struct ggml_tensor* q_reshaped = ggml_reshape_3d(ctx, q, head_dim, n_head, tokens.size());
        q_reshaped = ggml_permute(ctx, q_reshaped, 0, 2, 1, 3); // [head_dim, tokens, n_head, 1]
        
        // Reshape K for attention (grouped-query attention)
        struct ggml_tensor* k_reshaped = ggml_reshape_3d(ctx, k, kv_dim, n_kv_head, tokens.size());
        k_reshaped = ggml_permute(ctx, k_reshaped, 0, 2, 1, 3); // [kv_dim, tokens, n_kv_head, 1]
        
        // Reshape V for attention (grouped-query attention)
        struct ggml_tensor* v_reshaped = ggml_reshape_3d(ctx, v, kv_dim, n_kv_head, tokens.size());
        v_reshaped = ggml_permute(ctx, v_reshaped, 1, 2, 0, 3); // [tokens, n_kv_head, kv_dim, 1]
        
        // Apply rotary embeddings (RoPE) to Q and K
        q_reshaped = ggml_rope_inplace(ctx, q_reshaped, input_positions, head_dim/2, 0);
        k_reshaped = ggml_rope_inplace(ctx, k_reshaped, input_positions, kv_dim/2, 0);
        
        // Update KV cache
        struct ggml_tensor* k_cache = backbone_kv_cache_->k_cache(layer_idx);
        struct ggml_tensor* v_cache = backbone_kv_cache_->v_cache(layer_idx);
        
        // Update K cache - copy the new k values into the cache
        for (size_t i = 0; i < positions.size(); i++) {
            int pos = positions[i];
            struct ggml_tensor* k_cur = ggml_view_1d(ctx, k_reshaped, kv_dim * n_kv_head, 
                                                    i * kv_dim * n_kv_head * sizeof(float));
            struct ggml_tensor* k_cache_view = ggml_view_2d(ctx, k_cache, kv_dim, n_kv_head, 
                                                         kv_dim * n_kv_head * sizeof(float), 
                                                         pos * kv_dim * n_kv_head * sizeof(float));
            ggml_build_forward_expand(graph, ggml_cpy(ctx, k_cur, k_cache_view));
        }
        
        // Update V cache - copy the new v values into the cache
        for (size_t i = 0; i < positions.size(); i++) {
            int pos = positions[i];
            struct ggml_tensor* v_cur = ggml_view_1d(ctx, v_reshaped, kv_dim * n_kv_head, 
                                                    i * kv_dim * n_kv_head * sizeof(float));
            struct ggml_tensor* v_cache_view = ggml_view_2d(ctx, v_cache, kv_dim, n_kv_head,
                                                         kv_dim * n_kv_head * sizeof(float), 
                                                         pos * kv_dim * n_kv_head * sizeof(float));
            ggml_build_forward_expand(graph, ggml_cpy(ctx, v_cur, v_cache_view));
        }
        
        // Use K and V from the cache to perform attention calculation
        struct ggml_tensor* k_seq = ggml_view_3d(ctx, k_cache, kv_dim, n_kv_head, positions.back() + 1,
                                             kv_dim * sizeof(float), kv_dim * n_kv_head * sizeof(float), 0);
        struct ggml_tensor* v_seq = ggml_view_3d(ctx, v_cache, kv_dim, n_kv_head, positions.back() + 1,
                                             kv_dim * sizeof(float), kv_dim * n_kv_head * sizeof(float), 0);
        
        // Reshape K and V for attention
        k_seq = ggml_permute(ctx, k_seq, 0, 2, 1, 3); // [kv_dim, seq_len, n_kv_head, 1]
        v_seq = ggml_permute(ctx, v_seq, 1, 2, 0, 3); // [seq_len, n_kv_head, kv_dim, 1]
        
        // Grouped-query attention
        // Each query head attends to a specific KV head
        // We need to repeat KV heads to match the number of query heads
        struct ggml_tensor* kv_heads_repeated;
        if (n_head > n_kv_head) {
            // Repeat KV heads to match the number of query heads
            int repeats = n_head / n_kv_head;
            kv_heads_repeated = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kv_dim, positions.back() + 1, n_head);
            
            for (int head = 0; head < n_head; head++) {
                int kv_head = head / repeats;
                struct ggml_tensor* src = ggml_view_2d(ctx, k_seq, kv_dim, positions.back() + 1,
                                                      kv_dim * (positions.back() + 1) * sizeof(float),
                                                      kv_head * kv_dim * (positions.back() + 1) * sizeof(float));
                struct ggml_tensor* dst = ggml_view_2d(ctx, kv_heads_repeated, kv_dim, positions.back() + 1,
                                                      kv_dim * (positions.back() + 1) * sizeof(float),
                                                      head * kv_dim * (positions.back() + 1) * sizeof(float));
                ggml_build_forward_expand(graph, ggml_cpy(ctx, src, dst));
            }
            k_seq = kv_heads_repeated;
            
            // Repeat V heads in the same way
            kv_heads_repeated = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, positions.back() + 1, n_head, kv_dim);
            
            for (int head = 0; head < n_head; head++) {
                int kv_head = head / repeats;
                struct ggml_tensor* src = ggml_view_2d(ctx, v_seq, positions.back() + 1, kv_dim,
                                                      positions.back() + 1 * kv_dim * sizeof(float),
                                                      kv_head * positions.back() + 1 * kv_dim * sizeof(float));
                struct ggml_tensor* dst = ggml_view_2d(ctx, kv_heads_repeated, positions.back() + 1, kv_dim,
                                                      positions.back() + 1 * kv_dim * sizeof(float),
                                                      head * positions.back() + 1 * kv_dim * sizeof(float));
                ggml_build_forward_expand(graph, ggml_cpy(ctx, src, dst));
            }
            v_seq = kv_heads_repeated;
        }
        
        // QK attention
        struct ggml_tensor* att_scores = ggml_mul_mat(ctx, k_seq, q_reshaped);
        
        // Scale attention scores
        float scale = 1.0f / sqrtf(head_dim);
        att_scores = ggml_scale_inplace(ctx, att_scores, scale);
        
        // Apply causal mask: we need to mask future tokens
        // Create causal mask [seq_len, seq_len]
        struct ggml_tensor* causal_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, positions.back() + 1, tokens.size());
        float* mask_data = (float*)causal_mask->data;
        
        for (size_t i = 0; i < tokens.size(); i++) {
            int pos_i = positions[i];
            for (int j = 0; j <= positions.back(); j++) {
                // If j > pos_i, token j is in the future for token i
                mask_data[i * (positions.back() + 1) + j] = j > pos_i ? -INFINITY : 0.0f;
            }
        }
        
        // Apply mask to attention scores
        att_scores = ggml_add(ctx, att_scores, causal_mask);
        
        // Softmax
        att_scores = ggml_soft_max_inplace(ctx, att_scores);
        
        // Apply attention to value vectors
        struct ggml_tensor* output = ggml_mul_mat(ctx, v_seq, att_scores);
        
        // Reshape and permute to get the original shape
        output = ggml_permute(ctx, output, 0, 2, 1, 3); // [head_dim, n_head, tokens, 1]
        output = ggml_reshape_2d(ctx, output, n_embd, tokens.size()); // [n_embd, tokens]
        
        // Projection
        struct ggml_tensor* wo = get_weight(layer_prefix + "attention.wo.weight");
        output = ggml_mul_mat(ctx, wo, output);
        
        // Add the residual connection
        embd = ggml_add(ctx, output, residual);
        
        // Second residual connection
        residual = embd;
        
        // Second normalization (FFN norm)
        struct ggml_tensor* ffn_norm_weight = get_weight(layer_prefix + "ffn_norm.weight");
        struct ggml_tensor* ffn_norm_bias = get_weight(layer_prefix + "ffn_norm.bias");
        cur = ggml_rms_norm(ctx, embd, 1e-5f);
        cur = ggml_mul(ctx, cur, ffn_norm_weight);
        cur = ggml_add(ctx, cur, ffn_norm_bias);
        
        // SwiGLU Feed-Forward Network
        // SwiGLU: Act(xW1) * xW3
        struct ggml_tensor* w1 = get_weight(layer_prefix + "feed_forward.w1.weight");
        struct ggml_tensor* w3 = get_weight(layer_prefix + "feed_forward.w3.weight");
        struct ggml_tensor* w2 = get_weight(layer_prefix + "feed_forward.w2.weight");
        
        struct ggml_tensor* ffn_out1 = ggml_mul_mat(ctx, w1, cur);
        struct ggml_tensor* ffn_out3 = ggml_mul_mat(ctx, w3, cur);
        
        // SiLU activation for SwiGLU: x * sigmoid(x)
        struct ggml_tensor* ffn_out1_silu = ggml_silu(ctx, ffn_out1);
        
        // Element-wise multiplication
        struct ggml_tensor* ffn_out = ggml_mul(ctx, ffn_out1_silu, ffn_out3);
        
        // Final projection
        ffn_out = ggml_mul_mat(ctx, w2, ffn_out);
        
        // Add residual
        embd = ggml_add(ctx, ffn_out, residual);
    }
    
    // Final normalization
    struct ggml_tensor* norm_weight = get_weight("model.norm.weight");
    struct ggml_tensor* norm_bias = get_weight("model.norm.bias");
    struct ggml_tensor* normalized = ggml_rms_norm(ctx, embd, 1e-5f);
    normalized = ggml_mul(ctx, normalized, norm_weight);
    normalized = ggml_add(ctx, normalized, norm_bias);
    
    // Take only the last token's embedding
    normalized = ggml_view_1d(ctx, normalized, n_embd, (tokens.size() - 1) * n_embd * sizeof(float));
    
    // Output projection (to audio vocabulary)
    struct ggml_tensor* output_weight = get_weight("lm_head.weight");
    
    // Reshape output weight for audio vocabulary if needed
    struct ggml_tensor* audio_output_weight;
    if (output_weight->ne[0] != n_audio_vocab) {
        // If the output weight shape doesn't match the audio vocabulary size,
        // create a view of the appropriate size
        if (output_weight->ne[0] < n_audio_vocab) {
            // If the weight is smaller, pad with zeros
            audio_output_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_audio_vocab, n_embd);
            ggml_set_zero(audio_output_weight);
            
            struct ggml_tensor* weight_view = ggml_view_2d(
                ctx, audio_output_weight, output_weight->ne[0], n_embd, 
                n_audio_vocab * sizeof(float), 0
            );
            
            ggml_build_forward_expand(graph, ggml_cpy(ctx, output_weight, weight_view));
        } else {
            // If the weight is larger, use a smaller view
            audio_output_weight = ggml_view_2d(
                ctx, output_weight, n_audio_vocab, n_embd,
                output_weight->ne[0] * sizeof(float), 0
            );
        }
    } else {
        audio_output_weight = output_weight;
    }
    
    // Project to audio vocabulary
    struct ggml_tensor* logits = ggml_mul_mat(ctx, audio_output_weight, normalized);
    
    // Build forward graph with the final tensor (logits)
    ggml_build_forward_expand(graph, logits);
    
    return {graph, logits};
}

std::pair<struct ggml_cgraph*, struct ggml_tensor*> GGMLModel::build_decoder_graph(
    struct ggml_context* ctx,
    const std::vector<int>& tokens,
    const std::vector<int>& positions,
    int codebook) {
    
    // Make sure we have tokens
    if (tokens.empty()) {
        throw std::runtime_error("Empty token sequence provided to decoder model");
    }
    
    // Create computation graph
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    
    // Shorthand for configuration values
    const int n_embd = config_.d_model;
    const int n_layer = config_.n_audio_layers;
    const int n_head = config_.n_heads;
    const int n_kv_head = config_.n_kv_heads;
    const int n_vocab = config_.audio_vocab_size;
    const int head_dim = n_embd / n_head;
    const int kv_dim = n_embd / n_head;
    
    // Make sure KV cache is properly sized
    decoder_kv_cache_->resize(positions.back() + 1);
    
    // Create input tokens tensor
    struct ggml_tensor* input_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tokens.size());
    std::memcpy(input_tokens->data, tokens.data(), tokens.size() * sizeof(int));
    
    // Create input positions tensor
    struct ggml_tensor* input_positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, positions.size());
    std::memcpy(input_positions->data, positions.data(), positions.size() * sizeof(int));
    
    // Token embedding
    struct ggml_tensor* tok_embeddings = get_weight("decoder.tok_embeddings.weight");
    struct ggml_tensor* embd = ggml_get_rows(ctx, tok_embeddings, input_tokens);
    
    // Pre-compute sin/cos for RoPE
    struct ggml_tensor* rope_cos;
    struct ggml_tensor* rope_sin;
    float theta = config_.rope_theta;
    
    // Sin and cos tables for rotary embeddings
    rope_cos = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim / 2);
    rope_sin = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim / 2);
    
    for (int i = 0; i < head_dim / 2; i++) {
        float freq = 1.0f / powf(theta, (2.0f * i) / head_dim);
        
        float* cos_data = (float*)rope_cos->data;
        float* sin_data = (float*)rope_sin->data;
        
        cos_data[i] = freq;
        sin_data[i] = freq;
    }
    
    // Transformer layers
    for (int layer_idx = 0; layer_idx < n_layer; layer_idx++) {
        // Layer prefix for weight names
        std::string layer_prefix = "decoder.layers." + std::to_string(layer_idx) + ".";
        
        // Add residual connection
        struct ggml_tensor* residual = embd;
        
        // First normalization (attention norm)
        struct ggml_tensor* attn_norm_weight = get_weight(layer_prefix + "attention_norm.weight");
        struct ggml_tensor* attn_norm_bias = get_weight(layer_prefix + "attention_norm.bias");
        struct ggml_tensor* cur = ggml_rms_norm(ctx, embd, 1e-5f);
        cur = ggml_mul(ctx, cur, attn_norm_weight);
        cur = ggml_add(ctx, cur, attn_norm_bias);
        
        // QKV projections
        struct ggml_tensor* wq = get_weight(layer_prefix + "attention.wq.weight");
        struct ggml_tensor* wk = get_weight(layer_prefix + "attention.wk.weight");
        struct ggml_tensor* wv = get_weight(layer_prefix + "attention.wv.weight");
        
        // Project to Q, K, V
        struct ggml_tensor* q = ggml_mul_mat(ctx, wq, cur);
        struct ggml_tensor* k = ggml_mul_mat(ctx, wk, cur);
        struct ggml_tensor* v = ggml_mul_mat(ctx, wv, cur);
        
        // Reshape Q for attention
        struct ggml_tensor* q_reshaped = ggml_reshape_3d(ctx, q, head_dim, n_head, tokens.size());
        q_reshaped = ggml_permute(ctx, q_reshaped, 0, 2, 1, 3); // [head_dim, tokens, n_head, 1]
        
        // Reshape K for attention (grouped-query attention)
        struct ggml_tensor* k_reshaped = ggml_reshape_3d(ctx, k, kv_dim, n_kv_head, tokens.size());
        k_reshaped = ggml_permute(ctx, k_reshaped, 0, 2, 1, 3); // [kv_dim, tokens, n_kv_head, 1]
        
        // Reshape V for attention (grouped-query attention)
        struct ggml_tensor* v_reshaped = ggml_reshape_3d(ctx, v, kv_dim, n_kv_head, tokens.size());
        v_reshaped = ggml_permute(ctx, v_reshaped, 1, 2, 0, 3); // [tokens, n_kv_head, kv_dim, 1]
        
        // Apply rotary embeddings (RoPE) to Q and K
        q_reshaped = ggml_rope_inplace(ctx, q_reshaped, input_positions, head_dim/2, 0);
        k_reshaped = ggml_rope_inplace(ctx, k_reshaped, input_positions, kv_dim/2, 0);
        
        // Update KV cache
        struct ggml_tensor* k_cache = decoder_kv_cache_->k_cache(layer_idx);
        struct ggml_tensor* v_cache = decoder_kv_cache_->v_cache(layer_idx);
        
        // Update K cache - copy the new k values into the cache
        for (size_t i = 0; i < positions.size(); i++) {
            int pos = positions[i];
            struct ggml_tensor* k_cur = ggml_view_1d(ctx, k_reshaped, kv_dim * n_kv_head, 
                                                    i * kv_dim * n_kv_head * sizeof(float));
            struct ggml_tensor* k_cache_view = ggml_view_2d(ctx, k_cache, kv_dim, n_kv_head, 
                                                         kv_dim * n_kv_head * sizeof(float), 
                                                         pos * kv_dim * n_kv_head * sizeof(float));
            ggml_build_forward_expand(graph, ggml_cpy(ctx, k_cur, k_cache_view));
        }
        
        // Update V cache - copy the new v values into the cache
        for (size_t i = 0; i < positions.size(); i++) {
            int pos = positions[i];
            struct ggml_tensor* v_cur = ggml_view_1d(ctx, v_reshaped, kv_dim * n_kv_head, 
                                                    i * kv_dim * n_kv_head * sizeof(float));
            struct ggml_tensor* v_cache_view = ggml_view_2d(ctx, v_cache, kv_dim, n_kv_head,
                                                         kv_dim * n_kv_head * sizeof(float), 
                                                         pos * kv_dim * n_kv_head * sizeof(float));
            ggml_build_forward_expand(graph, ggml_cpy(ctx, v_cur, v_cache_view));
        }
        
        // Use K and V from the cache to perform attention calculation
        struct ggml_tensor* k_seq = ggml_view_3d(ctx, k_cache, kv_dim, n_kv_head, positions.back() + 1,
                                             kv_dim * sizeof(float), kv_dim * n_kv_head * sizeof(float), 0);
        struct ggml_tensor* v_seq = ggml_view_3d(ctx, v_cache, kv_dim, n_kv_head, positions.back() + 1,
                                             kv_dim * sizeof(float), kv_dim * n_kv_head * sizeof(float), 0);
        
        // Reshape K and V for attention
        k_seq = ggml_permute(ctx, k_seq, 0, 2, 1, 3); // [kv_dim, seq_len, n_kv_head, 1]
        v_seq = ggml_permute(ctx, v_seq, 1, 2, 0, 3); // [seq_len, n_kv_head, kv_dim, 1]
        
        // Grouped-query attention 
        // Each query head attends to a specific KV head
        // We need to repeat KV heads to match the number of query heads
        struct ggml_tensor* kv_heads_repeated;
        if (n_head > n_kv_head) {
            // Repeat KV heads to match the number of query heads
            int repeats = n_head / n_kv_head;
            kv_heads_repeated = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kv_dim, positions.back() + 1, n_head);
            
            for (int head = 0; head < n_head; head++) {
                int kv_head = head / repeats;
                struct ggml_tensor* src = ggml_view_2d(ctx, k_seq, kv_dim, positions.back() + 1,
                                                      kv_dim * (positions.back() + 1) * sizeof(float),
                                                      kv_head * kv_dim * (positions.back() + 1) * sizeof(float));
                struct ggml_tensor* dst = ggml_view_2d(ctx, kv_heads_repeated, kv_dim, positions.back() + 1,
                                                      kv_dim * (positions.back() + 1) * sizeof(float),
                                                      head * kv_dim * (positions.back() + 1) * sizeof(float));
                ggml_build_forward_expand(graph, ggml_cpy(ctx, src, dst));
            }
            k_seq = kv_heads_repeated;
            
            // Repeat V heads in the same way
            kv_heads_repeated = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, positions.back() + 1, n_head, kv_dim);
            
            for (int head = 0; head < n_head; head++) {
                int kv_head = head / repeats;
                struct ggml_tensor* src = ggml_view_2d(ctx, v_seq, positions.back() + 1, kv_dim,
                                                      positions.back() + 1 * kv_dim * sizeof(float),
                                                      kv_head * positions.back() + 1 * kv_dim * sizeof(float));
                struct ggml_tensor* dst = ggml_view_2d(ctx, kv_heads_repeated, positions.back() + 1, kv_dim,
                                                      positions.back() + 1 * kv_dim * sizeof(float),
                                                      head * positions.back() + 1 * kv_dim * sizeof(float));
                ggml_build_forward_expand(graph, ggml_cpy(ctx, src, dst));
            }
            v_seq = kv_heads_repeated;
        }
        
        // QK attention
        struct ggml_tensor* att_scores = ggml_mul_mat(ctx, k_seq, q_reshaped);
        
        // Scale attention scores
        float scale = 1.0f / sqrtf(head_dim);
        att_scores = ggml_scale_inplace(ctx, att_scores, scale);
        
        // Apply causal mask: we need to mask future tokens
        // Create causal mask [seq_len, seq_len]
        struct ggml_tensor* causal_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, positions.back() + 1, tokens.size());
        float* mask_data = (float*)causal_mask->data;
        
        for (size_t i = 0; i < tokens.size(); i++) {
            int pos_i = positions[i];
            for (int j = 0; j <= positions.back(); j++) {
                // If j > pos_i, token j is in the future for token i
                mask_data[i * (positions.back() + 1) + j] = j > pos_i ? -INFINITY : 0.0f;
            }
        }
        
        // Apply mask to attention scores
        att_scores = ggml_add(ctx, att_scores, causal_mask);
        
        // Softmax
        att_scores = ggml_soft_max_inplace(ctx, att_scores);
        
        // Apply attention to value vectors
        struct ggml_tensor* output = ggml_mul_mat(ctx, v_seq, att_scores);
        
        // Reshape and permute to get the original shape
        output = ggml_permute(ctx, output, 0, 2, 1, 3); // [head_dim, n_head, tokens, 1]
        output = ggml_reshape_2d(ctx, output, n_embd, tokens.size()); // [n_embd, tokens]
        
        // Projection
        struct ggml_tensor* wo = get_weight(layer_prefix + "attention.wo.weight");
        output = ggml_mul_mat(ctx, wo, output);
        
        // Add the residual connection
        embd = ggml_add(ctx, output, residual);
        
        // Second residual connection
        residual = embd;
        
        // Second normalization (FFN norm)
        struct ggml_tensor* ffn_norm_weight = get_weight(layer_prefix + "ffn_norm.weight");
        struct ggml_tensor* ffn_norm_bias = get_weight(layer_prefix + "ffn_norm.bias");
        cur = ggml_rms_norm(ctx, embd, 1e-5f);
        cur = ggml_mul(ctx, cur, ffn_norm_weight);
        cur = ggml_add(ctx, cur, ffn_norm_bias);
        
        // SwiGLU Feed-Forward Network
        // SwiGLU: Act(xW1) * xW3
        struct ggml_tensor* w1 = get_weight(layer_prefix + "feed_forward.w1.weight");
        struct ggml_tensor* w3 = get_weight(layer_prefix + "feed_forward.w3.weight");
        struct ggml_tensor* w2 = get_weight(layer_prefix + "feed_forward.w2.weight");
        
        struct ggml_tensor* ffn_out1 = ggml_mul_mat(ctx, w1, cur);
        struct ggml_tensor* ffn_out3 = ggml_mul_mat(ctx, w3, cur);
        
        // SiLU activation for SwiGLU: x * sigmoid(x)
        struct ggml_tensor* ffn_out1_silu = ggml_silu(ctx, ffn_out1);
        
        // Element-wise multiplication
        struct ggml_tensor* ffn_out = ggml_mul(ctx, ffn_out1_silu, ffn_out3);
        
        // Final projection
        ffn_out = ggml_mul_mat(ctx, w2, ffn_out);
        
        // Add residual
        embd = ggml_add(ctx, ffn_out, residual);
    }
    
    // Final normalization
    struct ggml_tensor* norm_weight = get_weight("decoder.norm.weight");
    struct ggml_tensor* norm_bias = get_weight("decoder.norm.bias");
    struct ggml_tensor* normalized = ggml_rms_norm(ctx, embd, 1e-5f);
    normalized = ggml_mul(ctx, normalized, norm_weight);
    normalized = ggml_add(ctx, normalized, norm_bias);
    
    // Take only the last token's embedding
    normalized = ggml_view_1d(ctx, normalized, n_embd, (tokens.size() - 1) * n_embd * sizeof(float));
    
    // Output projection (specific to each codebook)
    std::string output_name;
    if (codebook == 0) {
        // Use the same output weight as the backbone for the semantic codebook
        output_name = "lm_head.weight";
    } else {
        // Use the decoder's specific output weights for other codebooks
        output_name = "decoder.output." + std::to_string(codebook) + ".weight";
    }
    
    struct ggml_tensor* output_weight;
    try {
        output_weight = get_weight(output_name);
    } catch (const std::runtime_error& e) {
        // Fallback to common output weight if specific one not found
        CCSM_WARNING("Output weight not found for codebook ", codebook, ", using fallback");
        output_name = "decoder.output.weight";
        output_weight = get_weight(output_name);
    }
    
    // Make sure the output weight has the right shape for the audio vocabulary
    struct ggml_tensor* audio_output_weight;
    if (output_weight->ne[0] != n_vocab) {
        // If the output weight shape doesn't match the audio vocabulary size,
        // create a view of the appropriate size
        if (output_weight->ne[0] < n_vocab) {
            // If the weight is smaller, pad with zeros
            audio_output_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_vocab, n_embd);
            ggml_set_zero(audio_output_weight);
            
            struct ggml_tensor* weight_view = ggml_view_2d(
                ctx, audio_output_weight, output_weight->ne[0], n_embd, 
                n_vocab * sizeof(float), 0
            );
            
            ggml_build_forward_expand(graph, ggml_cpy(ctx, output_weight, weight_view));
        } else {
            // If the weight is larger, use a smaller view
            audio_output_weight = ggml_view_2d(
                ctx, output_weight, n_vocab, n_embd,
                output_weight->ne[0] * sizeof(float), 0
            );
        }
    } else {
        audio_output_weight = output_weight;
    }
    
    // Project to audio vocabulary
    struct ggml_tensor* logits = ggml_mul_mat(ctx, audio_output_weight, normalized);
    
    // Build forward graph with the final tensor (logits)
    ggml_build_forward_expand(graph, logits);
    
    return {graph, logits};
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