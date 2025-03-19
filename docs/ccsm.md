# CCSM: C++ Inference Engine for Conversational Speech Model

## Overview

This document outlines the implementation plan for developing a C++ based inference engine for the Conversational Speech Model (CSM). The goal is to create efficient, portable binaries that can run CSM inference across multiple hardware configurations with a focus on performance and low resource consumption.

## Model Architecture Understanding

Before diving into implementation details, it's crucial to understand CSM's dual-transformer architecture:

1. **Backbone Transformer** (~1B parameters in CSM-1B):
   - A LLaMA 3.2-based Transformer that processes both text and audio tokens
   - Maintains a 2048-token context window (roughly 2 minutes of conversation)
   - Generates "semantic audio tokens" (codebook 0) that represent high-level speech content

2. **Audio Decoder Transformer** (~100M parameters in CSM-1B):
   - A smaller Transformer that takes the semantic token from the backbone
   - Generates the remaining acoustic codebook tokens (codebooks 1 through N-1)
   - These tokens contain fine-grained acoustic details like timbre and intonation

3. **Tokenization System**:
   - Text is tokenized using a SentencePiece tokenizer (Llama 3-style)
   - Audio is tokenized using the Mimi codec (operating at 12.5Hz, 1 frame every 80ms)
   - Each audio frame is represented by 32 codebook tokens (1 semantic + 31 acoustic)

4. **Generation Loop**:
   - The backbone predicts the next semantic token (codebook 0)
   - The decoder predicts the remaining acoustic tokens for that frame (codebooks 1-31)
   - The complete set of tokens for a frame is decoded to speech and fed back into the model
   - This continues until an end-of-utterance token is produced

## Goals

1. Create a modular C++ implementation that supports multiple backend configurations:
   - CPU-only implementation (`ccsm-generate`)
   - MLX acceleration for Apple Silicon (`ccsm-generate-mlx`)
   - Future extensibility for CUDA and Vulkan backends

2. Achieve feature parity with the Python implementation while delivering:
   - Lower memory footprint
   - Faster inference
   - No Python dependencies
   - Smaller distributable binaries

3. Design a flexible architecture that allows for:
   - Easy addition of new hardware backends
   - Shared code for model loading, tokenization, and audio processing
   - Backend-specific optimizations

## Implementation Plan Checklist

### Phase 1: Foundation and Core Components (Week 1)

- [ ] **1.1 Project Structure and Build System**
  - [ ] Set up CMake configuration with flexible backend support:
    ```cmake
    option(WITH_MLX "Build with MLX support" ON)
    option(WITH_CUDA "Build with CUDA support" OFF)
    option(WITH_VULKAN "Build with Vulkan support" OFF)
    ```
  - [ ] Configure header-only GGML integration from llama.cpp (vendored)
  - [ ] Set up MLX-C integration for Apple Silicon builds
  - [ ] Create platform detection macros for conditional compilation

- [ ] **1.2 Tensor Abstraction Layer**
  - [ ] Design a unified tensor interface supporting both GGML and MLX:
    ```cpp
    class Tensor {
    public:
        virtual ~Tensor() = default;
        virtual size_t shape(int dim) const = 0;
        virtual int ndim() const = 0;
        virtual void* data() = 0;
        virtual const void* data() const = 0;
        // Backend-specific methods
    };

    class GGMLTensor : public Tensor {
    private:
        struct ggml_tensor* tensor;
        // Implementation
    };

    class MLXTensor : public Tensor {
    private:
        mlx_array array;
        // Implementation
    };
    ```
  - [ ] Implement tensor factory for backend-agnostic creation
  - [ ] Implement tensor conversion utilities between backends

- [ ] **1.3 Common Utilities**
  - [ ] Implement logging system with configurable verbosity levels
  - [ ] Create memory management utilities for both backends
  - [ ] Implement file I/O utilities for model loading
  - [ ] Create error handling system with proper error propagation

- [ ] **1.4 Model Weight Management**
  - [ ] Design a unified weight storage system compatible with both backends
  - [ ] Implement GGUF loader from llama.cpp (for compatibility)
  - [ ] Create checkpoint conversion utility for PyTorch → GGUF format
  - [ ] Implement direct loading from PyTorch checkpoints (`.pt` files)

### Phase 2: CPU Backend Implementation (Week 2-3)

- [ ] **2.1 GGML Integration**
  - [ ] Set up minimal GGML context with appropriate allocators
  - [ ] Implement GGML tensor allocation with proper quantization:
    ```cpp
    struct ggml_tensor* allocate_tensor(struct ggml_context* ctx, 
                                     enum ggml_type type,
                                     int n_dims, const int64_t* dims) {
        // Properly handle memory allocation for tensors
    }
    ```
  - [ ] Create GGML computation graph builder

- [ ] **2.2 Transformer Implementation for CPU**
  - [ ] Implement Llama 3.2-style attention mechanism using GGML:
    ```cpp
    // Multi-head attention with rotary embeddings
    struct ggml_tensor* llama_attention(struct ggml_context* ctx,
                                     struct ggml_tensor* q,
                                     struct ggml_tensor* k,
                                     struct ggml_tensor* v,
                                     struct ggml_tensor* mask,
                                     int n_heads, int n_kv_heads,
                                     float scale) {
        // Implementation based on llama.cpp patterns
    }
    ```
  - [ ] Implement SwiGLU feed-forward network with GGML
  - [ ] Implement rotary position embeddings using GGML primitives
  - [ ] Create backbone and decoder transformer implementations
  - [ ] Implement KV-caching strategy for efficient inference

- [ ] **2.3 Tokenization Module**
  - [ ] Implement SentencePiece-compatible text tokenizer:
    ```cpp
    class TextTokenizer {
    public:
        TextTokenizer(const std::string& model_path);
        std::vector<int> encode(const std::string& text);
        std::string decode(const std::vector<int>& tokens);
    private:
        // SentencePiece model internals
    };
    ```
  - [ ] Create Mimi-compatible audio tokenizer:
    ```cpp
    class AudioTokenizer {
    public:
        AudioTokenizer(const std::string& model_path);
        // Encode audio to tokens (for context)
        std::vector<std::vector<int>> encode(const std::vector<float>& audio);
        // Decode tokens to audio
        std::vector<float> decode(const std::vector<std::vector<int>>& tokens);
    private:
        // Mimi codec internals (integrate from Kyutai public code)
    };
    ```

- [ ] **2.4 Sampling Implementation**
  - [ ] Implement top-k sampling for token generation:
    ```cpp
    int sample_token(const float* logits, int vocab_size, 
                     int top_k, float temperature, 
                     std::mt19937& rng) {
        // Implement efficient top-k sampling without unnecessary copies
    }
    ```
  - [ ] Implement efficient logits processor with temperature scaling

- [ ] **2.5 CPU-Specific Optimizations**
  - [ ] Implement SIMD optimizations for critical paths
  - [ ] Optimize memory layout for cache efficiency
  - [ ] Implement thread-pool for parallel computation when possible

### Phase 3: MLX Backend Implementation (Week 3-4)

- [ ] **3.1 MLX-C Integration**
  - [ ] Create MLX array wrappers for tensors
  - [ ] Implement MLX tensor operations interface
  - [ ] Set up MLX device management:
    ```cpp
    class MLXDevice {
    public:
        static bool is_available();
        static const char* name();
        static void synchronize();
    private:
        // MLX device management
    };
    ```

- [ ] **3.2 Transformer Implementation for MLX**
  - [ ] Implement Llama 3.2-style attention using MLX-C:
    ```cpp
    mlx_array mlx_attention(mlx_array query, mlx_array key, 
                          mlx_array value, mlx_array mask,
                          int n_heads, int n_kv_heads,
                          float scale) {
        // Implementation using MLX-C primitives
    }
    ```
  - [ ] Implement SwiGLU feed-forward network using MLX operations
  - [ ] Create MLX-specific implementation of rotary embeddings
  - [ ] Build MLX transformer layers for backbone and decoder
  - [ ] Implement MLX KV-caching for efficient inference

- [ ] **3.3 MLX-Specific Optimizations**
  - [ ] Leverage MLX's parallel processing capabilities
  - [ ] Optimize memory transfers between CPU and MLX
  - [ ] Implement MLX-specific tensor layout optimizations
  - [ ] Configure MLX compute precision for best performance/quality balance

- [ ] **3.4 PyTorch -> MLX Weight Conversion**
  - [ ] Create direct PyTorch checkpoint → MLX weight conversion:
    ```cpp
    void convert_pytorch_to_mlx(const std::string& pt_path,
                              const std::string& mlx_path) {
        // Load PyTorch weights and convert to MLX format
    }
    ```
  - [ ] Implement proper data type conversion (bfloat16, etc.)

### Phase 4: Model Generation Pipeline (Week 4-5)

- [ ] **4.1 Model Interface**
  - [ ] Design unified model interface supporting both backends:
    ```cpp
    class Model {
    public:
        virtual ~Model() = default;
        virtual void load_weights(const std::string& path) = 0;
        virtual std::vector<int> generate_frame(const std::vector<int>& tokens,
                                            const std::vector<int>& positions,
                                            float temperature,
                                            int top_k) = 0;
        virtual void reset_caches() = 0;
    };
    ```
  - [ ] Implement CPU and MLX model variants
  - [ ] Create model factory for dynamic backend selection

- [ ] **4.2 Token Generation Logic**
  - [ ] Implement complete token generation pipeline:
    ```cpp
    std::vector<std::vector<int>> generate_tokens(
        Model& model, const std::string& text, int speaker_id,
        float temperature, int top_k) {
        // Text tokenization
        // Backbone inference for semantic tokens
        // Decoder inference for acoustic tokens
        // Frame by frame generation
    }
    ```
  - [ ] Implement backbone token generation for semantic tokens
  - [ ] Implement decoder token generation for acoustic tokens
  - [ ] Create end-to-end pipeline connecting both transformers

- [ ] **4.3 Context Management**
  - [ ] Implement context window management:
    ```cpp
    class ContextManager {
    public:
        void add_segment(const std::string& text, int speaker_id,
                        const std::vector<float>& audio = {});
        std::vector<int> get_tokens();
        std::vector<int> get_positions();
    private:
        // Context tracking and management
    };
    ```
  - [ ] Create token position tracking for transformer
  - [ ] Implement efficient context pruning for long conversations

### Phase 5: Audio Processing (Week 5-6)

- [ ] **5.1 Implement Mimi Codec Integration**
  - [ ] Port or integrate Mimi encoder/decoder to C++:
    ```cpp
    class MimiCodec {
    public:
        MimiCodec(const std::string& model_path);
        // Encode audio to tokens (32 codebooks)
        std::vector<std::vector<int>> encode(const std::vector<float>& audio);
        // Decode tokens to audio
        std::vector<float> decode(const std::vector<std::vector<int>>& tokens);
    private:
        // Internal state for the codec
    };
    ```
  - [ ] Ensure compatibility with Kyutai's implementation
  - [ ] Optimize for efficient CPU/GPU operation

- [ ] **5.2 Audio Output Processing**
  - [ ] Implement audio normalization and post-processing
  - [ ] Create WAV file output functionality
  - [ ] Add real-time audio output capabilities (optional)

- [ ] **5.3 Audio Watermarking**
  - [ ] Port SilentCipher watermarking to C++:
    ```cpp
    class Watermarker {
    public:
        Watermarker(const std::string& key);
        std::vector<float> apply_watermark(const std::vector<float>& audio);
        bool detect_watermark(const std::vector<float>& audio);
    private:
        // SilentCipher implementation details
    };
    ```
  - [ ] Ensure compatibility with original implementation
  - [ ] Optimize for minimal impact on audio quality

### Phase 6: Command-Line Interface (Week 6-7)

- [ ] **6.1 CLI Arguments Parser**
  - [ ] Create argument parser with all necessary options:
    ```cpp
    struct CLIArgs {
        std::string model_path;
        std::string text;
        int speaker_id;
        std::string output_path;
        std::vector<std::string> context_audio;
        std::vector<std::string> context_text;
        std::vector<int> context_speaker;
        int max_audio_length_ms;
        float temperature;
        int top_k;
        int seed;
        bool debug;
        // Backend-specific args
    };

    CLIArgs parse_args(int argc, char** argv);
    ```
  - [ ] Implement help text and usage information
  - [ ] Add proper error handling for invalid arguments

- [ ] **6.2 Main Application Logic**
  - [ ] Implement entry points for both binaries
  - [ ] Create progress reporting mechanism
  - [ ] Add performance metrics reporting

- [ ] **6.3 Model Loading Infrastructure**
  - [ ] Implement model discovery and auto-download:
    ```cpp
    std::string find_or_download_model(const std::string& model_name) {
        // Check local paths
        // Download from Hugging Face if not found
        // Return path to model
    }
    ```
  - [ ] Implement Hugging Face Hub integration for model downloading
  - [ ] Create model caching system

### Phase 7: Testing and Optimization (Week 7-8)

- [ ] **7.1 Testing Framework**
  - [ ] Set up unit testing infrastructure
  - [ ] Create component tests for critical functions
  - [ ] Implement end-to-end tests with reference outputs

- [ ] **7.2 Performance Benchmarking**
  - [ ] Create benchmarking system for measuring:
    - Token generation speed
    - Memory usage
    - Real-time factor (RTF)
    - Initialization time

- [ ] **7.3 Final Optimizations**
  - [ ] Profile and optimize critical paths
  - [ ] Implement quantization options
  - [ ] Optimize binary size and memory footprint

## Technical Implementation Details

### Tensor Operations for GGML

For CPU implementation, we'll use GGML from llama.cpp for tensor operations. Key operations to implement include:

```cpp
// Rotary position embeddings
struct ggml_tensor* apply_rotary_embeddings(
    struct ggml_context* ctx,
    struct ggml_tensor* x, 
    struct ggml_tensor* sin, 
    struct ggml_tensor* cos) {
    
    // Get q and k tensors from x
    struct ggml_tensor* q = ggml_view_2d(ctx, x, n_embd/2, n_tokens, 
                                      ggml_element_size(x)*n_embd, 0);
    struct ggml_tensor* k = ggml_view_2d(ctx, x, n_embd/2, n_tokens, 
                                      ggml_element_size(x)*n_embd, 
                                      ggml_element_size(x)*n_embd/2);
    
    // Apply RoPE to q
    struct ggml_tensor* q_interleaved = ggml_rope(ctx, q, sin, cos, pos, n_rot, 0);
    // Apply RoPE to k
    struct ggml_tensor* k_interleaved = ggml_rope(ctx, k, sin, cos, pos, n_rot, 0);
    
    // Concatenate
    return ggml_concat(ctx, q_interleaved, k_interleaved);
}

// Multi-head attention
struct ggml_tensor* multi_head_attention(
    struct ggml_context* ctx,
    struct ggml_tensor* q,
    struct ggml_tensor* k,
    struct ggml_tensor* v,
    struct ggml_tensor* mask,
    int n_heads) {
    
    // Split q, k, v into heads
    struct ggml_tensor* q_heads = ggml_reshape_3d(ctx, q, q->ne[0]/n_heads, n_heads, q->ne[1]);
    struct ggml_tensor* k_heads = ggml_reshape_3d(ctx, k, k->ne[0]/n_heads, n_heads, k->ne[1]);
    struct ggml_tensor* v_heads = ggml_reshape_3d(ctx, v, v->ne[0]/n_heads, n_heads, v->ne[1]);
    
    // QK attention
    float scale = 1.0f / sqrtf(q->ne[0]/n_heads);
    struct ggml_tensor* qk = ggml_mul_mat(ctx, k_heads, q_heads);
    qk = ggml_scale(ctx, qk, scale);
    
    // Apply mask
    if (mask) {
        qk = ggml_add(ctx, qk, mask);
    }
    
    // Softmax
    struct ggml_tensor* qk_soft = ggml_soft_max(ctx, qk);
    
    // Apply to values
    struct ggml_tensor* qkv = ggml_mul_mat(ctx, v_heads, qk_soft);
    
    // Merge heads
    return ggml_reshape_2d(ctx, qkv, qkv->ne[0]*n_heads, qkv->ne[2]);
}

// SwiGLU feed-forward network
struct ggml_tensor* swiglu_ffn(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* w1,
    struct ggml_tensor* w2,
    struct ggml_tensor* w3) {
    
    // First projection
    struct ggml_tensor* a = ggml_mul_mat(ctx, w1, x);
    
    // Second projection
    struct ggml_tensor* b = ggml_mul_mat(ctx, w3, x);
    
    // SwiGLU activation
    struct ggml_tensor* ab = ggml_mul(ctx, a, ggml_silu(ctx, b));
    
    // Output projection
    return ggml_mul_mat(ctx, w2, ab);
}
```

### MLX Tensor Operations

For MLX implementation, we'll use the mlx-c API:

```cpp
// Rotary position embeddings with MLX
mlx_array mlx_rotary_embeddings(mlx_array x, mlx_array sin, mlx_array cos, int pos) {
    // Extract q and k from x
    int n_embd = mlx_array_dim(x, 0);
    mlx_array q = mlx_slice(x, 0, 0, n_embd/2);
    mlx_array k = mlx_slice(x, 0, n_embd/2, n_embd);
    
    // Apply RoPE to q and k
    mlx_array q_rotated = mlx_fast_rope(q, sin, cos, pos);
    mlx_array k_rotated = mlx_fast_rope(k, sin, cos, pos);
    
    // Concatenate and return
    return mlx_concatenate(q_rotated, k_rotated, 0);
}

// Multi-head attention with MLX
mlx_array mlx_multi_head_attention(
    mlx_array q, mlx_array k, mlx_array v, 
    mlx_array mask, int n_heads) {
    
    // Reshape into heads
    int head_dim = mlx_array_dim(q, 0) / n_heads;
    mlx_array q_heads = mlx_reshape(q, {head_dim, n_heads, -1});
    mlx_array k_heads = mlx_reshape(k, {head_dim, n_heads, -1});
    mlx_array v_heads = mlx_reshape(v, {head_dim, n_heads, -1});
    
    // QK attention
    float scale = 1.0f / sqrtf(head_dim);
    mlx_array qk = mlx_matmul(q_heads, k_heads, true);
    qk = mlx_multiply(qk, mlx_scalar(scale));
    
    // Apply mask
    if (mask.ctx != NULL) {
        qk = mlx_add(qk, mask);
    }
    
    // Softmax
    mlx_array qk_soft = mlx_softmax(qk, -1);
    
    // Apply to values
    mlx_array qkv = mlx_matmul(qk_soft, v_heads);
    
    // Reshape back
    return mlx_reshape(qkv, {head_dim * n_heads, -1});
}

// SwiGLU feed-forward network with MLX
mlx_array mlx_swiglu_ffn(
    mlx_array x, mlx_array w1, mlx_array w2, mlx_array w3) {
    
    // First projection
    mlx_array a = mlx_matmul(x, w1);
    
    // Second projection
    mlx_array b = mlx_matmul(x, w3);
    
    // SwiGLU activation
    mlx_array silu_b = mlx_silu(b);
    mlx_array ab = mlx_multiply(a, silu_b);
    
    // Output projection
    return mlx_matmul(ab, w2);
}
```

### Model Loading and Weight Management

Both backends will need to load weights from PyTorch checkpoints. We'll implement a common weight loading system:

```cpp
class WeightLoader {
public:
    // Load from PyTorch checkpoint
    static bool load_from_pytorch(const std::string& path, WeightMap& weights) {
        // Open file
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            return false;
        }
        
        // Parse PyTorch serialization format
        // Extract weights into WeightMap
        // Handle different parameter naming conventions
        
        return true;
    }
    
    // Load from GGUF format (for CPU backend)
    static bool load_from_gguf(const std::string& path, WeightMap& weights) {
        // Use llama.cpp's GGUF reader
        struct gguf_context* ctx = gguf_init_from_file(path.c_str());
        if (!ctx) {
            return false;
        }
        
        // Extract tensors and metadata
        // Convert to our internal format
        
        gguf_free(ctx);
        return true;
    }
};
```

### Audio Tokenization with Mimi

For audio tokenization, we'll need to port or integrate Kyutai's Mimi codec:

```cpp
class MimiCodec {
private:
    // Internal model parameters
    std::vector<Tensor> codebooks;
    std::vector<Tensor> codebook_embeddings;
    std::vector<Tensor> quantizers;
    Tensor decoder;
    
    // Sample rate and other parameters
    int sample_rate = 24000;
    int audio_vocab_size = 2051;
    int num_codebooks = 32;
    
public:
    // Initialize from model files
    MimiCodec(const std::string& model_path) {
        // Load model weights
        // Initialize necessary components
    }
    
    // Encode audio to tokens
    std::vector<std::vector<int>> encode(const std::vector<float>& audio) {
        // Preprocess audio (resample if needed)
        // Apply mel filterbank
        // Encode through RVQ quantizers
        // Return tokens for each codebook
    }
    
    // Decode tokens to audio
    std::vector<float> decode(const std::vector<std::vector<int>>& tokens) {
        // Convert tokens to codebook embeddings
        // Apply decoder network
        // Postprocess audio
        // Return waveform
    }
};
```

### Generation Pipeline

The full generation pipeline will tie all components together:

```cpp
class Generator {
private:
    std::shared_ptr<Model> model;
    std::shared_ptr<TextTokenizer> text_tokenizer;
    std::shared_ptr<MimiCodec> audio_codec;
    std::shared_ptr<Watermarker> watermarker;
    
public:
    Generator(std::shared_ptr<Model> model,
             std::shared_ptr<TextTokenizer> text_tokenizer,
             std::shared_ptr<MimiCodec> audio_codec,
             std::shared_ptr<Watermarker> watermarker)
        : model(model), text_tokenizer(text_tokenizer),
          audio_codec(audio_codec), watermarker(watermarker) {}
    
    // Main generation function
    std::vector<float> generate_speech(
        const std::string& text,
        int speaker_id,
        const std::vector<Segment>& context,
        float temperature = 0.9f,
        int top_k = 50,
        int max_audio_length_ms = 10000,
        std::function<void(int, int)> progress_callback = nullptr) {
        
        // 1. Tokenize text and prepare context
        std::vector<int> text_tokens = text_tokenizer->encode(text);
        
        // 2. Build context tokens
        ContextManager context_mgr;
        for (const auto& segment : context) {
            context_mgr.add_segment(segment.text, segment.speaker_id, segment.audio);
        }
        context_mgr.add_segment(text, speaker_id);
        
        std::vector<int> context_tokens = context_mgr.get_tokens();
        std::vector<int> positions = context_mgr.get_positions();
        
        // 3. Generate frames
        model->reset_caches();
        
        std::vector<std::vector<int>> audio_tokens;
        int max_frames = max_audio_length_ms / 80; // 80ms per frame
        
        for (int i = 0; i < max_frames; i++) {
            // Generate next frame of tokens
            std::vector<int> frame = model->generate_frame(
                context_tokens, positions, temperature, top_k);
            
            // Check for EOS
            if (is_eos_frame(frame)) {
                break;
            }
            
            // Add frame to results
            audio_tokens.push_back(frame);
            
            // Update context with new frame
            context_tokens = update_context(context_tokens, frame);
            positions = update_positions(positions);
            
            // Report progress
            if (progress_callback) {
                progress_callback(i + 1, max_frames);
            }
        }
        
        // 4. Decode audio tokens to waveform
        std::vector<float> audio = audio_codec->decode(audio_tokens);
        
        // 5. Apply watermark
        std::vector<float> watermarked_audio = watermarker->apply_watermark(audio);
        
        return watermarked_audio;
    }
};
```

## Development Strategy

Our development strategy will follow these principles:

1. **Incremental Implementation**: Start with minimal working components and gradually add features.

2. **Continuous Testing**: Test each component thoroughly as it's developed.

3. **Performance-First Mindset**: Design with performance in mind from the beginning.

4. **MLX/CPU Parity**: Ensure both backends provide the same output quality.

5. **Modular Design**: Create components that can be reused and extended.

Initial focus will be on getting a minimal working CPU-based implementation, followed by the MLX backend and then refinements to both.

## Future Extension Points

The architecture will support future extensions for:

1. **CUDA Backend**: For NVIDIA GPU acceleration using CUDA kernels.

2. **Vulkan Backend**: For cross-platform GPU acceleration.

3. **Quantization Schemes**: Supporting different precision levels (4-bit, 8-bit).

4. **Streaming API**: For real-time generation and playback.

5. **Language Model Integration**: For seamless integration with LLM-based systems.

## Conclusion

This implementation plan provides a comprehensive roadmap for developing a C++ inference engine for CSM. By following this detailed plan with a focus on both the CPU and MLX backends, we'll create a high-performance, flexible solution that can run efficiently across different hardware platforms while maintaining the quality of the original Python implementation.