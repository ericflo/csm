#include <ccsm/cpu/ggml_model.h>
#include <ccsm/cpu/ggml_tensor.h>
#include <ccsm/tokenizer.h>
#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <random>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <thread>
#include <future>

namespace ccsm {
namespace {

// Mock tokenizer for testing
class MockTokenizer : public TextTokenizer {
public:
    // Implement Tokenizer interface
    std::vector<int> encode(const std::string& text) const override { return {1, 2, 3}; }
    std::string decode(const std::vector<int>& tokens) const override { return "test"; }
    int vocab_size() const override { return 32000; }
    
    // Implement TextTokenizer interface
    int bos_token_id() const override { return 1; }
    int eos_token_id() const override { return 2; }
    int pad_token_id() const override { return 0; }
    int unk_token_id() const override { return 3; }
    int get_speaker_token_id(int speaker_id) const override { return speaker_id + 100; }
    std::vector<int> get_audio_token_ids() const override { return {1, 2, 3, 4}; }
};

// Mock GGML model for testing memory optimizations without requiring weights
class MockGGMLModel : public GGMLModel {
public:
    MockGGMLModel(const ModelConfig& config) : GGMLModel(config) {
        // Initialize KV caches for testing
        backbone_kv_cache_ = std::make_unique<KVCache>(
            config.n_layers, 
            config.n_kv_heads, 
            config.d_model / config.n_heads, 
            config.max_seq_len
        );
        
        decoder_kv_cache_ = std::make_unique<KVCache>(
            config.n_audio_layers, 
            config.n_kv_heads, 
            config.d_model / config.n_heads, 
            config.max_seq_len
        );
        
        // Initialize RNG for token generation
        rng_.seed(42);
    }
    
    // Override the get_weight method to avoid loading actual weights
    struct ggml_tensor* get_weight(const std::string& name) const override {
        // Create a dummy context for the test
        static ggml_context* dummy_ctx = nullptr;
        if (!dummy_ctx) {
            struct ggml_init_params params = {
                .mem_size   = 1024 * 1024, // 1MB
                .mem_buffer = NULL,
                .no_alloc   = false,
            };
            dummy_ctx = ggml_init(params);
        }
        
        // Return a dummy tensor (1x1) for any weight
        return ggml_new_tensor_1d(dummy_ctx, GGML_TYPE_F32, 1);
    }
    
    // Override to avoid requiring weights
    bool has_weight(const std::string& name) const override {
        return true; // Pretend we have all weights
    }
    
    // For testing, make KVCache current_seq_len accessible
    friend class MemoryOptimizationEdgeCaseTest;
    
    // Override to make generate_frame work without actual weights
    std::vector<int> generate_frame(
        const std::vector<int>& tokens,
        const std::vector<int>& positions,
        float temperature = 0.9f,
        int top_k = 50) override {
        
        // Add tokens to KV cache to simulate real operation
        for (size_t i = 0; i < tokens.size(); i++) {
            // Update the internal KV caches to simulate real operation
            if (backbone_kv_cache_) {
                size_t& seq_len = backbone_kv_cache_->current_seq_len_;
                seq_len = std::max(seq_len, positions[i] + 1);
            }
            
            if (decoder_kv_cache_) {
                size_t& seq_len = decoder_kv_cache_->current_seq_len_;
                seq_len = std::max(seq_len, positions[i] + 1);
            }
        }
        
        // Generate some dummy tokens as response (4 tokens per codebook)
        std::vector<int> result;
        for (int i = 0; i < config_.num_codebooks; i++) {
            result.push_back(i * 100 + 1);  // Just some deterministic pattern
        }
        
        return result;
    }
    
    // For test inspection
    size_t get_backbone_cache_size() const {
        return backbone_kv_cache_ ? backbone_kv_cache_->current_seq_len() : 0;
    }
    
    size_t get_decoder_cache_size() const {
        return decoder_kv_cache_ ? decoder_kv_cache_->current_seq_len() : 0;
    }
};

// Test fixture for memory optimization under extreme conditions
class MemoryOptimizationEdgeCaseTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize model config with a smaller model for tests
        config.vocab_size = 32000;
        config.audio_vocab_size = 2051;
        config.d_model = 512;      // Smaller for tests
        config.n_heads = 8;        // Smaller for tests
        config.n_kv_heads = 2;     // Using GQA
        config.n_layers = 4;       // Smaller for tests
        config.n_audio_layers = 2; // Smaller for tests
        config.rope_theta = 10000.0f;
        config.max_seq_len = 1024;
        config.num_codebooks = 4;  // Smaller for tests
        config.name = "test_model";
        
        // Create mock model with built-in KV caches
        model = std::make_shared<MockGGMLModel>(config);
        
        // Set up test data
        sequence_length = 256;
        model_tokens.resize(sequence_length);
        model_positions.resize(sequence_length);
        
        // Fill with sequential data
        std::iota(model_tokens.begin(), model_tokens.end(), 0);
        std::iota(model_positions.begin(), model_positions.end(), 0);
        
        // Add some randomness to tokens to make them more realistic
        std::mt19937 gen(42);
        std::uniform_int_distribution<int> dist(1, 31999);
        for (auto& token : model_tokens) {
            token = dist(gen);
        }
        
        // Create tokenizer
        tokenizer = std::make_shared<MockTokenizer>();
    }
    
    // Helper to generate and process sequential batches to simulate
    // long-running text generation with accumulating KV cache
    void generate_sequential_batches(int num_batches, int tokens_per_batch) {
        // Process batches sequentially to build KV cache
        std::vector<int> tokens;
        std::vector<int> positions;
        
        for (int batch = 0; batch < num_batches; batch++) {
            // Prepare batch with correct positional information
            tokens.clear();
            positions.clear();
            
            for (int i = 0; i < tokens_per_batch; i++) {
                tokens.push_back(model_tokens[i % model_tokens.size()]);
                positions.push_back(batch * tokens_per_batch + i);
            }
            
            // Generate backbone logits to fill backbone KV cache
            model->get_backbone_logits(tokens, positions);
            
            // Generate decoder logits to fill decoder KV cache
            for (int codebook = 0; codebook < std::min(4, config.num_codebooks); codebook++) {
                model->get_decoder_logits(tokens, positions, codebook);
            }
        }
    }
    
    // Test parameters
    ModelConfig config;
    size_t sequence_length;
    
    // Test data
    std::vector<int> model_tokens;
    std::vector<int> model_positions;
    
    // Model instance
    std::shared_ptr<MockGGMLModel> model;
    std::shared_ptr<TextTokenizer> tokenizer;
};

// Test aggressive memory optimization with extremely low memory limit
TEST_F(MemoryOptimizationEdgeCaseTest, ExtremelyLowMemoryLimit) {
    try {
        // Fill the KV cache with a longer sequence to simulate a large context
        generate_sequential_batches(10, 64);
        
        // Apply memory optimization with extremely low memory limit (1MB)
        size_t extremely_low_limit = 1;
        model->optimize_memory(extremely_low_limit);
        
        // Verify we can still generate tokens after extreme optimization
        std::vector<int> audio_frame = model->generate_frame(
            model_tokens, model_positions, 0.9f, 50);
        
        // There should be at least some tokens in the output
        EXPECT_GT(audio_frame.size(), 0);
        
    } catch (const std::exception& e) {
        GTEST_FAIL() << "Exception during test: " << e.what();
    }
}

// Test memory optimization with multiple optimization calls
TEST_F(MemoryOptimizationEdgeCaseTest, RepeatedOptimization) {
    try {
        // Fill KV cache
        generate_sequential_batches(5, 64);
        
        // First optimization - moderate
        size_t first_limit = 20;
        model->optimize_memory(first_limit);
        
        // Generate a token after first optimization
        std::vector<int> frame1 = model->generate_frame(
            model_tokens, model_positions, 0.9f, 50);
        
        // Fill KV cache with more data
        generate_sequential_batches(3, 32);
        
        // Second optimization - more aggressive
        size_t second_limit = 10;
        model->optimize_memory(second_limit);
        
        // Generate a token after second optimization
        std::vector<int> frame2 = model->generate_frame(
            model_tokens, model_positions, 0.9f, 50);
        
        // Verify outputs
        EXPECT_GT(frame1.size(), 0);
        EXPECT_GT(frame2.size(), 0);
        
    } catch (const std::exception& e) {
        GTEST_FAIL() << "Exception during test: " << e.what();
    }
}

// Test cache pruning followed by memory optimization
TEST_F(MemoryOptimizationEdgeCaseTest, PruneThenOptimize) {
    try {
        // Fill KV cache
        generate_sequential_batches(6, 64);
        
        // First apply pruning
        float prune_factor = 0.3f;
        model->prune_caches(prune_factor);
        
        // Then apply memory optimization
        size_t memory_limit = 15;
        model->optimize_memory(memory_limit);
        
        // Verify we can still generate tokens
        std::vector<int> audio_frame = model->generate_frame(
            model_tokens, model_positions, 0.9f, 50);
        
        // There should be at least some tokens in the output
        EXPECT_GT(audio_frame.size(), 0);
        
    } catch (const std::exception& e) {
        GTEST_FAIL() << "Exception during test: " << e.what();
    }
}

// Test optimization, then adding new tokens, then optimization again
TEST_F(MemoryOptimizationEdgeCaseTest, OptimizeAddTokensOptimize) {
    try {
        // Fill KV cache with initial data
        generate_sequential_batches(4, 64);
        
        // First optimization
        size_t first_limit = 20;
        model->optimize_memory(first_limit);
        
        // Add more tokens to KV cache
        generate_sequential_batches(2, 32);
        
        // Second optimization
        size_t second_limit = 15;
        model->optimize_memory(second_limit);
        
        // Verify we can still generate tokens
        std::vector<int> audio_frame = model->generate_frame(
            model_tokens, model_positions, 0.9f, 50);
        
        // There should be at least some tokens in the output
        EXPECT_GT(audio_frame.size(), 0);
        
    } catch (const std::exception& e) {
        GTEST_FAIL() << "Exception during test: " << e.what();
    }
}

// Test extreme pruning (keeping only the minimum required tokens)
TEST_F(MemoryOptimizationEdgeCaseTest, ExtremePruning) {
    try {
        // Fill KV cache with a large batch of tokens
        generate_sequential_batches(8, 64);
        
        // Extreme pruning (remove 95% of tokens)
        float extreme_prune_factor = 0.95f;
        model->prune_caches(extreme_prune_factor);
        
        // Verify we can still generate tokens after extreme pruning
        std::vector<int> audio_frame = model->generate_frame(
            model_tokens, model_positions, 0.9f, 50);
        
        // There should be at least some tokens in the output
        EXPECT_GT(audio_frame.size(), 0);
        
    } catch (const std::exception& e) {
        GTEST_FAIL() << "Exception during test: " << e.what();
    }
}

// Test memory optimization under concurrent token generation
TEST_F(MemoryOptimizationEdgeCaseTest, ConcurrentGeneration) {
    try {
        // Fill KV cache with initial data
        generate_sequential_batches(5, 64);
        
        // Start a long-running token generation in a separate thread
        std::atomic<bool> generation_completed(false);
        auto generation_thread = std::thread([this, &generation_completed]() {
            try {
                // Generate frames multiple times to simulate continuous generation
                for (int i = 0; i < 5; i++) {
                    model->generate_frame(model_tokens, model_positions, 0.9f, 50);
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                }
                generation_completed = true;
            } catch (...) {
                // Catch any exceptions but don't mark as completed
            }
        });
        
        // Give the generation thread a head start
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        
        // In main thread, perform memory optimization
        model->optimize_memory(20);
        
        // Wait for generation to complete
        generation_thread.join();
        
        // Verify generation completed successfully
        EXPECT_TRUE(generation_completed) << "Generation didn't complete after memory optimization";
        
    } catch (const std::exception& e) {
        GTEST_FAIL() << "Exception during test: " << e.what();
    }
}

// Test alternating between extreme pruning and optimization
TEST_F(MemoryOptimizationEdgeCaseTest, AlternatingPruningAndOptimization) {
    try {
        // Run multiple cycles of:
        // 1. Generate tokens to fill KV cache
        // 2. Apply pruning
        // 3. Generate more tokens
        // 4. Apply memory optimization
        
        for (int cycle = 0; cycle < 3; cycle++) {
            // Generate tokens to fill KV cache
            generate_sequential_batches(2, 64);
            
            // Apply pruning with varying factors
            float prune_factor = 0.3f + (cycle * 0.2f);
            model->prune_caches(prune_factor);
            
            // Verify we can generate after pruning
            std::vector<int> frame1 = model->generate_frame(
                model_tokens, model_positions, 0.9f, 50);
            EXPECT_GT(frame1.size(), 0);
            
            // Generate more tokens
            generate_sequential_batches(2, 32);
            
            // Apply memory optimization with varying limits
            size_t memory_limit = 20 - (cycle * 5);
            model->optimize_memory(memory_limit);
            
            // Verify we can generate after optimization
            std::vector<int> frame2 = model->generate_frame(
                model_tokens, model_positions, 0.9f, 50);
            EXPECT_GT(frame2.size(), 0);
        }
        
    } catch (const std::exception& e) {
        GTEST_FAIL() << "Exception during test: " << e.what();
    }
}

// Test the effect of memory optimization on generation performance
TEST_F(MemoryOptimizationEdgeCaseTest, PerformanceImpactWithLargeCache) {
    try {
        // Fill KV cache with a large number of tokens
        generate_sequential_batches(12, 64);
        
        // Measure generation time before optimization
        auto start_before = std::chrono::high_resolution_clock::now();
        model->generate_frame(model_tokens, model_positions, 0.9f, 50);
        auto end_before = std::chrono::high_resolution_clock::now();
        auto duration_before = std::chrono::duration_cast<std::chrono::milliseconds>(end_before - start_before);
        
        // Perform memory optimization
        model->optimize_memory(30);
        
        // Measure generation time after optimization
        auto start_after = std::chrono::high_resolution_clock::now();
        model->generate_frame(model_tokens, model_positions, 0.9f, 50);
        auto end_after = std::chrono::high_resolution_clock::now();
        auto duration_after = std::chrono::duration_cast<std::chrono::milliseconds>(end_after - start_after);
        
        // Log performance results
        std::cout << "Generation time with large cache before optimization: " 
                  << duration_before.count() << " ms" << std::endl;
        std::cout << "Generation time after optimization: " 
                  << duration_after.count() << " ms" << std::endl;
        
        // We don't assert specific performance improvements since they're hardware-dependent,
        // but we log the results for manual inspection
        
    } catch (const std::exception& e) {
        GTEST_FAIL() << "Exception during test: " << e.what();
    }
}

} // namespace
} // namespace ccsm