#include <ccsm/cpu/ggml_model.h>
#include <ccsm/cpu/ggml_tensor.h>
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

// Test fixture for memory optimization edge cases directly using KVCache
class MemoryOptimizationEdgeCaseTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Test parameters
        n_layers = 4;
        n_heads = 8;
        n_kv_heads = 2;  // Using GQA
        head_dim = 64;
        max_seq_len = 1024;
        
        // Create KV caches for backbone and decoder
        backbone_kv_cache = std::make_shared<KVCache>(n_layers, n_heads, n_kv_heads, head_dim, max_seq_len);
        decoder_kv_cache = std::make_shared<KVCache>(2, n_heads, n_kv_heads, head_dim, max_seq_len);
        
        // Initialize random number generator
        gen.seed(42);  // Fixed seed for reproducibility
    }
    
    // Helper to simulate growing the KV cache to a specific length
    void grow_kv_cache(std::shared_ptr<KVCache> cache, size_t target_seq_len) {
        if (!cache) return;
        
        size_t current = cache->current_seq_len();
        if (current >= target_seq_len) return;
        
        // Add enough tokens to reach target length
        // In a real model, this would happen by processing input tokens
        // Here we just update the internal sequence length counter
        while (current < target_seq_len) {
            current++;
            // In a real model, we would update the key and value tensors
            // but for testing memory optimization, we just need the sequence length
        }
        
        // Force update the internal sequence length
        cache->resize(current);
    }
    
    // Test parameters
    size_t n_layers;
    size_t n_heads;
    size_t n_kv_heads;
    size_t head_dim;
    size_t max_seq_len;
    
    // KV caches
    std::shared_ptr<KVCache> backbone_kv_cache;
    std::shared_ptr<KVCache> decoder_kv_cache;
    
    // Random number generator
    std::mt19937 gen;
};

// Test aggressive memory optimization with extremely low memory limit
TEST_F(MemoryOptimizationEdgeCaseTest, ExtremelyLowMemoryLimit) {
    try {
        // Fill KV caches with a longer sequence to simulate a large context
        grow_kv_cache(backbone_kv_cache, 640);
        grow_kv_cache(decoder_kv_cache, 640);
        
        // Get initial memory usage
        size_t initial_memory = backbone_kv_cache->memory_usage() + decoder_kv_cache->memory_usage();
        size_t initial_backbone_len = backbone_kv_cache->current_seq_len();
        size_t initial_decoder_len = decoder_kv_cache->current_seq_len();
        
        EXPECT_GT(initial_memory, 0);
        EXPECT_GT(initial_backbone_len, 0);
        EXPECT_GT(initial_decoder_len, 0);
        
        // Apply memory optimization with extremely low memory limit (1MB)
        size_t extremely_low_limit = 1 * 1024 * 1024; // 1MB
        
        // Calculate reduction factor
        float reduction_factor = static_cast<float>(extremely_low_limit) / initial_memory;
        
        // Apply to backbone cache
        size_t new_backbone_len = std::max(static_cast<size_t>(initial_backbone_len * reduction_factor), 
                                         static_cast<size_t>(64));
        backbone_kv_cache->resize(new_backbone_len);
        
        // Apply to decoder cache
        size_t new_decoder_len = std::max(static_cast<size_t>(initial_decoder_len * reduction_factor), 
                                        static_cast<size_t>(64));
        decoder_kv_cache->resize(new_decoder_len);
        
        // Check new memory usage and sequence lengths
        size_t final_memory = backbone_kv_cache->memory_usage() + decoder_kv_cache->memory_usage();
        size_t final_backbone_len = backbone_kv_cache->current_seq_len();
        size_t final_decoder_len = decoder_kv_cache->current_seq_len();
        
        // Verify the KV caches were reduced but still have a minimum context size
        EXPECT_LT(final_memory, initial_memory);
        EXPECT_LT(final_backbone_len, initial_backbone_len);
        EXPECT_LT(final_decoder_len, initial_decoder_len);
        
        // Minimum context size should be preserved
        EXPECT_GE(final_backbone_len, 64);
        EXPECT_GE(final_decoder_len, 64);
        
    } catch (const std::exception& e) {
        FAIL() << "Exception during test: " << e.what();
    }
}

// Test memory optimization with multiple optimization calls
TEST_F(MemoryOptimizationEdgeCaseTest, RepeatedOptimization) {
    try {
        // Fill KV caches
        grow_kv_cache(backbone_kv_cache, 320);
        grow_kv_cache(decoder_kv_cache, 320);
        
        // Initial state
        size_t initial_backbone_len = backbone_kv_cache->current_seq_len();
        size_t initial_decoder_len = decoder_kv_cache->current_seq_len();
        
        // First optimization - moderate
        float first_reduction = 0.8f;
        size_t new_backbone_len1 = static_cast<size_t>(initial_backbone_len * first_reduction);
        size_t new_decoder_len1 = static_cast<size_t>(initial_decoder_len * first_reduction);
        
        backbone_kv_cache->resize(new_backbone_len1);
        decoder_kv_cache->resize(new_decoder_len1);
        
        // Verify first optimization
        EXPECT_EQ(backbone_kv_cache->current_seq_len(), new_backbone_len1);
        EXPECT_EQ(decoder_kv_cache->current_seq_len(), new_decoder_len1);
        
        // Grow the caches again
        grow_kv_cache(backbone_kv_cache, new_backbone_len1 + 64);
        grow_kv_cache(decoder_kv_cache, new_decoder_len1 + 64);
        
        // Second optimization - more aggressive
        float second_reduction = 0.6f;
        size_t intermediate_backbone_len = backbone_kv_cache->current_seq_len();
        size_t intermediate_decoder_len = decoder_kv_cache->current_seq_len();
        
        size_t new_backbone_len2 = static_cast<size_t>(intermediate_backbone_len * second_reduction);
        size_t new_decoder_len2 = static_cast<size_t>(intermediate_decoder_len * second_reduction);
        
        backbone_kv_cache->resize(new_backbone_len2);
        decoder_kv_cache->resize(new_decoder_len2);
        
        // Verify second optimization
        EXPECT_EQ(backbone_kv_cache->current_seq_len(), new_backbone_len2);
        EXPECT_EQ(decoder_kv_cache->current_seq_len(), new_decoder_len2);
        
        // Ensure overall reduction
        EXPECT_LT(new_backbone_len2, initial_backbone_len);
        EXPECT_LT(new_decoder_len2, initial_decoder_len);
        
    } catch (const std::exception& e) {
        FAIL() << "Exception during test: " << e.what();
    }
}

// Test cache pruning with importance scores
TEST_F(MemoryOptimizationEdgeCaseTest, ImportanceBasedPruning) {
    try {
        // Fill KV caches
        size_t seq_len = 512;
        grow_kv_cache(backbone_kv_cache, seq_len);
        
        // Create importance scores - higher values for more important tokens
        std::vector<float> importance(seq_len);
        
        // Pattern: important tokens at beginning and end, with some important tokens scattered
        for (size_t i = 0; i < seq_len; i++) {
            if (i < 50 || i > seq_len - 50) {
                // Higher importance for tokens at beginning and end
                importance[i] = 1.0f + 0.1f * (std::min(i, seq_len - i - 1) % 10);
            } else if (i % 37 == 0) {
                // Some scattered important tokens
                importance[i] = 1.2f;
            } else {
                // Background tokens with lower importance
                importance[i] = 0.5f + 0.1f * (i % 5);
            }
        }
        
        // Target length after pruning - aggressive pruning
        size_t target_len = seq_len / 3;
        
        // Keep last N tokens regardless of importance
        size_t keep_last_n = 20;
        
        // Before pruning
        size_t before_len = backbone_kv_cache->current_seq_len();
        EXPECT_EQ(before_len, seq_len);
        
        // Apply pruning
        size_t after_len = backbone_kv_cache->prune(target_len, importance, keep_last_n);
        
        // Verify pruning results
        EXPECT_EQ(after_len, target_len);
        EXPECT_EQ(backbone_kv_cache->current_seq_len(), target_len);
        
        // Overall reduction happened
        EXPECT_LT(after_len, before_len);
        
    } catch (const std::exception& e) {
        FAIL() << "Exception during test: " << e.what();
    }
}

// Test extreme pruning with very high pruning factor
TEST_F(MemoryOptimizationEdgeCaseTest, ExtremePruning) {
    try {
        // Fill KV cache with a large number of tokens
        size_t seq_len = 1024;
        grow_kv_cache(backbone_kv_cache, seq_len);
        
        // Create importance scores with uniform values
        std::vector<float> importance(seq_len, 1.0f);
        
        // Extreme pruning - keep only 5% of tokens
        float extreme_prune_factor = 0.95f;
        size_t target_len = static_cast<size_t>(seq_len * (1.0f - extreme_prune_factor));
        
        // Ensure we keep a minimum number of tokens
        target_len = std::max(target_len, static_cast<size_t>(64));
        
        // Apply pruning with the minimum required token count
        size_t after_len = backbone_kv_cache->prune(target_len, importance, 0);
        
        // Verify pruning results
        EXPECT_EQ(after_len, target_len);
        EXPECT_EQ(backbone_kv_cache->current_seq_len(), target_len);
        
        // Test that we can still grow the cache after extreme pruning
        grow_kv_cache(backbone_kv_cache, target_len + 32);
        EXPECT_EQ(backbone_kv_cache->current_seq_len(), target_len + 32);
        
    } catch (const std::exception& e) {
        FAIL() << "Exception during test: " << e.what();
    }
}

// Test alternating between pruning and resizing
TEST_F(MemoryOptimizationEdgeCaseTest, AlternatingPruningAndResizing) {
    try {
        // Run multiple cycles of:
        // 1. Grow KV cache
        // 2. Apply pruning
        // 3. Grow again
        // 4. Apply resizing
        
        size_t initial_size = 256;
        size_t growth_size = 64;
        
        for (int cycle = 0; cycle < 3; cycle++) {
            // Grow KV cache
            grow_kv_cache(backbone_kv_cache, initial_size + (cycle * growth_size));
            size_t current_len = backbone_kv_cache->current_seq_len();
            
            // Create importance scores
            std::vector<float> importance(current_len, 1.0f);
            // Add some variation
            for (size_t i = 0; i < current_len; i++) {
                importance[i] = 0.8f + 0.4f * ((i + cycle) % 10) / 10.0f;
            }
            
            // Apply pruning with varying factors
            float prune_factor = 0.3f + (cycle * 0.1f);
            size_t target_len_prune = static_cast<size_t>(current_len * (1.0f - prune_factor));
            
            // Apply pruning
            size_t after_prune_len = backbone_kv_cache->prune(target_len_prune, importance, target_len_prune / 4);
            
            // Verify pruning results
            EXPECT_EQ(after_prune_len, target_len_prune);
            EXPECT_EQ(backbone_kv_cache->current_seq_len(), target_len_prune);
            
            // Grow again
            grow_kv_cache(backbone_kv_cache, after_prune_len + growth_size / 2);
            current_len = backbone_kv_cache->current_seq_len();
            
            // Apply resizing with varying factors
            float resize_factor = 0.8f - (cycle * 0.1f);
            size_t target_len_resize = static_cast<size_t>(current_len * resize_factor);
            
            // Apply resizing
            backbone_kv_cache->resize(target_len_resize);
            
            // Verify resizing results
            EXPECT_EQ(backbone_kv_cache->current_seq_len(), target_len_resize);
        }
        
    } catch (const std::exception& e) {
        FAIL() << "Exception during test: " << e.what();
    }
}

// Test memory optimization under concurrent access
TEST_F(MemoryOptimizationEdgeCaseTest, ConcurrentAccess) {
    try {
        // Fill KV cache with initial data
        grow_kv_cache(backbone_kv_cache, 320);
        
        // Get initial sequence length
        size_t initial_length = backbone_kv_cache->current_seq_len();
        
        // Start thread for continuously updating the cache
        std::atomic<bool> stop_signal(false);
        std::atomic<bool> resize_completed(false);
        std::atomic<bool> thread_running(true);
        
        auto updater_thread = std::thread([this, &stop_signal, &resize_completed, initial_length]() {
            // Keep adding to the cache until signaled to stop
            size_t new_length = initial_length;
            
            while (!stop_signal) {
                // Simulate adding tokens to the cache
                if (!resize_completed) {
                    // Before resize, grow slowly
                    new_length += 1;
                } else {
                    // After resize, try to grow more aggressively
                    new_length += 5;
                }
                
                // Grow the cache
                grow_kv_cache(backbone_kv_cache, new_length);
                
                // Simulate some processing time
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
        
        // Wait a bit for the thread to start adding tokens
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        
        // Apply resize in main thread while the updater is running
        size_t target_len = initial_length / 2;
        backbone_kv_cache->resize(target_len);
        
        // Mark that resize is completed
        resize_completed = true;
        
        // Let the thread keep running for a bit after resize
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        
        // Signal the thread to stop and wait for it
        stop_signal = true;
        updater_thread.join();
        
        // Check final length - should be larger than target_len due to continued growth
        size_t final_length = backbone_kv_cache->current_seq_len();
        EXPECT_GT(final_length, target_len);
        
        // Final length should be smaller than what it would have been without resizing
        EXPECT_LT(final_length, initial_length + 200);  // rough estimate
        
    } catch (const std::exception& e) {
        FAIL() << "Exception during test: " << e.what();
    }
}

// Test repeated resizing to verify memory stability
TEST_F(MemoryOptimizationEdgeCaseTest, RepeatedResizing) {
    try {
        // Start with a moderate-sized cache
        grow_kv_cache(backbone_kv_cache, 256);
        
        // Track memory usage
        std::vector<size_t> memory_usage;
        std::vector<size_t> sequence_lengths;
        
        // Perform multiple resize operations
        for (int i = 0; i < 10; i++) {
            // Record pre-resize state
            sequence_lengths.push_back(backbone_kv_cache->current_seq_len());
            memory_usage.push_back(backbone_kv_cache->memory_usage());
            
            // Calculate a new target length
            // Oscillate between growing and shrinking
            float factor = (i % 2 == 0) ? 0.7f : 1.3f;
            size_t target_len = static_cast<size_t>(backbone_kv_cache->current_seq_len() * factor);
            
            // Ensure we don't exceed max_seq_len
            target_len = std::min(target_len, max_seq_len);
            
            // Apply resize
            backbone_kv_cache->resize(target_len);
            
            // Verify resize was effective
            EXPECT_EQ(backbone_kv_cache->current_seq_len(), target_len);
        }
        
        // Verify memory usage patterns are consistent with sequence lengths
        for (size_t i = 1; i < memory_usage.size(); i++) {
            float seq_len_ratio = static_cast<float>(sequence_lengths[i]) / sequence_lengths[i-1];
            float memory_ratio = static_cast<float>(memory_usage[i]) / memory_usage[i-1];
            
            // Memory usage should scale roughly linearly with sequence length
            // Allow for some overhead/efficiency differences
            float ratio_difference = std::abs(seq_len_ratio - memory_ratio);
            EXPECT_LT(ratio_difference, 0.3f);
        }
        
    } catch (const std::exception& e) {
        FAIL() << "Exception during test: " << e.what();
    }
}

// Test combining pruning and resizing with a complex pattern
TEST_F(MemoryOptimizationEdgeCaseTest, ComplexPruningAndResizing) {
    try {
        // Fill initial cache
        grow_kv_cache(backbone_kv_cache, 512);
        
        // First, apply a moderate pruning
        std::vector<float> importance(512);
        // Create an interesting pattern of importance
        for (size_t i = 0; i < 512; i++) {
            // Tokens at specific positions are more important
            if (i % 50 < 10) {
                importance[i] = 1.5f;  // Important group of tokens
            } else if (i > 400) {
                importance[i] = 1.2f;  // Recent tokens are important
            } else {
                importance[i] = 0.8f;  // Less important tokens
            }
        }
        
        // Apply pruning with 30% reduction
        size_t target_after_prune = 360;
        backbone_kv_cache->prune(target_after_prune, importance, 40);
        
        // Verify pruning result
        EXPECT_EQ(backbone_kv_cache->current_seq_len(), target_after_prune);
        
        // Grow the cache again
        grow_kv_cache(backbone_kv_cache, 450);
        
        // Now apply a resize operation to shrink it differently
        size_t target_after_resize = 300;
        backbone_kv_cache->resize(target_after_resize);
        
        // Verify resize result
        EXPECT_EQ(backbone_kv_cache->current_seq_len(), target_after_resize);
        
        // Grow again
        grow_kv_cache(backbone_kv_cache, 400);
        
        // Final check
        EXPECT_EQ(backbone_kv_cache->current_seq_len(), 400);
        
    } catch (const std::exception& e) {
        FAIL() << "Exception during test: " << e.what();
    }
}

} // namespace
} // namespace ccsm