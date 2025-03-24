#include <ccsm/cpu/ggml_model.h>
#include <ccsm/cpu/ggml_tensor.h>
#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <random>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace ccsm {
namespace {

class KVCacheMemoryOptimizationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a GGML context
        ggml_ctx = std::make_shared<GGMLContext>();
        
        // Initialize random number generator
        std::mt19937 gen(42); // Fixed seed for reproducibility
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        
        // Test parameters
        n_layers = 4;        // More layers than in quantization tests
        n_heads = 8;
        n_kv_heads = 8;      // No GQA for simplicity in testing
        head_dim = 64;
        max_seq_len = 2048;  // Real-world transformer sequence length
        
        // Create KV cache
        kv_cache = std::make_shared<KVCache>(n_layers, n_heads, n_kv_heads, head_dim, max_seq_len);
        
        // Generate random test data - only generate a portion to save memory
        size_t sample_seq_len = 128; // Use a smaller sample for initial tests
        size_t k_sample_size = n_layers * n_kv_heads * head_dim * sample_seq_len;
        size_t v_sample_size = n_layers * n_kv_heads * head_dim * sample_seq_len;
        
        // Generate random key and value tensors for samples
        k_data.resize(k_sample_size);
        v_data.resize(v_sample_size);
        
        for (size_t i = 0; i < k_sample_size; i++) {
            k_data[i] = dist(gen);
        }
        
        for (size_t i = 0; i < v_sample_size; i++) {
            v_data[i] = dist(gen);
        }
    }
    
    // Test data
    size_t n_layers;
    size_t n_heads;
    size_t n_kv_heads;
    size_t head_dim;
    size_t max_seq_len;
    
    std::vector<float> k_data;
    std::vector<float> v_data;
    
    // GGML context
    std::shared_ptr<GGMLContext> ggml_ctx;
    
    // KV cache
    std::shared_ptr<KVCache> kv_cache;
    
    // Helper function to calculate memory usage of tensors
    size_t calculate_tensor_memory(struct ggml_tensor* tensor) {
        if (!tensor) return 0;
        
        size_t type_size = 0;
        switch (tensor->type) {
            case GGML_TYPE_F32: type_size = sizeof(float); break;
            case GGML_TYPE_F16: type_size = sizeof(uint16_t); break;
            case GGML_TYPE_Q8_0: type_size = sizeof(int8_t); break;
            case GGML_TYPE_Q4_0: type_size = sizeof(int8_t) / 2; break; // 4 bits
            case GGML_TYPE_Q4_1: type_size = sizeof(int8_t) / 2; break; // 4 bits
            default: type_size = sizeof(uint8_t); break;
        }
        
        // Calculate total size based on dimensions
        size_t nelements = 1;
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            nelements *= tensor->ne[i];
        }
        
        // Account for extra data needed by quantized types
        size_t extra_data = 0;
        if (tensor->type == GGML_TYPE_Q8_0) {
            // Q8_0 typically has one float per 32 elements (block size 32)
            extra_data = (nelements / 32) * sizeof(float);
        } else if (tensor->type == GGML_TYPE_Q4_0 || tensor->type == GGML_TYPE_Q4_1) {
            // Q4 types typically have one float per 32 elements (block size 32)
            extra_data = (nelements / 32) * sizeof(float);
            // Q4_1 has an additional scale factor
            if (tensor->type == GGML_TYPE_Q4_1) {
                extra_data += (nelements / 32) * sizeof(float);
            }
        }
        
        return nelements * type_size + extra_data;
    }
    
    // Helper function to calculate memory usage of KV cache
    size_t calculate_kv_cache_memory(const std::shared_ptr<KVCache>& cache) {
        size_t total_memory = 0;
        
        for (size_t i = 0; i < n_layers; i++) {
            total_memory += calculate_tensor_memory(cache->k_cache(i));
            total_memory += calculate_tensor_memory(cache->v_cache(i));
        }
        
        return total_memory;
    }
    
    // Helper function to copy data to a tensor with specified offset
    void copy_to_tensor(struct ggml_tensor* tensor, const std::vector<float>& data, 
                       size_t seq_offset, size_t seq_len) {
        if (!tensor) return;
        
        float* tensor_data = (float*)tensor->data;
        
        // Calculate dimensions
        size_t dim0 = tensor->ne[0]; // head_dim
        size_t dim1 = tensor->ne[1]; // seq_len
        size_t dim2 = tensor->ne[2]; // n_kv_heads
        
        // Determine the amount of data to copy (limited by available sample data)
        size_t copy_seq_len = std::min(seq_len, data.size() / (dim0 * dim2));
        
        // Copy data to the specified sequence position
        for (size_t h = 0; h < dim2; h++) {          // For each head
            for (size_t s = 0; s < copy_seq_len; s++) {  // For each sequence position
                for (size_t d = 0; d < dim0; d++) {      // For each dimension
                    // Calculate source and destination indices
                    size_t src_idx = h * copy_seq_len * dim0 + s * dim0 + d;
                    size_t dst_idx = h * dim1 * dim0 + (s + seq_offset) * dim0 + d;
                    
                    if (src_idx < data.size() && dst_idx < dim0 * dim1 * dim2) {
                        tensor_data[dst_idx] = data[src_idx];
                    }
                }
            }
        }
    }
    
    // Helper function to fill KV cache with test data
    void fill_kv_cache(size_t seq_len) {
        for (size_t i = 0; i < n_layers; i++) {
            copy_to_tensor(kv_cache->k_cache(i), k_data, 0, seq_len);
            copy_to_tensor(kv_cache->v_cache(i), v_data, 0, seq_len);
        }
    }
};

// Test basic memory allocation of KV cache
TEST_F(KVCacheMemoryOptimizationTest, BasicMemoryAllocation) {
    // Verify initial cache state
    EXPECT_EQ(kv_cache->current_seq_len(), 0);
    EXPECT_EQ(kv_cache->max_seq_len(), max_seq_len);
    
    // Calculate initial memory usage (should be for max_seq_len)
    size_t initial_memory = calculate_kv_cache_memory(kv_cache);
    
    // Print initial memory usage
    std::cout << "Initial KV cache memory allocation (max_seq_len=" << max_seq_len << "): " 
              << initial_memory << " bytes" << std::endl;
    
    // Expected memory usage for F32 KV cache
    // Each tensor is: n_kv_heads * head_dim * max_seq_len * sizeof(float)
    // Total for all layers: 2 * n_layers * n_kv_heads * head_dim * max_seq_len * sizeof(float)
    size_t expected_memory = 2 * n_layers * n_kv_heads * head_dim * max_seq_len * sizeof(float);
    
    // Verify memory usage is close to expected (allowing for some overhead)
    const float overhead_factor = 1.1f; // Allow 10% overhead
    EXPECT_LE(initial_memory, expected_memory * overhead_factor);
    
    // Now resize to a smaller sequence length
    size_t small_seq_len = 512;
    kv_cache->resize(small_seq_len);
    
    // Verify sequence length was updated
    EXPECT_EQ(kv_cache->current_seq_len(), small_seq_len);
    
    // Calculate new memory usage
    size_t resized_memory = calculate_kv_cache_memory(kv_cache);
    
    // Print memory usage after resize
    std::cout << "Resized KV cache memory allocation (seq_len=" << small_seq_len << "): " 
              << resized_memory << " bytes" << std::endl;
    
    // Calculate expected memory after resize
    // Expected memory should be adaptive to sequence length
    size_t expected_resized_memory = 2 * n_layers * n_kv_heads * head_dim * small_seq_len * sizeof(float);
    
    // Verify memory usage is proportional to sequence length
    float memory_ratio = static_cast<float>(resized_memory) / initial_memory;
    float seq_len_ratio = static_cast<float>(small_seq_len) / max_seq_len;
    
    std::cout << "Memory ratio: " << memory_ratio << std::endl;
    std::cout << "Sequence length ratio: " << seq_len_ratio << std::endl;
    
    // With our memory-efficient implementation, memory usage should be proportional to sequence length
    // Allow for some variance due to implementation details
    EXPECT_NEAR(memory_ratio, seq_len_ratio, 0.2f);
    
    // Verify the resized memory is close to expected
    EXPECT_NEAR(static_cast<float>(resized_memory), static_cast<float>(expected_resized_memory), 
               expected_resized_memory * 0.2f);
    
    // Clear the cache and verify memory is minimized
    kv_cache->clear();
    EXPECT_EQ(kv_cache->current_seq_len(), 0);
}

// Test the memory_usage method
TEST_F(KVCacheMemoryOptimizationTest, MemoryUsage) {
    // Initially cache is empty
    EXPECT_EQ(kv_cache->current_seq_len(), 0);
    
    // Fill cache with data
    size_t test_seq_len = 128;
    fill_kv_cache(test_seq_len);
    kv_cache->resize(test_seq_len);
    
    // Verify current sequence length
    EXPECT_EQ(kv_cache->current_seq_len(), test_seq_len);
    
    // Get memory usage using the built-in method
    size_t memory_usage = kv_cache->memory_usage();
    
    // Calculate expected memory
    size_t expected_memory = 2 * n_layers * n_kv_heads * head_dim * test_seq_len * sizeof(float);
    
    // Verify memory usage is close to expected (allowing for some overhead)
    std::cout << "Memory usage: " << memory_usage << " bytes" << std::endl;
    std::cout << "Expected memory: " << expected_memory << " bytes" << std::endl;
    
    // Allow for some variance due to implementation details
    EXPECT_NEAR(static_cast<float>(memory_usage), static_cast<float>(expected_memory), 
                expected_memory * 0.2f);
}

// Test cache pruning with importance scores
TEST_F(KVCacheMemoryOptimizationTest, PruningWithImportance) {
    // Fill cache with test data and resize
    size_t original_seq_len = 100;
    fill_kv_cache(original_seq_len);
    kv_cache->resize(original_seq_len);
    
    // Verify cache size
    EXPECT_EQ(kv_cache->current_seq_len(), original_seq_len);
    
    // Create importance scores - make positions 10, 20, 30, 40, 50 important
    std::vector<float> importance(original_seq_len, 0.1f);
    importance[10] = 0.9f;
    importance[20] = 0.9f;
    importance[30] = 0.9f;
    importance[40] = 0.9f;
    importance[50] = 0.9f;
    
    // Also make the last few tokens important (recency bias)
    importance[original_seq_len - 1] = 0.8f;
    importance[original_seq_len - 2] = 0.8f;
    importance[original_seq_len - 3] = 0.8f;
    
    // Target a smaller cache size
    size_t target_len = 20;
    
    // Prune the cache
    size_t new_len = kv_cache->prune(target_len, importance, 3);
    
    // Verify we kept target_len positions
    EXPECT_EQ(new_len, target_len);
    EXPECT_EQ(kv_cache->current_seq_len(), target_len);
    
    // Calculate memory after pruning
    size_t pruned_memory = kv_cache->memory_usage();
    
    // Expected memory after pruning
    size_t expected_pruned_memory = 2 * n_layers * n_kv_heads * head_dim * target_len * sizeof(float);
    
    // Verify memory usage is proportional to target length
    std::cout << "Pruned memory: " << pruned_memory << " bytes" << std::endl;
    std::cout << "Expected pruned memory: " << expected_pruned_memory << " bytes" << std::endl;
    
    // Allow for some implementation variance
    EXPECT_NEAR(static_cast<float>(pruned_memory), static_cast<float>(expected_pruned_memory), 
                expected_pruned_memory * 0.2f);
}

// Test pruning with only keeping recent tokens
TEST_F(KVCacheMemoryOptimizationTest, PruningKeepRecent) {
    // Fill cache with test data
    size_t original_seq_len = 100;
    fill_kv_cache(original_seq_len);
    kv_cache->resize(original_seq_len);
    
    // Verify cache size
    EXPECT_EQ(kv_cache->current_seq_len(), original_seq_len);
    
    // Create importance scores - all equal
    std::vector<float> importance(original_seq_len, 0.5f);
    
    // Target a smaller cache size, but keep all tokens as recent
    size_t target_len = 20;
    
    // Prune the cache - keeping all tokens as recent (no importance-based selection)
    size_t new_len = kv_cache->prune(target_len, importance, target_len);
    
    // Verify we kept target_len positions
    EXPECT_EQ(new_len, target_len);
    EXPECT_EQ(kv_cache->current_seq_len(), target_len);
    
    // Verify memory usage matches target length
    size_t pruned_memory = kv_cache->memory_usage();
    size_t expected_pruned_memory = 2 * n_layers * n_kv_heads * head_dim * target_len * sizeof(float);
    
    // Allow for some implementation variance
    EXPECT_NEAR(static_cast<float>(pruned_memory), static_cast<float>(expected_pruned_memory), 
                expected_pruned_memory * 0.2f);
    
    // Verify we kept the most recent tokens
    // This would require tracking which tokens were kept, which we don't have in this test
    // In a real application, we would verify the tokens kept are the most recent ones
}

} // namespace
} // namespace ccsm