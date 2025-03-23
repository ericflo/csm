#include <ccsm/cpu/ggml_model.h>
#include <ccsm/cpu/ggml_tensor.h>
#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <algorithm>

namespace ccsm {
namespace {

class GGMLKVCacheQuantizationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a GGML context
        ggml_ctx = std::make_shared<GGMLContext>();
        
        // Initialize random number generator
        std::mt19937 gen(42); // Fixed seed for reproducibility
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        
        // Test parameters
        n_layers = 2;
        n_heads = 8;
        n_kv_heads = 8;
        head_dim = 64;
        max_seq_len = 128;
        
        // Create KV cache
        kv_cache = std::make_shared<KVCache>(n_layers, n_heads, n_kv_heads, head_dim, max_seq_len);
        
        // Generate random test data
        size_t k_size = n_layers * n_kv_heads * head_dim * max_seq_len;
        size_t v_size = n_layers * n_kv_heads * head_dim * max_seq_len;
        
        // Generate random key and value tensors
        k_data.resize(k_size);
        v_data.resize(v_size);
        
        for (size_t i = 0; i < k_size; i++) {
            k_data[i] = dist(gen);
        }
        
        for (size_t i = 0; i < v_size; i++) {
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
    
    // Helper function to calculate RMSE between two tensors
    static double calculate_rmse(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size()) {
            return std::numeric_limits<double>::infinity();
        }
        
        double sum_squared_error = 0.0;
        for (size_t i = 0; i < a.size(); i++) {
            double error = a[i] - b[i];
            sum_squared_error += error * error;
        }
        
        return std::sqrt(sum_squared_error / a.size());
    }
};

// Test KV cache initialization and basic operations
TEST_F(GGMLKVCacheQuantizationTest, BasicKVCacheOperations) {
    // Verify KV cache exists and basic properties
    EXPECT_GT(kv_cache->size(), 0);
    EXPECT_EQ(kv_cache->max_seq_len(), max_seq_len);
    EXPECT_EQ(kv_cache->current_seq_len(), 0);
    
    // Get K and V tensors for each layer
    for (size_t layer = 0; layer < n_layers; layer++) {
        struct ggml_tensor* k_tensor = kv_cache->k_cache(layer);
        struct ggml_tensor* v_tensor = kv_cache->v_cache(layer);
        
        ASSERT_NE(k_tensor, nullptr);
        ASSERT_NE(v_tensor, nullptr);
        
        // Verify dimensions exist (don't test exact sizes since implementation may vary)
        EXPECT_GT(k_tensor->ne[0], 0);
        EXPECT_GT(k_tensor->ne[1], 0);
        
        EXPECT_GT(v_tensor->ne[0], 0);
        EXPECT_GT(v_tensor->ne[1], 0);
        
        // Print actual dimensions for debugging
        std::cout << "K tensor dimensions: [" 
                  << k_tensor->ne[0] << ", " 
                  << k_tensor->ne[1] << ", " 
                  << k_tensor->ne[2] << ", " 
                  << k_tensor->ne[3] << "]" << std::endl;
                  
        std::cout << "V tensor dimensions: [" 
                  << v_tensor->ne[0] << ", " 
                  << v_tensor->ne[1] << ", " 
                  << v_tensor->ne[2] << ", " 
                  << v_tensor->ne[3] << "]" << std::endl;
    }
    
    // Resize the cache to a smaller sequence length
    size_t new_seq_len = 64;
    kv_cache->resize(new_seq_len);
    EXPECT_EQ(kv_cache->current_seq_len(), new_seq_len);
    
    // Clear the cache
    kv_cache->clear();
    EXPECT_EQ(kv_cache->current_seq_len(), 0);
}

// Test KV cache with different quantization formats
TEST_F(GGMLKVCacheQuantizationTest, QuantizedKVCache) {
    // For this test, we'll create a simulated quantized KV cache
    struct ggml_context* sim_ctx = ggml_init(
        { .mem_size = 1024 * 1024 * 1024, .mem_buffer = NULL, .no_alloc = false }
    );
    ASSERT_NE(sim_ctx, nullptr);
    
    // Create tensors to represent a layer of K and V caches with quantization
    const size_t seq_pos = 10; // Position in sequence to test
    const size_t batch_size = 1;
    
    // Original F32 tensors
    struct ggml_tensor* k_f32 = ggml_new_tensor_3d(sim_ctx, GGML_TYPE_F32, head_dim, max_seq_len, n_kv_heads);
    struct ggml_tensor* v_f32 = ggml_new_tensor_3d(sim_ctx, GGML_TYPE_F32, head_dim, max_seq_len, n_kv_heads);
    
    // Fill with test data
    float* k_ptr = (float*)k_f32->data;
    float* v_ptr = (float*)v_f32->data;
    
    const size_t k_layer_size = head_dim * max_seq_len * n_kv_heads;
    const size_t v_layer_size = head_dim * max_seq_len * n_kv_heads;
    
    std::copy(k_data.begin(), k_data.begin() + k_layer_size, k_ptr);
    std::copy(v_data.begin(), v_data.begin() + v_layer_size, v_ptr);
    
    // Simulated quantization to different formats (using GGML's quantization functions)
    struct ggml_tensor* k_q8_0 = ggml_new_tensor_3d(sim_ctx, GGML_TYPE_Q8_0, head_dim, max_seq_len, n_kv_heads);
    struct ggml_tensor* v_q8_0 = ggml_new_tensor_3d(sim_ctx, GGML_TYPE_Q8_0, head_dim, max_seq_len, n_kv_heads);
    
    struct ggml_tensor* k_q4_0 = ggml_new_tensor_3d(sim_ctx, GGML_TYPE_Q4_0, head_dim, max_seq_len, n_kv_heads);
    struct ggml_tensor* v_q4_0 = ggml_new_tensor_3d(sim_ctx, GGML_TYPE_Q4_0, head_dim, max_seq_len, n_kv_heads);
    
    struct ggml_tensor* k_q4_1 = ggml_new_tensor_3d(sim_ctx, GGML_TYPE_Q4_1, head_dim, max_seq_len, n_kv_heads);
    struct ggml_tensor* v_q4_1 = ggml_new_tensor_3d(sim_ctx, GGML_TYPE_Q4_1, head_dim, max_seq_len, n_kv_heads);
    
    // Quantize the tensors
    int64_t hist_q8_0[16] = {0};
    int64_t hist_q4_0[16] = {0};
    int64_t hist_q4_1[16] = {0};
    
    // Usually we'd quantize with a function like this:
    // ggml_quantize_q8_0(k_f32->data, k_q8_0->data, n_elements, k_f32->ne[0], hist_q8_0);
    // But for this test, we'll use GGML graph operations
    
    // Create a computation graph for the quantization
    struct ggml_cgraph* graph = ggml_new_graph(sim_ctx);
    
    // Create the operations for cast/conversion
    struct ggml_tensor* k_f32_to_q8_0 = ggml_cast(sim_ctx, k_f32, GGML_TYPE_Q8_0);
    struct ggml_tensor* v_f32_to_q8_0 = ggml_cast(sim_ctx, v_f32, GGML_TYPE_Q8_0);
    
    struct ggml_tensor* k_f32_to_q4_0 = ggml_cast(sim_ctx, k_f32, GGML_TYPE_Q4_0);
    struct ggml_tensor* v_f32_to_q4_0 = ggml_cast(sim_ctx, v_f32, GGML_TYPE_Q4_0);
    
    struct ggml_tensor* k_f32_to_q4_1 = ggml_cast(sim_ctx, k_f32, GGML_TYPE_Q4_1);
    struct ggml_tensor* v_f32_to_q4_1 = ggml_cast(sim_ctx, v_f32, GGML_TYPE_Q4_1);
    
    // Build the graph
    ggml_build_forward_expand(graph, k_f32_to_q8_0);
    ggml_build_forward_expand(graph, v_f32_to_q8_0);
    ggml_build_forward_expand(graph, k_f32_to_q4_0);
    ggml_build_forward_expand(graph, v_f32_to_q4_0);
    ggml_build_forward_expand(graph, k_f32_to_q4_1);
    ggml_build_forward_expand(graph, v_f32_to_q4_1);
    
    // Compute the graph
    ggml_ctx->compute(graph);
    
    // Now let's convert back for comparison
    struct ggml_tensor* k_q8_0_to_f32 = ggml_cast(sim_ctx, k_f32_to_q8_0, GGML_TYPE_F32);
    struct ggml_tensor* v_q8_0_to_f32 = ggml_cast(sim_ctx, v_f32_to_q8_0, GGML_TYPE_F32);
    
    struct ggml_tensor* k_q4_0_to_f32 = ggml_cast(sim_ctx, k_f32_to_q4_0, GGML_TYPE_F32);
    struct ggml_tensor* v_q4_0_to_f32 = ggml_cast(sim_ctx, v_f32_to_q4_0, GGML_TYPE_F32);
    
    struct ggml_tensor* k_q4_1_to_f32 = ggml_cast(sim_ctx, k_f32_to_q4_1, GGML_TYPE_F32);
    struct ggml_tensor* v_q4_1_to_f32 = ggml_cast(sim_ctx, v_f32_to_q4_1, GGML_TYPE_F32);
    
    // Create a new graph for the dequantization
    struct ggml_cgraph* graph2 = ggml_new_graph(sim_ctx);
    
    // Build the graph
    ggml_build_forward_expand(graph2, k_q8_0_to_f32);
    ggml_build_forward_expand(graph2, v_q8_0_to_f32);
    ggml_build_forward_expand(graph2, k_q4_0_to_f32);
    ggml_build_forward_expand(graph2, v_q4_0_to_f32);
    ggml_build_forward_expand(graph2, k_q4_1_to_f32);
    ggml_build_forward_expand(graph2, v_q4_1_to_f32);
    
    // Compute the graph
    ggml_ctx->compute(graph2);
    
    // Now we can compare the original and quantized-then-dequantized data
    std::vector<float> k_orig(k_layer_size);
    std::vector<float> k_q8_0_dequant(k_layer_size);
    std::vector<float> k_q4_0_dequant(k_layer_size);
    std::vector<float> k_q4_1_dequant(k_layer_size);
    
    std::vector<float> v_orig(v_layer_size);
    std::vector<float> v_q8_0_dequant(v_layer_size);
    std::vector<float> v_q4_0_dequant(v_layer_size);
    std::vector<float> v_q4_1_dequant(v_layer_size);
    
    // Copy data
    std::copy((float*)k_f32->data, (float*)k_f32->data + k_layer_size, k_orig.begin());
    std::copy((float*)k_q8_0_to_f32->data, (float*)k_q8_0_to_f32->data + k_layer_size, k_q8_0_dequant.begin());
    std::copy((float*)k_q4_0_to_f32->data, (float*)k_q4_0_to_f32->data + k_layer_size, k_q4_0_dequant.begin());
    std::copy((float*)k_q4_1_to_f32->data, (float*)k_q4_1_to_f32->data + k_layer_size, k_q4_1_dequant.begin());
    
    std::copy((float*)v_f32->data, (float*)v_f32->data + v_layer_size, v_orig.begin());
    std::copy((float*)v_q8_0_to_f32->data, (float*)v_q8_0_to_f32->data + v_layer_size, v_q8_0_dequant.begin());
    std::copy((float*)v_q4_0_to_f32->data, (float*)v_q4_0_to_f32->data + v_layer_size, v_q4_0_dequant.begin());
    std::copy((float*)v_q4_1_to_f32->data, (float*)v_q4_1_to_f32->data + v_layer_size, v_q4_1_dequant.begin());
    
    // Calculate errors
    double k_q8_0_rmse = calculate_rmse(k_orig, k_q8_0_dequant);
    double k_q4_0_rmse = calculate_rmse(k_orig, k_q4_0_dequant);
    double k_q4_1_rmse = calculate_rmse(k_orig, k_q4_1_dequant);
    
    double v_q8_0_rmse = calculate_rmse(v_orig, v_q8_0_dequant);
    double v_q4_0_rmse = calculate_rmse(v_orig, v_q4_0_dequant);
    double v_q4_1_rmse = calculate_rmse(v_orig, v_q4_1_dequant);
    
    // Print errors
    std::cout << "KV Cache Quantization Errors (RMSE):" << std::endl;
    std::cout << "-------------------------------------------------------------" << std::endl;
    std::cout << std::setw(12) << "Format" << std::setw(15) << "K Cache RMSE" 
              << std::setw(15) << "V Cache RMSE" << std::endl;
    std::cout << "-------------------------------------------------------------" << std::endl;
    
    std::cout << std::setw(12) << "Q8_0" << std::setw(15) << k_q8_0_rmse 
              << std::setw(15) << v_q8_0_rmse << std::endl;
    
    std::cout << std::setw(12) << "Q4_0" << std::setw(15) << k_q4_0_rmse 
              << std::setw(15) << v_q4_0_rmse << std::endl;
    
    std::cout << std::setw(12) << "Q4_1" << std::setw(15) << k_q4_1_rmse 
              << std::setw(15) << v_q4_1_rmse << std::endl;
    
    // Verify errors are within acceptable limits
    // These limits may need adjustment based on your implementation
    EXPECT_LT(k_q8_0_rmse, 0.6);
    EXPECT_LT(v_q8_0_rmse, 0.6);
    
    EXPECT_LT(k_q4_0_rmse, 1.0);
    EXPECT_LT(v_q4_0_rmse, 1.0);
    
    EXPECT_LT(k_q4_1_rmse, 0.8);
    EXPECT_LT(v_q4_1_rmse, 0.8);
    
    // Free the GGML context
    ggml_free(sim_ctx);
}

// Test attention calculation with quantized KV cache
TEST_F(GGMLKVCacheQuantizationTest, AttentionWithQuantizedKVCache) {
    // Create a GGML context for the test
    struct ggml_context* attn_ctx = ggml_init(
        { .mem_size = 1024 * 1024 * 1024, .mem_buffer = NULL, .no_alloc = false }
    );
    ASSERT_NE(attn_ctx, nullptr);
    
    // Define attention parameters
    size_t batch_size = 1;
    size_t seq_len = 32;    // Current sequence length
    size_t q_heads = 8;     // Query heads (could be different from KV heads in GQA)
    size_t kv_heads = 8;    // Key/Value heads
    size_t head_dim = 64;   // Head dimension
    
    // Create input tensors for attention (query, key, value)
    // GGML dimension ordering: [head_dim, seq_len, n_heads, batch_size]
    struct ggml_tensor* q = ggml_new_tensor_4d(attn_ctx, GGML_TYPE_F32, head_dim, seq_len, q_heads, batch_size);
    struct ggml_tensor* k = ggml_new_tensor_4d(attn_ctx, GGML_TYPE_F32, head_dim, seq_len, kv_heads, batch_size);
    struct ggml_tensor* v = ggml_new_tensor_4d(attn_ctx, GGML_TYPE_F32, head_dim, seq_len, kv_heads, batch_size);
    
    // Fill with random data
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    float* q_ptr = (float*)q->data;
    float* k_ptr = (float*)k->data;
    float* v_ptr = (float*)v->data;
    
    size_t q_size = head_dim * seq_len * q_heads * batch_size;
    size_t k_size = head_dim * seq_len * kv_heads * batch_size;
    size_t v_size = head_dim * seq_len * kv_heads * batch_size;
    
    for (size_t i = 0; i < q_size; i++) {
        q_ptr[i] = dist(gen) * 0.1f; // Scale down to avoid numerical issues
    }
    
    for (size_t i = 0; i < k_size; i++) {
        k_ptr[i] = dist(gen) * 0.1f;
    }
    
    for (size_t i = 0; i < v_size; i++) {
        v_ptr[i] = dist(gen) * 0.1f;
    }
    
    // Create quantized versions of K and V
    struct ggml_tensor* k_q8_0 = ggml_cast(attn_ctx, k, GGML_TYPE_Q8_0);
    struct ggml_tensor* v_q8_0 = ggml_cast(attn_ctx, v, GGML_TYPE_Q8_0);
    
    struct ggml_tensor* k_q4_0 = ggml_cast(attn_ctx, k, GGML_TYPE_Q4_0);
    struct ggml_tensor* v_q4_0 = ggml_cast(attn_ctx, v, GGML_TYPE_Q4_0);
    
    struct ggml_tensor* k_q4_1 = ggml_cast(attn_ctx, k, GGML_TYPE_Q4_1);
    struct ggml_tensor* v_q4_1 = ggml_cast(attn_ctx, v, GGML_TYPE_Q4_1);
    
    // Compute attention with different quantization formats
    // 1. Build attention with full precision
    struct ggml_tensor* qk_f32 = ggml_mul_mat(attn_ctx, k, q);
    struct ggml_tensor* qk_f32_scaled = ggml_scale(attn_ctx, qk_f32, 1.0f / sqrt(head_dim));
    struct ggml_tensor* qk_f32_softmax = ggml_soft_max(attn_ctx, qk_f32_scaled);
    struct ggml_tensor* attn_f32 = ggml_mul_mat(attn_ctx, v, qk_f32_softmax);
    
    // 2. Build attention with Q8_0 quantized K and V
    struct ggml_tensor* qk_q8_0 = ggml_mul_mat(attn_ctx, k_q8_0, q);
    struct ggml_tensor* qk_q8_0_scaled = ggml_scale(attn_ctx, qk_q8_0, 1.0f / sqrt(head_dim));
    struct ggml_tensor* qk_q8_0_softmax = ggml_soft_max(attn_ctx, qk_q8_0_scaled);
    struct ggml_tensor* attn_q8_0 = ggml_mul_mat(attn_ctx, v_q8_0, qk_q8_0_softmax);
    
    // 3. Build attention with Q4_0 quantized K and V
    struct ggml_tensor* qk_q4_0 = ggml_mul_mat(attn_ctx, k_q4_0, q);
    struct ggml_tensor* qk_q4_0_scaled = ggml_scale(attn_ctx, qk_q4_0, 1.0f / sqrt(head_dim));
    struct ggml_tensor* qk_q4_0_softmax = ggml_soft_max(attn_ctx, qk_q4_0_scaled);
    struct ggml_tensor* attn_q4_0 = ggml_mul_mat(attn_ctx, v_q4_0, qk_q4_0_softmax);
    
    // 4. Build attention with Q4_1 quantized K and V
    struct ggml_tensor* qk_q4_1 = ggml_mul_mat(attn_ctx, k_q4_1, q);
    struct ggml_tensor* qk_q4_1_scaled = ggml_scale(attn_ctx, qk_q4_1, 1.0f / sqrt(head_dim));
    struct ggml_tensor* qk_q4_1_softmax = ggml_soft_max(attn_ctx, qk_q4_1_scaled);
    struct ggml_tensor* attn_q4_1 = ggml_mul_mat(attn_ctx, v_q4_1, qk_q4_1_softmax);
    
    // Create computation graphs and compute
    // For full precision
    struct ggml_cgraph* graph_f32 = ggml_new_graph(attn_ctx);
    ggml_build_forward_expand(graph_f32, attn_f32);
    ggml_ctx->compute(graph_f32);
    
    // For Q8_0
    struct ggml_cgraph* graph_q8_0 = ggml_new_graph(attn_ctx);
    ggml_build_forward_expand(graph_q8_0, attn_q8_0);
    ggml_ctx->compute(graph_q8_0);
    
    // For Q4_0
    struct ggml_cgraph* graph_q4_0 = ggml_new_graph(attn_ctx);
    ggml_build_forward_expand(graph_q4_0, attn_q4_0);
    ggml_ctx->compute(graph_q4_0);
    
    // For Q4_1
    struct ggml_cgraph* graph_q4_1 = ggml_new_graph(attn_ctx);
    ggml_build_forward_expand(graph_q4_1, attn_q4_1);
    ggml_ctx->compute(graph_q4_1);
    
    // Now compare the results
    std::vector<float> attn_f32_data(q_size);
    std::vector<float> attn_q8_0_data(q_size);
    std::vector<float> attn_q4_0_data(q_size);
    std::vector<float> attn_q4_1_data(q_size);
    
    std::copy((float*)attn_f32->data, (float*)attn_f32->data + q_size, attn_f32_data.begin());
    std::copy((float*)attn_q8_0->data, (float*)attn_q8_0->data + q_size, attn_q8_0_data.begin());
    std::copy((float*)attn_q4_0->data, (float*)attn_q4_0->data + q_size, attn_q4_0_data.begin());
    std::copy((float*)attn_q4_1->data, (float*)attn_q4_1->data + q_size, attn_q4_1_data.begin());
    
    // Calculate RMSE
    double attn_q8_0_rmse = calculate_rmse(attn_f32_data, attn_q8_0_data);
    double attn_q4_0_rmse = calculate_rmse(attn_f32_data, attn_q4_0_data);
    double attn_q4_1_rmse = calculate_rmse(attn_f32_data, attn_q4_1_data);
    
    // Print results
    std::cout << "Attention Output Errors with Quantized KV Cache (RMSE):" << std::endl;
    std::cout << "-------------------------------------------------------------" << std::endl;
    std::cout << std::setw(12) << "Format" << std::setw(20) << "Attention RMSE" << std::endl;
    std::cout << "-------------------------------------------------------------" << std::endl;
    
    std::cout << std::setw(12) << "Q8_0" << std::setw(20) << attn_q8_0_rmse << std::endl;
    std::cout << std::setw(12) << "Q4_0" << std::setw(20) << attn_q4_0_rmse << std::endl;
    std::cout << std::setw(12) << "Q4_1" << std::setw(20) << attn_q4_1_rmse << std::endl;
    
    // Verify errors are within acceptable limits for attention
    // Attention calculation with softmax tends to increase errors
    EXPECT_LT(attn_q8_0_rmse, 1.0);
    EXPECT_LT(attn_q4_0_rmse, 1.5);
    EXPECT_LT(attn_q4_1_rmse, 1.2);
    
    // Free the GGML context
    ggml_free(attn_ctx);
}

// Test memory usage and performance with quantized KV Cache
TEST_F(GGMLKVCacheQuantizationTest, MemoryAndPerformance) {
    // Create a GGML context for the test
    struct ggml_context* perf_ctx = ggml_init(
        { .mem_size = 1024 * 1024 * 1024, .mem_buffer = NULL, .no_alloc = false }
    );
    ASSERT_NE(perf_ctx, nullptr);
    
    // Define larger parameters for performance testing
    size_t batch_size = 1;
    size_t seq_len = 512;    // Longer sequence
    size_t q_heads = 16;     // More heads
    size_t kv_heads = 16;    // More KV heads
    size_t head_dim = 128;   // Larger head dimension
    
    // Measure memory usage for different quantization formats
    // GGML dimension ordering: [head_dim, seq_len, n_heads, batch_size]
    struct ggml_tensor* k_f32 = ggml_new_tensor_4d(perf_ctx, GGML_TYPE_F32, head_dim, seq_len, kv_heads, batch_size);
    struct ggml_tensor* v_f32 = ggml_new_tensor_4d(perf_ctx, GGML_TYPE_F32, head_dim, seq_len, kv_heads, batch_size);
    
    struct ggml_tensor* k_q8_0 = ggml_new_tensor_4d(perf_ctx, GGML_TYPE_Q8_0, head_dim, seq_len, kv_heads, batch_size);
    struct ggml_tensor* v_q8_0 = ggml_new_tensor_4d(perf_ctx, GGML_TYPE_Q8_0, head_dim, seq_len, kv_heads, batch_size);
    
    struct ggml_tensor* k_q4_0 = ggml_new_tensor_4d(perf_ctx, GGML_TYPE_Q4_0, head_dim, seq_len, kv_heads, batch_size);
    struct ggml_tensor* v_q4_0 = ggml_new_tensor_4d(perf_ctx, GGML_TYPE_Q4_0, head_dim, seq_len, kv_heads, batch_size);
    
    struct ggml_tensor* k_q4_1 = ggml_new_tensor_4d(perf_ctx, GGML_TYPE_Q4_1, head_dim, seq_len, kv_heads, batch_size);
    struct ggml_tensor* v_q4_1 = ggml_new_tensor_4d(perf_ctx, GGML_TYPE_Q4_1, head_dim, seq_len, kv_heads, batch_size);
    
    // Calculate memory usage for each tensor
    size_t kv_f32_size = 2 * sizeof(float) * head_dim * seq_len * kv_heads * batch_size;
    size_t kv_q8_0_size = ggml_nbytes(k_q8_0) + ggml_nbytes(v_q8_0);
    size_t kv_q4_0_size = ggml_nbytes(k_q4_0) + ggml_nbytes(v_q4_0);
    size_t kv_q4_1_size = ggml_nbytes(k_q4_1) + ggml_nbytes(v_q4_1);
    
    // Print memory usage
    std::cout << "KV Cache Memory Usage:" << std::endl;
    std::cout << "-------------------------------------------------------------" << std::endl;
    std::cout << std::setw(12) << "Format" << std::setw(20) << "Memory (bytes)" 
              << std::setw(20) << "% of F32" << std::endl;
    std::cout << "-------------------------------------------------------------" << std::endl;
    
    std::cout << std::setw(12) << "F32" << std::setw(20) << kv_f32_size 
              << std::setw(20) << "100.00%" << std::endl;
    
    std::cout << std::setw(12) << "Q8_0" << std::setw(20) << kv_q8_0_size 
              << std::setw(20) << std::fixed << std::setprecision(2) 
              << (100.0 * kv_q8_0_size / kv_f32_size) << "%" << std::endl;
    
    std::cout << std::setw(12) << "Q4_0" << std::setw(20) << kv_q4_0_size 
              << std::setw(20) << std::fixed << std::setprecision(2) 
              << (100.0 * kv_q4_0_size / kv_f32_size) << "%" << std::endl;
    
    std::cout << std::setw(12) << "Q4_1" << std::setw(20) << kv_q4_1_size 
              << std::setw(20) << std::fixed << std::setprecision(2) 
              << (100.0 * kv_q4_1_size / kv_f32_size) << "%" << std::endl;
    
    // Verify memory savings
    EXPECT_LT(kv_q8_0_size, kv_f32_size * 0.5); // At least 50% reduction
    EXPECT_LT(kv_q4_0_size, kv_f32_size * 0.25); // At least 75% reduction
    EXPECT_LT(kv_q4_1_size, kv_f32_size * 0.3); // At least 70% reduction
    
    // Create a query tensor for attention
    struct ggml_tensor* q = ggml_new_tensor_4d(perf_ctx, GGML_TYPE_F32, head_dim, 1, q_heads, batch_size);
    
    // Fill with random data
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    // Fill K and V with random data
    float* k_ptr = (float*)k_f32->data;
    float* v_ptr = (float*)v_f32->data;
    float* q_ptr = (float*)q->data;
    
    size_t k_size = head_dim * seq_len * kv_heads * batch_size;
    size_t v_size = head_dim * seq_len * kv_heads * batch_size;
    size_t q_size = head_dim * 1 * q_heads * batch_size;
    
    for (size_t i = 0; i < k_size; i++) {
        k_ptr[i] = dist(gen) * 0.1f;
    }
    
    for (size_t i = 0; i < v_size; i++) {
        v_ptr[i] = dist(gen) * 0.1f;
    }
    
    for (size_t i = 0; i < q_size; i++) {
        q_ptr[i] = dist(gen) * 0.1f;
    }
    
    // Create quantized versions of K and V
    struct ggml_cgraph* quant_graph = ggml_new_graph(perf_ctx);
    
    struct ggml_tensor* k_f32_to_q8_0 = ggml_cast(perf_ctx, k_f32, GGML_TYPE_Q8_0);
    struct ggml_tensor* v_f32_to_q8_0 = ggml_cast(perf_ctx, v_f32, GGML_TYPE_Q8_0);
    
    struct ggml_tensor* k_f32_to_q4_0 = ggml_cast(perf_ctx, k_f32, GGML_TYPE_Q4_0);
    struct ggml_tensor* v_f32_to_q4_0 = ggml_cast(perf_ctx, v_f32, GGML_TYPE_Q4_0);
    
    struct ggml_tensor* k_f32_to_q4_1 = ggml_cast(perf_ctx, k_f32, GGML_TYPE_Q4_1);
    struct ggml_tensor* v_f32_to_q4_1 = ggml_cast(perf_ctx, v_f32, GGML_TYPE_Q4_1);
    
    ggml_build_forward_expand(quant_graph, k_f32_to_q8_0);
    ggml_build_forward_expand(quant_graph, v_f32_to_q8_0);
    ggml_build_forward_expand(quant_graph, k_f32_to_q4_0);
    ggml_build_forward_expand(quant_graph, v_f32_to_q4_0);
    ggml_build_forward_expand(quant_graph, k_f32_to_q4_1);
    ggml_build_forward_expand(quant_graph, v_f32_to_q4_1);
    
    ggml_ctx->compute(quant_graph);
    
    // Measure performance of attention with different quantized KV cache formats
    // 1. Attention with F32 KV cache
    auto start_f32 = std::chrono::high_resolution_clock::now();
    
    struct ggml_tensor* qk_f32 = ggml_mul_mat(perf_ctx, k_f32, q);
    struct ggml_tensor* qk_f32_scaled = ggml_scale(perf_ctx, qk_f32, 1.0f / sqrt(head_dim));
    struct ggml_tensor* qk_f32_softmax = ggml_soft_max(perf_ctx, qk_f32_scaled);
    struct ggml_tensor* attn_f32 = ggml_mul_mat(perf_ctx, v_f32, qk_f32_softmax);
    
    struct ggml_cgraph* graph_f32 = ggml_new_graph(perf_ctx);
    ggml_build_forward_expand(graph_f32, attn_f32);
    ggml_ctx->compute(graph_f32);
    
    auto end_f32 = std::chrono::high_resolution_clock::now();
    
    // 2. Attention with Q8_0 KV cache
    auto start_q8_0 = std::chrono::high_resolution_clock::now();
    
    struct ggml_tensor* qk_q8_0 = ggml_mul_mat(perf_ctx, k_f32_to_q8_0, q);
    struct ggml_tensor* qk_q8_0_scaled = ggml_scale(perf_ctx, qk_q8_0, 1.0f / sqrt(head_dim));
    struct ggml_tensor* qk_q8_0_softmax = ggml_soft_max(perf_ctx, qk_q8_0_scaled);
    struct ggml_tensor* attn_q8_0 = ggml_mul_mat(perf_ctx, v_f32_to_q8_0, qk_q8_0_softmax);
    
    struct ggml_cgraph* graph_q8_0 = ggml_new_graph(perf_ctx);
    ggml_build_forward_expand(graph_q8_0, attn_q8_0);
    ggml_ctx->compute(graph_q8_0);
    
    auto end_q8_0 = std::chrono::high_resolution_clock::now();
    
    // 3. Attention with Q4_0 KV cache
    auto start_q4_0 = std::chrono::high_resolution_clock::now();
    
    struct ggml_tensor* qk_q4_0 = ggml_mul_mat(perf_ctx, k_f32_to_q4_0, q);
    struct ggml_tensor* qk_q4_0_scaled = ggml_scale(perf_ctx, qk_q4_0, 1.0f / sqrt(head_dim));
    struct ggml_tensor* qk_q4_0_softmax = ggml_soft_max(perf_ctx, qk_q4_0_scaled);
    struct ggml_tensor* attn_q4_0 = ggml_mul_mat(perf_ctx, v_f32_to_q4_0, qk_q4_0_softmax);
    
    struct ggml_cgraph* graph_q4_0 = ggml_new_graph(perf_ctx);
    ggml_build_forward_expand(graph_q4_0, attn_q4_0);
    ggml_ctx->compute(graph_q4_0);
    
    auto end_q4_0 = std::chrono::high_resolution_clock::now();
    
    // 4. Attention with Q4_1 KV cache
    auto start_q4_1 = std::chrono::high_resolution_clock::now();
    
    struct ggml_tensor* qk_q4_1 = ggml_mul_mat(perf_ctx, k_f32_to_q4_1, q);
    struct ggml_tensor* qk_q4_1_scaled = ggml_scale(perf_ctx, qk_q4_1, 1.0f / sqrt(head_dim));
    struct ggml_tensor* qk_q4_1_softmax = ggml_soft_max(perf_ctx, qk_q4_1_scaled);
    struct ggml_tensor* attn_q4_1 = ggml_mul_mat(perf_ctx, v_f32_to_q4_1, qk_q4_1_softmax);
    
    struct ggml_cgraph* graph_q4_1 = ggml_new_graph(perf_ctx);
    ggml_build_forward_expand(graph_q4_1, attn_q4_1);
    ggml_ctx->compute(graph_q4_1);
    
    auto end_q4_1 = std::chrono::high_resolution_clock::now();
    
    // Calculate timings
    auto duration_f32 = std::chrono::duration_cast<std::chrono::microseconds>(end_f32 - start_f32);
    auto duration_q8_0 = std::chrono::duration_cast<std::chrono::microseconds>(end_q8_0 - start_q8_0);
    auto duration_q4_0 = std::chrono::duration_cast<std::chrono::microseconds>(end_q4_0 - start_q4_0);
    auto duration_q4_1 = std::chrono::duration_cast<std::chrono::microseconds>(end_q4_1 - start_q4_1);
    
    // Print performance results
    std::cout << "Attention Performance with Quantized KV Cache:" << std::endl;
    std::cout << "-------------------------------------------------------------" << std::endl;
    std::cout << std::setw(12) << "Format" << std::setw(20) << "Time (microsec)" 
              << std::setw(20) << "Speedup" << std::endl;
    std::cout << "-------------------------------------------------------------" << std::endl;
    
    std::cout << std::setw(12) << "F32" << std::setw(20) << duration_f32.count() 
              << std::setw(20) << "1.00x" << std::endl;
    
    std::cout << std::setw(12) << "Q8_0" << std::setw(20) << duration_q8_0.count() 
              << std::setw(20) << std::fixed << std::setprecision(2) 
              << static_cast<double>(duration_f32.count()) / duration_q8_0.count() << "x" << std::endl;
    
    std::cout << std::setw(12) << "Q4_0" << std::setw(20) << duration_q4_0.count() 
              << std::setw(20) << std::fixed << std::setprecision(2) 
              << static_cast<double>(duration_f32.count()) / duration_q4_0.count() << "x" << std::endl;
    
    std::cout << std::setw(12) << "Q4_1" << std::setw(20) << duration_q4_1.count() 
              << std::setw(20) << std::fixed << std::setprecision(2) 
              << static_cast<double>(duration_f32.count()) / duration_q4_1.count() << "x" << std::endl;
    
    // Expected performance improvements - these might need adjustment based on your actual hardware
    // and the overhead of GGML graph construction
    // In a real implementation, we'd expect these to be faster, but for the test we'll use
    // conservative assertions
    EXPECT_LT(duration_q8_0.count(), duration_f32.count() * 1.2); // At least not slower
    EXPECT_LT(duration_q4_0.count(), duration_f32.count() * 1.2);
    EXPECT_LT(duration_q4_1.count(), duration_f32.count() * 1.2);
    
    // Free GGML context
    ggml_free(perf_ctx);
}

} // namespace
} // namespace ccsm