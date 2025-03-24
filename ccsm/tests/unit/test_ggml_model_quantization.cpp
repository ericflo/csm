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

class GGMLModelQuantizationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a GGML context
        ggml_ctx = std::make_shared<GGMLContext>();
        
        // Initialize random number generator
        std::mt19937 gen(42); // Fixed seed for reproducibility
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        
        // Create simple model parameters
        // We'll simulate a small transformer model with:
        // - Embedding layer
        // - One transformer block (attention + feedforward)
        // - Output projection
        embedding_dim = 64;
        hidden_dim = 256;
        num_heads = 4;
        vocab_size = 1000;
        
        // Create embedding weights
        embedding_weights.resize(vocab_size * embedding_dim);
        for (size_t i = 0; i < embedding_weights.size(); i++) {
            embedding_weights[i] = dist(gen) * 0.1f;
        }
        
        // Create attention weights
        q_proj_weights.resize(embedding_dim * embedding_dim);
        k_proj_weights.resize(embedding_dim * embedding_dim);
        v_proj_weights.resize(embedding_dim * embedding_dim);
        o_proj_weights.resize(embedding_dim * embedding_dim);
        
        for (size_t i = 0; i < embedding_dim * embedding_dim; i++) {
            q_proj_weights[i] = dist(gen) * 0.1f;
            k_proj_weights[i] = dist(gen) * 0.1f;
            v_proj_weights[i] = dist(gen) * 0.1f;
            o_proj_weights[i] = dist(gen) * 0.1f;
        }
        
        // Create feedforward weights
        ff1_weights.resize(embedding_dim * hidden_dim);
        ff2_weights.resize(hidden_dim * embedding_dim);
        
        for (size_t i = 0; i < embedding_dim * hidden_dim; i++) {
            ff1_weights[i] = dist(gen) * 0.1f;
        }
        
        for (size_t i = 0; i < hidden_dim * embedding_dim; i++) {
            ff2_weights[i] = dist(gen) * 0.1f;
        }
        
        // Create output projection weights
        output_weights.resize(embedding_dim * vocab_size);
        
        for (size_t i = 0; i < embedding_dim * vocab_size; i++) {
            output_weights[i] = dist(gen) * 0.1f;
        }
    }
    
    // Helper function to calculate RMSE between two vectors
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
    
    // Model parameters
    size_t embedding_dim;
    size_t hidden_dim;
    size_t num_heads;
    size_t vocab_size;
    
    // Weight matrices
    std::vector<float> embedding_weights;
    std::vector<float> q_proj_weights;
    std::vector<float> k_proj_weights;
    std::vector<float> v_proj_weights;
    std::vector<float> o_proj_weights;
    std::vector<float> ff1_weights;
    std::vector<float> ff2_weights;
    std::vector<float> output_weights;
    
    // GGML context
    std::shared_ptr<GGMLContext> ggml_ctx;
};

// Test quantization of a small model
TEST_F(GGMLModelQuantizationTest, QuantizeSmallModel) {
    // Create tensor objects for model parameters in F32
    Tensor embedding_tensor = ggml_ctx->create_tensor({embedding_dim, vocab_size}, DataType::F32);
    Tensor q_proj_tensor = ggml_ctx->create_tensor({embedding_dim, embedding_dim}, DataType::F32);
    Tensor k_proj_tensor = ggml_ctx->create_tensor({embedding_dim, embedding_dim}, DataType::F32);
    Tensor v_proj_tensor = ggml_ctx->create_tensor({embedding_dim, embedding_dim}, DataType::F32);
    Tensor o_proj_tensor = ggml_ctx->create_tensor({embedding_dim, embedding_dim}, DataType::F32);
    Tensor ff1_tensor = ggml_ctx->create_tensor({hidden_dim, embedding_dim}, DataType::F32);
    Tensor ff2_tensor = ggml_ctx->create_tensor({embedding_dim, hidden_dim}, DataType::F32);
    Tensor output_tensor = ggml_ctx->create_tensor({vocab_size, embedding_dim}, DataType::F32);
    
    // Copy data to tensors
    std::memcpy(embedding_tensor.data(), embedding_weights.data(), embedding_weights.size() * sizeof(float));
    std::memcpy(q_proj_tensor.data(), q_proj_weights.data(), q_proj_weights.size() * sizeof(float));
    std::memcpy(k_proj_tensor.data(), k_proj_weights.data(), k_proj_weights.size() * sizeof(float));
    std::memcpy(v_proj_tensor.data(), v_proj_weights.data(), v_proj_weights.size() * sizeof(float));
    std::memcpy(o_proj_tensor.data(), o_proj_weights.data(), o_proj_weights.size() * sizeof(float));
    std::memcpy(ff1_tensor.data(), ff1_weights.data(), ff1_weights.size() * sizeof(float));
    std::memcpy(ff2_tensor.data(), ff2_weights.data(), ff2_weights.size() * sizeof(float));
    std::memcpy(output_tensor.data(), output_weights.data(), output_weights.size() * sizeof(float));
    
    // Quantize all weight matrices to different formats
    // 1. Q8_0 quantization
    Tensor embedding_q8_0 = ggml_ctx->cast(embedding_tensor, DataType::Q8_0);
    Tensor q_proj_q8_0 = ggml_ctx->cast(q_proj_tensor, DataType::Q8_0);
    Tensor k_proj_q8_0 = ggml_ctx->cast(k_proj_tensor, DataType::Q8_0);
    Tensor v_proj_q8_0 = ggml_ctx->cast(v_proj_tensor, DataType::Q8_0);
    Tensor o_proj_q8_0 = ggml_ctx->cast(o_proj_tensor, DataType::Q8_0);
    Tensor ff1_q8_0 = ggml_ctx->cast(ff1_tensor, DataType::Q8_0);
    Tensor ff2_q8_0 = ggml_ctx->cast(ff2_tensor, DataType::Q8_0);
    Tensor output_q8_0 = ggml_ctx->cast(output_tensor, DataType::Q8_0);
    
    // 2. Q4_0 quantization
    Tensor embedding_q4_0 = ggml_ctx->cast(embedding_tensor, DataType::Q4_0);
    Tensor q_proj_q4_0 = ggml_ctx->cast(q_proj_tensor, DataType::Q4_0);
    Tensor k_proj_q4_0 = ggml_ctx->cast(k_proj_tensor, DataType::Q4_0);
    Tensor v_proj_q4_0 = ggml_ctx->cast(v_proj_tensor, DataType::Q4_0);
    Tensor o_proj_q4_0 = ggml_ctx->cast(o_proj_tensor, DataType::Q4_0);
    Tensor ff1_q4_0 = ggml_ctx->cast(ff1_tensor, DataType::Q4_0);
    Tensor ff2_q4_0 = ggml_ctx->cast(ff2_tensor, DataType::Q4_0);
    Tensor output_q4_0 = ggml_ctx->cast(output_tensor, DataType::Q4_0);
    
    // 3. Q4_1 quantization
    Tensor embedding_q4_1 = ggml_ctx->cast(embedding_tensor, DataType::Q4_1);
    Tensor q_proj_q4_1 = ggml_ctx->cast(q_proj_tensor, DataType::Q4_1);
    Tensor k_proj_q4_1 = ggml_ctx->cast(k_proj_tensor, DataType::Q4_1);
    Tensor v_proj_q4_1 = ggml_ctx->cast(v_proj_tensor, DataType::Q4_1);
    Tensor o_proj_q4_1 = ggml_ctx->cast(o_proj_tensor, DataType::Q4_1);
    Tensor ff1_q4_1 = ggml_ctx->cast(ff1_tensor, DataType::Q4_1);
    Tensor ff2_q4_1 = ggml_ctx->cast(ff2_tensor, DataType::Q4_1);
    Tensor output_q4_1 = ggml_ctx->cast(output_tensor, DataType::Q4_1);
    
    // Verify all tensors have the right data type
    EXPECT_EQ(embedding_q8_0.dtype(), DataType::Q8_0);
    EXPECT_EQ(q_proj_q8_0.dtype(), DataType::Q8_0);
    EXPECT_EQ(k_proj_q8_0.dtype(), DataType::Q8_0);
    EXPECT_EQ(v_proj_q8_0.dtype(), DataType::Q8_0);
    EXPECT_EQ(o_proj_q8_0.dtype(), DataType::Q8_0);
    EXPECT_EQ(ff1_q8_0.dtype(), DataType::Q8_0);
    EXPECT_EQ(ff2_q8_0.dtype(), DataType::Q8_0);
    EXPECT_EQ(output_q8_0.dtype(), DataType::Q8_0);
    
    EXPECT_EQ(embedding_q4_0.dtype(), DataType::Q4_0);
    EXPECT_EQ(q_proj_q4_0.dtype(), DataType::Q4_0);
    EXPECT_EQ(k_proj_q4_0.dtype(), DataType::Q4_0);
    EXPECT_EQ(v_proj_q4_0.dtype(), DataType::Q4_0);
    EXPECT_EQ(o_proj_q4_0.dtype(), DataType::Q4_0);
    EXPECT_EQ(ff1_q4_0.dtype(), DataType::Q4_0);
    EXPECT_EQ(ff2_q4_0.dtype(), DataType::Q4_0);
    EXPECT_EQ(output_q4_0.dtype(), DataType::Q4_0);
    
    EXPECT_EQ(embedding_q4_1.dtype(), DataType::Q4_1);
    EXPECT_EQ(q_proj_q4_1.dtype(), DataType::Q4_1);
    EXPECT_EQ(k_proj_q4_1.dtype(), DataType::Q4_1);
    EXPECT_EQ(v_proj_q4_1.dtype(), DataType::Q4_1);
    EXPECT_EQ(o_proj_q4_1.dtype(), DataType::Q4_1);
    EXPECT_EQ(ff1_q4_1.dtype(), DataType::Q4_1);
    EXPECT_EQ(ff2_q4_1.dtype(), DataType::Q4_1);
    EXPECT_EQ(output_q4_1.dtype(), DataType::Q4_1);
    
    // Calculate memory savings
    size_t model_size_f32 = 
        (embedding_weights.size() + 
         q_proj_weights.size() + 
         k_proj_weights.size() + 
         v_proj_weights.size() + 
         o_proj_weights.size() + 
         ff1_weights.size() + 
         ff2_weights.size() + 
         output_weights.size()) * sizeof(float);
    
    size_t model_size_q8_0 = 
        embedding_q8_0.size() * sizeof(float) + 
        q_proj_q8_0.size() * sizeof(float) + 
        k_proj_q8_0.size() * sizeof(float) + 
        v_proj_q8_0.size() * sizeof(float) + 
        o_proj_q8_0.size() * sizeof(float) + 
        ff1_q8_0.size() * sizeof(float) + 
        ff2_q8_0.size() * sizeof(float) + 
        output_q8_0.size() * sizeof(float);
    
    size_t model_size_q4_0 = 
        embedding_q4_0.size() * sizeof(float) / 2 + 
        q_proj_q4_0.size() * sizeof(float) / 2 + 
        k_proj_q4_0.size() * sizeof(float) / 2 + 
        v_proj_q4_0.size() * sizeof(float) / 2 + 
        o_proj_q4_0.size() * sizeof(float) / 2 + 
        ff1_q4_0.size() * sizeof(float) / 2 + 
        ff2_q4_0.size() * sizeof(float) / 2 + 
        output_q4_0.size() * sizeof(float) / 2;
    
    size_t model_size_q4_1 = 
        embedding_q4_1.size() * sizeof(float) / 2 + 
        q_proj_q4_1.size() * sizeof(float) / 2 + 
        k_proj_q4_1.size() * sizeof(float) / 2 + 
        v_proj_q4_1.size() * sizeof(float) / 2 + 
        o_proj_q4_1.size() * sizeof(float) / 2 + 
        ff1_q4_1.size() * sizeof(float) / 2 + 
        ff2_q4_1.size() * sizeof(float) / 2 + 
        output_q4_1.size() * sizeof(float) / 2;
    
    // Print memory usage summary
    std::cout << "Model Memory Usage:" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << std::setw(10) << "Format" << std::setw(15) << "Size (bytes)" 
              << std::setw(15) << "% of F32" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    
    std::cout << std::setw(10) << "F32" << std::setw(15) << model_size_f32 
              << std::setw(15) << "100.00%" << std::endl;
    
    std::cout << std::setw(10) << "Q8_0" << std::setw(15) << model_size_q8_0 
              << std::setw(15) << std::fixed << std::setprecision(2) 
              << (100.0 * model_size_q8_0 / model_size_f32) << "%" << std::endl;
    
    std::cout << std::setw(10) << "Q4_0" << std::setw(15) << model_size_q4_0 
              << std::setw(15) << std::fixed << std::setprecision(2) 
              << (100.0 * model_size_q4_0 / model_size_f32) << "%" << std::endl;
    
    std::cout << std::setw(10) << "Q4_1" << std::setw(15) << model_size_q4_1 
              << std::setw(15) << std::fixed << std::setprecision(2) 
              << (100.0 * model_size_q4_1 / model_size_f32) << "%" << std::endl;
    
    // Validate significant memory reduction
    EXPECT_LT(model_size_q8_0, model_size_f32 * 0.5);  // At least 50% reduction
    EXPECT_LT(model_size_q4_0, model_size_f32 * 0.25); // At least 75% reduction
    EXPECT_LT(model_size_q4_1, model_size_f32 * 0.3);  // At least 70% reduction
    
    // Dequantize and compare to original
    // For brevity, we'll just check one representative matrix
    Tensor ff1_q8_0_dequant = ggml_ctx->cast(ff1_q8_0, DataType::F32);
    Tensor ff1_q4_0_dequant = ggml_ctx->cast(ff1_q4_0, DataType::F32);
    Tensor ff1_q4_1_dequant = ggml_ctx->cast(ff1_q4_1, DataType::F32);
    
    // Extract data
    std::vector<float> ff1_orig(ff1_weights.size());
    std::vector<float> ff1_q8_0_data(ff1_weights.size());
    std::vector<float> ff1_q4_0_data(ff1_weights.size());
    std::vector<float> ff1_q4_1_data(ff1_weights.size());
    
    std::memcpy(ff1_orig.data(), ff1_tensor.data(), ff1_weights.size() * sizeof(float));
    std::memcpy(ff1_q8_0_data.data(), ff1_q8_0_dequant.data(), ff1_weights.size() * sizeof(float));
    std::memcpy(ff1_q4_0_data.data(), ff1_q4_0_dequant.data(), ff1_weights.size() * sizeof(float));
    std::memcpy(ff1_q4_1_data.data(), ff1_q4_1_dequant.data(), ff1_weights.size() * sizeof(float));
    
    // Calculate errors
    double ff1_q8_0_rmse = calculate_rmse(ff1_orig, ff1_q8_0_data);
    double ff1_q4_0_rmse = calculate_rmse(ff1_orig, ff1_q4_0_data);
    double ff1_q4_1_rmse = calculate_rmse(ff1_orig, ff1_q4_1_data);
    
    // Print error metrics
    std::cout << "Weight Matrix Quantization Errors (RMSE):" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << std::setw(10) << "Format" << std::setw(15) << "RMSE" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    
    std::cout << std::setw(10) << "Q8_0" << std::setw(15) << ff1_q8_0_rmse << std::endl;
    std::cout << std::setw(10) << "Q4_0" << std::setw(15) << ff1_q4_0_rmse << std::endl;
    std::cout << std::setw(10) << "Q4_1" << std::setw(15) << ff1_q4_1_rmse << std::endl;
    
    // Validate errors are within acceptable bounds
    EXPECT_LT(ff1_q8_0_rmse, 0.1);
    EXPECT_LT(ff1_q4_0_rmse, 0.3);
    EXPECT_LT(ff1_q4_1_rmse, 0.2);
}

// Test simple forward pass with mixed precision (some layers quantized, some not)
TEST_F(GGMLModelQuantizationTest, MixedPrecisionForward) {
    // Context for building the computational graph
    struct ggml_context* mp_ctx = ggml_init(
        { .mem_size = 1024 * 1024 * 1024, .mem_buffer = NULL, .no_alloc = false }
    );
    ASSERT_NE(mp_ctx, nullptr);
    
    // Create tensor objects for model parameters
    struct ggml_tensor* emb = ggml_new_tensor_2d(mp_ctx, GGML_TYPE_F32, embedding_dim, vocab_size);
    struct ggml_tensor* q_proj = ggml_new_tensor_2d(mp_ctx, GGML_TYPE_F32, embedding_dim, embedding_dim);
    struct ggml_tensor* k_proj = ggml_new_tensor_2d(mp_ctx, GGML_TYPE_F32, embedding_dim, embedding_dim);
    struct ggml_tensor* v_proj = ggml_new_tensor_2d(mp_ctx, GGML_TYPE_F32, embedding_dim, embedding_dim);
    struct ggml_tensor* o_proj = ggml_new_tensor_2d(mp_ctx, GGML_TYPE_F32, embedding_dim, embedding_dim);
    struct ggml_tensor* ff1 = ggml_new_tensor_2d(mp_ctx, GGML_TYPE_F32, hidden_dim, embedding_dim);
    struct ggml_tensor* ff2 = ggml_new_tensor_2d(mp_ctx, GGML_TYPE_F32, embedding_dim, hidden_dim);
    struct ggml_tensor* output = ggml_new_tensor_2d(mp_ctx, GGML_TYPE_F32, vocab_size, embedding_dim);
    
    // Populate with data
    std::memcpy(emb->data, embedding_weights.data(), embedding_weights.size() * sizeof(float));
    std::memcpy(q_proj->data, q_proj_weights.data(), q_proj_weights.size() * sizeof(float));
    std::memcpy(k_proj->data, k_proj_weights.data(), k_proj_weights.size() * sizeof(float));
    std::memcpy(v_proj->data, v_proj_weights.data(), v_proj_weights.size() * sizeof(float));
    std::memcpy(o_proj->data, o_proj_weights.data(), o_proj_weights.size() * sizeof(float));
    std::memcpy(ff1->data, ff1_weights.data(), ff1_weights.size() * sizeof(float));
    std::memcpy(ff2->data, ff2_weights.data(), ff2_weights.size() * sizeof(float));
    std::memcpy(output->data, output_weights.data(), output_weights.size() * sizeof(float));
    
    // Generate a mixed precision model - quantize only some layers
    // Keep embedding and output in FP32, quantize attention and FF
    struct ggml_tensor* q_proj_q8_0 = ggml_cast(mp_ctx, q_proj, GGML_TYPE_Q8_0);
    struct ggml_tensor* k_proj_q8_0 = ggml_cast(mp_ctx, k_proj, GGML_TYPE_Q8_0);
    struct ggml_tensor* v_proj_q8_0 = ggml_cast(mp_ctx, v_proj, GGML_TYPE_Q8_0);
    struct ggml_tensor* o_proj_q8_0 = ggml_cast(mp_ctx, o_proj, GGML_TYPE_Q8_0);
    
    struct ggml_tensor* ff1_q4_0 = ggml_cast(mp_ctx, ff1, GGML_TYPE_Q4_0);
    struct ggml_tensor* ff2_q4_0 = ggml_cast(mp_ctx, ff2, GGML_TYPE_Q4_0);
    
    // Create test input - a small batch of token IDs
    size_t batch_size = 2;
    size_t seq_len = 4;
    std::vector<int32_t> input_tokens = {42, 100, 200, 300, 150, 250, 350, 400};
    struct ggml_tensor* input = ggml_new_tensor_2d(mp_ctx, GGML_TYPE_I32, seq_len, batch_size);
    std::memcpy(input->data, input_tokens.data(), input_tokens.size() * sizeof(int32_t));
    
    // Build a forward pass computation graph for the full precision model
    struct ggml_tensor* cur = ggml_get_rows(mp_ctx, emb, input);
    
    // Apply full precision attention
    struct ggml_tensor* q = ggml_mul_mat(mp_ctx, q_proj, cur);
    struct ggml_tensor* k = ggml_mul_mat(mp_ctx, k_proj, cur);
    struct ggml_tensor* v = ggml_mul_mat(mp_ctx, v_proj, cur);
    
    // Simplified attention - in reality there'd be more steps here
    struct ggml_tensor* qk = ggml_mul_mat(mp_ctx, k, q);
    struct ggml_tensor* qk_scaled = ggml_scale(mp_ctx, qk, 1.0f / sqrt(embedding_dim));
    struct ggml_tensor* qk_soft = ggml_soft_max(mp_ctx, qk_scaled);
    struct ggml_tensor* attn = ggml_mul_mat(mp_ctx, v, qk_soft);
    
    struct ggml_tensor* attn_out = ggml_mul_mat(mp_ctx, o_proj, attn);
    
    // Apply feedforward
    struct ggml_tensor* ff_mid = ggml_mul_mat(mp_ctx, ff1, attn_out);
    struct ggml_tensor* ff_act = ggml_relu(mp_ctx, ff_mid);
    struct ggml_tensor* ff_out = ggml_mul_mat(mp_ctx, ff2, ff_act);
    
    // Output projection
    struct ggml_tensor* logits = ggml_mul_mat(mp_ctx, output, ff_out);
    
    // Now build a mixed precision model forward pass
    struct ggml_tensor* cur_mp = ggml_get_rows(mp_ctx, emb, input);
    
    // Apply quantized attention
    struct ggml_tensor* q_mp = ggml_mul_mat(mp_ctx, q_proj_q8_0, cur_mp);
    struct ggml_tensor* k_mp = ggml_mul_mat(mp_ctx, k_proj_q8_0, cur_mp);
    struct ggml_tensor* v_mp = ggml_mul_mat(mp_ctx, v_proj_q8_0, cur_mp);
    
    // Simplified attention
    struct ggml_tensor* qk_mp = ggml_mul_mat(mp_ctx, k_mp, q_mp);
    struct ggml_tensor* qk_scaled_mp = ggml_scale(mp_ctx, qk_mp, 1.0f / sqrt(embedding_dim));
    struct ggml_tensor* qk_soft_mp = ggml_soft_max(mp_ctx, qk_scaled_mp);
    struct ggml_tensor* attn_mp = ggml_mul_mat(mp_ctx, v_mp, qk_soft_mp);
    
    struct ggml_tensor* attn_out_mp = ggml_mul_mat(mp_ctx, o_proj_q8_0, attn_mp);
    
    // Apply quantized feedforward
    struct ggml_tensor* ff_mid_mp = ggml_mul_mat(mp_ctx, ff1_q4_0, attn_out_mp);
    struct ggml_tensor* ff_act_mp = ggml_relu(mp_ctx, ff_mid_mp);
    struct ggml_tensor* ff_out_mp = ggml_mul_mat(mp_ctx, ff2_q4_0, ff_act_mp);
    
    // Output projection (full precision)
    struct ggml_tensor* logits_mp = ggml_mul_mat(mp_ctx, output, ff_out_mp);
    
    // Compute both graphs
    struct ggml_cgraph* graph_fp = ggml_new_graph(mp_ctx);
    ggml_build_forward_expand(graph_fp, logits);
    ggml_ctx->compute(graph_fp);
    
    struct ggml_cgraph* graph_mp = ggml_new_graph(mp_ctx);
    ggml_build_forward_expand(graph_mp, logits_mp);
    ggml_ctx->compute(graph_mp);
    
    // Compare results
    size_t logits_size = batch_size * seq_len * vocab_size;
    std::vector<float> logits_fp(logits_size);
    std::vector<float> logits_mp(logits_size);
    
    std::memcpy(logits_fp.data(), logits->data, logits_size * sizeof(float));
    std::memcpy(logits_mp.data(), logits_mp->data, logits_size * sizeof(float));
    
    // Calculate error
    double logits_rmse = calculate_rmse(logits_fp, logits_mp);
    
    // Print error metrics
    std::cout << "Mixed Precision Forward Pass:" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "Forward Pass RMSE: " << logits_rmse << std::endl;
    
    // Verify errors are within acceptable bounds for model output
    // Error will be larger due to compounding errors through model layers
    EXPECT_LT(logits_rmse, 1.0);
    
    // Check for NaN/Inf values in the output
    bool has_nan_inf = false;
    for (float val : logits_mp) {
        if (std::isnan(val) || std::isinf(val)) {
            has_nan_inf = true;
            break;
        }
    }
    EXPECT_FALSE(has_nan_inf) << "Mixed precision output contains NaN or Inf values";
    
    // Free GGML context
    ggml_free(mp_ctx);
}

// Test calibration-based weight quantization (simulated)
TEST_F(GGMLModelQuantizationTest, CalibrationQuantization) {
    // This test simulates a more advanced quantization approach where we:
    // 1. Generate calibration data (fake activations)
    // 2. Use activation statistics to guide quantization
    // 3. Compare results with standard quantization
    
    // Create a context for this test
    struct ggml_context* calib_ctx = ggml_init(
        { .mem_size = 1024 * 1024 * 1024, .mem_buffer = NULL, .no_alloc = false }
    );
    ASSERT_NE(calib_ctx, nullptr);
    
    // We'll focus on a single weight matrix for simplicity: ff1
    struct ggml_tensor* ff1 = ggml_new_tensor_2d(calib_ctx, GGML_TYPE_F32, hidden_dim, embedding_dim);
    std::memcpy(ff1->data, ff1_weights.data(), ff1_weights.size() * sizeof(float));
    
    // Generate "calibration data" that simulates activations going through this layer
    // In a real calibration scenario, we'd use real model inputs
    std::mt19937 gen(42);
    std::normal_distribution<float> normal_dist(0.0f, 1.0f);
    
    const size_t num_samples = 100;
    const size_t sample_dim = embedding_dim;
    
    std::vector<float> calibration_data(num_samples * sample_dim);
    for (size_t i = 0; i < calibration_data.size(); i++) {
        calibration_data[i] = normal_dist(gen) * 0.1f;
    }
    
    struct ggml_tensor* activations = ggml_new_tensor_2d(calib_ctx, GGML_TYPE_F32, sample_dim, num_samples);
    std::memcpy(activations->data, calibration_data.data(), calibration_data.size() * sizeof(float));
    
    // 1. Regular quantization - no calibration
    struct ggml_tensor* ff1_q8_0 = ggml_cast(calib_ctx, ff1, GGML_TYPE_Q8_0);
    struct ggml_tensor* ff1_q4_0 = ggml_cast(calib_ctx, ff1, GGML_TYPE_Q4_0);
    struct ggml_tensor* ff1_q4_1 = ggml_cast(calib_ctx, ff1, GGML_TYPE_Q4_1);
    
    // 2. Perform forward passes with calibration data, compute output
    struct ggml_tensor* output_f32 = ggml_mul_mat(calib_ctx, ff1, activations);
    struct ggml_tensor* output_q8_0 = ggml_mul_mat(calib_ctx, ff1_q8_0, activations);
    struct ggml_tensor* output_q4_0 = ggml_mul_mat(calib_ctx, ff1_q4_0, activations);
    struct ggml_tensor* output_q4_1 = ggml_mul_mat(calib_ctx, ff1_q4_1, activations);
    
    // Compute all operations
    struct ggml_cgraph* graph = ggml_new_graph(calib_ctx);
    ggml_build_forward_expand(graph, output_f32);
    ggml_build_forward_expand(graph, output_q8_0);
    ggml_build_forward_expand(graph, output_q4_0);
    ggml_build_forward_expand(graph, output_q4_1);
    ggml_ctx->compute(graph);
    
    // Get output data
    const size_t output_size = num_samples * hidden_dim;
    std::vector<float> output_f32_data(output_size);
    std::vector<float> output_q8_0_data(output_size);
    std::vector<float> output_q4_0_data(output_size);
    std::vector<float> output_q4_1_data(output_size);
    
    std::memcpy(output_f32_data.data(), output_f32->data, output_size * sizeof(float));
    std::memcpy(output_q8_0_data.data(), output_q8_0->data, output_size * sizeof(float));
    std::memcpy(output_q4_0_data.data(), output_q4_0->data, output_size * sizeof(float));
    std::memcpy(output_q4_1_data.data(), output_q4_1->data, output_size * sizeof(float));
    
    // 3. Simulate per-channel quantization by analyzing individual rows/columns
    // In a real implementation, we'd do more sophisticated per-channel calibration
    
    // Calculate error metrics for each quantization method
    double output_q8_0_rmse = calculate_rmse(output_f32_data, output_q8_0_data);
    double output_q4_0_rmse = calculate_rmse(output_f32_data, output_q4_0_data);
    double output_q4_1_rmse = calculate_rmse(output_f32_data, output_q4_1_data);
    
    // Calculate per-channel errors (analyze each output neuron)
    std::vector<double> channel_errors_q8_0(hidden_dim, 0.0);
    std::vector<double> channel_errors_q4_0(hidden_dim, 0.0);
    std::vector<double> channel_errors_q4_1(hidden_dim, 0.0);
    
    for (size_t channel = 0; channel < hidden_dim; channel++) {
        double sum_squared_error_q8_0 = 0.0;
        double sum_squared_error_q4_0 = 0.0;
        double sum_squared_error_q4_1 = 0.0;
        
        for (size_t sample = 0; sample < num_samples; sample++) {
            size_t idx = sample * hidden_dim + channel;
            
            double err_q8_0 = output_f32_data[idx] - output_q8_0_data[idx];
            double err_q4_0 = output_f32_data[idx] - output_q4_0_data[idx];
            double err_q4_1 = output_f32_data[idx] - output_q4_1_data[idx];
            
            sum_squared_error_q8_0 += err_q8_0 * err_q8_0;
            sum_squared_error_q4_0 += err_q4_0 * err_q4_0;
            sum_squared_error_q4_1 += err_q4_1 * err_q4_1;
        }
        
        channel_errors_q8_0[channel] = std::sqrt(sum_squared_error_q8_0 / num_samples);
        channel_errors_q4_0[channel] = std::sqrt(sum_squared_error_q4_0 / num_samples);
        channel_errors_q4_1[channel] = std::sqrt(sum_squared_error_q4_1 / num_samples);
    }
    
    // Find max/min error per channel for each quantization method
    double max_channel_error_q8_0 = *std::max_element(channel_errors_q8_0.begin(), channel_errors_q8_0.end());
    double min_channel_error_q8_0 = *std::min_element(channel_errors_q8_0.begin(), channel_errors_q8_0.end());
    double max_channel_error_q4_0 = *std::max_element(channel_errors_q4_0.begin(), channel_errors_q4_0.end());
    double min_channel_error_q4_0 = *std::min_element(channel_errors_q4_0.begin(), channel_errors_q4_0.end());
    double max_channel_error_q4_1 = *std::max_element(channel_errors_q4_1.begin(), channel_errors_q4_1.end());
    double min_channel_error_q4_1 = *std::min_element(channel_errors_q4_1.begin(), channel_errors_q4_1.end());
    
    // Find channels with largest error (candidates for higher precision)
    std::vector<size_t> worst_channels_q4_0(5);
    std::partial_sort_copy(
        std::begin(std::vector<size_t>(hidden_dim)), 
        std::end(std::vector<size_t>(hidden_dim)), 
        std::begin(worst_channels_q4_0), 
        std::end(worst_channels_q4_0),
        [&channel_errors_q4_0](size_t i1, size_t i2) {
            return channel_errors_q4_0[i1] > channel_errors_q4_0[i2];
        }
    );
    
    // Print quantization error metrics
    std::cout << "Calibration-based Quantization Analysis:" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "Overall RMSE:" << std::endl;
    std::cout << "  Q8_0: " << output_q8_0_rmse << std::endl;
    std::cout << "  Q4_0: " << output_q4_0_rmse << std::endl;
    std::cout << "  Q4_1: " << output_q4_1_rmse << std::endl;
    std::cout << std::endl;
    
    std::cout << "Per-Channel Error Range:" << std::endl;
    std::cout << "  Q8_0: " << min_channel_error_q8_0 << " to " << max_channel_error_q8_0 << std::endl;
    std::cout << "  Q4_0: " << min_channel_error_q4_0 << " to " << max_channel_error_q4_0 << std::endl;
    std::cout << "  Q4_1: " << min_channel_error_q4_1 << " to " << max_channel_error_q4_1 << std::endl;
    std::cout << std::endl;
    
    // Simulate mixed-precision quantization guided by calibration
    // In a real implementation, we would actually quantize to different precisions
    // Here we just measure the potential improvement
    
    // Compute model size with mixed precision (Q8_0 for sensitive channels, Q4_0 for others)
    // For simulation, we'll use worst_channels_q4_0 (5 channels) at Q8_0, rest at Q4_0
    size_t mixed_precision_size = 0;
    
    // Calculate size for 5 most sensitive channels at Q8_0
    mixed_precision_size += 5 * embedding_dim * (sizeof(int8_t) + sizeof(float) / 32);
    
    // Calculate size for remaining channels at Q4_0
    mixed_precision_size += (hidden_dim - 5) * embedding_dim * (4/8.0 + sizeof(float) / 32);
    
    // Compare with uniform quantization
    size_t uniform_q4_0_size = hidden_dim * embedding_dim * (4/8.0 + sizeof(float) / 32);
    
    std::cout << "Mixed Precision Efficiency:" << std::endl;
    std::cout << "  Uniform Q4_0 Size: " << uniform_q4_0_size << " bytes" << std::endl;
    std::cout << "  Mixed-Precision Size: " << mixed_precision_size << " bytes" << std::endl;
    std::cout << "  Size Overhead: " << (100.0 * mixed_precision_size / uniform_q4_0_size - 100.0) << "%" << std::endl;
    
    // Estimate error reduction from mixed precision
    // In reality, we'd actually implement mixed precision and measure the true improvement
    std::cout << std::endl;
    std::cout << "Expected Error Reduction from Mixed Precision:" << std::endl;
    
    // Calculate expected RMSE if 5 worst channels used Q8_0 instead of Q4_0
    double est_mixed_rmse = 0.0;
    size_t count = 0;
    
    for (size_t channel = 0; channel < hidden_dim; channel++) {
        bool is_sensitive = std::find(worst_channels_q4_0.begin(), worst_channels_q4_0.end(), channel) != worst_channels_q4_0.end();
        double channel_rmse = is_sensitive ? channel_errors_q8_0[channel] : channel_errors_q4_0[channel];
        
        // Accumulate squared error
        est_mixed_rmse += channel_rmse * channel_rmse;
        count++;
    }
    
    est_mixed_rmse = std::sqrt(est_mixed_rmse / count);
    
    std::cout << "  Uniform Q4_0 RMSE: " << output_q4_0_rmse << std::endl;
    std::cout << "  Estimated Mixed-Precision RMSE: " << est_mixed_rmse << std::endl;
    std::cout << "  Estimated Error Reduction: " << (100.0 * (output_q4_0_rmse - est_mixed_rmse) / output_q4_0_rmse) << "%" << std::endl;
    
    // Verify error is within acceptable bounds
    EXPECT_LT(output_q8_0_rmse, 0.1);
    EXPECT_LT(output_q4_0_rmse, 0.3);
    EXPECT_LT(output_q4_1_rmse, 0.2);
    
    // Free GGML context
    ggml_free(calib_ctx);
}

} // namespace
} // namespace ccsm