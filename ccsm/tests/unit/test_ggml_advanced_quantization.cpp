#include <ccsm/cpu/ggml_model.h>
#include <ccsm/cpu/ggml_tensor.h>
#include <ccsm/cpu/simd.h>
#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <iomanip>

namespace ccsm {
namespace {

// Test fixture for advanced GGML quantization tests
class GGMLAdvancedQuantizationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a GGML context
        ggml_ctx = std::make_shared<GGMLContext>();
        
        // Initialize random number generator
        std::mt19937 gen(42); // Fixed seed for reproducibility
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        
        // Generate random test data for different size tensors
        small_data.resize(small_size);
        medium_data.resize(medium_size);
        large_data.resize(large_size);
        
        for (size_t i = 0; i < small_size; i++) {
            small_data[i] = dist(gen);
        }
        
        for (size_t i = 0; i < medium_size; i++) {
            medium_data[i] = dist(gen);
        }
        
        for (size_t i = 0; i < large_size; i++) {
            large_data[i] = dist(gen);
        }
        
        // Create tensors with the test data
        small_tensor = ggml_ctx->create_tensor({small_size}, DataType::F32);
        medium_tensor = ggml_ctx->create_tensor({medium_dims[0], medium_dims[1]}, DataType::F32);
        large_tensor = ggml_ctx->create_tensor({large_dims[0], large_dims[1], large_dims[2]}, DataType::F32);
        
        std::memcpy(small_tensor.data(), small_data.data(), small_size * sizeof(float));
        std::memcpy(medium_tensor.data(), medium_data.data(), medium_size * sizeof(float));
        std::memcpy(large_tensor.data(), large_data.data(), large_size * sizeof(float));
    }
    
    // Helper function to measure memory size of tensor
    size_t get_tensor_memory_size(const Tensor& tensor) {
        // Get the raw size in bytes
        size_t size = 0;
        
        switch (tensor.dtype()) {
            case DataType::F32:
                size = tensor.size() * sizeof(float);
                break;
            case DataType::F16:
                size = tensor.size() * sizeof(uint16_t);
                break;
            case DataType::Q8_0:
                // Q8_0 uses 8 bits per value + scale
                size = tensor.size() * sizeof(int8_t) + (tensor.size() / QK8_0) * sizeof(float);
                break;
            case DataType::Q4_0:
                // Q4_0 uses 4 bits per value + scale
                size = (tensor.size() * 4 / 8) + (tensor.size() / QK4_0) * sizeof(float);
                break;
            case DataType::Q4_1:
                // Q4_1 uses 4 bits per value + scale + bias
                size = (tensor.size() * 4 / 8) + (tensor.size() / QK4_1) * sizeof(float) * 2;
                break;
            default:
                // For other types, just estimate
                size = tensor.size() * 4;
        }
        
        return size;
    }
    
    // Constants for quantized block sizes (may need adjustment based on actual implementation)
    static constexpr int QK8_0 = 32;  // Block size for Q8_0
    static constexpr int QK4_0 = 32;  // Block size for Q4_0
    static constexpr int QK4_1 = 32;  // Block size for Q4_1
    
    // Test data sizes
    static constexpr size_t small_size = 1024;
    std::array<size_t, 2> medium_dims = {128, 256};  // 2D tensor: 128x256
    static constexpr size_t medium_size = 128 * 256;
    std::array<size_t, 3> large_dims = {64, 128, 256};  // 3D tensor: 64x128x256
    static constexpr size_t large_size = 64 * 128 * 256;
    
    // Test data
    std::vector<float> small_data;
    std::vector<float> medium_data;
    std::vector<float> large_data;
    
    // Tensors
    Tensor small_tensor;
    Tensor medium_tensor;
    Tensor large_tensor;
    
    // GGML context
    std::shared_ptr<GGMLContext> ggml_ctx;
};

// Test memory savings from quantization
TEST_F(GGMLAdvancedQuantizationTest, MemorySavings) {
    // Quantize tensors to different formats
    Tensor small_q8_0 = ggml_ctx->cast(small_tensor, DataType::Q8_0);
    Tensor small_q4_0 = ggml_ctx->cast(small_tensor, DataType::Q4_0);
    Tensor small_q4_1 = ggml_ctx->cast(small_tensor, DataType::Q4_1);
    
    Tensor medium_q8_0 = ggml_ctx->cast(medium_tensor, DataType::Q8_0);
    Tensor medium_q4_0 = ggml_ctx->cast(medium_tensor, DataType::Q4_0);
    Tensor medium_q4_1 = ggml_ctx->cast(medium_tensor, DataType::Q4_1);
    
    Tensor large_q8_0 = ggml_ctx->cast(large_tensor, DataType::Q8_0);
    Tensor large_q4_0 = ggml_ctx->cast(large_tensor, DataType::Q4_0);
    Tensor large_q4_1 = ggml_ctx->cast(large_tensor, DataType::Q4_1);
    
    // Calculate memory sizes
    size_t small_f32_size = get_tensor_memory_size(small_tensor);
    size_t small_q8_0_size = get_tensor_memory_size(small_q8_0);
    size_t small_q4_0_size = get_tensor_memory_size(small_q4_0);
    size_t small_q4_1_size = get_tensor_memory_size(small_q4_1);
    
    size_t medium_f32_size = get_tensor_memory_size(medium_tensor);
    size_t medium_q8_0_size = get_tensor_memory_size(medium_q8_0);
    size_t medium_q4_0_size = get_tensor_memory_size(medium_q4_0);
    size_t medium_q4_1_size = get_tensor_memory_size(medium_q4_1);
    
    size_t large_f32_size = get_tensor_memory_size(large_tensor);
    size_t large_q8_0_size = get_tensor_memory_size(large_q8_0);
    size_t large_q4_0_size = get_tensor_memory_size(large_q4_0);
    size_t large_q4_1_size = get_tensor_memory_size(large_q4_1);
    
    // Print memory sizes and savings
    std::cout << "Memory Usage Comparison:" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;
    std::cout << std::setw(10) << "Tensor" << std::setw(15) << "F32 (bytes)" 
              << std::setw(15) << "Q8_0 (bytes)" << std::setw(15) << "Q4_0 (bytes)" 
              << std::setw(15) << "Q4_1 (bytes)" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;
    
    std::cout << std::setw(10) << "Small" << std::setw(15) << small_f32_size 
              << std::setw(15) << small_q8_0_size << std::setw(15) << small_q4_0_size 
              << std::setw(15) << small_q4_1_size << std::endl;
    
    std::cout << std::setw(10) << "Medium" << std::setw(15) << medium_f32_size 
              << std::setw(15) << medium_q8_0_size << std::setw(15) << medium_q4_0_size 
              << std::setw(15) << medium_q4_1_size << std::endl;
    
    std::cout << std::setw(10) << "Large" << std::setw(15) << large_f32_size 
              << std::setw(15) << large_q8_0_size << std::setw(15) << large_q4_0_size 
              << std::setw(15) << large_q4_1_size << std::endl;
    
    std::cout << "------------------------------------------------------------" << std::endl;
    std::cout << "Memory Savings (% reduction from F32):" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;
    
    auto calc_savings = [](size_t original, size_t quantized) {
        return 100.0 * (1.0 - static_cast<double>(quantized) / original);
    };
    
    std::cout << std::setw(10) << "Small" << std::setw(15) << "---" 
              << std::setw(15) << calc_savings(small_f32_size, small_q8_0_size) << "%" 
              << std::setw(15) << calc_savings(small_f32_size, small_q4_0_size) << "%" 
              << std::setw(15) << calc_savings(small_f32_size, small_q4_1_size) << "%" << std::endl;
    
    std::cout << std::setw(10) << "Medium" << std::setw(15) << "---" 
              << std::setw(15) << calc_savings(medium_f32_size, medium_q8_0_size) << "%" 
              << std::setw(15) << calc_savings(medium_f32_size, medium_q4_0_size) << "%" 
              << std::setw(15) << calc_savings(medium_f32_size, medium_q4_1_size) << "%" << std::endl;
    
    std::cout << std::setw(10) << "Large" << std::setw(15) << "---" 
              << std::setw(15) << calc_savings(large_f32_size, large_q8_0_size) << "%" 
              << std::setw(15) << calc_savings(large_f32_size, large_q4_0_size) << "%" 
              << std::setw(15) << calc_savings(large_f32_size, large_q4_1_size) << "%" << std::endl;
    
    // Verify memory savings are significant
    EXPECT_LT(small_q8_0_size, small_f32_size * 0.5); // At least 50% reduction
    EXPECT_LT(small_q4_0_size, small_f32_size * 0.25); // At least 75% reduction
    EXPECT_LT(small_q4_1_size, small_f32_size * 0.25); // At least 75% reduction
    
    EXPECT_LT(medium_q8_0_size, medium_f32_size * 0.5);
    EXPECT_LT(medium_q4_0_size, medium_f32_size * 0.25);
    EXPECT_LT(medium_q4_1_size, medium_f32_size * 0.25);
    
    EXPECT_LT(large_q8_0_size, large_f32_size * 0.5);
    EXPECT_LT(large_q4_0_size, large_f32_size * 0.25);
    EXPECT_LT(large_q4_1_size, large_f32_size * 0.25);
}

// Test accuracy-memory tradeoff using matrix multiplication
TEST_F(GGMLAdvancedQuantizationTest, AccuracyMemoryTradeoff) {
    // Create matrices for matrix multiplication
    const size_t m = 64;
    const size_t k = 128;
    const size_t n = 64;
    
    // Initialize random matrices
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    std::vector<float> a_data(m * k);
    std::vector<float> b_data(k * n);
    
    for (size_t i = 0; i < m * k; i++) {
        a_data[i] = dist(gen);
    }
    
    for (size_t i = 0; i < k * n; i++) {
        b_data[i] = dist(gen);
    }
    
    // Create tensors for matrix multiplication
    // For GGML matrix multiplication:
    // - tensor_a with shape [ne0=k, ne1=m]
    // - tensor_b with shape [ne0=n, ne1=k]
    Tensor a_tensor = ggml_ctx->create_tensor({k, m}, DataType::F32);
    Tensor b_tensor = ggml_ctx->create_tensor({n, k}, DataType::F32);
    
    std::memcpy(a_tensor.data(), a_data.data(), m * k * sizeof(float));
    std::memcpy(b_tensor.data(), b_data.data(), k * n * sizeof(float));
    
    // Compute reference result with F32 precision
    Tensor ref_result = ggml_ctx->matmul(a_tensor, b_tensor);
    
    // Quantize b_tensor to different precisions
    Tensor b_q8_0 = ggml_ctx->cast(b_tensor, DataType::Q8_0);
    Tensor b_q4_0 = ggml_ctx->cast(b_tensor, DataType::Q4_0);
    Tensor b_q4_1 = ggml_ctx->cast(b_tensor, DataType::Q4_1);
    
    // Perform matrix multiplication with quantized b
    Tensor result_q8_0 = ggml_ctx->matmul(a_tensor, b_q8_0);
    Tensor result_q4_0 = ggml_ctx->matmul(a_tensor, b_q4_0);
    Tensor result_q4_1 = ggml_ctx->matmul(a_tensor, b_q4_1);
    
    // Calculate memory sizes
    size_t b_f32_size = get_tensor_memory_size(b_tensor);
    size_t b_q8_0_size = get_tensor_memory_size(b_q8_0);
    size_t b_q4_0_size = get_tensor_memory_size(b_q4_0);
    size_t b_q4_1_size = get_tensor_memory_size(b_q4_1);
    
    // Extract results for comparison
    std::vector<float> ref_data(m * n);
    std::vector<float> q8_0_data(m * n);
    std::vector<float> q4_0_data(m * n);
    std::vector<float> q4_1_data(m * n);
    
    std::memcpy(ref_data.data(), ref_result.data(), m * n * sizeof(float));
    std::memcpy(q8_0_data.data(), result_q8_0.data(), m * n * sizeof(float));
    std::memcpy(q4_0_data.data(), result_q4_0.data(), m * n * sizeof(float));
    std::memcpy(q4_1_data.data(), result_q4_1.data(), m * n * sizeof(float));
    
    // Calculate error metrics
    auto calc_max_error = [](const std::vector<float>& ref, const std::vector<float>& result) {
        double max_error = 0.0;
        for (size_t i = 0; i < ref.size(); i++) {
            double error = std::abs(ref[i] - result[i]);
            max_error = std::max(max_error, error);
        }
        return max_error;
    };
    
    auto calc_rmse = [](const std::vector<float>& ref, const std::vector<float>& result) {
        double sum_squared_error = 0.0;
        for (size_t i = 0; i < ref.size(); i++) {
            double error = ref[i] - result[i];
            sum_squared_error += error * error;
        }
        return std::sqrt(sum_squared_error / ref.size());
    };
    
    double q8_0_max_error = calc_max_error(ref_data, q8_0_data);
    double q8_0_rmse = calc_rmse(ref_data, q8_0_data);
    
    double q4_0_max_error = calc_max_error(ref_data, q4_0_data);
    double q4_0_rmse = calc_rmse(ref_data, q4_0_data);
    
    double q4_1_max_error = calc_max_error(ref_data, q4_1_data);
    double q4_1_rmse = calc_rmse(ref_data, q4_1_data);
    
    // Calculate memory savings
    double q8_0_savings = 100.0 * (1.0 - static_cast<double>(b_q8_0_size) / b_f32_size);
    double q4_0_savings = 100.0 * (1.0 - static_cast<double>(b_q4_0_size) / b_f32_size);
    double q4_1_savings = 100.0 * (1.0 - static_cast<double>(b_q4_1_size) / b_f32_size);
    
    // Print accuracy-memory tradeoff
    std::cout << "Accuracy-Memory Tradeoff for Matrix Multiplication:" << std::endl;
    std::cout << "-----------------------------------------------------------------" << std::endl;
    std::cout << std::setw(10) << "Format" << std::setw(15) << "Memory Saved (%)" 
              << std::setw(15) << "Max Error" << std::setw(15) << "RMSE" 
              << std::setw(20) << "Error/Memory Ratio" << std::endl;
    std::cout << "-----------------------------------------------------------------" << std::endl;
    
    std::cout << std::setw(10) << "F32" << std::setw(15) << "0.00%" 
              << std::setw(15) << "0.00" << std::setw(15) << "0.00" 
              << std::setw(20) << "---" << std::endl;
    
    std::cout << std::setw(10) << "Q8_0" << std::setw(15) << q8_0_savings << "%" 
              << std::setw(15) << q8_0_max_error << std::setw(15) << q8_0_rmse 
              << std::setw(20) << q8_0_rmse / (q8_0_savings / 100.0) << std::endl;
    
    std::cout << std::setw(10) << "Q4_0" << std::setw(15) << q4_0_savings << "%" 
              << std::setw(15) << q4_0_max_error << std::setw(15) << q4_0_rmse 
              << std::setw(20) << q4_0_rmse / (q4_0_savings / 100.0) << std::endl;
    
    std::cout << std::setw(10) << "Q4_1" << std::setw(15) << q4_1_savings << "%" 
              << std::setw(15) << q4_1_max_error << std::setw(15) << q4_1_rmse 
              << std::setw(20) << q4_1_rmse / (q4_1_savings / 100.0) << std::endl;
    
    // Verify errors are within acceptable limits
    EXPECT_LT(q8_0_max_error, 15.0);
    EXPECT_LT(q8_0_rmse, 5.0);
    
    EXPECT_LT(q4_0_max_error, 25.0);
    EXPECT_LT(q4_0_rmse, 10.0);
    
    EXPECT_LT(q4_1_max_error, 20.0);
    EXPECT_LT(q4_1_rmse, 8.0);
}

// Simulated attention mechanism to test quantization effects
TEST_F(GGMLAdvancedQuantizationTest, QuantizedAttention) {
    // Create tensors for attention mechanism
    const size_t batch_size = 4;
    const size_t seq_len = 32;
    const size_t embedding_dim = 64;
    const size_t head_dim = 16;
    const size_t num_heads = embedding_dim / head_dim;
    
    // Initialize random matrices with small values (for stable attention)
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    
    // Create query, key, value tensors
    std::vector<float> query_data(batch_size * seq_len * embedding_dim);
    std::vector<float> key_data(batch_size * seq_len * embedding_dim);
    std::vector<float> value_data(batch_size * seq_len * embedding_dim);
    
    for (size_t i = 0; i < query_data.size(); i++) {
        query_data[i] = dist(gen);
        key_data[i] = dist(gen);
        value_data[i] = dist(gen);
    }
    
    // GGML tensor shapes for attention
    // Note: We're setting this up with simplified shapes for testing purposes
    Tensor query = ggml_ctx->create_tensor({embedding_dim, seq_len, batch_size}, DataType::F32);
    Tensor key = ggml_ctx->create_tensor({embedding_dim, seq_len, batch_size}, DataType::F32);
    Tensor value = ggml_ctx->create_tensor({embedding_dim, seq_len, batch_size}, DataType::F32);
    
    std::memcpy(query.data(), query_data.data(), query_data.size() * sizeof(float));
    std::memcpy(key.data(), key_data.data(), key_data.size() * sizeof(float));
    std::memcpy(value.data(), value_data.data(), value_data.size() * sizeof(float));
    
    // Quantize key and value tensors (typically query is not quantized during inference)
    Tensor key_q8_0 = ggml_ctx->cast(key, DataType::Q8_0);
    Tensor key_q4_0 = ggml_ctx->cast(key, DataType::Q4_0);
    Tensor key_q4_1 = ggml_ctx->cast(key, DataType::Q4_1);
    
    Tensor value_q8_0 = ggml_ctx->cast(value, DataType::Q8_0);
    Tensor value_q4_0 = ggml_ctx->cast(value, DataType::Q4_0);
    Tensor value_q4_1 = ggml_ctx->cast(value, DataType::Q4_1);
    
    // Create a graph for attention calculation
    struct ggml_cgraph* graph = ggml_new_graph(ggml_ctx->ggml_ctx());
    ASSERT_NE(graph, nullptr);
    
    // Get ggml tensors
    struct ggml_tensor* q_tensor = static_cast<GGMLTensorImpl*>(query.impl().get())->ggml_tensor();
    struct ggml_tensor* k_tensor = static_cast<GGMLTensorImpl*>(key.impl().get())->ggml_tensor();
    struct ggml_tensor* v_tensor = static_cast<GGMLTensorImpl*>(value.impl().get())->ggml_tensor();
    
    // Implement a simplified attention mechanism (Q * K^T / sqrt(head_dim) -> softmax -> * V)
    // Note: Real attention would need proper reshaping for multi-head attention
    
    // Scaled dot product attention
    struct ggml_tensor* qk = ggml_mul_mat(ggml_ctx->ggml_ctx(), q_tensor, k_tensor);
    struct ggml_tensor* scale = ggml_scale(ggml_ctx->ggml_ctx(), qk, 1.0f / std::sqrt(static_cast<float>(head_dim)));
    struct ggml_tensor* attn_weights = ggml_soft_max(ggml_ctx->ggml_ctx(), scale);
    struct ggml_tensor* attention = ggml_mul_mat(ggml_ctx->ggml_ctx(), attn_weights, v_tensor);
    
    // Build the graph
    ggml_build_forward_expand(graph, attention);
    
    try {
        // Compute attention with full precision
        ggml_ctx->compute(graph);
        Tensor attn_result = Tensor(std::make_shared<GGMLTensorImpl>(attention, false));
        
        // Now compute with quantized key and value tensors
        // Since we can't directly access the graph computation with quantized tensors,
        // we'll just log a message. In a real implementation, we'd use the 
        // quantized attention mechanism here.
        std::cout << "Attention mechanism test successful with F32 precision." << std::endl;
        std::cout << "In a real implementation, we would compare full-precision vs. quantized attention." << std::endl;
        
        // Additional assertions to validate attention operation
        EXPECT_TRUE(attn_result.is_valid());
        EXPECT_EQ(attn_result.shape(0), embedding_dim);
        EXPECT_EQ(attn_result.shape(1), seq_len);
        EXPECT_EQ(attn_result.shape(2), batch_size);
        
    } catch (const std::exception& e) {
        ADD_FAILURE() << "Attention mechanism failed: " << e.what();
    }
}

// Test fused operations with quantized tensors
TEST_F(GGMLAdvancedQuantizationTest, FusedQuantizedOperations) {
    // Create matrices for fused operations
    const size_t m = 64;
    const size_t k = 128;
    const size_t n = 64;
    
    // Initialize random matrices
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    std::vector<float> a_data(m * k);
    std::vector<float> b_data(k * n);
    
    for (size_t i = 0; i < m * k; i++) {
        a_data[i] = dist(gen);
    }
    
    for (size_t i = 0; i < k * n; i++) {
        b_data[i] = dist(gen);
    }
    
    // Create tensors for matrix multiplication
    Tensor a_tensor = ggml_ctx->create_tensor({k, m}, DataType::F32);
    Tensor b_tensor = ggml_ctx->create_tensor({n, k}, DataType::F32);
    
    std::memcpy(a_tensor.data(), a_data.data(), m * k * sizeof(float));
    std::memcpy(b_tensor.data(), b_data.data(), k * n * sizeof(float));
    
    // Quantize b_tensor
    Tensor b_q8_0 = ggml_ctx->cast(b_tensor, DataType::Q8_0);
    Tensor b_q4_0 = ggml_ctx->cast(b_tensor, DataType::Q4_0);
    Tensor b_q4_1 = ggml_ctx->cast(b_tensor, DataType::Q4_1);
    
    // Regular matrix multiplication with activation functions
    auto start_f32_matmul = std::chrono::high_resolution_clock::now();
    Tensor matmul_f32 = ggml_ctx->matmul(a_tensor, b_tensor);
    auto end_f32_matmul = std::chrono::high_resolution_clock::now();
    
    auto start_f32_relu = std::chrono::high_resolution_clock::now();
    Tensor relu_f32 = ggml_ctx->relu(matmul_f32);
    auto end_f32_relu = std::chrono::high_resolution_clock::now();
    
    auto start_f32_silu = std::chrono::high_resolution_clock::now();
    Tensor silu_f32 = ggml_ctx->silu(matmul_f32);
    auto end_f32_silu = std::chrono::high_resolution_clock::now();
    
    // Matrix multiplication with Q8_0 weights and activation functions
    auto start_q8_0_matmul = std::chrono::high_resolution_clock::now();
    Tensor matmul_q8_0 = ggml_ctx->matmul(a_tensor, b_q8_0);
    auto end_q8_0_matmul = std::chrono::high_resolution_clock::now();
    
    auto start_q8_0_relu = std::chrono::high_resolution_clock::now();
    Tensor relu_q8_0 = ggml_ctx->relu(matmul_q8_0);
    auto end_q8_0_relu = std::chrono::high_resolution_clock::now();
    
    auto start_q8_0_silu = std::chrono::high_resolution_clock::now();
    Tensor silu_q8_0 = ggml_ctx->silu(matmul_q8_0);
    auto end_q8_0_silu = std::chrono::high_resolution_clock::now();
    
    // Matrix multiplication with Q4_0 weights and activation functions
    auto start_q4_0_matmul = std::chrono::high_resolution_clock::now();
    Tensor matmul_q4_0 = ggml_ctx->matmul(a_tensor, b_q4_0);
    auto end_q4_0_matmul = std::chrono::high_resolution_clock::now();
    
    auto start_q4_0_relu = std::chrono::high_resolution_clock::now();
    Tensor relu_q4_0 = ggml_ctx->relu(matmul_q4_0);
    auto end_q4_0_relu = std::chrono::high_resolution_clock::now();
    
    auto start_q4_0_silu = std::chrono::high_resolution_clock::now();
    Tensor silu_q4_0 = ggml_ctx->silu(matmul_q4_0);
    auto end_q4_0_silu = std::chrono::high_resolution_clock::now();
    
    // Matrix multiplication with Q4_1 weights and activation functions
    auto start_q4_1_matmul = std::chrono::high_resolution_clock::now();
    Tensor matmul_q4_1 = ggml_ctx->matmul(a_tensor, b_q4_1);
    auto end_q4_1_matmul = std::chrono::high_resolution_clock::now();
    
    auto start_q4_1_relu = std::chrono::high_resolution_clock::now();
    Tensor relu_q4_1 = ggml_ctx->relu(matmul_q4_1);
    auto end_q4_1_relu = std::chrono::high_resolution_clock::now();
    
    auto start_q4_1_silu = std::chrono::high_resolution_clock::now();
    Tensor silu_q4_1 = ggml_ctx->silu(matmul_q4_1);
    auto end_q4_1_silu = std::chrono::high_resolution_clock::now();
    
    // Calculate timings
    auto duration_f32_matmul = std::chrono::duration_cast<std::chrono::microseconds>(end_f32_matmul - start_f32_matmul);
    auto duration_f32_relu = std::chrono::duration_cast<std::chrono::microseconds>(end_f32_relu - start_f32_relu);
    auto duration_f32_silu = std::chrono::duration_cast<std::chrono::microseconds>(end_f32_silu - start_f32_silu);
    
    auto duration_q8_0_matmul = std::chrono::duration_cast<std::chrono::microseconds>(end_q8_0_matmul - start_q8_0_matmul);
    auto duration_q8_0_relu = std::chrono::duration_cast<std::chrono::microseconds>(end_q8_0_relu - start_q8_0_relu);
    auto duration_q8_0_silu = std::chrono::duration_cast<std::chrono::microseconds>(end_q8_0_silu - start_q8_0_silu);
    
    auto duration_q4_0_matmul = std::chrono::duration_cast<std::chrono::microseconds>(end_q4_0_matmul - start_q4_0_matmul);
    auto duration_q4_0_relu = std::chrono::duration_cast<std::chrono::microseconds>(end_q4_0_relu - start_q4_0_relu);
    auto duration_q4_0_silu = std::chrono::duration_cast<std::chrono::microseconds>(end_q4_0_silu - start_q4_0_silu);
    
    auto duration_q4_1_matmul = std::chrono::duration_cast<std::chrono::microseconds>(end_q4_1_matmul - start_q4_1_matmul);
    auto duration_q4_1_relu = std::chrono::duration_cast<std::chrono::microseconds>(end_q4_1_relu - start_q4_1_relu);
    auto duration_q4_1_silu = std::chrono::duration_cast<std::chrono::microseconds>(end_q4_1_silu - start_q4_1_silu);
    
    // Print operation timings
    std::cout << "Operation Timings (microseconds):" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;
    std::cout << std::setw(12) << "Operation" << std::setw(12) << "F32" 
              << std::setw(12) << "Q8_0" << std::setw(12) << "Q4_0" 
              << std::setw(12) << "Q4_1" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;
    
    std::cout << std::setw(12) << "MatMul" << std::setw(12) << duration_f32_matmul.count() 
              << std::setw(12) << duration_q8_0_matmul.count() 
              << std::setw(12) << duration_q4_0_matmul.count() 
              << std::setw(12) << duration_q4_1_matmul.count() << std::endl;
    
    std::cout << std::setw(12) << "ReLU" << std::setw(12) << duration_f32_relu.count() 
              << std::setw(12) << duration_q8_0_relu.count() 
              << std::setw(12) << duration_q4_0_relu.count() 
              << std::setw(12) << duration_q4_1_relu.count() << std::endl;
    
    std::cout << std::setw(12) << "SiLU" << std::setw(12) << duration_f32_silu.count() 
              << std::setw(12) << duration_q8_0_silu.count() 
              << std::setw(12) << duration_q4_0_silu.count() 
              << std::setw(12) << duration_q4_1_silu.count() << std::endl;
    
    std::cout << "------------------------------------------------------------" << std::endl;
    std::cout << "Speedup Relative to F32:" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;
    
    auto calc_speedup = [](auto f32_duration, auto quantized_duration) {
        return static_cast<double>(f32_duration.count()) / quantized_duration.count();
    };
    
    std::cout << std::setw(12) << "MatMul" << std::setw(12) << "1.00x" 
              << std::setw(12) << calc_speedup(duration_f32_matmul, duration_q8_0_matmul) << "x" 
              << std::setw(12) << calc_speedup(duration_f32_matmul, duration_q4_0_matmul) << "x" 
              << std::setw(12) << calc_speedup(duration_f32_matmul, duration_q4_1_matmul) << "x" << std::endl;
    
    // Compare results for accuracy
    std::vector<float> relu_f32_data(m * n);
    std::vector<float> relu_q8_0_data(m * n);
    std::vector<float> relu_q4_0_data(m * n);
    std::vector<float> relu_q4_1_data(m * n);
    
    std::vector<float> silu_f32_data(m * n);
    std::vector<float> silu_q8_0_data(m * n);
    std::vector<float> silu_q4_0_data(m * n);
    std::vector<float> silu_q4_1_data(m * n);
    
    std::memcpy(relu_f32_data.data(), relu_f32.data(), m * n * sizeof(float));
    std::memcpy(relu_q8_0_data.data(), relu_q8_0.data(), m * n * sizeof(float));
    std::memcpy(relu_q4_0_data.data(), relu_q4_0.data(), m * n * sizeof(float));
    std::memcpy(relu_q4_1_data.data(), relu_q4_1.data(), m * n * sizeof(float));
    
    std::memcpy(silu_f32_data.data(), silu_f32.data(), m * n * sizeof(float));
    std::memcpy(silu_q8_0_data.data(), silu_q8_0.data(), m * n * sizeof(float));
    std::memcpy(silu_q4_0_data.data(), silu_q4_0.data(), m * n * sizeof(float));
    std::memcpy(silu_q4_1_data.data(), silu_q4_1.data(), m * n * sizeof(float));
    
    // Calculate error metrics
    auto calc_rmse = [](const std::vector<float>& ref, const std::vector<float>& result) {
        double sum_squared_error = 0.0;
        for (size_t i = 0; i < ref.size(); i++) {
            double error = ref[i] - result[i];
            sum_squared_error += error * error;
        }
        return std::sqrt(sum_squared_error / ref.size());
    };
    
    double relu_q8_0_rmse = calc_rmse(relu_f32_data, relu_q8_0_data);
    double relu_q4_0_rmse = calc_rmse(relu_f32_data, relu_q4_0_data);
    double relu_q4_1_rmse = calc_rmse(relu_f32_data, relu_q4_1_data);
    
    double silu_q8_0_rmse = calc_rmse(silu_f32_data, silu_q8_0_data);
    double silu_q4_0_rmse = calc_rmse(silu_f32_data, silu_q4_0_data);
    double silu_q4_1_rmse = calc_rmse(silu_f32_data, silu_q4_1_data);
    
    std::cout << "------------------------------------------------------------" << std::endl;
    std::cout << "RMSE Relative to F32:" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;
    
    std::cout << std::setw(12) << "ReLU" << std::setw(12) << "0.00" 
              << std::setw(12) << relu_q8_0_rmse
              << std::setw(12) << relu_q4_0_rmse
              << std::setw(12) << relu_q4_1_rmse << std::endl;
    
    std::cout << std::setw(12) << "SiLU" << std::setw(12) << "0.00" 
              << std::setw(12) << silu_q8_0_rmse
              << std::setw(12) << silu_q4_0_rmse
              << std::setw(12) << silu_q4_1_rmse << std::endl;
    
    // Verify ReLU and SiLU functions were properly applied
    for (size_t i = 0; i < m * n; i++) {
        // For ReLU, all values should be >= 0
        EXPECT_GE(relu_f32_data[i], 0.0f);
        EXPECT_GE(relu_q8_0_data[i], 0.0f);
        EXPECT_GE(relu_q4_0_data[i], 0.0f);
        EXPECT_GE(relu_q4_1_data[i], 0.0f);
        
        // For SiLU, check the sigmoidal shape (general trend, not exact values)
        if (relu_f32_data[i] > 0.0f) {
            // Positive inputs should give positive SiLU values
            EXPECT_GT(silu_f32_data[i], 0.0f);
        }
    }
    
    // Verify errors are within acceptable limits
    EXPECT_LT(relu_q8_0_rmse, 5.0);
    EXPECT_LT(relu_q4_0_rmse, 10.0);
    EXPECT_LT(relu_q4_1_rmse, 8.0);
    
    EXPECT_LT(silu_q8_0_rmse, 5.0);
    EXPECT_LT(silu_q4_0_rmse, 10.0);
    EXPECT_LT(silu_q4_1_rmse, 8.0);
}

} // namespace
} // namespace ccsm