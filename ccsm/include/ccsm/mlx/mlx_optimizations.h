#ifndef CCSM_MLX_OPTIMIZATIONS_H
#define CCSM_MLX_OPTIMIZATIONS_H

#include <ccsm/tensor.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

#ifdef CCSM_WITH_MLX
#include "mlx/c/array.h"
#include "mlx/c/device.h"
#include "mlx/c/stream.h"
#endif

namespace ccsm {

// Configuration for MLX optimizations
struct MLXOptimizationConfig {
    // Compute precision options
    enum class ComputePrecision {
        FLOAT32,    // Full precision (slower but more accurate)
        BFLOAT16,   // Brain float (good balance of speed and accuracy)
        FLOAT16     // Half precision (fastest but less accurate)
    };
    
    // Memory usage options
    enum class MemoryUsage {
        MINIMAL,    // Minimize memory usage, potentially slower
        BALANCED,   // Balance memory usage and performance
        PERFORMANCE // Prioritize performance, use more memory
    };
    
    // Default configuration
    ComputePrecision compute_precision = ComputePrecision::BFLOAT16;
    MemoryUsage memory_usage = MemoryUsage::BALANCED;
    int num_compute_threads = 0; // 0 means use system default
    bool use_autotune = true;
    bool use_async_compute = true;
    
    // Helper methods to configure from environment variables
    static MLXOptimizationConfig from_env();
};

#ifdef CCSM_WITH_MLX
// MLX-specific optimization functions

// Configure MLX for optimal performance based on the current device
void configure_mlx_for_device(const MLXOptimizationConfig& config = MLXOptimizationConfig());

// Convert MLX arrays to the specified precision
mlx_array convert_precision(mlx_array array, MLXOptimizationConfig::ComputePrecision precision, mlx_stream stream);

// Memory optimization: release resources not needed for current operation
void optimize_memory_usage(std::vector<mlx_array>& arrays_to_keep, MLXOptimizationConfig::MemoryUsage usage_level);

// Fast computation for common operations 
namespace mlx_fast {
    // Optimized matrix multiplication that avoids unnecessary intermediate memory allocations
    mlx_array matmul_optimized(mlx_array a, mlx_array b, bool transpose_b, mlx_stream stream);
    
    // Fused layer norm + linear operation
    mlx_array fused_layer_norm_linear(
        mlx_array x, 
        mlx_array norm_weight, 
        mlx_array norm_bias,
        mlx_array linear_weight,
        float eps,
        mlx_stream stream);
    
    // Fused attention operation that computes QKV projections and attention in one operation
    mlx_array fused_attention(
        mlx_array x,
        mlx_array wq,
        mlx_array wk,
        mlx_array wv,
        mlx_array wo,
        const std::vector<int>& positions,
        float rope_theta,
        int n_heads,
        int n_kv_heads,
        mlx_stream stream);
        
    // Optimized rotary positional embedding implementation
    mlx_array fast_rope(
        mlx_array x,
        const std::vector<int>& positions,
        float theta,
        mlx_stream stream);
}

// Batch processing to minimize data transfers
class MLXBatchProcessor {
public:
    MLXBatchProcessor();
    ~MLXBatchProcessor();
    
    // Add a tensor operation to the batch
    void add_operation(const std::function<mlx_array()>& op);
    
    // Execute all operations in the batch
    std::vector<mlx_array> execute(mlx_stream stream);
    
    // Clear all operations
    void clear();
    
private:
    std::vector<std::function<mlx_array()>> operations_;
};

// Memory-efficient tensor operations
namespace mlx_memory {
    // In-place operations where possible
    mlx_array add_inplace(mlx_array& a, mlx_array b, mlx_stream stream);
    mlx_array multiply_inplace(mlx_array& a, mlx_array b, mlx_stream stream);
    
    // Tensor reuse pool for minimizing allocations
    class TensorPool {
    public:
        TensorPool(size_t max_pool_size = 64);
        ~TensorPool();
        
        // Get a tensor of the specified shape and dtype from the pool or create a new one
        mlx_array get(const std::vector<int>& shape, mlx_dtype dtype);
        
        // Return a tensor to the pool for reuse
        void recycle(mlx_array& tensor);
        
        // Clear the pool
        void clear();
        
    private:
        size_t max_pool_size_;
        std::vector<mlx_array> pool_;
        
        // Helper to check if two shapes match
        bool shapes_match(mlx_array tensor, const std::vector<int>& shape);
    };
}

#endif // CCSM_WITH_MLX

} // namespace ccsm

#endif // CCSM_MLX_OPTIMIZATIONS_H