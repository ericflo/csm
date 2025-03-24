# KV Cache Quantization

This document outlines the plan for implementing quantization of KV (Key-Value) caches in the CCSM model to further optimize memory usage.

## 1. Overview

KV cache quantization can significantly reduce memory usage during inference with minimal impact on model quality. Combined with our existing memory optimization techniques (resizing and pruning), quantization can further reduce memory requirements for long context deployments.

## 2. Goals

- Implement KV cache quantization support for Q8_0, Q4_0, and Q4_1 formats
- Minimize accuracy loss while maximizing memory savings
- Provide easy-to-use API for controlling the quantization
- Integrate with existing memory optimization methods

## 3. Implementation Plan

### 3.1 API Design

Add the capability to quantize KV caches with a simple API:

```cpp
// In the Model class (model.h)
virtual void quantize_kv_cache(DataType quantization_type = DataType::Q8_0) = 0;

// In GGMLModel
void quantize_kv_cache(DataType quantization_type = DataType::Q8_0) override;
```

### 3.2 KVCache Class Extensions

Modify the KVCache class to support quantized storage:

```cpp
class KVCache {
public:
    // Existing methods...
    
    // New methods for quantization
    void quantize(DataType quantization_type);
    DataType current_quantization_type() const;
    
private:
    // Existing members...
    
    // New members for quantization
    DataType quantization_type_ = DataType::F32;
    std::vector<struct ggml_tensor*> k_caches_quantized_;
    std::vector<struct ggml_tensor*> v_caches_quantized_;
    
    // Helper methods for quantization
    void create_quantized_tensors(DataType quantization_type);
    void convert_to_quantized();
    void cleanup_original_tensors();
};
```

### 3.3 Integration with Memory Optimization

Modify the `GGMLModel::optimize_memory` method to use quantization when advantageous:

```cpp
void GGMLModel::optimize_memory(size_t max_memory_mb) {
    // Existing code...
    
    // Calculate current memory usage
    size_t backbone_memory = backbone_kv_cache_ ? backbone_kv_cache_->memory_usage() : 0;
    size_t decoder_memory = decoder_kv_cache_ ? decoder_kv_cache_->memory_usage() : 0;
    size_t total_memory = backbone_memory + decoder_memory;
    
    // If memory usage is already under the limit, nothing to optimize
    if (total_memory <= max_memory_bytes) {
        return;
    }
    
    // Calculate the reduction needed
    float reduction_factor = static_cast<float>(max_memory_bytes) / total_memory;
    
    // Choose strategy based on the reduction needed:
    if (reduction_factor < 0.25f) {
        // Need very aggressive reduction (>75% reduction)
        // Try Q4_0 quantization + pruning
        quantize_kv_cache(DataType::Q4_0);
        
        // Recalculate memory usage after quantization
        backbone_memory = backbone_kv_cache_ ? backbone_kv_cache_->memory_usage() : 0;
        decoder_memory = decoder_kv_cache_ ? decoder_kv_cache_->memory_usage() : 0;
        total_memory = backbone_memory + decoder_memory;
        
        // If still not enough, apply pruning
        if (total_memory > max_memory_bytes) {
            float prune_factor = 1.0f - (static_cast<float>(max_memory_bytes) / total_memory);
            prune_caches(prune_factor);
        }
    }
    else if (reduction_factor < 0.5f) {
        // Need significant reduction (50-75% reduction)
        // Try Q8_0 quantization
        quantize_kv_cache(DataType::Q8_0);
        
        // Recalculate memory usage after quantization
        backbone_memory = backbone_kv_cache_ ? backbone_kv_cache_->memory_usage() : 0;
        decoder_memory = decoder_kv_cache_ ? decoder_kv_cache_->memory_usage() : 0;
        total_memory = backbone_memory + decoder_memory;
        
        // If still not enough, apply resizing
        if (total_memory > max_memory_bytes) {
            // Apply resizing with new reduction factor
            float new_reduction_factor = static_cast<float>(max_memory_bytes) / total_memory;
            // ... resize logic ...
        }
    }
    else {
        // Moderate reduction (< 50% reduction)
        // Use standard resizing approach (current implementation)
        // ... existing resize logic ...
    }
}
```

### 3.4 Attention Calculation with Quantized KV Cache

Modify the attention calculation to handle quantized KV caches:

```cpp
// In GGMLModel methods that access KV cache
// Instead of directly using k_cache/v_cache tensors, add a layer of indirection:

struct ggml_tensor* get_k_tensor(int layer) {
    return backbone_kv_cache_->k_tensor(layer); // This method returns the appropriate tensor
}

struct ggml_tensor* get_v_tensor(int layer) {
    return backbone_kv_cache_->v_tensor(layer); // This method returns the appropriate tensor
}

// KVCache::k_tensor and v_tensor methods would return the quantized or original tensor as appropriate
struct ggml_tensor* KVCache::k_tensor(int layer) {
    if (quantization_type_ != DataType::F32 && layer < k_caches_quantized_.size()) {
        return k_caches_quantized_[layer];
    }
    return k_caches_[layer];
}
```

## 4. Memory Usage Estimation

Approximate memory usage for different quantization types:

| Format | Bits per Element | Overhead | Memory Reduction |
|--------|------------------|----------|------------------|
| F32    | 32 bits          | None     | 1x (baseline)    |
| Q8_0   | 8 bits + scale   | ~3%      | ~4x              |
| Q4_0   | 4 bits + scale   | ~6%      | ~7.5x            |
| Q4_1   | 4 bits + scale + bias | ~12% | ~6.8x           |

With combined techniques:
- Q8_0 + 50% pruning: ~8x reduction
- Q4_0 + 50% pruning: ~15x reduction

## 5. Testing Strategy

### 5.1 Unit Tests

- Test KV cache quantization with different formats
- Test accuracy impact on attention calculations
- Test memory usage before and after quantization
- Test performance impact of quantized operations

### 5.2 Integration Tests

- Test combined quantization and pruning
- Test memory optimization strategy selection
- Test extreme memory constraints

## 6. Implementation Timeline

1. **Phase 1**: Implement KVCache quantization support
   - Add quantization methods to KVCache class
   - Implement tensor conversion between formats
   - Add memory usage calculation for quantized tensors

2. **Phase 2**: Integrate with Model interface
   - Implement quantize_kv_cache in GGMLModel
   - Update attention calculation to support quantized tensors
   - Modify cache reset and handling

3. **Phase 3**: Enhance memory optimization
   - Update optimize_memory to use quantization
   - Implement strategy selection based on memory constraints
   - Optimize combined quantization and pruning

4. **Phase 4**: Testing and Optimization
   - Implement comprehensive test suite
   - Benchmark memory usage and performance
   - Optimize for minimal accuracy loss

## 7. Future Enhancements

- Implement on-the-fly quantization during inference
- Support for additional quantization formats (e.g., Q5_K, Q2_K)
- Adaptive quantization based on context importance
- Mixed-precision KV cache (different quantization for different layers)