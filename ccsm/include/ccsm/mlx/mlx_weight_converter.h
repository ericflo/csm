#ifndef CCSM_MLX_WEIGHT_CONVERTER_H
#define CCSM_MLX_WEIGHT_CONVERTER_H

#include <ccsm/tensor.h>
#include <ccsm/model_loader.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <functional>

#ifdef CCSM_WITH_MLX
#include "mlx/c/array.h"
#include "mlx/c/device.h"
#include "mlx/c/stream.h"
#endif

namespace ccsm {

// Configuration for PyTorch to MLX weight conversion
struct MLXWeightConversionConfig {
    // Whether to use BFloat16 precision instead of Float32
    bool use_bfloat16 = true;
    
    // Whether to use quantization when available
    bool use_quantization = false;
    
    // Whether to cache the converted weights
    bool cache_converted_weights = true;
    
    // Custom parameter mapping for different model types
    std::unordered_map<std::string, std::string> parameter_mapping;
    
    // Progress callback
    std::function<void(float)> progress_callback = nullptr;
};

#ifdef CCSM_WITH_MLX
// Convert PyTorch weights to MLX arrays
std::unordered_map<std::string, mlx_array> convert_pytorch_to_mlx(
    const std::string& pytorch_model_path,
    const MLXWeightConversionConfig& config = MLXWeightConversionConfig());

// Convert PyTorch weights to MLX arrays from a weight map
std::unordered_map<std::string, mlx_array> convert_pytorch_to_mlx(
    const WeightMap& weights,
    const MLXWeightConversionConfig& config = MLXWeightConversionConfig());

// Convert a single PyTorch tensor to MLX array
mlx_array convert_tensor_to_mlx_array(
    const Tensor& tensor,
    bool use_bfloat16 = true);

// Save the converted weights to a cache file
bool save_mlx_weights_to_cache(
    const std::string& cache_path,
    const std::unordered_map<std::string, mlx_array>& weights);

// Load the converted weights from a cache file
std::unordered_map<std::string, mlx_array> load_mlx_weights_from_cache(
    const std::string& cache_path);

// Check if cached MLX weights exist for a PyTorch model
bool has_cached_mlx_weights(const std::string& pytorch_model_path);

// Get the path to cached MLX weights for a PyTorch model
std::string get_cached_mlx_weights_path(const std::string& pytorch_model_path);
#endif

// High-level utilities for model conversion (usable even without MLX compiled in)
class MLXWeightConverter {
public:
    // Initialize the weight converter
    MLXWeightConverter(const MLXWeightConversionConfig& config = MLXWeightConversionConfig());
    
    // Convert PyTorch checkpoint to MLX model
    bool convert_checkpoint(
        const std::string& pytorch_path,
        const std::string& mlx_output_path);
    
    // Check if MLX is available and usable on this system
    static bool is_mlx_available();
    
private:
    MLXWeightConversionConfig config_;
};

} // namespace ccsm

#endif // CCSM_MLX_WEIGHT_CONVERTER_H