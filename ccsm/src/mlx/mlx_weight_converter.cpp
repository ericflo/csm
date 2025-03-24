#include <ccsm/mlx/mlx_weight_converter.h>
#include <ccsm/mlx/mlx_tensor.h>
#include <ccsm/utils.h>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <stdexcept>

namespace ccsm {

#ifdef CCSM_WITH_MLX

// Helper function to generate a hash for caching
std::string generate_model_hash(const std::string& model_path) {
    // Simple hash to identify the model file
    // In a production system, this would be a proper hash of the file content
    std::string filename = std::filesystem::path(model_path).filename().string();
    std::string hash = filename;
    
    // Get file size and modification time for more unique hash
    try {
        auto file_info = std::filesystem::status(model_path);
        auto file_size = std::filesystem::file_size(model_path);
        auto mod_time = std::filesystem::last_write_time(model_path);
        
        // Convert to string representation
        std::stringstream ss;
        ss << filename << "_" << file_size << "_";
        ss << std::hex << std::setw(16) << std::setfill('0');
        ss << static_cast<uint64_t>(mod_time.time_since_epoch().count());
        hash = ss.str();
    } catch (const std::exception& e) {
        CCSM_INFO("Error generating model hash: ");
        CCSM_INFO(e.what());
    }
    
    return hash;
}

// Convert a CCSM tensor to MLX array
mlx_array convert_tensor_to_mlx_array(const Tensor& tensor, bool use_bfloat16) {
    CCSM_INFO("Converting tensor to MLX array, use_bfloat16=");
    CCSM_INFO(use_bfloat16 ? "true" : "false");
    
    // If this is already an MLX tensor, extract the array directly
    if (auto mlx_impl = std::dynamic_pointer_cast<MLXTensorImpl>(tensor.impl())) {
        CCSM_INFO("Input is already an MLX tensor, extracting array");
        mlx_array result = mlx_impl->mlx_array_handle();
        
        // Convert to bfloat16 if requested and not already in that format
        if (use_bfloat16 && tensor.dtype() != DataType::BF16) {
            CCSM_INFO("Converting to BFloat16");
            mlx_array converted;
            // In a real implementation, we would call mlx_array_astype
            // For now, just return the original array since this is a stub
            // mlx_array_astype(result, MLX_BFLOAT16, &converted);
            return result;
        }
        
        return result;
    }
    
    // Otherwise, need to convert from raw data
    CCSM_INFO("Converting from raw tensor data");
    
    // Prepare shape for MLX
    std::vector<int64_t> mlx_shape;
    auto shape = tensor.shape();
    mlx_shape.reserve(shape.size());
    for (auto dim : shape) {
        mlx_shape.push_back(static_cast<int64_t>(dim));
    }
    
    // Determine target data type
    DataType target_dtype = tensor.dtype();
    mlx_dtype mlx_type;
    
    // Apply bfloat16 conversion if requested
    if (use_bfloat16 && (target_dtype == DataType::F32 || target_dtype == DataType::F16)) {
        mlx_type = MLX_BFLOAT16;
    } else {
        mlx_type = MLXTensorImpl::to_mlx_dtype(target_dtype);
    }
    
    // Create MLX array from raw data
    mlx_array result;
    if (tensor.data() == nullptr) {
        CCSM_INFO("Tensor data is null, creating empty array");
        // In a real implementation, we would call mlx_array_zeros
        // For now, return an empty array since this is a stub
        // mlx_array_zeros(mlx_shape.data(), mlx_shape.size(), mlx_type, &result);
        return result;
    }
    
    // Create array from data
    // In a real implementation, we would call mlx_array_from_data
    // For now, return an empty array since this is a stub
    // mlx_array_from_data(tensor.data(), mlx_shape.data(), mlx_shape.size(), 
    //                    MLXTensorImpl::to_mlx_dtype(tensor.dtype()), &result);
    
    // Convert to bfloat16 if needed
    if (use_bfloat16 && tensor.dtype() != DataType::BF16) {
        CCSM_INFO("Converting to BFloat16");
        // In a real implementation, we would call mlx_array_astype
        // mlx_array converted;
        // mlx_array_astype(result, MLX_BFLOAT16, &converted);
        // Free the original array
        // mlx_array_free(result);
        // return converted;
    }
    
    return result;
}

// Convert PyTorch weights from a file path
std::unordered_map<std::string, mlx_array> convert_pytorch_to_mlx(
    const std::string& pytorch_model_path,
    const MLXWeightConversionConfig& config) {
    
    CCSM_INFO("Converting PyTorch model to MLX: ");
    CCSM_INFO(pytorch_model_path);
    
    // Check if cached weights exist and should be used
    if (config.cache_converted_weights && has_cached_mlx_weights(pytorch_model_path)) {
        CCSM_INFO("Using cached MLX weights");
        return load_mlx_weights_from_cache(get_cached_mlx_weights_path(pytorch_model_path));
    }
    
    // Create PyTorch loader
    auto loader = std::make_shared<PyTorchLoader>(pytorch_model_path);
    if (!loader) {
        throw std::runtime_error("Failed to create PyTorch loader for " + pytorch_model_path);
    }
    
    // Load weights
    WeightMap weights;
    if (!loader->load(weights)) {
        throw std::runtime_error("Failed to load weights from " + pytorch_model_path);
    }
    
    // Convert weights
    auto mlx_weights = convert_pytorch_to_mlx(weights, config);
    
    // Cache the weights if requested
    if (config.cache_converted_weights) {
        CCSM_INFO("Caching converted MLX weights");
        std::string cache_path = get_cached_mlx_weights_path(pytorch_model_path);
        save_mlx_weights_to_cache(cache_path, mlx_weights);
    }
    
    return mlx_weights;
}

// Convert PyTorch weights from a weight map
std::unordered_map<std::string, mlx_array> convert_pytorch_to_mlx(
    const WeightMap& weights,
    const MLXWeightConversionConfig& config) {
    
    CCSM_INFO("Converting weight map to MLX arrays");
    std::unordered_map<std::string, mlx_array> mlx_weights;
    
    // Total number of weights for progress tracking
    size_t total_weights = weights.size();
    size_t processed = 0;
    
    // Process each weight
    for (const auto& [name, tensor] : weights) {
        CCSM_INFO("Converting weight: ");
        CCSM_INFO(name);
        
        // Check if parameter should be remapped based on config
        std::string target_name = name;
        if (config.parameter_mapping.count(name) > 0) {
            target_name = config.parameter_mapping.at(name);
            CCSM_INFO("Remapped parameter name: ");
            CCSM_INFO(name);
            CCSM_INFO(" -> ");
            CCSM_INFO(target_name);
        }
        
        // Convert tensor to MLX array
        mlx_weights[target_name] = convert_tensor_to_mlx_array(tensor, config.use_bfloat16);
        
        // Update progress if callback provided
        processed++;
        if (config.progress_callback) {
            float progress = static_cast<float>(processed) / total_weights;
            config.progress_callback(progress);
        }
    }
    
    CCSM_INFO("Converted weights to MLX arrays: ");
    CCSM_INFO(std::to_string(processed));
    return mlx_weights;
}

// Save converted weights to cache
bool save_mlx_weights_to_cache(
    const std::string& cache_path,
    const std::unordered_map<std::string, mlx_array>& weights) {
    
    CCSM_INFO("Saving MLX weights to cache: ");
    CCSM_INFO(cache_path);
    
    // Create directory if it doesn't exist
    std::filesystem::path path(cache_path);
    std::filesystem::create_directories(path.parent_path());
    
    // Open file for writing
    std::ofstream file(cache_path, std::ios::binary);
    if (!file) {
        CCSM_INFO("Failed to open cache file for writing: ");
        CCSM_INFO(cache_path);
        return false;
    }
    
    // Write number of weights
    uint64_t num_weights = weights.size();
    file.write(reinterpret_cast<const char*>(&num_weights), sizeof(num_weights));
    
    // Write each weight
    for (const auto& [name, array] : weights) {
        // Write name length and name
        uint64_t name_length = name.size();
        file.write(reinterpret_cast<const char*>(&name_length), sizeof(name_length));
        file.write(name.c_str(), name_length);
        
        // In a real implementation, we would get array information
        // and write that information to the cache file
        
        // Stub implementation
        size_t ndim = 0;
        file.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
    }
    
    return true;
}

// Load converted weights from cache
std::unordered_map<std::string, mlx_array> load_mlx_weights_from_cache(
    const std::string& cache_path) {
    
    CCSM_INFO("Loading MLX weights from cache: ");
    CCSM_INFO(cache_path);
    std::unordered_map<std::string, mlx_array> weights;
    
    // Open file for reading
    std::ifstream file(cache_path, std::ios::binary);
    if (!file) {
        CCSM_INFO("Failed to open cache file for reading: ");
        CCSM_INFO(cache_path);
        return weights;
    }
    
    // Read number of weights
    uint64_t num_weights;
    file.read(reinterpret_cast<char*>(&num_weights), sizeof(num_weights));
    
    // Read each weight - stub implementation
    for (uint64_t i = 0; i < num_weights && file.good(); i++) {
        // Read name
        uint64_t name_length;
        file.read(reinterpret_cast<char*>(&name_length), sizeof(name_length));
        std::string name(name_length, '\0');
        file.read(&name[0], name_length);
        
        // Read ndim
        size_t ndim;
        file.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
        
        // In a real implementation, we would read shape, dtype, and data
        // and create a real MLX array
        
        // For the stub implementation, we just add an empty array to the map
        weights[name] = mlx_array{};
    }
    
    CCSM_INFO("Loaded weights from cache: ");
    CCSM_INFO(std::to_string(weights.size()));
    return weights;
}

// Check if cached MLX weights exist for a PyTorch model
bool has_cached_mlx_weights(const std::string& pytorch_model_path) {
    std::string cache_path = get_cached_mlx_weights_path(pytorch_model_path);
    return std::filesystem::exists(cache_path);
}

// Get the path to cached MLX weights for a PyTorch model
std::string get_cached_mlx_weights_path(const std::string& pytorch_model_path) {
    // Create cache directory in user's home directory
    std::string home_dir;
    
    #ifdef _WIN32
    home_dir = std::getenv("USERPROFILE");
    #else
    home_dir = std::getenv("HOME");
    #endif
    
    if (home_dir.empty()) {
        home_dir = "."; // Use current directory as fallback
    }
    
    std::string cache_dir = home_dir + "/.cache/ccsm/mlx_weights";
    
    // Create hash from model path
    std::string model_hash = generate_model_hash(pytorch_model_path);
    
    // Return full cache path
    return cache_dir + "/" + model_hash + ".mlxcache";
}

#endif // CCSM_WITH_MLX

// High-level utilities for model conversion
MLXWeightConverter::MLXWeightConverter(const MLXWeightConversionConfig& config)
    : config_(config) {
    CCSM_INFO("Created MLXWeightConverter with config:");
    CCSM_INFO("  use_bfloat16: ");
    CCSM_INFO(config.use_bfloat16 ? "true" : "false");
    CCSM_INFO("  use_quantization: ");
    CCSM_INFO(config.use_quantization ? "true" : "false");
    CCSM_INFO("  cache_converted_weights: ");
    CCSM_INFO(config.cache_converted_weights ? "true" : "false");
}

bool MLXWeightConverter::convert_checkpoint(
    const std::string& pytorch_path,
    const std::string& mlx_output_path) {
    
    CCSM_INFO("Converting PyTorch checkpoint to MLX: ");
    CCSM_INFO(pytorch_path);
    CCSM_INFO(" -> ");
    CCSM_INFO(mlx_output_path);
    
    #ifdef CCSM_WITH_MLX
    try {
        // Convert weights
        auto mlx_weights = convert_pytorch_to_mlx(pytorch_path, config_);
        
        // Save to output path
        return save_mlx_weights_to_cache(mlx_output_path, mlx_weights);
    } catch (const std::exception& e) {
        CCSM_INFO("Error converting checkpoint: ");
        CCSM_INFO(e.what());
        return false;
    }
    #else
    CCSM_INFO("MLX support not compiled in, cannot convert checkpoint");
    return false;
    #endif
}

bool MLXWeightConverter::is_mlx_available() {
    #ifdef CCSM_WITH_MLX
    return MLXDevice::is_available();
    #else
    return false;
    #endif
}

} // namespace ccsm