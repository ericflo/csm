#include <ccsm/model_loader.h>
#include <ccsm/tensor.h>
#include <ccsm/utils.h>
#include <ccsm/cpu/ggml_tensor.h>
#include <fstream>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <cstring>

#include "ggml.h"
#include "ggml-alloc.h"
#include "gguf.h"

namespace ccsm {

// Implementation for GGUFLoader
struct GGUFLoader::Impl {
    std::string path;
    struct gguf_context* ctx;
    struct ggml_context* ggml_ctx;
    std::vector<std::string> weight_names;
    ModelMetadata metadata;
    bool loaded;
    
    Impl(const std::string& file_path) 
        : path(file_path), ctx(nullptr), ggml_ctx(nullptr), loaded(false) {
        
        if (!FileUtils::file_exists(path)) {
            throw std::runtime_error("File does not exist: " + path);
        }
        
        // Initialize GGUF context
        ctx = gguf_init_from_file(path.c_str(), {});
        if (!ctx) {
            throw std::runtime_error("Failed to initialize GGUF context from: " + path);
        }
        
        // Load metadata
        metadata.name = gguf_get_name(ctx);
        
        // Try to get architecture from metadata
        int32_t arch_key_idx = gguf_find_key(ctx, "general.architecture");
        if (arch_key_idx >= 0) {
            enum gguf_type type = gguf_get_key_type(ctx, arch_key_idx);
            if (type == GGUF_TYPE_STRING) {
                metadata.architecture = gguf_get_val_str(ctx, arch_key_idx);
            }
        }
        
        // Try to get version from metadata
        int32_t version_key_idx = gguf_find_key(ctx, "general.version");
        if (version_key_idx >= 0) {
            enum gguf_type type = gguf_get_key_type(ctx, version_key_idx);
            if (type == GGUF_TYPE_STRING) {
                metadata.version = gguf_get_val_str(ctx, version_key_idx);
            }
        }
        
        // Get all tensor names
        size_t tensor_count = gguf_get_n_tensors(ctx);
        for (size_t i = 0; i < tensor_count; i++) {
            weight_names.push_back(gguf_get_tensor_name(ctx, i));
        }
        
        // Create GGML context for tensor operations
        struct ggml_init_params params = {
            .mem_size   = 16 * 1024 * 1024, // 16 MB for metadata
            .mem_buffer = NULL,
            .no_alloc   = false,
        };
        
        ggml_ctx = ggml_init(params);
        if (!ggml_ctx) {
            gguf_free(ctx);
            throw std::runtime_error("Failed to initialize GGML context");
        }
    }
    
    ~Impl() {
        if (ggml_ctx) {
            ggml_free(ggml_ctx);
        }
        if (ctx) {
            gguf_free(ctx);
        }
    }
    
    bool load_weights(WeightMap& weights) {
        // Create a new context with enough memory for all tensors
        size_t total_size = 0;
        for (size_t i = 0; i < gguf_get_n_tensors(ctx); i++) {
            const char* name = gguf_get_tensor_name(ctx, i);
            struct gguf_tensor_info info = gguf_get_tensor_info(ctx, i);
            
            size_t tensor_size = ggml_nbytes_pad(info.dtype, info.n_elements);
            total_size += tensor_size;
        }
        
        // Add extra memory for tensor operations
        total_size += 16 * 1024 * 1024; // 16 MB extra
        
        // Free previous context if it exists
        if (ggml_ctx) {
            ggml_free(ggml_ctx);
        }
        
        // Create new context
        struct ggml_init_params params = {
            .mem_size   = total_size,
            .mem_buffer = NULL,
            .no_alloc   = false,
        };
        
        ggml_ctx = ggml_init(params);
        if (!ggml_ctx) {
            throw std::runtime_error("Failed to initialize GGML context for weights");
        }
        
        // Load tensors into context
        for (size_t i = 0; i < gguf_get_n_tensors(ctx); i++) {
            const char* name = gguf_get_tensor_name(ctx, i);
            struct gguf_tensor_info info = gguf_get_tensor_info(ctx, i);
            
            // Convert dimensions to int64_t array
            int64_t ne[GGML_MAX_DIMS];
            for (uint32_t j = 0; j < info.n_dims; j++) {
                ne[j] = info.ne[j];
            }
            
            // Create tensor
            struct ggml_tensor* tensor = ggml_new_tensor(ggml_ctx, info.dtype, info.n_dims, ne);
            if (!tensor) {
                throw std::runtime_error("Failed to create tensor: " + std::string(name));
            }
            
            // Load data
            gguf_get_tensor_data(ctx, i, tensor->data, ggml_nbytes(tensor));
            
            // Add to weight map
            weights[name] = Tensor(std::make_shared<GGMLTensorImpl>(tensor, false));
        }
        
        loaded = true;
        return true;
    }
    
    Tensor get_tensor(const std::string& name, std::shared_ptr<Context> context) {
        // Find tensor index
        int32_t tensor_idx = -1;
        for (size_t i = 0; i < gguf_get_n_tensors(ctx); i++) {
            if (name == gguf_get_tensor_name(ctx, i)) {
                tensor_idx = i;
                break;
            }
        }
        
        if (tensor_idx < 0) {
            throw std::runtime_error("Tensor not found: " + name);
        }
        
        // Get tensor info
        struct gguf_tensor_info info = gguf_get_tensor_info(ctx, tensor_idx);
        
        // Create tensor in the provided context
        std::vector<size_t> shape;
        for (uint32_t i = 0; i < info.n_dims; i++) {
            shape.push_back(info.ne[i]);
        }
        
        // Convert GGML type to our DataType
        DataType dtype = GGMLTensorImpl::from_ggml_type(info.dtype);
        
        // Create tensor
        Tensor tensor = context->zeros(shape, dtype);
        
        // Load data
        gguf_get_tensor_data(ctx, tensor_idx, tensor.data(), ggml_nbytes_from_shape(info.n_dims, info.ne, info.dtype));
        
        return tensor;
    }
};

GGUFLoader::GGUFLoader(const std::string& path) : impl_(std::make_unique<Impl>(path)) {}

GGUFLoader::~GGUFLoader() = default;

bool GGUFLoader::load(WeightMap& weights) {
    return impl_->load_weights(weights);
}

ModelMetadata GGUFLoader::get_metadata() const {
    return impl_->metadata;
}

bool GGUFLoader::has_weight(const std::string& name) const {
    return std::find(impl_->weight_names.begin(), impl_->weight_names.end(), name) != impl_->weight_names.end();
}

Tensor GGUFLoader::get_weight(const std::string& name, std::shared_ptr<Context> ctx) {
    return impl_->get_tensor(name, ctx);
}

std::vector<std::string> GGUFLoader::get_weight_names() const {
    return impl_->weight_names;
}

// Implementation for PyTorchLoader
struct PyTorchLoader::Impl {
    std::string path;
    bool loaded;
    std::vector<std::string> weight_names;
    ModelMetadata metadata;
    
    Impl(const std::string& file_path) : path(file_path), loaded(false) {
        // TODO: Implement PyTorch checkpoint loading
        throw std::runtime_error("PyTorch loader not implemented yet");
    }
    
    bool load_weights(WeightMap& weights) {
        // TODO: Implement weight loading
        return false;
    }
    
    Tensor get_tensor(const std::string& name, std::shared_ptr<Context> context) {
        // TODO: Implement tensor loading
        throw std::runtime_error("PyTorch loader not implemented yet");
    }
};

PyTorchLoader::PyTorchLoader(const std::string& path) : impl_(std::make_unique<Impl>(path)) {}

PyTorchLoader::~PyTorchLoader() = default;

bool PyTorchLoader::load(WeightMap& weights) {
    return impl_->load_weights(weights);
}

ModelMetadata PyTorchLoader::get_metadata() const {
    return impl_->metadata;
}

bool PyTorchLoader::has_weight(const std::string& name) const {
    return std::find(impl_->weight_names.begin(), impl_->weight_names.end(), name) != impl_->weight_names.end();
}

Tensor PyTorchLoader::get_weight(const std::string& name, std::shared_ptr<Context> ctx) {
    return impl_->get_tensor(name, ctx);
}

std::vector<std::string> PyTorchLoader::get_weight_names() const {
    return impl_->weight_names;
}

// ModelLoaderFactory implementation
std::shared_ptr<ModelLoader> ModelLoaderFactory::create(const std::string& path) {
    if (!FileUtils::file_exists(path)) {
        throw std::runtime_error("File does not exist: " + path);
    }
    
    // Check file extension
    std::filesystem::path fs_path(path);
    std::string extension = fs_path.extension().string();
    
    if (extension == ".pt" || extension == ".pth") {
        return std::make_shared<PyTorchLoader>(path);
    } else if (extension == ".gguf") {
        return std::make_shared<GGUFLoader>(path);
    } else {
        // Try to detect format based on file contents
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Failed to open file: " + path);
        }
        
        // Read first few bytes
        char header[8];
        file.read(header, sizeof(header));
        
        // Check for GGUF magic
        if (strncmp(header, "GGUF", 4) == 0) {
            return std::make_shared<GGUFLoader>(path);
        }
        
        // TODO: Add detection for PyTorch format
        
        throw std::runtime_error("Unsupported model format: " + path);
    }
}

bool ModelLoaderFactory::convert_pytorch_to_gguf(
    const std::string& src_path,
    const std::string& dst_path,
    std::function<void(float)> progress_callback) {
    
    // TODO: Implement conversion
    throw std::runtime_error("PyTorch to GGUF conversion not implemented yet");
}

// ModelDiscovery implementation
std::string ModelDiscovery::find_or_download_model(
    const std::string& model_name,
    std::function<void(float)> progress_callback) {
    
    // First try to find the model locally
    std::string local_path = find_model(model_name);
    if (!local_path.empty()) {
        return local_path;
    }
    
    // Model not found locally, try to download
    std::string home_dir;
    
#ifdef _WIN32
    home_dir = std::getenv("USERPROFILE");
#else
    home_dir = std::getenv("HOME");
#endif
    
    if (home_dir.empty()) {
        throw std::runtime_error("Failed to get home directory");
    }
    
    // Create models directory if it doesn't exist
    std::filesystem::path models_dir = std::filesystem::path(home_dir) / ".ccsm" / "models";
    if (!std::filesystem::exists(models_dir)) {
        std::filesystem::create_directories(models_dir);
    }
    
    std::string dest_path = (models_dir / model_name).string();
    
    // Download the model
    return download_model(model_name, dest_path, progress_callback);
}

std::string ModelDiscovery::find_model(const std::string& model_name) {
    // Define search paths
    std::vector<std::string> search_paths;
    
    // Add current directory
    search_paths.push_back(".");
    
    // Add home directory
    std::string home_dir;
#ifdef _WIN32
    home_dir = std::getenv("USERPROFILE");
#else
    home_dir = std::getenv("HOME");
#endif
    
    if (!home_dir.empty()) {
        search_paths.push_back(std::filesystem::path(home_dir) / ".ccsm" / "models");
    }
    
    // Add system-wide directories
#ifdef _WIN32
    search_paths.push_back("C:\\Program Files\\CCSM\\models");
#else
    search_paths.push_back("/usr/local/share/ccsm/models");
    search_paths.push_back("/usr/share/ccsm/models");
#endif
    
    // Search for model
    for (const auto& path : search_paths) {
        // Search for exact match
        std::filesystem::path full_path = std::filesystem::path(path) / model_name;
        if (FileUtils::file_exists(full_path.string())) {
            return full_path.string();
        }
        
        // Search for model with extensions
        for (const auto& ext : {".gguf", ".bin", ".pt", ".pth"}) {
            std::string path_with_ext = full_path.string() + ext;
            if (FileUtils::file_exists(path_with_ext)) {
                return path_with_ext;
            }
        }
        
        // Check if directory exists with model name
        if (std::filesystem::exists(full_path) && std::filesystem::is_directory(full_path)) {
            // Search for common model filenames
            for (const auto& filename : {"model.gguf", "model.bin", "model.pt", "pytorch_model.bin"}) {
                std::string model_path = (full_path / filename).string();
                if (FileUtils::file_exists(model_path)) {
                    return model_path;
                }
            }
        }
    }
    
    // Model not found
    return "";
}

std::string ModelDiscovery::download_model(
    const std::string& model_name,
    const std::string& dest_path,
    std::function<void(float)> progress_callback) {
    
    // TODO: Implement model downloading from Hugging Face Hub
    throw std::runtime_error("Model downloading not implemented yet: " + model_name);
}

} // namespace ccsm