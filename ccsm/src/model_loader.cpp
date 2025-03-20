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
        
        // Set a default model name
        metadata.name = "CSM Model";
        
        // Try to get architecture from metadata
        int64_t arch_key_idx = gguf_find_key(ctx, "general.architecture");
        if (arch_key_idx >= 0) {
            enum gguf_type type = gguf_get_kv_type(ctx, arch_key_idx);
            if (type == GGUF_TYPE_STRING) {
                metadata.architecture = gguf_get_val_str(ctx, arch_key_idx);
            }
        }
        
        // Try to get version from metadata
        int64_t version_key_idx = gguf_find_key(ctx, "general.version");
        if (version_key_idx >= 0) {
            enum gguf_type type = gguf_get_kv_type(ctx, version_key_idx);
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
        int64_t n_tensors = gguf_get_n_tensors(ctx);
        
        for (int64_t i = 0; i < n_tensors; i++) {
            const char* name = gguf_get_tensor_name(ctx, i);
            enum ggml_type dtype = gguf_get_tensor_type(ctx, i);
            size_t tensor_size = gguf_get_tensor_size(ctx, i);
            
            total_size += tensor_size + 128; // Add some padding
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
        for (int64_t i = 0; i < n_tensors; i++) {
            const char* name = gguf_get_tensor_name(ctx, i);
            enum ggml_type dtype = gguf_get_tensor_type(ctx, i);
            size_t offset = gguf_get_tensor_offset(ctx, i);
            
            // Get tensor dimensions from original tensor
            // We'll need to query this indirectly
            int n_dims = 0;
            int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
            
            // For now we only support up to 2D tensors
            size_t tensor_size = gguf_get_tensor_size(ctx, i);
            
            // Simple heuristic to guess dimensions - this should be improved
            // In a real implementation, we would need to read this from file metadata
            // or reconstruct from the actual GGUF format
            if (dtype == GGML_TYPE_F32) {
                ne[0] = tensor_size / sizeof(float);
                n_dims = 1;
            } else {
                // Just use a 1D tensor as fallback
                ne[0] = tensor_size;
                n_dims = 1;
            }
            
            // Create tensor
            struct ggml_tensor* tensor = ggml_new_tensor(ggml_ctx, dtype, n_dims, ne);
            if (!tensor) {
                throw std::runtime_error("Failed to create tensor: " + std::string(name));
            }
            
            // Read data directly from file
            // Calculate byte offset in the file
            size_t data_offset = gguf_get_data_offset(ctx) + offset;
            
            // Open the file directly for reading tensor data
            FILE* f = fopen(path.c_str(), "rb");
            if (!f) {
                throw std::runtime_error("Failed to open file for reading tensor data: " + path);
            }
            
            // Seek to the tensor data position
            fseek(f, data_offset, SEEK_SET);
            
            // Read data directly into tensor
            size_t bytes_read = fread(tensor->data, 1, ggml_nbytes(tensor), f);
            fclose(f);
            
            if (bytes_read != ggml_nbytes(tensor)) {
                throw std::runtime_error("Failed to read tensor data for: " + std::string(name));
            }
            
            // Add to weight map
            weights[name] = Tensor(std::make_shared<GGMLTensorImpl>(tensor, false));
        }
        
        loaded = true;
        return true;
    }
    
    Tensor get_tensor(const std::string& name, std::shared_ptr<Context> context) {
        // Find tensor index
        int64_t tensor_idx = gguf_find_tensor(ctx, name.c_str());
        
        if (tensor_idx < 0) {
            throw std::runtime_error("Tensor not found: " + name);
        }
        
        // Get tensor info - we need to reconstruct this since gguf_tensor_info is no longer available
        enum ggml_type dtype = gguf_get_tensor_type(ctx, tensor_idx);
        size_t tensor_size = gguf_get_tensor_size(ctx, tensor_idx);
        
        // Create tensor in the provided context
        // For simplicity, we'll treat it as a 1D tensor for now
        std::vector<size_t> shape;
        shape.push_back(tensor_size / ggml_type_size(dtype));
        
        // Convert GGML type to our DataType
        DataType our_dtype = GGMLTensorImpl::from_ggml_type(dtype);
        
        // Create tensor
        Tensor tensor = context->zeros(shape, our_dtype);
        
        // Load data directly from file
        size_t offset = gguf_get_tensor_offset(ctx, tensor_idx);
        size_t data_offset = gguf_get_data_offset(ctx) + offset;
        
        // Open the file directly for reading tensor data
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) {
            throw std::runtime_error("Failed to open file for reading tensor data: " + path);
        }
        
        // Seek to the tensor data position
        fseek(f, data_offset, SEEK_SET);
        
        // Read data directly into tensor
        size_t bytes_read = fread(tensor.data(), 1, tensor_size, f);
        fclose(f);
        
        if (bytes_read != tensor_size) {
            throw std::runtime_error("Failed to read tensor data for: " + name);
        }
        
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