#ifndef CCSM_MODEL_LOADER_H
#define CCSM_MODEL_LOADER_H

#include <ccsm/tensor.h>
#include <string>
#include <unordered_map>
#include <memory>
#include <vector>
#include <functional>

namespace ccsm {

// Forward declaration
class Context;

// Weight map to store model parameters
using WeightMap = std::unordered_map<std::string, Tensor>;

// Weight information for conversion
struct WeightInfo {
    std::string name;
    std::vector<size_t> shape;
    DataType dtype;
    size_t offset;
    size_t size;
};

// Model metadata
struct ModelMetadata {
    std::string name;
    std::string architecture;
    std::string version;
    std::unordered_map<std::string, std::string> params;
};

// Base class for all model loaders
class ModelLoader {
public:
    virtual ~ModelLoader() = default;
    
    // Load model weights
    virtual bool load(WeightMap& weights) = 0;
    
    // Get model metadata
    virtual ModelMetadata get_metadata() const = 0;
    
    // Check if a specific weight exists
    virtual bool has_weight(const std::string& name) const = 0;
    
    // Get a specific weight without loading everything
    virtual Tensor get_weight(const std::string& name, std::shared_ptr<Context> ctx) = 0;
    
    // Get all weight names
    virtual std::vector<std::string> get_weight_names() const = 0;
};

// PyTorch checkpoint loader
class PyTorchLoader : public ModelLoader {
public:
    PyTorchLoader(const std::string& path);
    ~PyTorchLoader();
    
    bool load(WeightMap& weights) override;
    ModelMetadata get_metadata() const override;
    bool has_weight(const std::string& name) const override;
    Tensor get_weight(const std::string& name, std::shared_ptr<Context> ctx) override;
    std::vector<std::string> get_weight_names() const override;
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// GGUF format loader
class GGUFLoader : public ModelLoader {
public:
    GGUFLoader(const std::string& path);
    ~GGUFLoader();
    
    bool load(WeightMap& weights) override;
    ModelMetadata get_metadata() const override;
    bool has_weight(const std::string& name) const override;
    Tensor get_weight(const std::string& name, std::shared_ptr<Context> ctx) override;
    std::vector<std::string> get_weight_names() const override;
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Factory for creating model loaders
class ModelLoaderFactory {
public:
    // Create a model loader based on file extension
    static std::shared_ptr<ModelLoader> create(const std::string& path);
    
    // Convert PyTorch checkpoint to GGUF
    static bool convert_pytorch_to_gguf(
        const std::string& src_path,
        const std::string& dst_path,
        std::function<void(float)> progress_callback = nullptr);
        
    // Load model with configuration
    static std::shared_ptr<Generator> load_model(
        const std::string& path,
        const ModelConfig& config);
};

// Model discovery and download
class ModelDiscovery {
public:
    // Find or download model
    static std::string find_or_download_model(
        const std::string& model_name,
        std::function<void(float)> progress_callback = nullptr);
    
    // Search for model in common paths
    static std::string find_model(const std::string& model_name);
    
    // Download model from Hugging Face Hub
    static std::string download_model(
        const std::string& model_name,
        const std::string& dest_path,
        std::function<void(float)> progress_callback = nullptr);
};

} // namespace ccsm

#endif // CCSM_MODEL_LOADER_H