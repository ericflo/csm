#include <ccsm/model.h>
#include <ccsm/utils.h>
#include <unordered_map>
#include <stdexcept>

namespace ccsm {

// Model implementation
Model::Model(const ModelConfig& config) : config_(config) {
    CCSM_INFO("Creating model with config: ", config_.name);
}

const ModelConfig& Model::config() const {
    return config_;
}

// ModelFactory implementation
std::shared_ptr<Model> ModelFactory::create(const std::string& backend, const ModelConfig& config) {
    if (!is_backend_available(backend)) {
        throw std::runtime_error("Backend '" + backend + "' is not available");
    }
    
    // TODO: Create specific model implementations
    throw std::runtime_error("ModelFactory::create not implemented yet for backend: " + backend);
}

bool ModelFactory::is_backend_available(const std::string& backend) {
    if (backend == "cpu" || backend == "ggml") {
        // CPU backend is always available
        return true;
    }
    
#ifdef CCSM_WITH_MLX
    if (backend == "mlx") {
        // TODO: Check MLX runtime availability
        return true;
    }
#endif

#ifdef CCSM_WITH_CUDA
    if (backend == "cuda") {
        // TODO: Check CUDA runtime availability
        return false;
    }
#endif

#ifdef CCSM_WITH_VULKAN
    if (backend == "vulkan") {
        // TODO: Check Vulkan runtime availability
        return false;
    }
#endif

    return false;
}

std::vector<std::string> ModelFactory::get_available_backends() {
    std::vector<std::string> backends;
    
    // CPU is always available
    backends.push_back("cpu");
    
#ifdef CCSM_WITH_MLX
    // Check MLX availability at runtime
    // TODO: Implement actual check
    backends.push_back("mlx");
#endif

#ifdef CCSM_WITH_CUDA
    // Check CUDA availability at runtime
    // TODO: Implement actual check
    // backends.push_back("cuda");
#endif

#ifdef CCSM_WITH_VULKAN
    // Check Vulkan availability at runtime
    // TODO: Implement actual check
    // backends.push_back("vulkan");
#endif

    return backends;
}

} // namespace ccsm