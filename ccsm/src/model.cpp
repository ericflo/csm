#include <ccsm/model.h>
#include <ccsm/utils.h>
#include <ccsm/cpu/ggml_model.h>

// Include the real MLX model if available
#ifdef CCSM_WITH_MLX
#include <ccsm/mlx/mlx_model.h>
#else
// Stub implementation for MLXModel to avoid link errors
namespace ccsm {
    class MLXModel : public Model {
    public:
        MLXModel(const ModelConfig& config) : Model(config) {
            throw std::runtime_error("MLXModel stub - not implemented");
        }
        
        bool load_weights(const std::string& path) override {
            return false;
        }
        
        bool load_weights(std::shared_ptr<ModelLoader> loader) override {
            return false;
        }
        
        bool load_weights(const WeightMap& weights) override {
            return false;
        }
        
        std::vector<int> generate_frame(
            const std::vector<int>& tokens,
            const std::vector<int>& positions,
            float temperature = 0.9f,
            int top_k = 50) override {
            return {};
        }
        
        void reset_caches() override {}
        
        void optimize_memory(size_t max_memory_mb = 0) override {}
        
        void prune_caches(float prune_factor = 0.5f) override {}
        
        std::vector<float> get_backbone_logits(
            const std::vector<int>& tokens,
            const std::vector<int>& positions) override {
            return {};
        }
        
        std::vector<float> get_decoder_logits(
            const std::vector<int>& tokens,
            const std::vector<int>& positions,
            int codebook) override {
            return {};
        }
    };
}
#endif

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
    
    if (backend == "cpu" || backend == "ggml") {
        // Create CPU/GGML backend model
        return std::make_shared<GGMLModel>(config);
    }
    else if (backend == "mlx") {
        // Always fall back to CPU for MLX until implementation is complete
        CCSM_WARNING("MLX backend requested but implementation is incomplete, using CPU/GGML backend");
        return std::make_shared<GGMLModel>(config);
    }
    
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