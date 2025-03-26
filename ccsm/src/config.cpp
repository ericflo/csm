#include <ccsm/config.h>
#include <ccsm/utils.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <optional>
#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>

// Simple JSON parser and serializer
#include <nlohmann/json.hpp>

namespace ccsm {

//=============================================================================
// ConfigValue implementation
//=============================================================================

std::string ConfigValue::as_string() const {
    if (is_string()) {
        return std::any_cast<std::string>(value_);
    } else if (is_int()) {
        return std::to_string(std::any_cast<int>(value_));
    } else if (is_float()) {
        return std::to_string(std::any_cast<float>(value_));
    } else if (is_bool()) {
        return std::any_cast<bool>(value_) ? "true" : "false";
    } 
    // For arrays or unsupported types
    return "";
}

int ConfigValue::as_int() const {
    if (is_int()) {
        return std::any_cast<int>(value_);
    } else if (is_float()) {
        return static_cast<int>(std::any_cast<float>(value_));
    } else if (is_string()) {
        try {
            return std::stoi(std::any_cast<std::string>(value_));
        } catch (const std::exception& e) {
            return 0;
        }
    } else if (is_bool()) {
        return std::any_cast<bool>(value_) ? 1 : 0;
    }
    return 0;
}

float ConfigValue::as_float() const {
    if (is_float()) {
        return std::any_cast<float>(value_);
    } else if (is_int()) {
        return static_cast<float>(std::any_cast<int>(value_));
    } else if (is_string()) {
        try {
            return std::stof(std::any_cast<std::string>(value_));
        } catch (const std::exception& e) {
            return 0.0f;
        }
    } else if (is_bool()) {
        return std::any_cast<bool>(value_) ? 1.0f : 0.0f;
    }
    return 0.0f;
}

bool ConfigValue::as_bool() const {
    if (is_bool()) {
        return std::any_cast<bool>(value_);
    } else if (is_int()) {
        return std::any_cast<int>(value_) != 0;
    } else if (is_float()) {
        return std::any_cast<float>(value_) != 0.0f;
    } else if (is_string()) {
        std::string str = std::any_cast<std::string>(value_);
        std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) { return std::tolower(c); });
        return str == "true" || str == "yes" || str == "1" || str == "on";
    }
    return false;
}

std::vector<std::string> ConfigValue::as_string_array() const {
    if (is_string_array()) {
        return std::any_cast<std::vector<std::string>>(value_);
    } else if (is_string()) {
        // Parse comma-separated list
        std::vector<std::string> result;
        std::string str = std::any_cast<std::string>(value_);
        std::istringstream ss(str);
        std::string item;
        while (std::getline(ss, item, ',')) {
            // Trim whitespace
            item.erase(0, item.find_first_not_of(" \t\n\r\f\v"));
            item.erase(item.find_last_not_of(" \t\n\r\f\v") + 1);
            result.push_back(item);
        }
        return result;
    }
    return {};
}

std::vector<int> ConfigValue::as_int_array() const {
    if (is_int_array()) {
        return std::any_cast<std::vector<int>>(value_);
    } else if (is_string_array()) {
        // Convert string array to int array
        std::vector<int> result;
        auto str_array = std::any_cast<std::vector<std::string>>(value_);
        for (const auto& str : str_array) {
            try {
                result.push_back(std::stoi(str));
            } catch (const std::exception& e) {
                result.push_back(0);
            }
        }
        return result;
    } else if (is_string()) {
        // Parse comma-separated list
        std::vector<int> result;
        std::string str = std::any_cast<std::string>(value_);
        std::istringstream ss(str);
        std::string item;
        while (std::getline(ss, item, ',')) {
            try {
                result.push_back(std::stoi(item));
            } catch (const std::exception& e) {
                result.push_back(0);
            }
        }
        return result;
    }
    return {};
}

std::vector<float> ConfigValue::as_float_array() const {
    if (is_float_array()) {
        return std::any_cast<std::vector<float>>(value_);
    } else if (is_int_array()) {
        // Convert int array to float array
        std::vector<float> result;
        auto int_array = std::any_cast<std::vector<int>>(value_);
        for (const auto& val : int_array) {
            result.push_back(static_cast<float>(val));
        }
        return result;
    } else if (is_string_array()) {
        // Convert string array to float array
        std::vector<float> result;
        auto str_array = std::any_cast<std::vector<std::string>>(value_);
        for (const auto& str : str_array) {
            try {
                result.push_back(std::stof(str));
            } catch (const std::exception& e) {
                result.push_back(0.0f);
            }
        }
        return result;
    } else if (is_string()) {
        // Parse comma-separated list
        std::vector<float> result;
        std::string str = std::any_cast<std::string>(value_);
        std::istringstream ss(str);
        std::string item;
        while (std::getline(ss, item, ',')) {
            try {
                result.push_back(std::stof(item));
            } catch (const std::exception& e) {
                result.push_back(0.0f);
            }
        }
        return result;
    }
    return {};
}

//=============================================================================
// Config implementation
//=============================================================================

struct Config::Impl {
    nlohmann::json config_data;
    
    // Helper for path-based access
    std::optional<nlohmann::json&> get_json_at_path(const std::string& path) {
        std::vector<std::string> parts;
        std::stringstream ss(path);
        std::string part;
        
        // Split path by dots
        while (std::getline(ss, part, '.')) {
            if (!part.empty()) {
                parts.push_back(part);
            }
        }
        
        if (parts.empty()) {
            return std::nullopt;
        }
        
        nlohmann::json* current = &config_data;
        for (const auto& p : parts) {
            if (!current->is_object() || !current->contains(p)) {
                return std::nullopt;
            }
            current = &(*current)[p];
        }
        
        return *current;
    }
    
    // Create parent path if it doesn't exist
    nlohmann::json& ensure_path(const std::string& path) {
        std::vector<std::string> parts;
        std::stringstream ss(path);
        std::string part;
        
        // Split path by dots
        while (std::getline(ss, part, '.')) {
            if (!part.empty()) {
                parts.push_back(part);
            }
        }
        
        if (parts.empty()) {
            return config_data;
        }
        
        nlohmann::json* current = &config_data;
        for (size_t i = 0; i < parts.size(); ++i) {
            const auto& p = parts[i];
            if (!current->contains(p) || !(*current)[p].is_object()) {
                if (i < parts.size() - 1) {
                    // Create intermediate objects
                    (*current)[p] = nlohmann::json::object();
                }
            }
            
            if (i < parts.size() - 1) {
                current = &(*current)[p];
            } else {
                return (*current)[p];
            }
        }
        
        return *current;
    }
};

Config::Config() : impl_(std::make_unique<Impl>()) {
    impl_->config_data = nlohmann::json::object();
}

bool Config::load_from_file(const std::string& path) {
    if (!std::filesystem::exists(path)) {
        return false;
    }
    
    try {
        std::ifstream file(path);
        impl_->config_data = nlohmann::json::parse(file);
        return true;
    } catch (const std::exception& e) {
        CCSM_ERROR("Failed to load configuration from file: ", e.what());
        return false;
    }
}

bool Config::save_to_file(const std::string& path) const {
    try {
        // Create directory if it doesn't exist
        std::filesystem::path fs_path(path);
        std::filesystem::create_directories(fs_path.parent_path());
        
        std::ofstream file(path);
        file << impl_->config_data.dump(4);  // Pretty printing with 4-space indent
        return true;
    } catch (const std::exception& e) {
        CCSM_ERROR("Failed to save configuration to file: ", e.what());
        return false;
    }
}

ConfigValue Config::get(const std::string& path, const ConfigValue& default_value) const {
    auto json_opt = impl_->get_json_at_path(path);
    if (!json_opt.has_value()) {
        return default_value;
    }
    
    const auto& json_value = json_opt.value();
    
    if (json_value.is_string()) {
        return ConfigValue(json_value.get<std::string>());
    } else if (json_value.is_number_integer()) {
        return ConfigValue(json_value.get<int>());
    } else if (json_value.is_number_float()) {
        return ConfigValue(json_value.get<float>());
    } else if (json_value.is_boolean()) {
        return ConfigValue(json_value.get<bool>());
    } else if (json_value.is_array()) {
        // Try to determine the array type
        if (json_value.size() > 0) {
            const auto& first = json_value[0];
            if (first.is_string()) {
                std::vector<std::string> result;
                for (const auto& item : json_value) {
                    result.push_back(item.get<std::string>());
                }
                return ConfigValue(result);
            } else if (first.is_number_integer()) {
                std::vector<int> result;
                for (const auto& item : json_value) {
                    result.push_back(item.get<int>());
                }
                return ConfigValue(result);
            } else if (first.is_number_float()) {
                std::vector<float> result;
                for (const auto& item : json_value) {
                    result.push_back(item.get<float>());
                }
                return ConfigValue(result);
            }
        }
    }
    
    // Fallback to default value for unsupported types
    return default_value;
}

void Config::set(const std::string& path, const ConfigValue& value) {
    auto& target = impl_->ensure_path(path);
    
    if (value.is_string()) {
        target = value.as_string();
    } else if (value.is_int()) {
        target = value.as_int();
    } else if (value.is_float()) {
        target = value.as_float();
    } else if (value.is_bool()) {
        target = value.as_bool();
    } else if (value.is_string_array()) {
        auto array = value.as_string_array();
        target = nlohmann::json::array();
        for (const auto& item : array) {
            target.push_back(item);
        }
    } else if (value.is_int_array()) {
        auto array = value.as_int_array();
        target = nlohmann::json::array();
        for (const auto& item : array) {
            target.push_back(item);
        }
    } else if (value.is_float_array()) {
        auto array = value.as_float_array();
        target = nlohmann::json::array();
        for (const auto& item : array) {
            target.push_back(item);
        }
    }
}

bool Config::has(const std::string& path) const {
    return impl_->get_json_at_path(path).has_value();
}

void Config::remove(const std::string& path) {
    std::vector<std::string> parts;
    std::stringstream ss(path);
    std::string part;
    
    // Split path by dots
    while (std::getline(ss, part, '.')) {
        if (!part.empty()) {
            parts.push_back(part);
        }
    }
    
    if (parts.empty()) {
        return;
    }
    
    nlohmann::json* current = &impl_->config_data;
    for (size_t i = 0; i < parts.size() - 1; ++i) {
        const auto& p = parts[i];
        if (!current->is_object() || !current->contains(p)) {
            return;
        }
        current = &(*current)[p];
    }
    
    if (current->is_object() && current->contains(parts.back())) {
        current->erase(parts.back());
    }
}

void Config::clear() {
    impl_->config_data.clear();
    impl_->config_data = nlohmann::json::object();
}

void Config::merge(const Config& other, bool overwrite) {
    // Helper to merge JSON objects
    auto merge_json = [overwrite](nlohmann::json& target, const nlohmann::json& source) {
        if (!source.is_object()) {
            if (overwrite) {
                target = source;
            }
            return;
        }
        
        if (!target.is_object()) {
            target = nlohmann::json::object();
        }
        
        for (auto it = source.begin(); it != source.end(); ++it) {
            if (it.value().is_object()) {
                merge_json(target[it.key()], it.value());
            } else if (overwrite || !target.contains(it.key())) {
                target[it.key()] = it.value();
            }
        }
    };
    
    merge_json(impl_->config_data, other.impl_->config_data);
}

bool Config::load_from_string(const std::string& json_string) {
    try {
        impl_->config_data = nlohmann::json::parse(json_string);
        return true;
    } catch (const std::exception& e) {
        CCSM_ERROR("Failed to parse JSON string: ", e.what());
        return false;
    }
}

std::string Config::to_string() const {
    return impl_->config_data.dump(4);  // Pretty printing with 4-space indent
}

void Config::load_defaults() {
    // Base Config class doesn't have defaults
    // Overridden by derived classes
}

std::unordered_map<std::string, ConfigValue> Config::get_all() const {
    std::unordered_map<std::string, ConfigValue> result;
    
    std::function<void(const std::string&, const nlohmann::json&)> traverse = 
        [&](const std::string& prefix, const nlohmann::json& json) {
            if (json.is_object()) {
                for (auto it = json.begin(); it != json.end(); ++it) {
                    std::string new_prefix = prefix.empty() ? it.key() : prefix + "." + it.key();
                    traverse(new_prefix, it.value());
                }
            } else if (json.is_string()) {
                result[prefix] = ConfigValue(json.get<std::string>());
            } else if (json.is_number_integer()) {
                result[prefix] = ConfigValue(json.get<int>());
            } else if (json.is_number_float()) {
                result[prefix] = ConfigValue(json.get<float>());
            } else if (json.is_boolean()) {
                result[prefix] = ConfigValue(json.get<bool>());
            } else if (json.is_array()) {
                // Try to determine the array type
                if (json.size() > 0) {
                    const auto& first = json[0];
                    if (first.is_string()) {
                        std::vector<std::string> array_result;
                        for (const auto& item : json) {
                            array_result.push_back(item.get<std::string>());
                        }
                        result[prefix] = ConfigValue(array_result);
                    } else if (first.is_number_integer()) {
                        std::vector<int> array_result;
                        for (const auto& item : json) {
                            array_result.push_back(item.get<int>());
                        }
                        result[prefix] = ConfigValue(array_result);
                    } else if (first.is_number_float()) {
                        std::vector<float> array_result;
                        for (const auto& item : json) {
                            array_result.push_back(item.get<float>());
                        }
                        result[prefix] = ConfigValue(array_result);
                    }
                }
            }
        };
    
    traverse("", impl_->config_data);
    return result;
}

//=============================================================================
// ModelConfig implementation
//=============================================================================

ModelConfig::ModelConfig() : Config() {
    load_defaults();
}

void ModelConfig::load_defaults() {
    // Set default values for model configuration
    set("model_path", ConfigValue(""));
    set("architecture", ConfigValue("ccsm"));
    set("tokenizer_path", ConfigValue(""));
    set("embedding_dim", ConfigValue(4096));
    set("num_layers", ConfigValue(32));
    set("num_heads", ConfigValue(32));
    set("vocab_size", ConfigValue(32000));
    set("use_kv_cache", ConfigValue(true));
}

std::string ModelConfig::get_model_path() const {
    return get("model_path").as_string();
}

std::string ModelConfig::get_architecture() const {
    return get("architecture").as_string();
}

std::string ModelConfig::get_tokenizer_path() const {
    return get("tokenizer_path").as_string();
}

int ModelConfig::get_embedding_dim() const {
    return get("embedding_dim").as_int();
}

int ModelConfig::get_num_layers() const {
    return get("num_layers").as_int();
}

int ModelConfig::get_num_heads() const {
    return get("num_heads").as_int();
}

int ModelConfig::get_vocab_size() const {
    return get("vocab_size").as_int();
}

bool ModelConfig::get_use_kv_cache() const {
    return get("use_kv_cache").as_bool();
}

void ModelConfig::set_model_path(const std::string& path) {
    set("model_path", ConfigValue(path));
}

void ModelConfig::set_architecture(const std::string& architecture) {
    set("architecture", ConfigValue(architecture));
}

void ModelConfig::set_tokenizer_path(const std::string& path) {
    set("tokenizer_path", ConfigValue(path));
}

void ModelConfig::set_embedding_dim(int dim) {
    set("embedding_dim", ConfigValue(dim));
}

void ModelConfig::set_num_layers(int layers) {
    set("num_layers", ConfigValue(layers));
}

void ModelConfig::set_num_heads(int heads) {
    set("num_heads", ConfigValue(heads));
}

void ModelConfig::set_vocab_size(int size) {
    set("vocab_size", ConfigValue(size));
}

void ModelConfig::set_use_kv_cache(bool use_cache) {
    set("use_kv_cache", ConfigValue(use_cache));
}

//=============================================================================
// GenerationConfig implementation
//=============================================================================

GenerationConfig::GenerationConfig() : Config() {
    load_defaults();
}

void GenerationConfig::load_defaults() {
    // Set default values for generation configuration
    set("temperature", ConfigValue(0.9f));
    set("top_k", ConfigValue(50));
    set("top_p", ConfigValue(1.0f));
    set("max_audio_length_ms", ConfigValue(10000));
    set("seed", ConfigValue(-1));
    set("enable_watermark", ConfigValue(true));
    set("repetition_penalty", ConfigValue(1.0f));
}

float GenerationConfig::get_temperature() const {
    return get("temperature").as_float();
}

int GenerationConfig::get_top_k() const {
    return get("top_k").as_int();
}

float GenerationConfig::get_top_p() const {
    return get("top_p").as_float();
}

int GenerationConfig::get_max_audio_length_ms() const {
    return get("max_audio_length_ms").as_int();
}

int GenerationConfig::get_seed() const {
    return get("seed").as_int();
}

bool GenerationConfig::get_enable_watermark() const {
    return get("enable_watermark").as_bool();
}

float GenerationConfig::get_repetition_penalty() const {
    return get("repetition_penalty").as_float();
}

void GenerationConfig::set_temperature(float temp) {
    set("temperature", ConfigValue(temp));
}

void GenerationConfig::set_top_k(int k) {
    set("top_k", ConfigValue(k));
}

void GenerationConfig::set_top_p(float p) {
    set("top_p", ConfigValue(p));
}

void GenerationConfig::set_max_audio_length_ms(int ms) {
    set("max_audio_length_ms", ConfigValue(ms));
}

void GenerationConfig::set_seed(int seed) {
    set("seed", ConfigValue(seed));
}

void GenerationConfig::set_enable_watermark(bool enable) {
    set("enable_watermark", ConfigValue(enable));
}

void GenerationConfig::set_repetition_penalty(float penalty) {
    set("repetition_penalty", ConfigValue(penalty));
}

//=============================================================================
// SystemConfig implementation
//=============================================================================

SystemConfig::SystemConfig() : Config() {
    load_defaults();
}

void SystemConfig::load_defaults() {
    // Set default values for system configuration
    set("num_threads", ConfigValue(4));
    set("cpu_only", ConfigValue(false));
    set("debug", ConfigValue(false));
    
    // Determine default directories
    std::string home_dir;
#ifdef _WIN32
    home_dir = std::getenv("USERPROFILE") ? std::getenv("USERPROFILE") : ".";
#else
    home_dir = std::getenv("HOME") ? std::getenv("HOME") : ".";
#endif
    
    std::string cache_dir = std::filesystem::path(home_dir) / ".ccsm" / "cache";
    std::string models_dir = std::filesystem::path(home_dir) / ".ccsm" / "models";
    
    set("cache_dir", ConfigValue(cache_dir));
    set("models_dir", ConfigValue(models_dir));
}

int SystemConfig::get_num_threads() const {
    return get("num_threads").as_int();
}

bool SystemConfig::get_cpu_only() const {
    return get("cpu_only").as_bool();
}

bool SystemConfig::get_debug() const {
    return get("debug").as_bool();
}

std::string SystemConfig::get_cache_dir() const {
    return get("cache_dir").as_string();
}

std::string SystemConfig::get_models_dir() const {
    return get("models_dir").as_string();
}

void SystemConfig::set_num_threads(int threads) {
    set("num_threads", ConfigValue(threads));
}

void SystemConfig::set_cpu_only(bool cpu_only) {
    set("cpu_only", ConfigValue(cpu_only));
}

void SystemConfig::set_debug(bool debug) {
    set("debug", ConfigValue(debug));
}

void SystemConfig::set_cache_dir(const std::string& dir) {
    set("cache_dir", ConfigValue(dir));
}

void SystemConfig::set_models_dir(const std::string& dir) {
    set("models_dir", ConfigValue(dir));
}

//=============================================================================
// ConfigManager implementation
//=============================================================================

ConfigManager& ConfigManager::instance() {
    static ConfigManager instance;
    return instance;
}

ConfigManager::ConfigManager()
    : model_config_(), generation_config_(), system_config_() {
    reset_to_defaults();
}

ModelConfig& ConfigManager::model_config() {
    return model_config_;
}

GenerationConfig& ConfigManager::generation_config() {
    return generation_config_;
}

SystemConfig& ConfigManager::system_config() {
    return system_config_;
}

bool ConfigManager::load_from_directory(const std::string& dir) {
    bool success = true;
    
    // Load model config
    std::string model_config_path = std::filesystem::path(dir) / "model.json";
    if (std::filesystem::exists(model_config_path)) {
        success &= model_config_.load_from_file(model_config_path);
    }
    
    // Load generation config
    std::string generation_config_path = std::filesystem::path(dir) / "generation.json";
    if (std::filesystem::exists(generation_config_path)) {
        success &= generation_config_.load_from_file(generation_config_path);
    }
    
    // Load system config
    std::string system_config_path = std::filesystem::path(dir) / "system.json";
    if (std::filesystem::exists(system_config_path)) {
        success &= system_config_.load_from_file(system_config_path);
    }
    
    return success;
}

bool ConfigManager::save_to_directory(const std::string& dir) const {
    bool success = true;
    
    // Create directory if it doesn't exist
    if (!std::filesystem::exists(dir)) {
        std::filesystem::create_directories(dir);
    }
    
    // Save model config
    std::string model_config_path = std::filesystem::path(dir) / "model.json";
    success &= model_config_.save_to_file(model_config_path);
    
    // Save generation config
    std::string generation_config_path = std::filesystem::path(dir) / "generation.json";
    success &= generation_config_.save_to_file(generation_config_path);
    
    // Save system config
    std::string system_config_path = std::filesystem::path(dir) / "system.json";
    success &= system_config_.save_to_file(system_config_path);
    
    return success;
}

void ConfigManager::load_from_args(int argc, char** argv) {
    // Parse command-line arguments and update configurations
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        // Skip help and version flags
        if (arg == "--help" || arg == "-h" || arg == "--version" || arg == "-v") {
            continue;
        }
        
        // Handle model options
        if (arg == "--model" || arg == "-m") {
            if (i + 1 < argc) {
                model_config_.set_model_path(argv[++i]);
            }
        }
        else if (arg == "--tokenizer") {
            if (i + 1 < argc) {
                model_config_.set_tokenizer_path(argv[++i]);
            }
        }
        
        // Handle generation options
        else if (arg == "--text" || arg == "-t") {
            if (i + 1 < argc) {
                ++i; // Skip the text value, it's handled elsewhere
            }
        }
        else if (arg == "--speaker" || arg == "-s") {
            if (i + 1 < argc) {
                ++i; // Skip the speaker value, it's handled elsewhere
            }
        }
        else if (arg == "--temperature") {
            if (i + 1 < argc) {
                try {
                    generation_config_.set_temperature(std::stof(argv[++i]));
                } catch (...) {
                    // Skip invalid value
                }
            }
        }
        else if (arg == "--topk") {
            if (i + 1 < argc) {
                try {
                    generation_config_.set_top_k(std::stoi(argv[++i]));
                } catch (...) {
                    // Skip invalid value
                }
            }
        }
        else if (arg == "--topp") {
            if (i + 1 < argc) {
                try {
                    generation_config_.set_top_p(std::stof(argv[++i]));
                } catch (...) {
                    // Skip invalid value
                }
            }
        }
        else if (arg == "--seed") {
            if (i + 1 < argc) {
                try {
                    generation_config_.set_seed(std::stoi(argv[++i]));
                } catch (...) {
                    // Skip invalid value
                }
            }
        }
        else if (arg == "--max-length") {
            if (i + 1 < argc) {
                try {
                    generation_config_.set_max_audio_length_ms(std::stoi(argv[++i]));
                } catch (...) {
                    // Skip invalid value
                }
            }
        }
        else if (arg == "--output" || arg == "-o") {
            if (i + 1 < argc) {
                ++i; // Skip the output path, it's handled elsewhere
            }
        }
        else if (arg == "--no-watermark") {
            generation_config_.set_enable_watermark(false);
        }
        else if (arg == "--repetition-penalty") {
            if (i + 1 < argc) {
                try {
                    generation_config_.set_repetition_penalty(std::stof(argv[++i]));
                } catch (...) {
                    // Skip invalid value
                }
            }
        }
        
        // Handle context options
        else if (arg == "--context-text" || arg == "--context-audio" || arg == "--context-speaker") {
            if (i + 1 < argc) {
                ++i; // Skip the context value, it's handled elsewhere
            }
        }
        
        // Handle system options
        else if (arg == "--threads") {
            if (i + 1 < argc) {
                try {
                    system_config_.set_num_threads(std::stoi(argv[++i]));
                } catch (...) {
                    // Skip invalid value
                }
            }
        }
        else if (arg == "--cpu-only") {
            system_config_.set_cpu_only(true);
        }
        else if (arg == "--debug" || arg == "-d") {
            system_config_.set_debug(true);
        }
        else if (arg == "--cache-dir") {
            if (i + 1 < argc) {
                system_config_.set_cache_dir(argv[++i]);
            }
        }
        else if (arg == "--models-dir") {
            if (i + 1 < argc) {
                system_config_.set_models_dir(argv[++i]);
            }
        }
        
        // Handle backend parameters
        else if (arg.substr(0, 9) == "--backend-") {
            // These are handled elsewhere
        }
    }
}

bool ConfigManager::load_from_json_string(const std::string& json) {
    try {
        // Parse the JSON string
        nlohmann::json json_data = nlohmann::json::parse(json);
        
        // Extract model config
        if (json_data.contains("model") && json_data["model"].is_object()) {
            model_config_.load_from_string(json_data["model"].dump());
        }
        
        // Extract generation config
        if (json_data.contains("generation") && json_data["generation"].is_object()) {
            generation_config_.load_from_string(json_data["generation"].dump());
        }
        
        // Extract system config
        if (json_data.contains("system") && json_data["system"].is_object()) {
            system_config_.load_from_string(json_data["system"].dump());
        }
        
        return true;
    } catch (const std::exception& e) {
        CCSM_ERROR("Failed to parse JSON configuration: ", e.what());
        return false;
    }
}

std::string ConfigManager::to_json_string() const {
    nlohmann::json json_data;
    
    // Add model config
    json_data["model"] = nlohmann::json::parse(model_config_.to_string());
    
    // Add generation config
    json_data["generation"] = nlohmann::json::parse(generation_config_.to_string());
    
    // Add system config
    json_data["system"] = nlohmann::json::parse(system_config_.to_string());
    
    return json_data.dump(4);  // Pretty printing with 4-space indent
}

void ConfigManager::reset_to_defaults() {
    model_config_.load_defaults();
    generation_config_.load_defaults();
    system_config_.load_defaults();
}

} // namespace ccsm