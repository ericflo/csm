#ifndef CCSM_CONFIG_H
#define CCSM_CONFIG_H

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <any>
#include <typeindex>
#include <functional>

namespace ccsm {

/**
 * ConfigValue - A class to store configuration values of different types
 */
class ConfigValue {
public:
    // Constructors for various types
    ConfigValue() : type_(typeid(void)) {}
    ConfigValue(const std::string& value) : value_(value), type_(typeid(std::string)) {}
    ConfigValue(int value) : value_(value), type_(typeid(int)) {}
    ConfigValue(float value) : value_(value), type_(typeid(float)) {}
    ConfigValue(bool value) : value_(value), type_(typeid(bool)) {}
    ConfigValue(const std::vector<std::string>& value) : value_(value), type_(typeid(std::vector<std::string>)) {}
    ConfigValue(const std::vector<int>& value) : value_(value), type_(typeid(std::vector<int>)) {}
    ConfigValue(const std::vector<float>& value) : value_(value), type_(typeid(std::vector<float>)) {}
    
    // Type checking
    bool is_string() const { return type_ == typeid(std::string); }
    bool is_int() const { return type_ == typeid(int); }
    bool is_float() const { return type_ == typeid(float); }
    bool is_bool() const { return type_ == typeid(bool); }
    bool is_string_array() const { return type_ == typeid(std::vector<std::string>); }
    bool is_int_array() const { return type_ == typeid(std::vector<int>); }
    bool is_float_array() const { return type_ == typeid(std::vector<float>); }
    
    // Value retrieval
    std::string as_string() const;
    int as_int() const;
    float as_float() const;
    bool as_bool() const;
    std::vector<std::string> as_string_array() const;
    std::vector<int> as_int_array() const;
    std::vector<float> as_float_array() const;
    
    // Get the type info
    const std::type_info& type() const { return type_; }
    
private:
    std::any value_;
    std::type_info const& type_;
};

/**
 * Config - A hierarchical configuration system for model and generation settings
 */
class Config {
public:
    // Constructor
    Config();
    
    // Load configuration from a file
    bool load_from_file(const std::string& path);
    
    // Save configuration to a file
    bool save_to_file(const std::string& path) const;
    
    // Get value with path-like syntax (e.g., "model.parameters.layers")
    ConfigValue get(const std::string& path, const ConfigValue& default_value = ConfigValue()) const;
    
    // Set value with path-like syntax
    void set(const std::string& path, const ConfigValue& value);
    
    // Check if a path exists
    bool has(const std::string& path) const;
    
    // Remove a path
    void remove(const std::string& path);
    
    // Clear all settings
    void clear();
    
    // Merge another configuration into this one
    void merge(const Config& other, bool overwrite = true);
    
    // Load settings from a string (JSON format)
    bool load_from_string(const std::string& json_string);
    
    // Convert to a string representation (JSON format)
    std::string to_string() const;
    
    // Load default values
    void load_defaults();
    
    // Get all settings as a flat map
    std::unordered_map<std::string, ConfigValue> get_all() const;
    
private:
    // Implementation details
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * ModelConfig - Specialized configuration for model settings
 */
class ModelConfig : public Config {
public:
    ModelConfig();
    
    // Pre-defined getters for common model settings
    std::string get_model_path() const;
    std::string get_architecture() const;
    std::string get_tokenizer_path() const;
    int get_embedding_dim() const;
    int get_num_layers() const;
    int get_num_heads() const;
    int get_vocab_size() const;
    bool get_use_kv_cache() const;
    
    // Pre-defined setters for common model settings
    void set_model_path(const std::string& path);
    void set_architecture(const std::string& architecture);
    void set_tokenizer_path(const std::string& path);
    void set_embedding_dim(int dim);
    void set_num_layers(int layers);
    void set_num_heads(int heads);
    void set_vocab_size(int size);
    void set_use_kv_cache(bool use_cache);
};

/**
 * GenerationConfig - Specialized configuration for generation settings
 */
class GenerationConfig : public Config {
public:
    GenerationConfig();
    
    // Pre-defined getters for common generation settings
    float get_temperature() const;
    int get_top_k() const;
    float get_top_p() const;
    int get_max_audio_length_ms() const;
    int get_seed() const;
    bool get_enable_watermark() const;
    float get_repetition_penalty() const;
    
    // Pre-defined setters for common generation settings
    void set_temperature(float temp);
    void set_top_k(int k);
    void set_top_p(float p);
    void set_max_audio_length_ms(int ms);
    void set_seed(int seed);
    void set_enable_watermark(bool enable);
    void set_repetition_penalty(float penalty);
};

/**
 * SystemConfig - Specialized configuration for system-wide settings
 */
class SystemConfig : public Config {
public:
    SystemConfig();
    
    // Pre-defined getters for common system settings
    int get_num_threads() const;
    bool get_cpu_only() const;
    bool get_debug() const;
    std::string get_cache_dir() const;
    std::string get_models_dir() const;
    
    // Pre-defined setters for common system settings
    void set_num_threads(int threads);
    void set_cpu_only(bool cpu_only);
    void set_debug(bool debug);
    void set_cache_dir(const std::string& dir);
    void set_models_dir(const std::string& dir);
};

/**
 * ConfigManager - Singleton to manage all configurations
 */
class ConfigManager {
public:
    // Get the singleton instance
    static ConfigManager& instance();
    
    // Delete copy and move constructors and assign operators
    ConfigManager(const ConfigManager&) = delete;
    ConfigManager& operator=(const ConfigManager&) = delete;
    ConfigManager(ConfigManager&&) = delete;
    ConfigManager& operator=(ConfigManager&&) = delete;
    
    // Get configurations
    ModelConfig& model_config();
    GenerationConfig& generation_config();
    SystemConfig& system_config();
    
    // Load all configurations from a directory
    bool load_from_directory(const std::string& dir);
    
    // Save all configurations to a directory
    bool save_to_directory(const std::string& dir) const;
    
    // Load configurations from command-line arguments
    void load_from_args(int argc, char** argv);
    
    // Load configurations from a JSON string
    bool load_from_json_string(const std::string& json);
    
    // Get as a JSON string
    std::string to_json_string() const;
    
    // Reset to default values
    void reset_to_defaults();
    
private:
    // Private constructor for singleton
    ConfigManager();
    
    // Member variables
    ModelConfig model_config_;
    GenerationConfig generation_config_;
    SystemConfig system_config_;
};

} // namespace ccsm

#endif // CCSM_CONFIG_H