#include <gtest/gtest.h>
#include <ccsm/config.h>
#include <filesystem>
#include <fstream>

using namespace ccsm;

// Test ConfigValue functionality
TEST(ConfigTest, ConfigValueTest) {
    // Test string values
    ConfigValue str_val("test");
    EXPECT_TRUE(str_val.is_string());
    EXPECT_EQ(str_val.as_string(), "test");
    EXPECT_EQ(str_val.as_int(), 0);  // Can't convert to int
    EXPECT_EQ(str_val.as_float(), 0.0f); // Can't convert to float
    EXPECT_FALSE(str_val.as_bool()); // "test" doesn't convert to true
    
    // Test numeric string values
    ConfigValue num_str_val("123");
    EXPECT_TRUE(num_str_val.is_string());
    EXPECT_EQ(num_str_val.as_string(), "123");
    EXPECT_EQ(num_str_val.as_int(), 123);
    EXPECT_FLOAT_EQ(num_str_val.as_float(), 123.0f);
    EXPECT_FALSE(num_str_val.as_bool());
    
    // Test boolean string values
    ConfigValue bool_str_val("true");
    EXPECT_TRUE(bool_str_val.is_string());
    EXPECT_EQ(bool_str_val.as_string(), "true");
    EXPECT_EQ(bool_str_val.as_int(), 0);
    EXPECT_FLOAT_EQ(bool_str_val.as_float(), 0.0f);
    EXPECT_TRUE(bool_str_val.as_bool());
    
    // Test int values
    ConfigValue int_val(42);
    EXPECT_TRUE(int_val.is_int());
    EXPECT_EQ(int_val.as_string(), "42");
    EXPECT_EQ(int_val.as_int(), 42);
    EXPECT_FLOAT_EQ(int_val.as_float(), 42.0f);
    EXPECT_TRUE(int_val.as_bool());
    
    // Test float values
    ConfigValue float_val(3.14f);
    EXPECT_TRUE(float_val.is_float());
    EXPECT_EQ(float_val.as_string(), "3.140000");
    EXPECT_EQ(float_val.as_int(), 3);
    EXPECT_FLOAT_EQ(float_val.as_float(), 3.14f);
    EXPECT_TRUE(float_val.as_bool());
    
    // Test bool values
    ConfigValue bool_val(true);
    EXPECT_TRUE(bool_val.is_bool());
    EXPECT_EQ(bool_val.as_string(), "true");
    EXPECT_EQ(bool_val.as_int(), 1);
    EXPECT_FLOAT_EQ(bool_val.as_float(), 1.0f);
    EXPECT_TRUE(bool_val.as_bool());
    
    // Test array values
    std::vector<std::string> str_array = {"one", "two", "three"};
    ConfigValue str_array_val(str_array);
    EXPECT_TRUE(str_array_val.is_string_array());
    EXPECT_EQ(str_array_val.as_string_array().size(), 3);
    EXPECT_EQ(str_array_val.as_string_array()[0], "one");
    
    std::vector<int> int_array = {1, 2, 3};
    ConfigValue int_array_val(int_array);
    EXPECT_TRUE(int_array_val.is_int_array());
    EXPECT_EQ(int_array_val.as_int_array().size(), 3);
    EXPECT_EQ(int_array_val.as_int_array()[0], 1);
    
    // Test array conversion
    EXPECT_EQ(int_array_val.as_float_array().size(), 3);
    EXPECT_FLOAT_EQ(int_array_val.as_float_array()[0], 1.0f);
}

// Test basic Config functionality
TEST(ConfigTest, BasicConfigTest) {
    Config config;
    
    // Set values
    config.set("test_string", ConfigValue("hello"));
    config.set("test_int", ConfigValue(42));
    config.set("test_float", ConfigValue(3.14f));
    config.set("test_bool", ConfigValue(true));
    
    // Get values
    EXPECT_EQ(config.get("test_string").as_string(), "hello");
    EXPECT_EQ(config.get("test_int").as_int(), 42);
    EXPECT_FLOAT_EQ(config.get("test_float").as_float(), 3.14f);
    EXPECT_TRUE(config.get("test_bool").as_bool());
    
    // Test default values
    EXPECT_EQ(config.get("non_existent", ConfigValue("default")).as_string(), "default");
    EXPECT_EQ(config.get("non_existent", ConfigValue(123)).as_int(), 123);
    
    // Test has/remove
    EXPECT_TRUE(config.has("test_string"));
    config.remove("test_string");
    EXPECT_FALSE(config.has("test_string"));
    
    // Test nested paths
    config.set("nested.value", ConfigValue("nested_value"));
    EXPECT_EQ(config.get("nested.value").as_string(), "nested_value");
    
    // Test very deep nesting
    config.set("a.b.c.d.e.f.g", ConfigValue("deep"));
    EXPECT_EQ(config.get("a.b.c.d.e.f.g").as_string(), "deep");
}

// Test saving and loading
TEST(ConfigTest, SaveLoadTest) {
    // Create a temporary file path
    std::string temp_file = std::filesystem::temp_directory_path().string() + "/config_test.json";
    
    // Create a config
    Config config;
    config.set("string_value", ConfigValue("hello"));
    config.set("int_value", ConfigValue(42));
    config.set("nested.value", ConfigValue(true));
    
    // Save to file
    EXPECT_TRUE(config.save_to_file(temp_file));
    
    // Load into a new config
    Config loaded_config;
    EXPECT_TRUE(loaded_config.load_from_file(temp_file));
    
    // Verify values
    EXPECT_EQ(loaded_config.get("string_value").as_string(), "hello");
    EXPECT_EQ(loaded_config.get("int_value").as_int(), 42);
    EXPECT_TRUE(loaded_config.get("nested.value").as_bool());
    
    // Clean up
    std::filesystem::remove(temp_file);
}

// Test ModelConfig
TEST(ConfigTest, ModelConfigTest) {
    ModelConfig config;
    
    // Check default values
    EXPECT_EQ(config.get_architecture(), "ccsm");
    EXPECT_EQ(config.get_embedding_dim(), 4096);
    EXPECT_EQ(config.get_num_layers(), 32);
    EXPECT_TRUE(config.get_use_kv_cache());
    
    // Set new values
    config.set_model_path("/path/to/model");
    config.set_embedding_dim(2048);
    config.set_num_heads(16);
    
    // Check new values
    EXPECT_EQ(config.get_model_path(), "/path/to/model");
    EXPECT_EQ(config.get_embedding_dim(), 2048);
    EXPECT_EQ(config.get_num_heads(), 16);
}

// Test GenerationConfig
TEST(ConfigTest, GenerationConfigTest) {
    GenerationConfig config;
    
    // Check default values
    EXPECT_FLOAT_EQ(config.get_temperature(), 0.9f);
    EXPECT_EQ(config.get_top_k(), 50);
    EXPECT_FLOAT_EQ(config.get_top_p(), 1.0f);
    EXPECT_EQ(config.get_max_audio_length_ms(), 10000);
    EXPECT_EQ(config.get_seed(), -1);
    EXPECT_TRUE(config.get_enable_watermark());
    
    // Set new values
    config.set_temperature(1.2f);
    config.set_top_k(100);
    config.set_max_audio_length_ms(20000);
    config.set_repetition_penalty(1.5f);
    
    // Check new values
    EXPECT_FLOAT_EQ(config.get_temperature(), 1.2f);
    EXPECT_EQ(config.get_top_k(), 100);
    EXPECT_EQ(config.get_max_audio_length_ms(), 20000);
    EXPECT_FLOAT_EQ(config.get_repetition_penalty(), 1.5f);
}

// Test SystemConfig
TEST(ConfigTest, SystemConfigTest) {
    SystemConfig config;
    
    // Check default values
    EXPECT_EQ(config.get_num_threads(), 4);
    EXPECT_FALSE(config.get_cpu_only());
    EXPECT_FALSE(config.get_debug());
    
    // Set new values
    config.set_num_threads(8);
    config.set_cpu_only(true);
    config.set_debug(true);
    
    // Check new values
    EXPECT_EQ(config.get_num_threads(), 8);
    EXPECT_TRUE(config.get_cpu_only());
    EXPECT_TRUE(config.get_debug());
}

// Test ConfigManager
TEST(ConfigTest, ConfigManagerTest) {
    auto& manager = ConfigManager::instance();
    
    // Reset to defaults to ensure consistent test state
    manager.reset_to_defaults();
    
    // Check default values
    EXPECT_FLOAT_EQ(manager.generation_config().get_temperature(), 0.9f);
    EXPECT_EQ(manager.system_config().get_num_threads(), 4);
    
    // Modify values
    manager.generation_config().set_temperature(1.5f);
    manager.system_config().set_num_threads(16);
    
    // Verify changes
    EXPECT_FLOAT_EQ(manager.generation_config().get_temperature(), 1.5f);
    EXPECT_EQ(manager.system_config().get_num_threads(), 16);
    
    // Test JSON string export/import
    std::string json = manager.to_json_string();
    
    // Reset
    manager.reset_to_defaults();
    EXPECT_FLOAT_EQ(manager.generation_config().get_temperature(), 0.9f);
    
    // Load from JSON
    EXPECT_TRUE(manager.load_from_json_string(json));
    EXPECT_FLOAT_EQ(manager.generation_config().get_temperature(), 1.5f);
    EXPECT_EQ(manager.system_config().get_num_threads(), 16);
    
    // Save and load from directory
    std::string temp_dir = std::filesystem::temp_directory_path().string() + "/config_test_dir";
    std::filesystem::create_directories(temp_dir);
    
    EXPECT_TRUE(manager.save_to_directory(temp_dir));
    
    // Reset
    manager.reset_to_defaults();
    EXPECT_FLOAT_EQ(manager.generation_config().get_temperature(), 0.9f);
    
    // Load back
    EXPECT_TRUE(manager.load_from_directory(temp_dir));
    EXPECT_FLOAT_EQ(manager.generation_config().get_temperature(), 1.5f);
    
    // Clean up
    std::filesystem::remove_all(temp_dir);
}

// Test loading from CLI args
TEST(ConfigTest, LoadFromArgsTest) {
    auto& manager = ConfigManager::instance();
    manager.reset_to_defaults();
    
    // Prepare mock CLI args
    const char* argv[] = {
        "program",
        "--temperature", "1.5",
        "--topk", "20",
        "--topp", "0.9",
        "--threads", "8",
        "--debug",
        "--text", "Test text",  // Should be ignored by ConfigManager
        nullptr
    };
    int argc = sizeof(argv) / sizeof(argv[0]) - 1;
    
    // Load from args
    manager.load_from_args(argc, const_cast<char**>(argv));
    
    // Verify values were updated
    EXPECT_FLOAT_EQ(manager.generation_config().get_temperature(), 1.5f);
    EXPECT_EQ(manager.generation_config().get_top_k(), 20);
    EXPECT_FLOAT_EQ(manager.generation_config().get_top_p(), 0.9f);
    EXPECT_EQ(manager.system_config().get_num_threads(), 8);
    EXPECT_TRUE(manager.system_config().get_debug());
}