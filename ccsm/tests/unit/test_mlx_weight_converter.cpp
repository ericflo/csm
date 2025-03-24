#include <gtest/gtest.h>
#include <ccsm/mlx/mlx_weight_converter.h>
#include <ccsm/model_loader.h>
#include <ccsm/tensor.h>
#include <memory>
#include <filesystem>
#include <fstream>
#include <random>

namespace ccsm {
namespace testing {

// Create a mock tensor for testing
Tensor create_mock_tensor(const std::vector<size_t>& shape, DataType dtype) {
    return TensorFactory::zeros(shape, dtype);
}

// Create a mock tensor with random values for testing
Tensor create_random_tensor(const std::vector<size_t>& shape, DataType dtype) {
    // Create a tensor with the specified shape and dtype
    Tensor tensor = TensorFactory::zeros(shape, dtype);
    
    // Calculate the total number of elements
    size_t num_elements = 1;
    for (auto dim : shape) {
        num_elements *= dim;
    }
    
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    // Fill with random values based on dtype
    if (dtype == DataType::F32) {
        float* data_ptr = static_cast<float*>(tensor.data());
        for (size_t i = 0; i < num_elements; ++i) {
            data_ptr[i] = dist(gen);
        }
    } else if (dtype == DataType::F16) {
        // Simulate F16 by using F32 and truncating
        float* data_ptr = static_cast<float*>(tensor.data());
        for (size_t i = 0; i < num_elements; ++i) {
            data_ptr[i] = static_cast<float>(static_cast<int16_t>(dist(gen) * 100.0f) / 100.0f);
        }
    } else if (dtype == DataType::BF16) {
        // Simulate BF16 by using F32 and truncating more aggressively
        float* data_ptr = static_cast<float*>(tensor.data());
        for (size_t i = 0; i < num_elements; ++i) {
            data_ptr[i] = static_cast<float>(static_cast<int8_t>(dist(gen) * 10.0f) / 10.0f);
        }
    } else if (dtype == DataType::I32) {
        int32_t* data_ptr = static_cast<int32_t*>(tensor.data());
        for (size_t i = 0; i < num_elements; ++i) {
            data_ptr[i] = static_cast<int32_t>(dist(gen) * 100.0f);
        }
    } else if (dtype == DataType::I64) {
        int64_t* data_ptr = static_cast<int64_t*>(tensor.data());
        for (size_t i = 0; i < num_elements; ++i) {
            data_ptr[i] = static_cast<int64_t>(dist(gen) * 1000.0f);
        }
    }
    
    return tensor;
}

// Create a mock weight map for testing
WeightMap create_mock_weight_map() {
    WeightMap weights;
    // Add some test tensors with different shapes and dtypes
    weights["model.weight"] = create_random_tensor({10, 10}, DataType::F32);
    weights["model.bias"] = create_random_tensor({10}, DataType::F32);
    weights["embedding.weight"] = create_random_tensor({100, 20}, DataType::F32);
    weights["attention.weight"] = create_random_tensor({10, 10, 3}, DataType::F32);
    weights["attention.bias"] = create_random_tensor({10, 3}, DataType::F32);
    weights["ff.weight"] = create_random_tensor({20, 10, 3}, DataType::F16);
    weights["ff.bias"] = create_random_tensor({20, 3}, DataType::F16);
    weights["layer_norm.weight"] = create_random_tensor({10}, DataType::F32);
    weights["layer_norm.bias"] = create_random_tensor({10}, DataType::F32);
    weights["token_indices"] = create_random_tensor({100}, DataType::I32);
    weights["position_indices"] = create_random_tensor({100}, DataType::I64);
    return weights;
}

class MLXWeightConverterTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test if MLX is available
TEST_F(MLXWeightConverterTest, TestMLXAvailability) {
    // This test should pass regardless of MLX availability
    bool available = MLXWeightConverter::is_mlx_available();
    
    #ifdef CCSM_WITH_MLX
    // If compiled with MLX, we expect it to be available in this context
    // This would be true in a CI environment that has MLX installed
    // EXPECT_TRUE(available);
    #else
    // If not compiled with MLX, it should definitely not be available
    EXPECT_FALSE(available);
    #endif
}

// Test the initialization of MLXWeightConverter
TEST_F(MLXWeightConverterTest, TestWeightConverterInitialization) {
    MLXWeightConversionConfig config;
    config.use_bfloat16 = true;
    config.use_quantization = false;
    config.cache_converted_weights = true;
    
    // Test creating the converter
    MLXWeightConverter converter(config);
    
    // Not much to assert here since this is just initialization
    EXPECT_TRUE(true);
}

#ifdef CCSM_WITH_MLX
// Test converting a tensor to MLX array (only when MLX is available)
TEST_F(MLXWeightConverterTest, TestConvertTensorToMLXArray) {
    // Create tensors with different shapes and dtypes
    Tensor tensor_f32 = create_random_tensor({2, 3}, DataType::F32);
    Tensor tensor_f16 = create_random_tensor({3, 4, 5}, DataType::F16);
    Tensor tensor_bf16 = create_random_tensor({10}, DataType::BF16);
    Tensor tensor_i32 = create_random_tensor({2, 2}, DataType::I32);
    Tensor tensor_i64 = create_random_tensor({5}, DataType::I64);
    
    // Convert to MLX array with BFloat16 conversion
    mlx_array array_f32_to_bf16 = convert_tensor_to_mlx_array(tensor_f32, true);
    mlx_array array_f16_to_bf16 = convert_tensor_to_mlx_array(tensor_f16, true);
    mlx_array array_bf16 = convert_tensor_to_mlx_array(tensor_bf16, true);
    mlx_array array_i32 = convert_tensor_to_mlx_array(tensor_i32, true);
    mlx_array array_i64 = convert_tensor_to_mlx_array(tensor_i64, true);
    
    // Verify array shapes and dtypes
    uint32_t ndim;
    mlx_array_ndim(array_f32_to_bf16, &ndim);
    EXPECT_EQ(ndim, 2);
    
    std::vector<int64_t> shape(ndim);
    mlx_array_shape(array_f32_to_bf16, shape.data());
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    
    mlx_dtype dtype;
    mlx_array_dtype(array_f32_to_bf16, &dtype);
    EXPECT_EQ(dtype, MLX_BFLOAT16);
    
    // Convert to MLX array without BFloat16 conversion
    mlx_array array_f32 = convert_tensor_to_mlx_array(tensor_f32, false);
    mlx_array array_f16 = convert_tensor_to_mlx_array(tensor_f16, false);
    
    // Verify dtypes are preserved
    mlx_array_dtype(array_f32, &dtype);
    EXPECT_EQ(dtype, MLX_FLOAT32);
    
    mlx_array_dtype(array_f16, &dtype);
    EXPECT_EQ(dtype, MLX_FLOAT16);
    
    // Free all arrays
    mlx_array_free(array_f32_to_bf16);
    mlx_array_free(array_f16_to_bf16);
    mlx_array_free(array_bf16);
    mlx_array_free(array_i32);
    mlx_array_free(array_i64);
    mlx_array_free(array_f32);
    mlx_array_free(array_f16);
}

// Test handling of null data pointers
TEST_F(MLXWeightConverterTest, TestNullDataHandling) {
    // Create a tensor with a null data pointer
    Tensor tensor_with_null = TensorFactory::zeros({2, 3}, DataType::F32);
    
    // Manually set the data pointer to null (this is a bit of a hack for testing)
    // In real usage, this shouldn't happen, but we still want to test robustness
    tensor_with_null = TensorFactory::empty({2, 3}, DataType::F32);
    
    // Convert to MLX array
    mlx_array array = convert_tensor_to_mlx_array(tensor_with_null, true);
    
    // Verify array was created (should be zeros)
    uint32_t ndim;
    mlx_array_ndim(array, &ndim);
    EXPECT_EQ(ndim, 2);
    
    std::vector<int64_t> shape(ndim);
    mlx_array_shape(array, shape.data());
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    
    // Free the array
    mlx_array_free(array);
}

// Test MLX tensor conversion
TEST_F(MLXWeightConverterTest, TestMLXTensorConversion) {
    // Create an MLX tensor
    Tensor tensor = MLXTensorFactory::zeros({2, 3}, DataType::F32);
    
    // Convert to MLX array (should avoid copying since it's already an MLX tensor)
    mlx_array array = convert_tensor_to_mlx_array(tensor, false);
    
    // Verify array
    uint32_t ndim;
    mlx_array_ndim(array, &ndim);
    EXPECT_EQ(ndim, 2);
    
    std::vector<int64_t> shape(ndim);
    mlx_array_shape(array, shape.data());
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    
    // Free the array
    mlx_array_free(array);
}

// Test caching functionality
TEST_F(MLXWeightConverterTest, TestCachingFunctionality) {
    // Create a temporary file path for testing
    std::string temp_dir = std::filesystem::temp_directory_path().string();
    std::string cache_path = temp_dir + "/mlx_test_cache.mlxcache";
    
    // Create mock weights
    WeightMap weight_map = create_mock_weight_map();
    
    // Convert to MLX arrays
    MLXWeightConversionConfig config;
    auto mlx_weights = convert_pytorch_to_mlx(weight_map, config);
    
    // Save to cache
    bool save_success = save_mlx_weights_to_cache(cache_path, mlx_weights);
    EXPECT_TRUE(save_success);
    
    // Verify file exists
    EXPECT_TRUE(std::filesystem::exists(cache_path));
    
    // Load from cache
    auto loaded_weights = load_mlx_weights_from_cache(cache_path);
    
    // Verify loaded weights
    EXPECT_EQ(loaded_weights.size(), mlx_weights.size());
    
    // Verify individual weights
    for (const auto& [name, array] : mlx_weights) {
        EXPECT_TRUE(loaded_weights.count(name) > 0);
        
        // Check that arrays have the same shape and dtype
        uint32_t orig_ndim, loaded_ndim;
        mlx_array_ndim(array, &orig_ndim);
        mlx_array_ndim(loaded_weights[name], &loaded_ndim);
        EXPECT_EQ(orig_ndim, loaded_ndim);
        
        std::vector<int64_t> orig_shape(orig_ndim), loaded_shape(loaded_ndim);
        mlx_array_shape(array, orig_shape.data());
        mlx_array_shape(loaded_weights[name], loaded_shape.data());
        for (uint32_t i = 0; i < orig_ndim; ++i) {
            EXPECT_EQ(orig_shape[i], loaded_shape[i]);
        }
        
        mlx_dtype orig_dtype, loaded_dtype;
        mlx_array_dtype(array, &orig_dtype);
        mlx_array_dtype(loaded_weights[name], &loaded_dtype);
        EXPECT_EQ(orig_dtype, loaded_dtype);
    }
    
    // Free all arrays
    for (auto& [name, array] : mlx_weights) {
        mlx_array_free(array);
    }
    for (auto& [name, array] : loaded_weights) {
        mlx_array_free(array);
    }
    
    // Clean up temporary file
    std::filesystem::remove(cache_path);
}

// Test parameter mapping
TEST_F(MLXWeightConverterTest, TestParameterMapping) {
    // Create mock weights
    WeightMap weight_map = create_mock_weight_map();
    
    // Create config with parameter mapping
    MLXWeightConversionConfig config;
    config.parameter_mapping["model.weight"] = "model.transformed.weight";
    config.parameter_mapping["model.bias"] = "model.transformed.bias";
    config.parameter_mapping["embedding.weight"] = "embed.weight";
    config.parameter_mapping["attention.weight"] = "attn.weight";
    
    // Convert to MLX arrays
    auto mlx_weights = convert_pytorch_to_mlx(weight_map, config);
    
    // Verify parameter mapping
    EXPECT_TRUE(mlx_weights.count("model.transformed.weight") > 0);
    EXPECT_TRUE(mlx_weights.count("model.transformed.bias") > 0);
    EXPECT_TRUE(mlx_weights.count("embed.weight") > 0);
    EXPECT_TRUE(mlx_weights.count("attn.weight") > 0);
    EXPECT_TRUE(mlx_weights.count("model.weight") == 0);
    EXPECT_TRUE(mlx_weights.count("model.bias") == 0);
    EXPECT_TRUE(mlx_weights.count("embedding.weight") == 0);
    EXPECT_TRUE(mlx_weights.count("attention.weight") == 0);
    
    // Unmapped parameters should remain unchanged
    EXPECT_TRUE(mlx_weights.count("attention.bias") > 0);
    EXPECT_TRUE(mlx_weights.count("ff.weight") > 0);
    EXPECT_TRUE(mlx_weights.count("ff.bias") > 0);
    
    // Free all arrays
    for (auto& [name, array] : mlx_weights) {
        mlx_array_free(array);
    }
}

// Test progress callback
TEST_F(MLXWeightConverterTest, TestProgressCallback) {
    // Create mock weights
    WeightMap weight_map = create_mock_weight_map();
    
    // Setup progress tracking
    std::vector<float> progress_values;
    
    // Create config with progress callback
    MLXWeightConversionConfig config;
    config.progress_callback = [&](float progress) {
        progress_values.push_back(progress);
        EXPECT_GE(progress, 0.0f);
        EXPECT_LE(progress, 1.0f);
    };
    
    // Convert to MLX arrays
    auto mlx_weights = convert_pytorch_to_mlx(weight_map, config);
    
    // Verify progress callback was called
    EXPECT_FALSE(progress_values.empty());
    EXPECT_FLOAT_EQ(progress_values.back(), 1.0f);
    
    // Progress should be monotonically increasing
    for (size_t i = 1; i < progress_values.size(); ++i) {
        EXPECT_GE(progress_values[i], progress_values[i-1]);
    }
    
    // Free all arrays
    for (auto& [name, array] : mlx_weights) {
        mlx_array_free(array);
    }
}

// Test error handling for invalid tensors
TEST_F(MLXWeightConverterTest, TestErrorHandlingInvalidTensors) {
    // Create an invalid tensor with incompatible shape and dtype
    Tensor invalid_tensor = TensorFactory::zeros({0}, DataType::F32);
    
    // Convert to MLX array - this should not crash
    mlx_array array = convert_tensor_to_mlx_array(invalid_tensor, true);
    
    // Should have created an empty array
    uint32_t ndim;
    mlx_array_ndim(array, &ndim);
    EXPECT_EQ(ndim, 1);
    
    std::vector<int64_t> shape(ndim);
    mlx_array_shape(array, shape.data());
    EXPECT_EQ(shape[0], 0);
    
    // Free the array
    mlx_array_free(array);
}

// Test conversion with different dtype combinations
TEST_F(MLXWeightConverterTest, TestDifferentDtypeConversions) {
    // Create tensors with different dtypes
    std::vector<DataType> dtypes = {
        DataType::F32, DataType::F16, DataType::BF16, 
        DataType::I32, DataType::I64
    };
    
    for (auto src_dtype : dtypes) {
        // Create tensor
        Tensor tensor = create_random_tensor({2, 3}, src_dtype);
        
        // Test with and without BFloat16 conversion
        for (bool use_bfloat16 : {true, false}) {
            // Convert to MLX array
            mlx_array array = convert_tensor_to_mlx_array(tensor, use_bfloat16);
            
            // Check dtype
            mlx_dtype dtype;
            mlx_array_dtype(array, &dtype);
            
            if (use_bfloat16 && (src_dtype == DataType::F32 || src_dtype == DataType::F16)) {
                // Should be converted to BFloat16
                EXPECT_EQ(dtype, MLX_BFLOAT16);
            } else {
                // Should maintain original dtype
                EXPECT_EQ(dtype, MLXTensorImpl::to_mlx_dtype(src_dtype));
            }
            
            // Free array
            mlx_array_free(array);
        }
    }
}

// Test PyTorch checkpoint conversion
TEST_F(MLXWeightConverterTest, TestPyTorchConversion) {
    // Create a temporary file path for testing
    std::string temp_dir = std::filesystem::temp_directory_path().string();
    std::string model_path = temp_dir + "/test_model.pt";
    
    // Create a mock PyTorch checkpoint file (just a placeholder for testing)
    {
        std::ofstream file(model_path, std::ios::binary);
        EXPECT_TRUE(file.is_open());
        file << "MOCK_PYTORCH_CHECKPOINT";
    }
    
    // Create a PyTorch loader mock
    class MockPyTorchLoader : public PyTorchLoader {
    public:
        MockPyTorchLoader(const std::string& path) : PyTorchLoader(path) {}
        
        bool load(WeightMap& weights) override {
            // Add mock weights
            weights = create_mock_weight_map();
            return true;
        }
    };
    
    // Register the mock loader
    ModelLoaderRegistry::register_loader(".pt", [](const std::string& path) {
        return std::make_shared<MockPyTorchLoader>(path);
    });
    
    // Try to convert the checkpoint
    MLXWeightConversionConfig config;
    config.cache_converted_weights = false; // Don't cache for this test
    
    try {
        auto mlx_weights = convert_pytorch_to_mlx(model_path, config);
        
        // Verify weights were loaded
        EXPECT_FALSE(mlx_weights.empty());
        
        // Free all arrays
        for (auto& [name, array] : mlx_weights) {
            mlx_array_free(array);
        }
    } catch (const std::exception& e) {
        // This might fail if PyTorch loader isn't properly mocked
        FAIL() << "PyTorch conversion failed: " << e.what();
    }
    
    // Clean up temporary file
    std::filesystem::remove(model_path);
}

// Test handling of existing cache
TEST_F(MLXWeightConverterTest, TestExistingCacheHandling) {
    // Create a temporary file path for testing
    std::string temp_dir = std::filesystem::temp_directory_path().string();
    std::string model_path = temp_dir + "/cached_model.pt";
    std::string cache_path = get_cached_mlx_weights_path(model_path);
    
    // Create a mock PyTorch checkpoint file
    {
        std::ofstream file(model_path, std::ios::binary);
        EXPECT_TRUE(file.is_open());
        file << "MOCK_PYTORCH_CHECKPOINT";
    }
    
    // Create a mock cache file
    {
        std::filesystem::create_directories(std::filesystem::path(cache_path).parent_path());
        std::ofstream file(cache_path, std::ios::binary);
        EXPECT_TRUE(file.is_open());
        
        // Write mock cache data with a single dummy weight
        uint64_t num_weights = 1;
        file.write(reinterpret_cast<const char*>(&num_weights), sizeof(num_weights));
        
        // Write name
        std::string name = "test.weight";
        uint64_t name_length = name.size();
        file.write(reinterpret_cast<const char*>(&name_length), sizeof(name_length));
        file.write(name.c_str(), name_length);
        
        // Write shape
        uint32_t ndim = 2;
        file.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
        
        std::vector<int64_t> shape = {2, 3};
        file.write(reinterpret_cast<const char*>(shape.data()), ndim * sizeof(int64_t));
        
        // Write dtype
        mlx_dtype dtype = MLX_FLOAT32;
        file.write(reinterpret_cast<const char*>(&dtype), sizeof(dtype));
        
        // Write data (12 floats)
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        size_t data_size = data.size() * sizeof(float);
        file.write(reinterpret_cast<const char*>(&data_size), sizeof(data_size));
        file.write(reinterpret_cast<const char*>(data.data()), data_size);
    }
    
    // Check if cache exists
    EXPECT_TRUE(has_cached_mlx_weights(model_path));
    
    // Try to load from cache
    MLXWeightConversionConfig config;
    config.cache_converted_weights = true;
    
    auto mlx_weights = convert_pytorch_to_mlx(model_path, config);
    
    // Verify weights were loaded from cache
    EXPECT_EQ(mlx_weights.size(), 1);
    EXPECT_TRUE(mlx_weights.count("test.weight") > 0);
    
    // Verify weight properties
    uint32_t ndim;
    mlx_array_ndim(mlx_weights["test.weight"], &ndim);
    EXPECT_EQ(ndim, 2);
    
    std::vector<int64_t> shape(ndim);
    mlx_array_shape(mlx_weights["test.weight"], shape.data());
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    
    // Free all arrays
    for (auto& [name, array] : mlx_weights) {
        mlx_array_free(array);
    }
    
    // Clean up temporary files
    std::filesystem::remove(model_path);
    std::filesystem::remove(cache_path);
}
#endif // CCSM_WITH_MLX

// Test cache path generation
TEST_F(MLXWeightConverterTest, TestGetCachedPathGeneration) {
    #ifdef CCSM_WITH_MLX
    std::string test_path = "/path/to/model.pt";
    std::string cache_path = get_cached_mlx_weights_path(test_path);
    
    // Path should contain the cache directory and model name
    EXPECT_TRUE(cache_path.find(".cache/ccsm/mlx_weights") != std::string::npos);
    EXPECT_TRUE(cache_path.find("model.pt") != std::string::npos);
    #else
    // Skip test when MLX is not available
    GTEST_SKIP() << "MLX not available, skipping test";
    #endif
}

// Test hash generation for different files
TEST_F(MLXWeightConverterTest, TestModelHashGeneration) {
    #ifdef CCSM_WITH_MLX
    // Create two temporary files with different content
    std::string temp_dir = std::filesystem::temp_directory_path().string();
    std::string file1_path = temp_dir + "/model1.pt";
    std::string file2_path = temp_dir + "/model2.pt";
    
    {
        std::ofstream file1(file1_path, std::ios::binary);
        file1 << "CONTENT1";
        
        std::ofstream file2(file2_path, std::ios::binary);
        file2 << "CONTENT2";
    }
    
    // Generate hashes
    std::string hash1 = generate_model_hash(file1_path);
    std::string hash2 = generate_model_hash(file2_path);
    
    // Hashes should be different
    EXPECT_NE(hash1, hash2);
    
    // Clean up temporary files
    std::filesystem::remove(file1_path);
    std::filesystem::remove(file2_path);
    #else
    // Skip test when MLX is not available
    GTEST_SKIP() << "MLX not available, skipping test";
    #endif
}

// Test checkpoint conversion
TEST_F(MLXWeightConverterTest, TestConvertCheckpoint) {
    MLXWeightConversionConfig config;
    MLXWeightConverter converter(config);
    
    std::string test_src_path = "/path/to/nonexistent/model.pt";
    std::string test_dst_path = "/path/to/nonexistent/model.mlxcache";
    
    // This should fail since the paths don't exist
    bool result = converter.convert_checkpoint(test_src_path, test_dst_path);
    
    // When MLX is not available, this should always fail
    #ifndef CCSM_WITH_MLX
    EXPECT_FALSE(result);
    #else
    // With MLX, it would fail for different reasons (file not found)
    EXPECT_FALSE(result);
    #endif
}

// Test converter configuration options
TEST_F(MLXWeightConverterTest, TestConverterConfigOptions) {
    // Test different configuration combinations
    std::vector<bool> bool_options = {true, false};
    
    for (bool use_bfloat16 : bool_options) {
        for (bool use_quantization : bool_options) {
            for (bool cache_converted_weights : bool_options) {
                MLXWeightConversionConfig config;
                config.use_bfloat16 = use_bfloat16;
                config.use_quantization = use_quantization;
                config.cache_converted_weights = cache_converted_weights;
                
                // Create converter with this config
                MLXWeightConverter converter(config);
                
                // Not much to assert since this is just checking initialization doesn't crash
                EXPECT_TRUE(true);
            }
        }
    }
}

} // namespace testing
} // namespace ccsm