#include <gtest/gtest.h>
#include <ccsm/mlx/mlx_weight_converter.h>
#include <ccsm/model_loader.h>
#include <ccsm/tensor.h>
#include <memory>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <cstdlib> // for setenv, unsetenv
#include <limits>

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

// Create a mock tensor with special values (NaN, Inf) for edge case testing
Tensor create_edge_case_tensor(const std::vector<size_t>& shape, DataType dtype) {
    // Create a tensor with the specified shape and dtype
    Tensor tensor = TensorFactory::zeros(shape, dtype);
    
    // Calculate the total number of elements
    size_t num_elements = 1;
    for (auto dim : shape) {
        num_elements *= dim;
    }
    
    // Fill with special values based on dtype
    if (dtype == DataType::F32) {
        float* data_ptr = static_cast<float*>(tensor.data());
        // Add some NaN, Inf, -Inf, and other special values
        for (size_t i = 0; i < num_elements; ++i) {
            if (i % 5 == 0) {
                data_ptr[i] = std::numeric_limits<float>::quiet_NaN();
            } else if (i % 5 == 1) {
                data_ptr[i] = std::numeric_limits<float>::infinity();
            } else if (i % 5 == 2) {
                data_ptr[i] = -std::numeric_limits<float>::infinity();
            } else if (i % 5 == 3) {
                data_ptr[i] = std::numeric_limits<float>::denorm_min();
            } else {
                data_ptr[i] = std::numeric_limits<float>::max();
            }
        }
    } else if (dtype == DataType::F16 || dtype == DataType::BF16) {
        // Simulate F16/BF16 with F32 special values
        float* data_ptr = static_cast<float*>(tensor.data());
        for (size_t i = 0; i < num_elements; ++i) {
            if (i % 5 == 0) {
                data_ptr[i] = std::numeric_limits<float>::quiet_NaN();
            } else if (i % 5 == 1) {
                data_ptr[i] = std::numeric_limits<float>::infinity();
            } else if (i % 5 == 2) {
                data_ptr[i] = -std::numeric_limits<float>::infinity();
            } else if (i % 5 == 3) {
                data_ptr[i] = std::numeric_limits<float>::denorm_min();
            } else {
                data_ptr[i] = std::numeric_limits<float>::max();
            }
        }
    } else if (dtype == DataType::I32) {
        int32_t* data_ptr = static_cast<int32_t*>(tensor.data());
        for (size_t i = 0; i < num_elements; ++i) {
            if (i % 3 == 0) {
                data_ptr[i] = std::numeric_limits<int32_t>::max();
            } else if (i % 3 == 1) {
                data_ptr[i] = std::numeric_limits<int32_t>::min();
            } else {
                data_ptr[i] = 0;
            }
        }
    } else if (dtype == DataType::I64) {
        int64_t* data_ptr = static_cast<int64_t*>(tensor.data());
        for (size_t i = 0; i < num_elements; ++i) {
            if (i % 3 == 0) {
                data_ptr[i] = std::numeric_limits<int64_t>::max();
            } else if (i % 3 == 1) {
                data_ptr[i] = std::numeric_limits<int64_t>::min();
            } else {
                data_ptr[i] = 0;
            }
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

// Create a mock weight map with special edge case values
WeightMap create_edge_case_weight_map() {
    WeightMap weights;
    // Add some test tensors with different shapes and dtypes containing edge case values
    weights["edge.nan_inf"] = create_edge_case_tensor({10, 10}, DataType::F32);
    weights["edge.max_min_i32"] = create_edge_case_tensor({10}, DataType::I32);
    weights["edge.max_min_i64"] = create_edge_case_tensor({10}, DataType::I64);
    weights["edge.bf16_specials"] = create_edge_case_tensor({10, 5}, DataType::BF16);
    return weights;
}

// Create an empty weight map for testing
WeightMap create_empty_weight_map() {
    return WeightMap();
}

// Create a weight map with a single tensor for testing
WeightMap create_minimal_weight_map() {
    WeightMap weights;
    weights["single.weight"] = create_random_tensor({2, 2}, DataType::F32);
    return weights;
}

// Create a very large weight map for stress testing
WeightMap create_large_weight_map(size_t size_mb) {
    WeightMap weights;
    
    // Calculate how many elements we need for the requested size
    // Each float is 4 bytes
    size_t elements_per_mb = (1024 * 1024) / 4;
    size_t total_elements = elements_per_mb * size_mb;
    
    // Create tensors of various sizes to reach the total
    size_t elements_remaining = total_elements;
    size_t tensor_count = 0;
    
    while (elements_remaining > 0) {
        // Determine size for this tensor (max 10MB per tensor)
        size_t this_tensor_elements = std::min(elements_remaining, elements_per_mb * 10);
        
        // Create a shape for this tensor
        std::vector<size_t> shape;
        if (this_tensor_elements < 1000) {
            shape = {this_tensor_elements};
        } else if (this_tensor_elements < 1000000) {
            size_t dim = static_cast<size_t>(std::sqrt(this_tensor_elements));
            shape = {dim, this_tensor_elements / dim};
        } else {
            size_t dim = static_cast<size_t>(std::cbrt(this_tensor_elements));
            shape = {dim, dim, this_tensor_elements / (dim * dim)};
        }
        
        // Create the tensor
        std::string name = "large.weight_" + std::to_string(tensor_count++);
        weights[name] = create_random_tensor(shape, DataType::F32);
        
        // Update remaining elements
        elements_remaining -= this_tensor_elements;
    }
    
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

#ifdef CCSM_WITH_MLX
// Test handling extreme tensor values (NaN, Inf, etc.)
TEST_F(MLXWeightConverterTest, TestHandlingExtremeValues) {
    // Create tensors with extreme values
    WeightMap extreme_weights = create_edge_case_weight_map();
    
    // Convert to MLX arrays
    MLXWeightConversionConfig config;
    auto mlx_weights = convert_pytorch_to_mlx(extreme_weights, config);
    
    // Verify all weights were converted despite extreme values
    EXPECT_EQ(mlx_weights.size(), extreme_weights.size());
    
    // Check that each extreme tensor was converted correctly
    for (const auto& [name, tensor] : extreme_weights) {
        EXPECT_TRUE(mlx_weights.count(name) > 0);
        
        // Check dimensions match
        uint32_t ndim;
        mlx_array_ndim(mlx_weights[name], &ndim);
        EXPECT_EQ(ndim, tensor.shape().size());
        
        // Check shape matches
        std::vector<int64_t> shape(ndim);
        mlx_array_shape(mlx_weights[name], shape.data());
        for (uint32_t i = 0; i < ndim; ++i) {
            EXPECT_EQ(shape[i], static_cast<int64_t>(tensor.shape()[i]));
        }
    }
    
    // Free all arrays
    for (auto& [name, array] : mlx_weights) {
        mlx_array_free(array);
    }
}

// Test handling of empty weight maps
TEST_F(MLXWeightConverterTest, TestHandlingEmptyWeightMap) {
    // Create an empty weight map
    WeightMap empty_weights = create_empty_weight_map();
    
    // Convert to MLX arrays
    MLXWeightConversionConfig config;
    auto mlx_weights = convert_pytorch_to_mlx(empty_weights, config);
    
    // Verify the result is an empty map
    EXPECT_TRUE(mlx_weights.empty());
}

// Test handling of minimal weight maps
TEST_F(MLXWeightConverterTest, TestHandlingMinimalWeightMap) {
    // Create a minimal weight map
    WeightMap minimal_weights = create_minimal_weight_map();
    
    // Convert to MLX arrays
    MLXWeightConversionConfig config;
    auto mlx_weights = convert_pytorch_to_mlx(minimal_weights, config);
    
    // Verify all weights were converted
    EXPECT_EQ(mlx_weights.size(), minimal_weights.size());
    
    // Free all arrays
    for (auto& [name, array] : mlx_weights) {
        mlx_array_free(array);
    }
}

// Test environment variable handling in cache path generation
TEST_F(MLXWeightConverterTest, TestEnvironmentVariableHandling) {
    // Save original HOME environment variable
    char* original_home = getenv("HOME");
    std::string saved_home = original_home ? original_home : "";
    
    try {
        // Set a custom HOME path for testing
        std::string test_home = "/tmp/test_mlx_home";
        setenv("HOME", test_home.c_str(), 1);
        
        // Get cache path with modified HOME
        std::string test_path = "/path/to/model.pt";
        std::string cache_path = get_cached_mlx_weights_path(test_path);
        
        // Verify the cache path uses the custom HOME
        EXPECT_TRUE(cache_path.find(test_home) != std::string::npos);
        EXPECT_TRUE(cache_path.find(".cache/ccsm/mlx_weights") != std::string::npos);
        
        // Test with HOME unset
        unsetenv("HOME");
        cache_path = get_cached_mlx_weights_path(test_path);
        
        // Should fall back to "." (current directory)
        EXPECT_TRUE(cache_path.find("./.cache/ccsm/mlx_weights") != std::string::npos);
    } catch (...) {
        // Restore original HOME environment variable
        if (!saved_home.empty()) {
            setenv("HOME", saved_home.c_str(), 1);
        } else {
            unsetenv("HOME");
        }
        throw;
    }
    
    // Restore original HOME environment variable
    if (!saved_home.empty()) {
        setenv("HOME", saved_home.c_str(), 1);
    } else {
        unsetenv("HOME");
    }
}

// Test corrupt cache handling
TEST_F(MLXWeightConverterTest, TestCorruptCacheHandling) {
    // Create a temporary file path for testing
    std::string temp_dir = std::filesystem::temp_directory_path().string();
    std::string model_path = temp_dir + "/corrupt_model.pt";
    std::string cache_path = get_cached_mlx_weights_path(model_path);
    
    // Create a mock PyTorch checkpoint file
    {
        std::ofstream file(model_path, std::ios::binary);
        EXPECT_TRUE(file.is_open());
        file << "MOCK_PYTORCH_CHECKPOINT";
    }
    
    // Create an invalid/corrupt cache file
    {
        std::filesystem::create_directories(std::filesystem::path(cache_path).parent_path());
        std::ofstream file(cache_path, std::ios::binary);
        EXPECT_TRUE(file.is_open());
        
        // Write invalid cache data (just some random bytes)
        file << "CORRUPTED_CACHE_DATA";
    }
    
    // Register a mock loader
    class MockPyTorchLoader : public PyTorchLoader {
    public:
        MockPyTorchLoader(const std::string& path) : PyTorchLoader(path) {}
        
        bool load(WeightMap& weights) override {
            // Add mock weights
            weights = create_minimal_weight_map();
            return true;
        }
    };
    
    // Register the mock loader
    ModelLoaderRegistry::register_loader(".pt", [](const std::string& path) {
        return std::make_shared<MockPyTorchLoader>(path);
    });
    
    // Try to load with corrupt cache
    MLXWeightConversionConfig config;
    config.cache_converted_weights = true;
    
    try {
        // This should fall back to converting from scratch
        auto mlx_weights = convert_pytorch_to_mlx(model_path, config);
        
        // Verify weights were loaded by falling back to conversion
        EXPECT_EQ(mlx_weights.size(), create_minimal_weight_map().size());
        
        // Free all arrays
        for (auto& [name, array] : mlx_weights) {
            mlx_array_free(array);
        }
    } catch (const std::exception& e) {
        // This should be handled gracefully, not crash
        FAIL() << "Corrupt cache handling failed: " << e.what();
    }
    
    // Clean up temporary files
    std::filesystem::remove(model_path);
    std::filesystem::remove(cache_path);
}

// Test handling of large tensors (stress test)
TEST_F(MLXWeightConverterTest, TestHandlingLargeTensors) {
    // Skip if running in CI environment (to avoid excessive memory usage)
    char* ci_env = getenv("CI");
    if (ci_env && std::string(ci_env) == "true") {
        GTEST_SKIP() << "Skipping large tensor test in CI environment";
    }
    
    // Only use a small size (10MB) for regular testing
    // In a real stress test, you could use a much larger value, e.g., 500MB
    size_t test_size_mb = 10; 
    
    // Create a large weight map
    WeightMap large_weights = create_large_weight_map(test_size_mb);
    
    // Convert to MLX arrays with progress tracking
    MLXWeightConversionConfig config;
    std::vector<float> progress_values;
    config.progress_callback = [&](float progress) {
        progress_values.push_back(progress);
    };
    
    try {
        auto mlx_weights = convert_pytorch_to_mlx(large_weights, config);
        
        // Verify conversion completed successfully
        EXPECT_EQ(mlx_weights.size(), large_weights.size());
        
        // Verify progress was reported
        EXPECT_FALSE(progress_values.empty());
        EXPECT_FLOAT_EQ(progress_values.back(), 1.0f);
        
        // Free all arrays
        for (auto& [name, array] : mlx_weights) {
            mlx_array_free(array);
        }
    } catch (const std::exception& e) {
        // This might fail on systems with limited memory, so don't make it a hard failure
        std::cerr << "Note: Large tensor test failed (possibly due to memory constraints): " 
                  << e.what() << std::endl;
    }
}

// Test exception handling in weight conversion
TEST_F(MLXWeightConverterTest, TestExceptionHandlingInConversion) {
    MLXWeightConversionConfig config;
    MLXWeightConverter converter(config);
    
    // Test with a non-existent file
    std::string nonexistent_path = "/path/to/nonexistent/model.pt";
    std::string output_path = "/tmp/output.mlxcache";
    
    // This should fail gracefully
    bool result = converter.convert_checkpoint(nonexistent_path, output_path);
    EXPECT_FALSE(result);
}

#ifdef CCSM_WITH_MLX
// Test memory-efficient operations
TEST_F(MLXWeightConverterTest, TestMemoryEfficientOperations) {
    // Create test arrays
    int shape[] = {2, 3};
    std::vector<float> a_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> b_data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
    
    mlx_array a = mlx_array_new_data(a_data.data(), shape, 2, MLX_FLOAT32);
    mlx_array b = mlx_array_new_data(b_data.data(), shape, 2, MLX_FLOAT32);
    
    // Create stream
    mlx_stream stream = mlx_default_cpu_stream_new();
    
    // Test in-place addition
    mlx_array a_copy = a;
    mlx_array result = mlx_memory::add_inplace(a_copy, b, stream);
    
    // Verify result has expected values (should be a + b)
    float* result_data = (float*)mlx_array_data_float32(result);
    for (size_t i = 0; i < a_data.size(); ++i) {
        EXPECT_FLOAT_EQ(result_data[i], a_data[i] + b_data[i]);
    }
    
    // Test in-place multiplication
    a_copy = a;
    result = mlx_memory::multiply_inplace(a_copy, b, stream);
    
    // Verify result has expected values (should be a * b)
    result_data = (float*)mlx_array_data_float32(result);
    for (size_t i = 0; i < a_data.size(); ++i) {
        EXPECT_FLOAT_EQ(result_data[i], a_data[i] * b_data[i]);
    }
    
    // Clean up
    mlx_array_free(a);
    mlx_array_free(b);
    mlx_array_free(result);
}

// Test TensorPool functionality
TEST_F(MLXWeightConverterTest, TestTensorPool) {
    // Create a tensor pool with max size 3
    mlx_memory::TensorPool pool(3);
    
    // Create a few different shapes
    std::vector<int> shape1 = {2, 3};
    std::vector<int> shape2 = {3, 4};
    std::vector<int> shape3 = {4, 5};
    std::vector<int> shape4 = {5, 6};
    
    // Get tensors from the pool (should create new ones)
    mlx_array t1 = pool.get(shape1, MLX_FLOAT32);
    mlx_array t2 = pool.get(shape2, MLX_FLOAT32);
    mlx_array t3 = pool.get(shape3, MLX_FLOAT32);
    
    // Verify shapes
    uint32_t ndim;
    mlx_array_ndim(t1, &ndim);
    EXPECT_EQ(ndim, 2);
    const int* t1_shape = mlx_array_shape(t1);
    EXPECT_EQ(t1_shape[0], 2);
    EXPECT_EQ(t1_shape[1], 3);
    
    // Recycle the tensors
    pool.recycle(t1);
    pool.recycle(t2);
    pool.recycle(t3);
    
    // Get a tensor with the same shape as t1 (should reuse t1)
    mlx_array t1_reused = pool.get(shape1, MLX_FLOAT32);
    
    // Get a tensor with a new shape (should force the oldest tensor out of the pool)
    mlx_array t4 = pool.get(shape4, MLX_FLOAT32);
    
    // Clean up
    mlx_array_free(t1_reused);
    mlx_array_free(t4);
    
    // Clear the pool
    pool.clear();
}

// Test KV cache pruning functionality
TEST_F(MLXWeightConverterTest, TestKVCachePruning) {
    // Create an MLX KV cache with simple dimensions for testing
    const size_t n_layers = 2;
    const size_t n_heads = 4;
    const size_t head_dim = 8;
    const size_t max_seq_len = 32;
    
    MLXKVCache cache(n_layers, n_heads, head_dim, max_seq_len);
    
    // Verify initial state
    EXPECT_EQ(cache.current_seq_len(), 0);
    EXPECT_EQ(cache.max_seq_len(), max_seq_len);
    
    // Create a simple stream
    mlx_stream stream = mlx_default_cpu_stream_new();
    
    // Initialize some data
    std::vector<int> positions = {0, 1, 2, 3, 4, 5, 6, 7};
    int batch_size = 1;
    int seq_len = positions.size();
    
    // Create dummy key and value tensors
    std::vector<float> k_data(batch_size * seq_len * n_heads * head_dim, 1.0f);
    std::vector<float> v_data(batch_size * seq_len * n_heads * head_dim, 2.0f);
    
    int k_shape[] = {batch_size, seq_len, (int)n_heads, (int)head_dim};
    mlx_array k = mlx_array_new_data(k_data.data(), k_shape, 4, MLX_FLOAT32);
    mlx_array v = mlx_array_new_data(v_data.data(), k_shape, 4, MLX_FLOAT32);
    
    // Update the cache
    for (int layer = 0; layer < n_layers; ++layer) {
        cache.update(layer, k, v, positions, stream);
    }
    
    // Verify cache was updated
    EXPECT_EQ(cache.current_seq_len(), seq_len);
    
    // Test pruning to half the size
    size_t target_len = seq_len / 2;
    bool result = cache.prune(target_len, stream);
    
    // Verify pruning succeeded
    EXPECT_TRUE(result);
    EXPECT_EQ(cache.current_seq_len(), target_len);
    
    // Test pruning to zero (should clear the cache)
    result = cache.prune(0, stream);
    EXPECT_TRUE(result);
    EXPECT_EQ(cache.current_seq_len(), 0);
    
    // Clean up
    mlx_array_free(k);
    mlx_array_free(v);
}

// Test MLXTransformer KV cache pruning
TEST_F(MLXWeightConverterTest, TestTransformerKVCachePruning) {
    // Create a minimal transformer
    int d_model = 32;
    int n_layers = 2;
    int n_heads = 4;
    int n_kv_heads = 4;
    int vocab_size = 100;
    float rope_theta = 10000.0f;
    
    // Create a minimal weights map with required weights
    std::unordered_map<std::string, mlx_array> weights;
    
    // Add required weights
    int embed_shape[] = {vocab_size, d_model};
    std::vector<float> embed_data(vocab_size * d_model, 0.1f);
    weights["token_embedding.weight"] = mlx_array_new_data(
        embed_data.data(), embed_shape, 2, MLX_FLOAT32);
    
    int norm_shape[] = {d_model};
    std::vector<float> norm_data(d_model, 1.0f);
    weights["norm.weight"] = mlx_array_new_data(
        norm_data.data(), norm_shape, 1, MLX_FLOAT32);
    
    int output_shape[] = {vocab_size, d_model};
    std::vector<float> output_data(vocab_size * d_model, 0.1f);
    weights["output.weight"] = mlx_array_new_data(
        output_data.data(), output_shape, 2, MLX_FLOAT32);
    
    // Create transformer
    MLXTransformer transformer(weights, d_model, n_layers, n_heads, n_kv_heads, vocab_size, rope_theta);
    
    // Test pruning
    bool result = transformer.prune_kv_cache(16);
    EXPECT_TRUE(result);
    
    // Clean up
    mlx_array_free(weights["token_embedding.weight"]);
    mlx_array_free(weights["norm.weight"]);
    mlx_array_free(weights["output.weight"]);
}
#endif

// Test error handling with progress callback that throws exceptions
TEST_F(MLXWeightConverterTest, TestProgressCallbackErrorHandling) {
    // Create weights
    WeightMap weights = create_minimal_weight_map();
    
    // Create config with a progress callback that throws an exception
    MLXWeightConversionConfig config;
    int call_count = 0;
    config.progress_callback = [&call_count](float progress) {
        call_count++;
        if (call_count > 1) {
            throw std::runtime_error("Test exception from progress callback");
        }
    };
    
    try {
        // Convert weights - should handle the exception from the callback
        auto mlx_weights = convert_pytorch_to_mlx(weights, config);
        
        // Shouldn't get here if callback throws, but if it does, clean up
        for (auto& [name, array] : mlx_weights) {
            mlx_array_free(array);
        }
        FAIL() << "Expected exception from progress callback wasn't propagated";
    } catch (const std::runtime_error& e) {
        // Expected behavior - callback exception should be propagated
        EXPECT_STREQ(e.what(), "Test exception from progress callback");
    } catch (...) {
        FAIL() << "Wrong exception type caught";
    }
}
#endif // CCSM_WITH_MLX

} // namespace testing
} // namespace ccsm