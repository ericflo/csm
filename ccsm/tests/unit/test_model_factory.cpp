#include <gtest/gtest.h>
#include <ccsm/model.h>
#include <ccsm/cpu/ggml_model.h>
#include <memory>
#include <string>

namespace ccsm {
namespace testing {

// Test fixture for ModelFactory tests
class ModelFactoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up any test data here
    }
    
    void TearDown() override {
        // Clean up test data here
    }
    
    // Helper function to create a model config
    ModelConfig create_test_config() {
        ModelConfig config;
        config.name = "test_model";
        config.d_model = 512;
        config.n_heads = 8;
        config.n_layers = 6;
        config.max_seq_len = 512;
        config.vocab_size = 32000;
        config.audio_vocab_size = 1024;
        config.n_audio_layers = 4;
        return config;
    }
};

// Test that CPU backend is always available
TEST_F(ModelFactoryTest, CPUBackendIsAvailable) {
    EXPECT_TRUE(ModelFactory::is_backend_available("cpu"));
    EXPECT_TRUE(ModelFactory::is_backend_available("ggml"));
}

// Test that unavailable backends are correctly identified
TEST_F(ModelFactoryTest, UnavailableBackends) {
    EXPECT_FALSE(ModelFactory::is_backend_available("nonexistent_backend"));
}

// Test MLX backend availability matches compilation status
TEST_F(ModelFactoryTest, MLXBackendAvailability) {
    #ifdef CCSM_WITH_MLX
    EXPECT_TRUE(ModelFactory::is_backend_available("mlx"));
    #else
    EXPECT_FALSE(ModelFactory::is_backend_available("mlx"));
    #endif
}

// Test that get_available_backends returns a non-empty list
TEST_F(ModelFactoryTest, GetAvailableBackends) {
    auto backends = ModelFactory::get_available_backends();
    EXPECT_FALSE(backends.empty());
    EXPECT_TRUE(std::find(backends.begin(), backends.end(), "cpu") != backends.end());
}

// Test creating a CPU model
TEST_F(ModelFactoryTest, CreateCPUModel) {
    auto config = create_test_config();
    auto model = ModelFactory::create("cpu", config);
    
    EXPECT_TRUE(model != nullptr);
    EXPECT_EQ(model->config().name, "test_model");
    
    // Verify we have a GGMLModel
    auto ggml_model = std::dynamic_pointer_cast<GGMLModel>(model);
    EXPECT_TRUE(ggml_model != nullptr);
}

// Test creating a GGML model
TEST_F(ModelFactoryTest, CreateGGMLModel) {
    auto config = create_test_config();
    auto model = ModelFactory::create("ggml", config);
    
    EXPECT_TRUE(model != nullptr);
    EXPECT_EQ(model->config().name, "test_model");
    
    // Verify we have a GGMLModel
    auto ggml_model = std::dynamic_pointer_cast<GGMLModel>(model);
    EXPECT_TRUE(ggml_model != nullptr);
}

// Test that creating a model with an unavailable backend throws
TEST_F(ModelFactoryTest, CreateUnavailableModel) {
    auto config = create_test_config();
    EXPECT_THROW(ModelFactory::create("nonexistent_backend", config), std::runtime_error);
}

// Test creating an MLX model (conditionally compiled)
TEST_F(ModelFactoryTest, CreateMLXModel) {
    #ifdef CCSM_WITH_MLX
    auto config = create_test_config();
    auto model = ModelFactory::create("mlx", config);
    
    EXPECT_TRUE(model != nullptr);
    EXPECT_EQ(model->config().name, "test_model");
    
    // Verify we have an MLXModel
    auto mlx_model = std::dynamic_pointer_cast<MLXModel>(model);
    EXPECT_TRUE(mlx_model != nullptr);
    #else
    GTEST_SKIP() << "MLX support not compiled in, skipping test";
    #endif
}

} // namespace testing
} // namespace ccsm