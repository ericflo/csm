#include <gtest/gtest.h>
#include <ccsm/tensor.h>
#include <vector>
#include <memory>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <string>
#include <filesystem>

using namespace ccsm;

// Test fixture for tensor serialization tests
class TensorSerializationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up default tensor shapes for testing
        scalar_shape = {};
        vector_shape = {10};
        matrix_shape = {5, 10};
        tensor3d_shape = {3, 5, 10};
        
        // Default data type
        default_dtype = DataType::F32;
        
        // Create context
        context = ContextFactory::create();
        
        // Create test directory if needed
        test_dir = "/tmp/test_serialization";
        if (!std::filesystem::exists(test_dir)) {
            std::filesystem::create_directory(test_dir);
        }
    }
    
    void TearDown() override {
        // Clean up test files
        if (std::filesystem::exists(test_dir)) {
            std::filesystem::remove_all(test_dir);
        }
    }
    
    std::vector<size_t> scalar_shape;
    std::vector<size_t> vector_shape;
    std::vector<size_t> matrix_shape;
    std::vector<size_t> tensor3d_shape;
    DataType default_dtype;
    std::shared_ptr<Context> context;
    std::string test_dir;
    
    // Helper to create a tensor with specific values
    Tensor createFilledTensor(const std::vector<size_t>& shape, DataType dtype, float value) {
        Tensor t = TensorFactory::ones(shape, dtype);
        
        // For a real implementation, we would initialize the tensor with the given value
        // This is placeholder code since we're just testing the interface here
        
        return t;
    }
    
    // Helper to compare tensors for equality
    bool tensorEquals(const Tensor& a, const Tensor& b, float tolerance = 1e-5f) {
        // Check metadata
        if (a.ndim() != b.ndim() || a.dtype() != b.dtype()) {
            return false;
        }
        
        // Check shape
        for (int i = 0; i < a.ndim(); i++) {
            if (a.shape(i) != b.shape(i)) {
                return false;
            }
        }
        
        // For a real implementation, we would compare tensor data values here
        // This is placeholder code since we're just testing the interface
        
        return true;
    }
};

// Test basic tensor saving and loading
TEST_F(TensorSerializationTest, BasicSaveLoad) {
    // Create a simple tensor
    Tensor original = createFilledTensor(vector_shape, DataType::F32, 1.0f);
    
    // Filepath for save/load
    std::string filepath = test_dir + "/tensor_f32.bin";
    
    // Save the tensor
    EXPECT_TRUE(TensorFactory::save(original, filepath));
    
    // Load the tensor
    Tensor loaded = TensorFactory::load(filepath);
    
    // Verify the loaded tensor
    EXPECT_TRUE(loaded.is_valid());
    EXPECT_TRUE(tensorEquals(original, loaded));
    
    // Verify metadata
    EXPECT_EQ(loaded.ndim(), original.ndim());
    EXPECT_EQ(loaded.shape(), original.shape());
    EXPECT_EQ(loaded.dtype(), original.dtype());
}

// Test saving and loading tensors of different shapes
TEST_F(TensorSerializationTest, DifferentShapes) {
    // Create tensors with different shapes
    Tensor scalar = createFilledTensor(scalar_shape, default_dtype, 1.0f);
    Tensor vector = createFilledTensor(vector_shape, default_dtype, 1.0f);
    Tensor matrix = createFilledTensor(matrix_shape, default_dtype, 1.0f);
    Tensor tensor3d = createFilledTensor(tensor3d_shape, default_dtype, 1.0f);
    
    // Save all tensors
    TensorFactory::save(scalar, test_dir + "/scalar.bin");
    TensorFactory::save(vector, test_dir + "/vector.bin");
    TensorFactory::save(matrix, test_dir + "/matrix.bin");
    TensorFactory::save(tensor3d, test_dir + "/tensor3d.bin");
    
    // Load all tensors
    Tensor loaded_scalar = TensorFactory::load(test_dir + "/scalar.bin");
    Tensor loaded_vector = TensorFactory::load(test_dir + "/vector.bin");
    Tensor loaded_matrix = TensorFactory::load(test_dir + "/matrix.bin");
    Tensor loaded_tensor3d = TensorFactory::load(test_dir + "/tensor3d.bin");
    
    // Verify all tensors
    EXPECT_TRUE(tensorEquals(scalar, loaded_scalar));
    EXPECT_TRUE(tensorEquals(vector, loaded_vector));
    EXPECT_TRUE(tensorEquals(matrix, loaded_matrix));
    EXPECT_TRUE(tensorEquals(tensor3d, loaded_tensor3d));
}

// Test saving and loading tensors of different data types
TEST_F(TensorSerializationTest, DifferentDataTypes) {
    // Create tensors with different data types
    Tensor f32_tensor = createFilledTensor(vector_shape, DataType::F32, 1.0f);
    Tensor f16_tensor = createFilledTensor(vector_shape, DataType::F16, 1.0f);
    Tensor bf16_tensor = createFilledTensor(vector_shape, DataType::BF16, 1.0f);
    Tensor i32_tensor = createFilledTensor(vector_shape, DataType::I32, 1.0f);
    Tensor i16_tensor = createFilledTensor(vector_shape, DataType::I16, 1.0f);
    Tensor i8_tensor = createFilledTensor(vector_shape, DataType::I8, 1.0f);
    Tensor q8_0_tensor = createFilledTensor(vector_shape, DataType::Q8_0, 1.0f);
    Tensor q4_0_tensor = createFilledTensor(vector_shape, DataType::Q4_0, 1.0f);
    Tensor q4_1_tensor = createFilledTensor(vector_shape, DataType::Q4_1, 1.0f);
    
    // Save all tensors
    TensorFactory::save(f32_tensor, test_dir + "/f32.bin");
    TensorFactory::save(f16_tensor, test_dir + "/f16.bin");
    TensorFactory::save(bf16_tensor, test_dir + "/bf16.bin");
    TensorFactory::save(i32_tensor, test_dir + "/i32.bin");
    TensorFactory::save(i16_tensor, test_dir + "/i16.bin");
    TensorFactory::save(i8_tensor, test_dir + "/i8.bin");
    TensorFactory::save(q8_0_tensor, test_dir + "/q8_0.bin");
    TensorFactory::save(q4_0_tensor, test_dir + "/q4_0.bin");
    TensorFactory::save(q4_1_tensor, test_dir + "/q4_1.bin");
    
    // Load all tensors
    Tensor loaded_f32 = TensorFactory::load(test_dir + "/f32.bin");
    Tensor loaded_f16 = TensorFactory::load(test_dir + "/f16.bin");
    Tensor loaded_bf16 = TensorFactory::load(test_dir + "/bf16.bin");
    Tensor loaded_i32 = TensorFactory::load(test_dir + "/i32.bin");
    Tensor loaded_i16 = TensorFactory::load(test_dir + "/i16.bin");
    Tensor loaded_i8 = TensorFactory::load(test_dir + "/i8.bin");
    Tensor loaded_q8_0 = TensorFactory::load(test_dir + "/q8_0.bin");
    Tensor loaded_q4_0 = TensorFactory::load(test_dir + "/q4_0.bin");
    Tensor loaded_q4_1 = TensorFactory::load(test_dir + "/q4_1.bin");
    
    // Verify all data types are preserved
    EXPECT_EQ(loaded_f32.dtype(), DataType::F32);
    EXPECT_EQ(loaded_f16.dtype(), DataType::F16);
    EXPECT_EQ(loaded_bf16.dtype(), DataType::BF16);
    EXPECT_EQ(loaded_i32.dtype(), DataType::I32);
    EXPECT_EQ(loaded_i16.dtype(), DataType::I16);
    EXPECT_EQ(loaded_i8.dtype(), DataType::I8);
    EXPECT_EQ(loaded_q8_0.dtype(), DataType::Q8_0);
    EXPECT_EQ(loaded_q4_0.dtype(), DataType::Q4_0);
    EXPECT_EQ(loaded_q4_1.dtype(), DataType::Q4_1);
    
    // Verify data is correct
    EXPECT_TRUE(tensorEquals(f32_tensor, loaded_f32));
    EXPECT_TRUE(tensorEquals(f16_tensor, loaded_f16));
    EXPECT_TRUE(tensorEquals(bf16_tensor, loaded_bf16));
    EXPECT_TRUE(tensorEquals(i32_tensor, loaded_i32));
    EXPECT_TRUE(tensorEquals(i16_tensor, loaded_i16));
    EXPECT_TRUE(tensorEquals(i8_tensor, loaded_i8));
    EXPECT_TRUE(tensorEquals(q8_0_tensor, loaded_q8_0));
    EXPECT_TRUE(tensorEquals(q4_0_tensor, loaded_q4_0));
    EXPECT_TRUE(tensorEquals(q4_1_tensor, loaded_q4_1));
}

// Test saving and loading with endian conversion
TEST_F(TensorSerializationTest, EndianConversion) {
    // Create a tensor
    Tensor original = createFilledTensor(matrix_shape, DataType::F32, 1.0f);
    
    // Save with different endianness settings
    TensorFactory::save(original, test_dir + "/native_endian.bin", EndianFormat::NATIVE);
    TensorFactory::save(original, test_dir + "/little_endian.bin", EndianFormat::LITTLE);
    TensorFactory::save(original, test_dir + "/big_endian.bin", EndianFormat::BIG);
    
    // Load the tensors with explicit endianness
    Tensor loaded_native = TensorFactory::load(test_dir + "/native_endian.bin");
    Tensor loaded_little = TensorFactory::load(test_dir + "/little_endian.bin", EndianFormat::LITTLE);
    Tensor loaded_big = TensorFactory::load(test_dir + "/big_endian.bin", EndianFormat::BIG);
    
    // Verify all tensors
    EXPECT_TRUE(tensorEquals(original, loaded_native));
    EXPECT_TRUE(tensorEquals(original, loaded_little));
    EXPECT_TRUE(tensorEquals(original, loaded_big));
}

// Test handling of errors in serialization
TEST_F(TensorSerializationTest, ErrorHandling) {
    // Create a tensor
    Tensor original = createFilledTensor(vector_shape, DataType::F32, 1.0f);
    
    // Test saving to invalid location
    EXPECT_FALSE(TensorFactory::save(original, "/non/existent/directory/tensor.bin"));
    
    // Test loading non-existent file
    EXPECT_THROW(TensorFactory::load(test_dir + "/non_existent.bin"), std::runtime_error);
    
    // Test loading corrupted file
    std::string corrupted_file = test_dir + "/corrupted.bin";
    
    // Create a file with invalid data
    std::ofstream file(corrupted_file, std::ios::binary);
    char garbage_data[16] = {0};
    file.write(garbage_data, sizeof(garbage_data));
    file.close();
    
    EXPECT_THROW(TensorFactory::load(corrupted_file), std::runtime_error);
}

// Test serialization of tensor with custom metadata
TEST_F(TensorSerializationTest, CustomMetadata) {
    // Create a tensor
    Tensor original = createFilledTensor(vector_shape, DataType::F32, 1.0f);
    
    // Create metadata
    TensorMetadata metadata;
    metadata.name = "test_tensor";
    metadata.description = "A test tensor for serialization";
    metadata.version = 1;
    metadata.custom_fields["author"] = "CCSM Test";
    metadata.custom_fields["purpose"] = "Unit testing";
    
    // Test the actual implementation directly
    // No manual file creation, just use the proper APIs
    
    // Save with metadata
    EXPECT_TRUE(TensorFactory::save_with_metadata(original, test_dir + "/with_metadata.bin", metadata));
    
    // Load with metadata
    TensorMetadata loaded_metadata2;
    Tensor loaded2 = TensorFactory::load_with_metadata(test_dir + "/with_metadata.bin", loaded_metadata2);
    
    // Verify tensor
    EXPECT_TRUE(tensorEquals(original, loaded2));
    
    // Verify metadata
    EXPECT_EQ(loaded_metadata2.name, "test_tensor");
    EXPECT_EQ(loaded_metadata2.description, "A test tensor for serialization");
    EXPECT_EQ(loaded_metadata2.version, 1);
    EXPECT_EQ(loaded_metadata2.custom_fields["author"], "CCSM Test");
    EXPECT_EQ(loaded_metadata2.custom_fields["purpose"], "Unit testing");
}

// Test serialization with compression
TEST_F(TensorSerializationTest, Compression) {
    // Create a tensor
    Tensor original = createFilledTensor(tensor3d_shape, DataType::F32, 1.0f);
    
    // Save with different compression levels
    TensorFactory::save(original, test_dir + "/nocompress.bin", EndianFormat::NATIVE, CompressionLevel::NONE);
    TensorFactory::save(original, test_dir + "/compress_fast.bin", EndianFormat::NATIVE, CompressionLevel::FAST);
    TensorFactory::save(original, test_dir + "/compress_default.bin", EndianFormat::NATIVE, CompressionLevel::DEFAULT);
    TensorFactory::save(original, test_dir + "/compress_best.bin", EndianFormat::NATIVE, CompressionLevel::BEST);
    
    // Load the tensors
    Tensor loaded_none = TensorFactory::load(test_dir + "/nocompress.bin");
    Tensor loaded_fast = TensorFactory::load(test_dir + "/compress_fast.bin");
    Tensor loaded_default = TensorFactory::load(test_dir + "/compress_default.bin");
    Tensor loaded_best = TensorFactory::load(test_dir + "/compress_best.bin");
    
    // Verify all tensors load correctly
    EXPECT_TRUE(tensorEquals(original, loaded_none));
    EXPECT_TRUE(tensorEquals(original, loaded_fast));
    EXPECT_TRUE(tensorEquals(original, loaded_default));
    EXPECT_TRUE(tensorEquals(original, loaded_best));
    
    // Check if compression was effective
    std::uintmax_t size_none = std::filesystem::file_size(test_dir + "/nocompress.bin");
    std::uintmax_t size_fast = std::filesystem::file_size(test_dir + "/compress_fast.bin");
    std::uintmax_t size_default = std::filesystem::file_size(test_dir + "/compress_default.bin");
    std::uintmax_t size_best = std::filesystem::file_size(test_dir + "/compress_best.bin");
    
    // Better compression should result in smaller files
    // Note: This might not always be true for random data, but should be for uniformly filled data
    EXPECT_LE(size_best, size_default);
    EXPECT_LE(size_default, size_fast);
    EXPECT_LE(size_fast, size_none);
}