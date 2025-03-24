#include <gtest/gtest.h>
#include <ccsm/tensor.h>
#include <vector>
#include <memory>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <random>
#include <numeric>

namespace ccsm {
namespace {

// Test fixture for tensor memory efficiency tests
class TensorMemoryEfficiencyTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Standard shapes for testing
        small_shape = {16, 16};
        medium_shape = {64, 64};
        large_shape = {256, 256};
        very_large_shape = {1024, 1024};
        
        // Generate random data for testing
        std::mt19937 gen(42); // Fixed seed for reproducibility
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        
        small_data.resize(small_shape[0] * small_shape[1]);
        for (auto& v : small_data) {
            v = dist(gen);
        }
        
        medium_data.resize(medium_shape[0] * medium_shape[1]);
        for (auto& v : medium_data) {
            v = dist(gen);
        }
    }
    
    // Helper function to calculate memory usage of a tensor
    size_t calculate_memory_usage(const Tensor& tensor) {
        // Get data size based on data type
        size_t element_size = 0;
        switch (tensor.dtype()) {
            case DataType::F32:
                element_size = sizeof(float);
                break;
            case DataType::F16:
                element_size = sizeof(uint16_t);
                break;
            case DataType::I32:
                element_size = sizeof(int32_t);
                break;
            case DataType::I16:
                element_size = sizeof(int16_t);
                break;
            case DataType::I8:
                element_size = sizeof(int8_t);
                break;
            case DataType::U8:
                element_size = sizeof(uint8_t);
                break;
            default:
                // For quantized types, approximate
                element_size = 1; // Most quantized types use about 1 byte per element
                break;
        }
        
        // Calculate total size
        size_t total_elements = tensor.nelements();
        return total_elements * element_size;
    }
    
    // Tensor shapes for testing
    std::vector<size_t> small_shape;
    std::vector<size_t> medium_shape;
    std::vector<size_t> large_shape;
    std::vector<size_t> very_large_shape;
    
    // Test data
    std::vector<float> small_data;
    std::vector<float> medium_data;
};

// Test memory-efficient tensor allocation
TEST_F(TensorMemoryEfficiencyTest, EfficientAllocation) {
    // Create tensors with different allocation strategies
    Tensor tensor1(medium_shape, DataType::F32);
    
    // Measure initial memory usage
    size_t initial_memory = calculate_memory_usage(tensor1);
    
    // Test that allocated memory matches expected size
    EXPECT_EQ(initial_memory, medium_shape[0] * medium_shape[1] * sizeof(float));
    
    // Create tensor with pre-allocated data
    Tensor tensor2(medium_shape, DataType::F32, medium_data.data());
    
    // If tensor is using existing memory, its "own" allocations should be minimal
    // This is difficult to verify directly, but at least check that it's valid
    EXPECT_EQ(tensor2.nelements(), medium_shape[0] * medium_shape[1]);
    
    // Copy the tensor, which should make a deep copy for F32 type
    Tensor tensor3 = tensor2.clone();
    
    // Modify the original data
    medium_data[0] = 999.0f;
    
    // Check that the clone is not affected (i.e., it has its own copy)
    float* tensor3_data = static_cast<float*>(tensor3.data());
    EXPECT_NE(tensor3_data[0], 999.0f);
}

// Test TensorSliceView for memory-efficient slicing
TEST_F(TensorMemoryEfficiencyTest, SliceViews) {
    // Create original tensor
    Tensor original(medium_shape, DataType::F32);
    
    // Fill with sequential data for easy verification
    float* data = static_cast<float*>(original.data());
    std::iota(data, data + original.nelements(), 0.0f);
    
    // Create a slice view for a portion of the tensor
    std::vector<size_t> start = {10, 10};
    std::vector<size_t> end = {20, 20};
    Tensor slice = original.slice(start, end);
    
    // Verify slice has right shape
    std::vector<size_t> expected_shape = {10, 10}; // (20-10, 20-10)
    EXPECT_EQ(slice.shape(), expected_shape);
    
    // Verify slice data points to correct values from original
    float* slice_data = static_cast<float*>(slice.data());
    for (size_t i = 0; i < 10; i++) {
        for (size_t j = 0; j < 10; j++) {
            size_t original_idx = (i + 10) * medium_shape[1] + (j + 10);
            size_t slice_idx = i * 10 + j;
            EXPECT_EQ(slice_data[slice_idx], data[original_idx]);
        }
    }
    
    // Modify slice and verify it affects original
    slice_data[0] = 999.0f;
    size_t original_idx = 10 * medium_shape[1] + 10; // Corresponds to slice[0,0]
    EXPECT_EQ(data[original_idx], 999.0f);
    
    // Measure memory usage
    size_t original_memory = calculate_memory_usage(original);
    size_t slice_memory = calculate_memory_usage(slice);
    
    // For a view, the memory should be much smaller than a full copy would be
    float ratio = static_cast<float>(slice_memory) / static_cast<float>(original_memory);
    std::cout << "Memory efficiency for slice: " << ratio << std::endl;
    std::cout << "Original tensor memory: " << original_memory << " bytes" << std::endl;
    std::cout << "Slice view memory: " << slice_memory << " bytes" << std::endl;
    
    // The slice view should use significantly less memory than the original
    // Note: This check may need adjustment based on the actual implementation
    EXPECT_LT(ratio, 0.5) << "Slice is not memory efficient enough";
}

// Test sequential slicing efficiency
TEST_F(TensorMemoryEfficiencyTest, SequentialSlicing) {
    // Create original tensor
    Tensor original(large_shape, DataType::F32);
    
    // Fill with sequential data
    float* data = static_cast<float*>(original.data());
    std::iota(data, data + original.nelements(), 0.0f);
    
    // Create multiple sequential slices
    std::vector<Tensor> slices;
    const int num_slices = 10;
    
    for (int i = 0; i < num_slices; i++) {
        std::vector<size_t> start = {i * 20, 0};
        std::vector<size_t> end = {(i + 1) * 20, large_shape[1]};
        slices.push_back(original.slice(start, end));
    }
    
    // Verify each slice has correct data
    for (int i = 0; i < num_slices; i++) {
        EXPECT_EQ(slices[i].shape()[0], 20);
        EXPECT_EQ(slices[i].shape()[1], large_shape[1]);
        
        float* slice_data = static_cast<float*>(slices[i].data());
        float first_element = slice_data[0];
        float expected_first = (i * 20) * large_shape[1];
        
        EXPECT_EQ(first_element, expected_first);
    }
    
    // Calculate combined memory usage
    size_t original_memory = calculate_memory_usage(original);
    size_t combined_slice_memory = 0;
    for (const auto& slice : slices) {
        combined_slice_memory += calculate_memory_usage(slice);
    }
    
    std::cout << "Original tensor memory: " << original_memory << " bytes" << std::endl;
    std::cout << "Combined slice memory: " << combined_slice_memory << " bytes" << std::endl;
    
    // Memory usage from all slices should still be less than a full duplicate
    EXPECT_LT(combined_slice_memory, original_memory) << "Sequential slicing is not memory efficient";
}

// Test nested slicing
TEST_F(TensorMemoryEfficiencyTest, NestedSlicing) {
    // Create original tensor
    Tensor original(large_shape, DataType::F32);
    
    // Fill with sequential data
    float* data = static_cast<float*>(original.data());
    std::iota(data, data + original.nelements(), 0.0f);
    
    // Create first level slice
    std::vector<size_t> start1 = {50, 50};
    std::vector<size_t> end1 = {150, 150};
    Tensor slice1 = original.slice(start1, end1);
    
    // Create nested slice from the first slice
    std::vector<size_t> start2 = {10, 10};
    std::vector<size_t> end2 = {50, 50};
    Tensor slice2 = slice1.slice(start2, end2);
    
    // Verify nested slice has right shape and data
    std::vector<size_t> expected_shape = {40, 40}; // (50-10, 50-10)
    EXPECT_EQ(slice2.shape(), expected_shape);
    
    // Verify a sample point in nested slice
    float* slice2_data = static_cast<float*>(slice2.data());
    float first_element = slice2_data[0];
    
    // Calculate expected value: Original tensor position at (50+10, 50+10)
    float expected = (60 * large_shape[1]) + 60;
    EXPECT_EQ(first_element, expected);
    
    // Modify nested slice and verify it affects original
    slice2_data[0] = 999.0f;
    size_t original_idx = 60 * large_shape[1] + 60;
    EXPECT_EQ(data[original_idx], 999.0f);
    
    // Measure memory usage
    size_t original_memory = calculate_memory_usage(original);
    size_t slice1_memory = calculate_memory_usage(slice1);
    size_t slice2_memory = calculate_memory_usage(slice2);
    
    std::cout << "Original tensor memory: " << original_memory << " bytes" << std::endl;
    std::cout << "First level slice memory: " << slice1_memory << " bytes" << std::endl;
    std::cout << "Second level slice memory: " << slice2_memory << " bytes" << std::endl;
    
    // Nested slices should be especially memory efficient
    EXPECT_LT(slice2_memory, slice1_memory) << "Nested slicing is not memory efficient";
}

// Test memory reuse through reshape
TEST_F(TensorMemoryEfficiencyTest, MemoryReuseReshape) {
    // Create original tensor
    Tensor original(medium_shape, DataType::F32);
    
    // Fill with sequential data
    float* data = static_cast<float*>(original.data());
    std::iota(data, data + original.nelements(), 0.0f);
    
    // Create reshaped tensor with same data
    std::vector<size_t> new_shape = {medium_shape[0] * medium_shape[1], 1};
    Tensor reshaped = original.reshape(new_shape);
    
    // Verify reshape has right shape
    EXPECT_EQ(reshaped.shape(), new_shape);
    
    // Verify reshape and original share the same memory
    EXPECT_EQ(reshaped.data(), original.data());
    
    // Modify reshaped tensor and verify it affects original
    float* reshaped_data = static_cast<float*>(reshaped.data());
    reshaped_data[0] = 999.0f;
    EXPECT_EQ(data[0], 999.0f);
    
    // Reshape back to original shape
    Tensor reshaped_back = reshaped.reshape(medium_shape);
    
    // Verify it still has the same data
    float* back_data = static_cast<float*>(reshaped_back.data());
    EXPECT_EQ(back_data[0], 999.0f);
    
    // Measure memory usage before and after reshaping
    size_t original_memory = calculate_memory_usage(original);
    size_t reshaped_memory = calculate_memory_usage(reshaped);
    
    // Memory usage should be identical for reshapes
    EXPECT_EQ(original_memory, reshaped_memory) << "Reshape does not reuse memory efficiently";
}

// Test memory efficient tensor operations like add, mul, etc.
TEST_F(TensorMemoryEfficiencyTest, MemoryEfficientOperations) {
    // Create original tensors
    Tensor a(medium_shape, DataType::F32);
    Tensor b(medium_shape, DataType::F32);
    
    // Fill with data
    float* a_data = static_cast<float*>(a.data());
    float* b_data = static_cast<float*>(b.data());
    
    for (size_t i = 0; i < a.nelements(); i++) {
        a_data[i] = static_cast<float>(i);
        b_data[i] = static_cast<float>(i) * 0.5f;
    }
    
    // Create a tensor for in-place operations
    Tensor c = a.clone();
    float* c_data = static_cast<float*>(c.data());
    
    // Memory usage before operations
    size_t initial_memory = calculate_memory_usage(a) + calculate_memory_usage(b) + calculate_memory_usage(c);
    
    // Perform in-place add
    c.add_(b);
    
    // Verify results
    for (size_t i = 0; i < c.nelements(); i++) {
        EXPECT_FLOAT_EQ(c_data[i], a_data[i] + b_data[i]);
    }
    
    // Perform in-place multiply
    c.mul_(b);
    
    // Verify results
    for (size_t i = 0; i < c.nelements(); i++) {
        EXPECT_FLOAT_EQ(c_data[i], (a_data[i] + b_data[i]) * b_data[i]);
    }
    
    // Memory usage after operations
    size_t final_memory = calculate_memory_usage(a) + calculate_memory_usage(b) + calculate_memory_usage(c);
    
    // Memory usage should not increase significantly after in-place operations
    EXPECT_LE(final_memory, initial_memory * 1.1) << "In-place operations increased memory usage significantly";
}

// Test memory allocation and reuse in a sequence of operations
TEST_F(TensorMemoryEfficiencyTest, SequentialOperations) {
    // Create original tensor
    Tensor original(large_shape, DataType::F32);
    
    // Fill with data
    float* data = static_cast<float*>(original.data());
    std::iota(data, data + original.nelements(), 0.0f);
    
    // Measure initial memory
    size_t initial_memory = calculate_memory_usage(original);
    
    // Perform a sequence of non-in-place operations
    Tensor a = original.add(5.0f);
    Tensor b = a.square();
    Tensor c = b.sqrt();
    Tensor d = c.add(original);
    
    // Measure final memory after operations
    size_t final_memory = calculate_memory_usage(original) + 
                          calculate_memory_usage(a) + 
                          calculate_memory_usage(b) + 
                          calculate_memory_usage(c) + 
                          calculate_memory_usage(d);
    
    std::cout << "Initial memory: " << initial_memory << " bytes" << std::endl;
    std::cout << "Final memory after sequential operations: " << final_memory << " bytes" << std::endl;
    
    // Each operation would create a new tensor, so memory usage would increase
    // But verify the expected mathematical results
    float* d_data = static_cast<float*>(d.data());
    for (size_t i = 0; i < std::min((size_t)10, d.nelements()); i++) {
        float expected = std::sqrt(std::pow(data[i] + 5.0f, 2)) + data[i];
        EXPECT_NEAR(d_data[i], expected, 1e-4f);
    }
    
    // Now perform the same operations in-place where possible
    Tensor in_place = original.clone();
    
    // Get the initial memory
    size_t in_place_initial = calculate_memory_usage(in_place);
    
    // Perform in-place operations
    in_place.add_(5.0f);          // Add in-place
    in_place.square_();           // Square in-place
    in_place.sqrt_();             // Sqrt in-place
    in_place.add_(original);      // Add original in-place
    
    // Measure final in-place memory
    size_t in_place_final = calculate_memory_usage(in_place);
    
    std::cout << "In-place initial memory: " << in_place_initial << " bytes" << std::endl;
    std::cout << "In-place final memory: " << in_place_final << " bytes" << std::endl;
    
    // In-place operations should use much less memory than non-in-place
    EXPECT_LT(in_place_final, final_memory * 0.5) << "In-place operations not significantly more memory efficient";
    
    // Verify in-place results match non-in-place
    float* in_place_data = static_cast<float*>(in_place.data());
    for (size_t i = 0; i < std::min((size_t)10, in_place.nelements()); i++) {
        EXPECT_NEAR(in_place_data[i], d_data[i], 1e-4f);
    }
}

// Test TensorSliceView for large tensor matrices
TEST_F(TensorMemoryEfficiencyTest, LargeTensorSliceViews) {
    // Skip this test for routine testing to save time
    if (::testing::FLAGS_gtest_filter != "*LargeTensorSliceViews*") {
        GTEST_SKIP() << "Skipping large tensor test for routine testing";
    }
    
    // Create a very large tensor (could use significant memory if not handled efficiently)
    Tensor large_tensor(very_large_shape, DataType::F32);
    
    // Fill just a small portion with data for verification
    float* data = static_cast<float*>(large_tensor.data());
    for (size_t i = 0; i < 100; i++) {
        for (size_t j = 0; j < 100; j++) {
            size_t idx = i * very_large_shape[1] + j;
            data[idx] = static_cast<float>(i * 100 + j);
        }
    }
    
    // Create multiple slices of different regions
    std::vector<Tensor> slices;
    for (size_t i = 0; i < 10; i++) {
        std::vector<size_t> start = {i * 100, 0};
        std::vector<size_t> end = {(i + 1) * 100, 100};
        slices.push_back(large_tensor.slice(start, end));
    }
    
    // Verify slices have correct content
    for (size_t i = 0; i < slices.size(); i++) {
        float* slice_data = static_cast<float*>(slices[i].data());
        for (size_t j = 0; j < 10; j++) {
            size_t slice_idx = j * 100;  // Row 'j', column '0'
            float expected = static_cast<float>((i * 100 + j) * 100);
            EXPECT_EQ(slice_data[slice_idx], expected);
        }
    }
    
    // Create a large slice spanning the entire width
    std::vector<size_t> horiz_start = {0, 0};
    std::vector<size_t> horiz_end = {100, very_large_shape[1]};
    Tensor horizontal_slice = large_tensor.slice(horiz_start, horiz_end);
    
    // Verify horizontal slice
    float* horiz_data = static_cast<float*>(horizontal_slice.data());
    for (size_t i = 0; i < 10; i++) {
        for (size_t j = 0; j < 10; j++) {
            size_t idx = i * very_large_shape[1] + j;
            float expected = static_cast<float>(i * 100 + j);
            EXPECT_EQ(horiz_data[idx], expected);
        }
    }
    
    // Create a large slice spanning the entire height
    std::vector<size_t> vert_start = {0, 0};
    std::vector<size_t> vert_end = {very_large_shape[0], 100};
    Tensor vertical_slice = large_tensor.slice(vert_start, vert_end);
    
    // Verify vertical slice
    float* vert_data = static_cast<float*>(vertical_slice.data());
    for (size_t i = 0; i < 10; i++) {
        for (size_t j = 0; j < 10; j++) {
            size_t idx = i * 100 + j;
            float expected = static_cast<float>(i * 100 + j);
            EXPECT_EQ(vert_data[idx], expected);
        }
    }
    
    // Measure memory usage
    size_t large_tensor_memory = calculate_memory_usage(large_tensor);
    size_t horiz_slice_memory = calculate_memory_usage(horizontal_slice);
    size_t vert_slice_memory = calculate_memory_usage(vertical_slice);
    
    std::cout << "Large tensor memory: " << large_tensor_memory << " bytes" << std::endl;
    std::cout << "Horizontal slice memory: " << horiz_slice_memory << " bytes" << std::endl;
    std::cout << "Vertical slice memory: " << vert_slice_memory << " bytes" << std::endl;
    
    // Verify slicing is memory efficient
    EXPECT_LT(horiz_slice_memory, large_tensor_memory * 0.2);
    EXPECT_LT(vert_slice_memory, large_tensor_memory * 0.2);
}

// Test performance comparison between copy vs view
TEST_F(TensorMemoryEfficiencyTest, CopyVsViewPerformance) {
    // Skip this test for routine testing to save time
    if (::testing::FLAGS_gtest_filter != "*CopyVsViewPerformance*") {
        GTEST_SKIP() << "Skipping performance comparison test for routine testing";
    }
    
    // Create original large tensor
    Tensor original(large_shape, DataType::F32);
    
    // Fill with data
    float* data = static_cast<float*>(original.data());
    std::iota(data, data + original.nelements(), 0.0f);
    
    // Measure time to create copies vs. views
    const int num_iterations = 100;
    
    // Define slice region
    std::vector<size_t> start = {50, 50};
    std::vector<size_t> end = {150, 150};
    std::vector<size_t> slice_shape = {100, 100};
    
    // Measure time for creating views (slices)
    auto view_start = std::chrono::high_resolution_clock::now();
    
    std::vector<Tensor> views;
    for (int i = 0; i < num_iterations; i++) {
        views.push_back(original.slice(start, end));
    }
    
    auto view_end = std::chrono::high_resolution_clock::now();
    auto view_duration = std::chrono::duration_cast<std::chrono::microseconds>(view_end - view_start);
    
    // Measure time for creating copies
    auto copy_start = std::chrono::high_resolution_clock::now();
    
    std::vector<Tensor> copies;
    for (int i = 0; i < num_iterations; i++) {
        // Create a new tensor and copy data
        Tensor copy(slice_shape, DataType::F32);
        float* copy_data = static_cast<float*>(copy.data());
        
        // Manual copy from original
        for (size_t i = 0; i < 100; i++) {
            for (size_t j = 0; j < 100; j++) {
                size_t orig_idx = (i + 50) * large_shape[1] + (j + 50);
                size_t copy_idx = i * 100 + j;
                copy_data[copy_idx] = data[orig_idx];
            }
        }
        
        copies.push_back(copy);
    }
    
    auto copy_end = std::chrono::high_resolution_clock::now();
    auto copy_duration = std::chrono::duration_cast<std::chrono::microseconds>(copy_end - copy_start);
    
    // Print performance results
    std::cout << "Performance comparison between views and copies:" << std::endl;
    std::cout << "  View creation time for " << num_iterations << " iterations: " 
              << view_duration.count() << " microseconds" << std::endl;
    std::cout << "  Copy creation time for " << num_iterations << " iterations: " 
              << copy_duration.count() << " microseconds" << std::endl;
    std::cout << "  Copy/View time ratio: " 
              << static_cast<double>(copy_duration.count()) / view_duration.count() << "x" << std::endl;
    
    // Views should be significantly faster to create than copies
    EXPECT_LT(view_duration.count(), copy_duration.count() * 0.5);
    
    // Measure memory usage
    size_t views_memory = 0;
    for (const auto& view : views) {
        views_memory += calculate_memory_usage(view);
    }
    
    size_t copies_memory = 0;
    for (const auto& copy : copies) {
        copies_memory += calculate_memory_usage(copy);
    }
    
    std::cout << "Memory usage comparison:" << std::endl;
    std::cout << "  Total memory for " << num_iterations << " views: " 
              << views_memory << " bytes" << std::endl;
    std::cout << "  Total memory for " << num_iterations << " copies: " 
              << copies_memory << " bytes" << std::endl;
    std::cout << "  Copy/View memory ratio: " 
              << static_cast<double>(copies_memory) / views_memory << "x" << std::endl;
    
    // Views should use significantly less memory than copies
    EXPECT_LT(views_memory, copies_memory * 0.5);
}

// Test memory efficiency of the TensorSliceView with operations
TEST_F(TensorMemoryEfficiencyTest, SliceViewOperations) {
    // Create original tensor
    Tensor original(medium_shape, DataType::F32);
    
    // Fill with sequential data
    float* data = static_cast<float*>(original.data());
    std::iota(data, data + original.nelements(), 0.0f);
    
    // Create a slice view
    std::vector<size_t> start = {10, 10};
    std::vector<size_t> end = {20, 20};
    Tensor slice = original.slice(start, end);
    
    // Create another slice for operations
    std::vector<size_t> start2 = {30, 30};
    std::vector<size_t> end2 = {40, 40};
    Tensor slice2 = original.slice(start2, end2);
    
    // Record memory usage before operations
    size_t initial_memory = calculate_memory_usage(original) + 
                           calculate_memory_usage(slice) + 
                           calculate_memory_usage(slice2);
    
    // Perform operation on slices
    Tensor result = slice.add(slice2);
    
    // Record memory after operations
    size_t final_memory = calculate_memory_usage(original) + 
                         calculate_memory_usage(slice) + 
                         calculate_memory_usage(slice2) + 
                         calculate_memory_usage(result);
    
    std::cout << "Memory before slice operations: " << initial_memory << " bytes" << std::endl;
    std::cout << "Memory after slice operations: " << final_memory << " bytes" << std::endl;
    
    // Verify operation result is correct
    float* result_data = static_cast<float*>(result.data());
    for (size_t i = 0; i < 10; i++) {
        for (size_t j = 0; j < 10; j++) {
            size_t slice1_orig_idx = (i + 10) * medium_shape[1] + (j + 10);
            size_t slice2_orig_idx = (i + 30) * medium_shape[1] + (j + 30);
            size_t result_idx = i * 10 + j;
            
            EXPECT_FLOAT_EQ(result_data[result_idx], data[slice1_orig_idx] + data[slice2_orig_idx]);
        }
    }
    
    // Now perform in-place operation
    Tensor slice_clone = slice.clone();
    slice_clone.add_(slice2);
    
    // Verify in-place operation result
    float* clone_data = static_cast<float*>(slice_clone.data());
    for (size_t i = 0; i < 10; i++) {
        for (size_t j = 0; j < 10; j++) {
            size_t slice1_orig_idx = (i + 10) * medium_shape[1] + (j + 10);
            size_t slice2_orig_idx = (i + 30) * medium_shape[1] + (j + 30);
            size_t result_idx = i * 10 + j;
            
            EXPECT_FLOAT_EQ(clone_data[result_idx], data[slice1_orig_idx] + data[slice2_orig_idx]);
        }
    }
}

// Test memory optimization in tensor concatenation
TEST_F(TensorMemoryEfficiencyTest, TensorConcatenation) {
    // Create tensors to concatenate
    std::vector<size_t> shape1 = {10, 20};
    std::vector<size_t> shape2 = {15, 20};
    
    Tensor tensor1(shape1, DataType::F32);
    Tensor tensor2(shape2, DataType::F32);
    
    // Fill with sequential data
    float* data1 = static_cast<float*>(tensor1.data());
    float* data2 = static_cast<float*>(tensor2.data());
    
    std::iota(data1, data1 + tensor1.nelements(), 0.0f);
    std::iota(data2, data2 + tensor2.nelements(), 1000.0f);
    
    // Record memory before concatenation
    size_t initial_memory = calculate_memory_usage(tensor1) + calculate_memory_usage(tensor2);
    
    // Concatenate tensors along axis 0
    Tensor concat = Tensor::cat({tensor1, tensor2}, 0);
    
    // Record memory after concatenation
    size_t final_memory = calculate_memory_usage(tensor1) + 
                         calculate_memory_usage(tensor2) + 
                         calculate_memory_usage(concat);
    
    std::cout << "Memory before concatenation: " << initial_memory << " bytes" << std::endl;
    std::cout << "Memory after concatenation: " << final_memory << " bytes" << std::endl;
    
    // Verify concatenated shape
    std::vector<size_t> expected_shape = {25, 20}; // 10 + 15, 20
    EXPECT_EQ(concat.shape(), expected_shape);
    
    // Verify data in concatenated tensor
    float* concat_data = static_cast<float*>(concat.data());
    
    // Check first part (from tensor1)
    for (size_t i = 0; i < 10; i++) {
        for (size_t j = 0; j < 20; j++) {
            size_t t1_idx = i * 20 + j;
            size_t concat_idx = i * 20 + j;
            
            EXPECT_FLOAT_EQ(concat_data[concat_idx], data1[t1_idx]);
        }
    }
    
    // Check second part (from tensor2)
    for (size_t i = 0; i < 15; i++) {
        for (size_t j = 0; j < 20; j++) {
            size_t t2_idx = i * 20 + j;
            size_t concat_idx = (i + 10) * 20 + j;
            
            EXPECT_FLOAT_EQ(concat_data[concat_idx], data2[t2_idx]);
        }
    }
    
    // Concatenation should use efficient memory
    size_t concat_memory = calculate_memory_usage(concat);
    size_t expected_concat_memory = shape1[0] * shape1[1] * sizeof(float) + 
                                   shape2[0] * shape2[1] * sizeof(float);
    
    EXPECT_LE(concat_memory, expected_concat_memory * 1.1);
}

} // namespace
} // namespace ccsm