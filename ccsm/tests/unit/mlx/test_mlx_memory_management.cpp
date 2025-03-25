#include <gtest/gtest.h>
#include <ccsm/mlx/mlx_tensor.h>
#include <ccsm/tensor.h>
#include <vector>
#include <random>
#include <thread>
#include <chrono>

namespace ccsm {
namespace testing {

class MLXMemoryManagementTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
    
    // Helper to create a random tensor with given shape
    Tensor create_random_tensor(const std::vector<size_t>& shape, DataType dtype = DataType::F32) {
        Tensor tensor = TensorFactory::zeros(shape, dtype);
        
        // Calculate total number of elements
        size_t total_elements = 1;
        for (auto dim : shape) {
            total_elements *= dim;
        }
        
        // Fill with random values
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        
        if (dtype == DataType::F32) {
            float* data = static_cast<float*>(tensor.data());
            for (size_t i = 0; i < total_elements; ++i) {
                data[i] = dist(gen);
            }
        }
        
        return tensor;
    }
};

#ifdef CCSM_WITH_MLX
// Create a MLX memory manager utility class
class MLXMemoryManager {
public:
    MLXMemoryManager() {
        // Initialize MLX device
        device_ = mlx_default_device();
    }
    
    ~MLXMemoryManager() {
        // Free all allocated tensors
        clear();
    }
    
    // Allocate a new tensor
    mlx_array allocate(const std::vector<int>& shape, mlx_dtype dtype = MLX_FLOAT32) {
        mlx_array array;
        mlx_array_zeros(shape.data(), shape.size(), dtype, &array);
        allocated_arrays_.push_back(array);
        return array;
    }
    
    // Free a tensor
    void free(mlx_array array) {
        for (auto it = allocated_arrays_.begin(); it != allocated_arrays_.end(); ++it) {
            if (mlx_array_equal(*it, array)) {
                mlx_array_free(*it);
                allocated_arrays_.erase(it);
                return;
            }
        }
    }
    
    // Clear all allocated tensors
    void clear() {
        for (auto& array : allocated_arrays_) {
            mlx_array_free(array);
        }
        allocated_arrays_.clear();
    }
    
    // Get current memory used
    size_t get_memory_used() {
        mlx_device_memory_info info;
        mlx_device_memory(device_, &info);
        return info.used;
    }
    
    // Get available memory
    size_t get_memory_available() {
        mlx_device_memory_info info;
        mlx_device_memory(device_, &info);
        return info.total - info.used;
    }
    
    // Synchronize the device
    void synchronize() {
        mlx_device_synchronize(device_);
    }
    
private:
    mlx_device device_;
    std::vector<mlx_array> allocated_arrays_;
};

// Test basic memory allocation and usage
TEST_F(MLXMemoryManagementTest, TestMemoryAllocation) {
    MLXMemoryManager memory_manager;
    
    // Get initial memory usage
    size_t initial_memory = memory_manager.get_memory_used();
    
    // Allocate some tensors
    std::vector<mlx_array> arrays;
    for (int i = 0; i < 5; ++i) {
        std::vector<int> shape = {100, 100};
        arrays.push_back(memory_manager.allocate(shape));
    }
    
    // Memory usage should have increased
    size_t after_allocation = memory_manager.get_memory_used();
    EXPECT_GT(after_allocation, initial_memory);
    
    // Free some tensors
    for (int i = 0; i < 3; ++i) {
        memory_manager.free(arrays[i]);
    }
    arrays.erase(arrays.begin(), arrays.begin() + 3);
    
    // Synchronize to ensure memory is freed
    memory_manager.synchronize();
    
    // Memory usage should have decreased
    size_t after_free = memory_manager.get_memory_used();
    EXPECT_LT(after_free, after_allocation);
    
    // Clear all tensors
    memory_manager.clear();
    memory_manager.synchronize();
    
    // Memory usage should be back to close to initial
    size_t final_memory = memory_manager.get_memory_used();
    EXPECT_NEAR(final_memory, initial_memory, initial_memory * 0.1); // Allow for some overhead
}

// Test memory limit and out of memory handling
TEST_F(MLXMemoryManagementTest, TestMemoryLimits) {
    MLXMemoryManager memory_manager;
    
    // Get available memory
    size_t available_memory = memory_manager.get_memory_available();
    
    // Try to allocate a reasonable amount of memory
    std::vector<mlx_array> arrays;
    size_t allocated_memory = 0;
    size_t chunk_size = 10 * 1024 * 1024; // 10MB chunks
    
    try {
        // Allocate up to 50% of available memory in chunks
        while (allocated_memory < available_memory * 0.5) {
            int dim = static_cast<int>(std::sqrt(chunk_size / sizeof(float)));
            std::vector<int> shape = {dim, dim};
            arrays.push_back(memory_manager.allocate(shape));
            allocated_memory += chunk_size;
        }
        
        // Should reach here without OOM
        EXPECT_GT(arrays.size(), 0);
        
        // Clean up
        memory_manager.clear();
    } catch (const std::exception& e) {
        // If we got an OOM exception, that's okay, just clean up
        memory_manager.clear();
        std::cerr << "OOM exception: " << e.what() << std::endl;
        // This isn't a test failure, as some systems might have very limited memory
    }
}

// Test tensor pool for memory reuse
class MLXTensorPool {
public:
    MLXTensorPool(size_t max_size = 10) : max_size_(max_size) {}
    
    ~MLXTensorPool() {
        clear();
    }
    
    // Get a tensor from the pool or create a new one
    mlx_array get(const std::vector<int>& shape, mlx_dtype dtype = MLX_FLOAT32) {
        // Check if we have a compatible tensor in the pool
        for (auto it = pool_.begin(); it != pool_.end(); ++it) {
            if (is_compatible(*it, shape, dtype)) {
                mlx_array result = *it;
                pool_.erase(it);
                return result;
            }
        }
        
        // No compatible tensor found, create a new one
        mlx_array result;
        mlx_array_zeros(shape.data(), shape.size(), dtype, &result);
        return result;
    }
    
    // Return a tensor to the pool
    void recycle(mlx_array array) {
        // Add to pool if we have space
        if (pool_.size() < max_size_) {
            pool_.push_back(array);
        } else {
            // Otherwise free the oldest tensor and add this one
            mlx_array_free(pool_.front());
            pool_.erase(pool_.begin());
            pool_.push_back(array);
        }
    }
    
    // Clear the pool
    void clear() {
        for (auto& array : pool_) {
            mlx_array_free(array);
        }
        pool_.clear();
    }
    
    // Get current pool size
    size_t size() const {
        return pool_.size();
    }
    
private:
    std::vector<mlx_array> pool_;
    size_t max_size_;
    
    // Check if an array is compatible with the requested shape and dtype
    bool is_compatible(mlx_array array, const std::vector<int>& shape, mlx_dtype dtype) {
        uint32_t ndim;
        mlx_array_ndim(array, &ndim);
        if (ndim != shape.size()) {
            return false;
        }
        
        mlx_dtype array_dtype;
        mlx_array_dtype(array, &array_dtype);
        if (array_dtype != dtype) {
            return false;
        }
        
        const int* array_shape = mlx_array_shape(array);
        for (uint32_t i = 0; i < ndim; ++i) {
            if (array_shape[i] != shape[i]) {
                return false;
            }
        }
        
        return true;
    }
};

// Test tensor pool functionality
TEST_F(MLXMemoryManagementTest, TestTensorPool) {
    MLXTensorPool pool(3);
    
    // Get tensors from the pool (should create new ones)
    std::vector<int> shape1 = {10, 10};
    std::vector<int> shape2 = {20, 20};
    std::vector<int> shape3 = {30, 30};
    
    mlx_array t1 = pool.get(shape1);
    mlx_array t2 = pool.get(shape2);
    mlx_array t3 = pool.get(shape3);
    
    // Pool should be empty
    EXPECT_EQ(pool.size(), 0);
    
    // Recycle tensors
    pool.recycle(t1);
    pool.recycle(t2);
    pool.recycle(t3);
    
    // Pool should have 3 tensors
    EXPECT_EQ(pool.size(), 3);
    
    // Get a tensor with the same shape as t1 (should reuse t1)
    mlx_array t1_reused = pool.get(shape1);
    
    // Pool should have 2 tensors
    EXPECT_EQ(pool.size(), 2);
    
    // Verify t1_reused has the expected shape
    uint32_t ndim;
    mlx_array_ndim(t1_reused, &ndim);
    EXPECT_EQ(ndim, 2);
    
    const int* shape = mlx_array_shape(t1_reused);
    EXPECT_EQ(shape[0], 10);
    EXPECT_EQ(shape[1], 10);
    
    // Add t1_reused back to the pool
    pool.recycle(t1_reused);
    
    // Add more tensors to exceed pool capacity
    std::vector<int> shape4 = {40, 40};
    std::vector<int> shape5 = {50, 50};
    
    mlx_array t4 = pool.get(shape4);
    mlx_array t5 = pool.get(shape5);
    
    pool.recycle(t4);
    pool.recycle(t5);
    
    // Pool should still have 3 tensors (max capacity)
    EXPECT_EQ(pool.size(), 3);
    
    // Clean up
    pool.clear();
    EXPECT_EQ(pool.size(), 0);
}

// Test memory-efficient operations
TEST_F(MLXMemoryManagementTest, TestMemoryEfficientOperations) {
    MLXMemoryManager memory_manager;
    MLXTensorPool tensor_pool(5);
    
    // Create a stream
    mlx_stream stream = mlx_default_cpu_stream_new();
    
    // Allocate some tensors
    std::vector<int> shape = {100, 100};
    mlx_array a = tensor_pool.get(shape);
    mlx_array b = tensor_pool.get(shape);
    
    // Fill with some data
    float* a_data = static_cast<float*>(mlx_array_data_float32(a));
    float* b_data = static_cast<float*>(mlx_array_data_float32(b));
    
    for (int i = 0; i < 10000; ++i) {
        a_data[i] = static_cast<float>(i) * 0.1f;
        b_data[i] = static_cast<float>(i) * 0.2f;
    }
    
    // Record initial memory
    size_t initial_memory = memory_manager.get_memory_used();
    
    // In-place addition (should be more memory efficient)
    // Create a copy of 'a' to use for in-place operation
    mlx_array a_copy = a;
    mlx_array c = mlx_array_add(a_copy, b, stream);
    
    // Memory usage should be relatively small (just one extra tensor)
    size_t after_op_memory = memory_manager.get_memory_used();
    EXPECT_LT(after_op_memory - initial_memory, 10 * 1024 * 1024); // Less than 10MB increase
    
    // Recycle tensors
    tensor_pool.recycle(c);
    
    // Multiple operations in sequence
    std::vector<mlx_array> intermediates;
    for (int i = 0; i < 5; ++i) {
        // Get a tensor for intermediate result
        mlx_array intermediate = tensor_pool.get(shape);
        
        // Fill with data
        float* data = static_cast<float*>(mlx_array_data_float32(intermediate));
        for (int j = 0; j < 10000; ++j) {
            data[j] = static_cast<float>(j) * (i + 1) * 0.1f;
        }
        
        intermediates.push_back(intermediate);
    }
    
    // Create complex operation (a + b * c - d / e)
    mlx_array result = mlx_array_add(
        a,
        mlx_array_subtract(
            mlx_array_multiply(b, intermediates[0], stream),
            mlx_array_divide(intermediates[1], intermediates[2], stream),
            stream
        ),
        stream
    );
    
    // Recycle all tensors
    for (auto& t : intermediates) {
        tensor_pool.recycle(t);
    }
    tensor_pool.recycle(result);
    
    // Clean up
    mlx_array_free(a);
    mlx_array_free(b);
    tensor_pool.clear();
}

// Test memory optimization strategies
TEST_F(MLXMemoryManagementTest, TestMemoryOptimizationStrategies) {
    MLXMemoryManager memory_manager;
    
    // Create a stream
    mlx_stream stream = mlx_default_cpu_stream_new();
    
    // Initial memory usage
    size_t initial_memory = memory_manager.get_memory_used();
    
    // Create baseline tensors
    std::vector<int> shape = {100, 100};
    mlx_array a = memory_manager.allocate(shape);
    mlx_array b = memory_manager.allocate(shape);
    
    // Fill with data
    float* a_data = static_cast<float*>(mlx_array_data_float32(a));
    float* b_data = static_cast<float*>(mlx_array_data_float32(b));
    
    for (int i = 0; i < 10000; ++i) {
        a_data[i] = static_cast<float>(i) * 0.1f;
        b_data[i] = static_cast<float>(i) * 0.2f;
    }
    
    // Memory usage should have increased
    size_t after_allocation = memory_manager.get_memory_used();
    EXPECT_GT(after_allocation, initial_memory);
    
    // Test strategy 1: Reuse intermediate tensors
    // First approach - creates many intermediates
    std::vector<mlx_array> results1;
    for (int i = 0; i < 10; ++i) {
        mlx_array result = mlx_array_add(a, b, stream);
        results1.push_back(result);
    }
    
    size_t after_strategy1 = memory_manager.get_memory_used();
    
    // Clean up strategy 1 results
    for (auto& result : results1) {
        memory_manager.free(result);
    }
    results1.clear();
    memory_manager.synchronize();
    
    // Test strategy 2: In-place operations where possible
    std::vector<mlx_array> results2;
    mlx_array base = memory_manager.allocate(shape);
    
    // Copy data from 'a' to 'base'
    mlx_array_copy(base, a, stream);
    
    for (int i = 0; i < 10; ++i) {
        // Use base as the output for each iteration
        mlx_array_copy(base, mlx_array_add(base, b, stream), stream);
        
        // Create a copy for the result
        mlx_array result;
        mlx_array_copy(result, base, stream);
        results2.push_back(result);
    }
    
    size_t after_strategy2 = memory_manager.get_memory_used();
    
    // Strategy 2 should use less memory than strategy 1
    EXPECT_LT(after_strategy2, after_strategy1);
    
    // Clean up strategy 2 results
    memory_manager.free(base);
    for (auto& result : results2) {
        memory_manager.free(result);
    }
    results2.clear();
    memory_manager.synchronize();
    
    // Test strategy 3: Multi-stream pipeline
    mlx_stream stream1 = mlx_default_cpu_stream_new();
    mlx_stream stream2 = mlx_default_cpu_stream_new();
    
    // Use multiple streams for parallelization
    mlx_array c = memory_manager.allocate(shape);
    mlx_array d = memory_manager.allocate(shape);
    
    // Operation 1 on stream1
    mlx_array_copy(c, mlx_array_add(a, b, stream1), stream1);
    
    // Operation 2 on stream2
    mlx_array_copy(d, mlx_array_multiply(a, b, stream2), stream2);
    
    // Wait for both streams
    memory_manager.synchronize();
    
    // Final operation combining both results
    mlx_array final_result = mlx_array_add(c, d, stream);
    
    // Clean up
    memory_manager.free(c);
    memory_manager.free(d);
    memory_manager.free(final_result);
    memory_manager.synchronize();
    
    // Final memory usage should be close to initial
    size_t final_memory = memory_manager.get_memory_used();
    EXPECT_NEAR(final_memory, after_allocation, after_allocation * 0.1);
    
    // Clean up everything
    memory_manager.clear();
}

// Test memory fragmentation and defragmentation
TEST_F(MLXMemoryManagementTest, TestMemoryFragmentation) {
    MLXMemoryManager memory_manager;
    
    // Initial memory usage
    size_t initial_memory = memory_manager.get_memory_used();
    
    // Create tensors of various sizes to potentially cause fragmentation
    std::vector<mlx_array> arrays;
    std::vector<size_t> sizes = {10, 100, 1000, 10000, 100000}; // Elements
    
    for (auto size : sizes) {
        std::vector<int> shape = {static_cast<int>(size)};
        arrays.push_back(memory_manager.allocate(shape));
    }
    
    // Free some tensors in the middle
    memory_manager.free(arrays[1]);
    memory_manager.free(arrays[3]);
    arrays.erase(arrays.begin() + 3);
    arrays.erase(arrays.begin() + 1);
    
    // Synchronize to ensure memory is freed
    memory_manager.synchronize();
    
    // Allocate new tensors that might fit in the freed spaces
    std::vector<int> shape1 = {150}; // Should fit in the space of the freed 100-element tensor
    std::vector<int> shape2 = {50000}; // Should fit in the space of the freed 10000-element tensor
    
    mlx_array new_array1 = memory_manager.allocate(shape1);
    mlx_array new_array2 = memory_manager.allocate(shape2);
    
    // Clean up all tensors
    memory_manager.clear();
    memory_manager.synchronize();
    
    // Final memory usage should be close to initial
    size_t final_memory = memory_manager.get_memory_used();
    EXPECT_NEAR(final_memory, initial_memory, initial_memory * 0.1);
}

// Test device-specific memory optimizations
TEST_F(MLXMemoryManagementTest, TestDeviceSpecificOptimizations) {
    MLXMemoryManager memory_manager;
    MLXDevice device;
    
    // Skip test if not on Apple Silicon
    if (!device.is_metal_supported()) {
        GTEST_SKIP() << "Test requires Metal support";
    }
    
    // Initial memory usage
    size_t initial_memory = memory_manager.get_memory_used();
    
    // Create tensors with different dtypes
    std::vector<int> shape = {100, 100};
    mlx_array f32_tensor = memory_manager.allocate(shape, MLX_FLOAT32);
    mlx_array f16_tensor = memory_manager.allocate(shape, MLX_FLOAT16);
    mlx_array bf16_tensor = memory_manager.allocate(shape, MLX_BFLOAT16);
    
    // Lower precision should use less memory
    size_t f32_size = mlx_array_nbytes(f32_tensor);
    size_t f16_size = mlx_array_nbytes(f16_tensor);
    size_t bf16_size = mlx_array_nbytes(bf16_tensor);
    
    EXPECT_GT(f32_size, f16_size);
    EXPECT_GT(f32_size, bf16_size);
    
    // Clean up tensors
    memory_manager.clear();
    memory_manager.synchronize();
    
    // Final memory usage should be close to initial
    size_t final_memory = memory_manager.get_memory_used();
    EXPECT_NEAR(final_memory, initial_memory, initial_memory * 0.1);
}

// Test memory bandwidth optimization
TEST_F(MLXMemoryManagementTest, TestMemoryBandwidthOptimization) {
    MLXMemoryManager memory_manager;
    
    // Create a stream
    mlx_stream stream = mlx_default_cpu_stream_new();
    
    // Create large tensors to test bandwidth
    std::vector<int> shape = {1000, 1000}; // 1M elements
    mlx_array a = memory_manager.allocate(shape);
    mlx_array b = memory_manager.allocate(shape);
    
    // Fill with data
    float* a_data = static_cast<float*>(mlx_array_data_float32(a));
    float* b_data = static_cast<float*>(mlx_array_data_float32(b));
    
    for (int i = 0; i < 1000000; ++i) {
        a_data[i] = static_cast<float>(i % 1000) * 0.1f;
        b_data[i] = static_cast<float>(i % 1000) * 0.2f;
    }
    
    // Measure time for traditional operations
    auto start_time = std::chrono::high_resolution_clock::now();
    
    mlx_array c1 = mlx_array_add(a, b, stream);
    mlx_array d1 = mlx_array_multiply(a, b, stream);
    mlx_array e1 = mlx_array_add(c1, d1, stream);
    
    // Synchronize to ensure operations are complete
    memory_manager.synchronize();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto traditional_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    
    // Free intermediate results
    memory_manager.free(c1);
    memory_manager.free(d1);
    memory_manager.free(e1);
    
    // Measure time for fused operations (simulated here)
    start_time = std::chrono::high_resolution_clock::now();
    
    // Simulated fused operation: e2 = a + b + a * b
    mlx_array temp = mlx_array_multiply(a, b, stream);
    mlx_array e2 = mlx_array_add(mlx_array_add(a, b, stream), temp, stream);
    memory_manager.free(temp);
    
    // Synchronize to ensure operations are complete
    memory_manager.synchronize();
    
    end_time = std::chrono::high_resolution_clock::now();
    auto fused_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    
    // Free final result
    memory_manager.free(e2);
    
    // Fused operations should be faster, but we're not making a strict assertion
    // because performance depends on many factors
    std::cout << "Traditional operation time: " << traditional_duration << " us" << std::endl;
    std::cout << "Fused operation time: " << fused_duration << " us" << std::endl;
    
    // Clean up
    memory_manager.clear();
}
#endif // CCSM_WITH_MLX

// Test MLXDevice utility functions (these should work regardless of MLX availability)
TEST_F(MLXMemoryManagementTest, TestMLXDeviceUtilities) {
    // This test should work with or without MLX
    MLXDevice device;
    
    // Test device API
    bool is_available = device.is_available();
    bool is_metal_supported = device.is_metal_supported();
    
    // Test device info
    std::string device_info = device.get_device_info();
    EXPECT_FALSE(device_info.empty());
    
    // Test device name
    std::string device_name = device.get_device_name();
    EXPECT_FALSE(device_name.empty());
    
    // Test memory info - this should work even if MLX is not available
    size_t total_memory = device.get_total_memory();
    size_t used_memory = device.get_memory_used();
    
    // These values should be reasonable
    EXPECT_GT(total_memory, 0);
    EXPECT_GE(used_memory, 0);
    EXPECT_LE(used_memory, total_memory);
}

} // namespace testing
} // namespace ccsm