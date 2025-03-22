#include <gtest/gtest.h>
#include <ccsm/tensor.h>
#include <ccsm/cpu/ggml_tensor.h>
#include <vector>
#include <memory>
#include <cmath>
#include <stdexcept>

using namespace ccsm;

// Test fixture for tensor tests - this uses a mock tensor implementation
// that we can use for testing the interface without relying on a specific backend
class MockTensorImpl : public TensorImpl {
public:
    MockTensorImpl(const std::vector<size_t>& shape, DataType dtype) 
        : shape_(shape), dtype_(dtype) {
        // Calculate size
        size_ = 1;
        for (auto dim : shape) {
            size_ *= dim;
        }
        
        // Allocate data (float32 for simplicity)
        data_.resize(size_ * sizeof(float));
    }
    
    size_t shape(int dim) const override {
        if (dim < 0 || dim >= static_cast<int>(shape_.size())) {
            throw std::out_of_range("Dimension index out of range");
        }
        return shape_[dim];
    }
    
    std::vector<size_t> shape() const override {
        return shape_;
    }
    
    int ndim() const override {
        return static_cast<int>(shape_.size());
    }
    
    size_t size() const override {
        return size_;
    }
    
    DataType dtype() const override {
        return dtype_;
    }
    
    void* data() override {
        return data_.data();
    }
    
    const void* data() const override {
        return data_.data();
    }
    
    std::shared_ptr<TensorImpl> reshape(const std::vector<size_t>& new_shape) const override {
        // Calculate new size
        size_t new_size = 1;
        for (auto dim : new_shape) {
            new_size *= dim;
        }
        
        // Check if sizes match
        if (new_size != size_) {
            throw std::runtime_error("Cannot reshape tensor: size mismatch");
        }
        
        // Create new tensor with same data
        auto result = std::make_shared<MockTensorImpl>(new_shape, dtype_);
        std::memcpy(result->data_.data(), data_.data(), data_.size());
        return result;
    }
    
    std::shared_ptr<TensorImpl> view(const std::vector<size_t>& new_shape) const override {
        // For simplicity, view is the same as reshape in this mock
        return reshape(new_shape);
    }
    
    std::shared_ptr<TensorImpl> slice(int dim, size_t start, size_t end) const override {
        if (dim < 0 || dim >= static_cast<int>(shape_.size())) {
            throw std::out_of_range("Dimension index out of range");
        }
        
        if (start >= shape_[dim] || end > shape_[dim] || start >= end) {
            throw std::runtime_error("Invalid slice range");
        }
        
        // Calculate new shape
        std::vector<size_t> new_shape = shape_;
        new_shape[dim] = end - start;
        
        // Create new tensor
        auto result = std::make_shared<MockTensorImpl>(new_shape, dtype_);
        
        // For simplicity, just set all values to 1.0f
        float* result_data = static_cast<float*>(result->data());
        for (size_t i = 0; i < result->size(); i++) {
            result_data[i] = 1.0f;
        }
        
        return result;
    }
    
    void print(const std::string& name = "") const override {
        std::cout << "Mock Tensor";
        if (!name.empty()) {
            std::cout << " " << name;
        }
        std::cout << ": shape=[";
        
        for (size_t i = 0; i < shape_.size(); i++) {
            std::cout << shape_[i];
            if (i < shape_.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "], dtype=" << static_cast<int>(dtype_) << std::endl;
    }
    
private:
    std::vector<size_t> shape_;
    DataType dtype_;
    size_t size_;
    std::vector<char> data_;
};

// Mock context for testing
class MockContext : public Context {
private:
    // Check if broadcasting is possible between two shapes
    bool can_broadcast(const std::vector<size_t>& shape_a, const std::vector<size_t>& shape_b, 
                       std::vector<size_t>& result_shape) {
        // Handle scalar case
        if (shape_a.empty() || shape_b.empty()) {
            result_shape = shape_a.empty() ? shape_b : shape_a;
            return true;
        }
        
        // Get the number of dimensions of each tensor
        size_t ndim_a = shape_a.size();
        size_t ndim_b = shape_b.size();
        
        // Get the maximum number of dimensions
        size_t max_ndim = std::max(ndim_a, ndim_b);
        
        // Resize result shape to maximum dimensions
        result_shape.resize(max_ndim);
        
        // Check if broadcasting is possible and calculate the result shape
        for (size_t i = 0; i < max_ndim; i++) {
            size_t dim_a = (i < ndim_a) ? shape_a[ndim_a - 1 - i] : 1;
            size_t dim_b = (i < ndim_b) ? shape_b[ndim_b - 1 - i] : 1;
            
            if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
                return false; // Incompatible dimensions
            }
            
            result_shape[max_ndim - 1 - i] = std::max(dim_a, dim_b);
        }
        
        return true;
    }

public:
    Tensor add(const Tensor& a, const Tensor& b) override {
        // Check for broadcasting compatibility
        std::vector<size_t> result_shape;
        if (!can_broadcast(a.shape(), b.shape(), result_shape)) {
            throw std::runtime_error("Shape mismatch for addition");
        }
        
        // Create result tensor with broadcasted shape
        auto result = std::make_shared<MockTensorImpl>(result_shape, a.dtype());
        return Tensor(result);
    }
    
    Tensor subtract(const Tensor& a, const Tensor& b) override {
        // Check for broadcasting compatibility
        std::vector<size_t> result_shape;
        if (!can_broadcast(a.shape(), b.shape(), result_shape)) {
            throw std::runtime_error("Shape mismatch for subtraction");
        }
        
        // Create result tensor with broadcasted shape
        auto result = std::make_shared<MockTensorImpl>(result_shape, a.dtype());
        return Tensor(result);
    }
    
    Tensor multiply(const Tensor& a, const Tensor& b) override {
        // Check for broadcasting compatibility
        std::vector<size_t> result_shape;
        if (!can_broadcast(a.shape(), b.shape(), result_shape)) {
            throw std::runtime_error("Shape mismatch for multiplication");
        }
        
        // Create result tensor with broadcasted shape
        auto result = std::make_shared<MockTensorImpl>(result_shape, a.dtype());
        return Tensor(result);
    }
    
    Tensor divide(const Tensor& a, const Tensor& b) override {
        // Check for broadcasting compatibility
        std::vector<size_t> result_shape;
        if (!can_broadcast(a.shape(), b.shape(), result_shape)) {
            throw std::runtime_error("Shape mismatch for division");
        }
        
        // Create result tensor with broadcasted shape
        auto result = std::make_shared<MockTensorImpl>(result_shape, a.dtype());
        return Tensor(result);
    }
    
    Tensor matmul(const Tensor& a, const Tensor& b) override {
        // Ensure dimensions are compatible for matrix multiplication
        if (a.ndim() < 2 || b.ndim() < 2 || a.shape(a.ndim() - 1) != b.shape(b.ndim() - 2)) {
            throw std::runtime_error("Incompatible dimensions for matrix multiplication");
        }
        
        // Calculate result shape
        std::vector<size_t> result_shape = a.shape();
        result_shape[result_shape.size() - 1] = b.shape(b.ndim() - 1);
        
        // Create result tensor
        auto result = std::make_shared<MockTensorImpl>(result_shape, a.dtype());
        return Tensor(result);
    }
    
    Tensor relu(const Tensor& x) override {
        return Tensor(std::make_shared<MockTensorImpl>(x.shape(), x.dtype()));
    }
    
    Tensor gelu(const Tensor& x) override {
        return Tensor(std::make_shared<MockTensorImpl>(x.shape(), x.dtype()));
    }
    
    Tensor silu(const Tensor& x) override {
        return Tensor(std::make_shared<MockTensorImpl>(x.shape(), x.dtype()));
    }
    
    Tensor softmax(const Tensor& x, int dim) override {
        if (dim < 0 || dim >= x.ndim()) {
            throw std::out_of_range("Dimension index out of range");
        }
        
        return Tensor(std::make_shared<MockTensorImpl>(x.shape(), x.dtype()));
    }
    
    Tensor zeros(const std::vector<size_t>& shape, DataType dtype) override {
        return Tensor(std::make_shared<MockTensorImpl>(shape, dtype));
    }
    
    Tensor ones(const std::vector<size_t>& shape, DataType dtype) override {
        auto result = std::make_shared<MockTensorImpl>(shape, dtype);
        
        // Set all values to 1.0f
        float* data = static_cast<float*>(result->data());
        for (size_t i = 0; i < result->size(); i++) {
            data[i] = 1.0f;
        }
        
        return Tensor(result);
    }
    
    Tensor sum(const Tensor& x, int dim) override {
        if (dim < 0 || dim >= x.ndim()) {
            throw std::out_of_range("Dimension index out of range");
        }
        
        // Calculate result shape
        std::vector<size_t> result_shape = x.shape();
        result_shape[dim] = 1;
        
        return Tensor(std::make_shared<MockTensorImpl>(result_shape, x.dtype()));
    }
    
    Tensor mean(const Tensor& x, int dim) override {
        if (dim < 0 || dim >= x.ndim()) {
            throw std::out_of_range("Dimension index out of range");
        }
        
        // Calculate result shape
        std::vector<size_t> result_shape = x.shape();
        result_shape[dim] = 1;
        
        return Tensor(std::make_shared<MockTensorImpl>(result_shape, x.dtype()));
    }
    
    std::string backend() const override {
        return "mock";
    }
};

// Test helper functions - create local versions for testing without conflicting with the library's implementation
// These functions are intentionally very similar to the real implementations but don't override the actual functions
namespace test_helpers {
    static Tensor createMockTensor(const std::vector<size_t>& shape, DataType dtype) {
        static MockContext ctx;
        return ctx.zeros(shape, dtype);
    }
    
    static Tensor createMockZeros(const std::vector<size_t>& shape, DataType dtype) {
        static MockContext ctx;
        return ctx.zeros(shape, dtype);
    }
    
    static Tensor createMockOnes(const std::vector<size_t>& shape, DataType dtype) {
        static MockContext ctx;
        return ctx.ones(shape, dtype);
    }
    
    static Tensor createMockFromData(const void* data, const std::vector<size_t>& shape, DataType dtype) {
        auto result = std::make_shared<MockTensorImpl>(shape, dtype);
        
        // Copy data
        size_t size_bytes = result->size() * sizeof(float); // Assuming float
        std::memcpy(result->data(), data, size_bytes);
        
        return Tensor(result);
    }
    
    static Tensor convertMockTensor(const Tensor& tensor, const std::string& to_backend) {
        if (to_backend == "mock") {
            return tensor; // Already a mock tensor
        } else if (to_backend == "ggml") {
            // Pretend to convert to GGML
            return tensor;
        } else {
            throw std::runtime_error("Unsupported backend: " + to_backend);
        }
    }
    
    static std::shared_ptr<Context> createMockContext() {
        return std::make_shared<MockContext>();
    }
}

// Test fixture for tensor tests
class TensorTest : public ::testing::Test {
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
        context = std::make_shared<MockContext>();
    }
    
    std::vector<size_t> scalar_shape;
    std::vector<size_t> vector_shape;
    std::vector<size_t> matrix_shape;
    std::vector<size_t> tensor3d_shape;
    DataType default_dtype;
    std::shared_ptr<Context> context;
};

// Test invalid tensor operations throw exceptions
TEST_F(TensorTest, InvalidTensorOperations) {
    // Create an uninitialized tensor
    Tensor tensor;
    
    // Verify that operations on an uninitialized tensor throw exceptions
    EXPECT_THROW(tensor.shape(0), std::runtime_error);
    EXPECT_THROW(tensor.shape(), std::runtime_error);
    EXPECT_THROW(tensor.ndim(), std::runtime_error);
    EXPECT_THROW(tensor.size(), std::runtime_error);
    EXPECT_THROW(tensor.dtype(), std::runtime_error);
    EXPECT_THROW(tensor.dtype_str(), std::runtime_error);
    EXPECT_THROW(tensor.data(), std::runtime_error);
    EXPECT_THROW(static_cast<const Tensor&>(tensor).data(), std::runtime_error);
    EXPECT_THROW(tensor.reshape(vector_shape), std::runtime_error);
    EXPECT_THROW(tensor.view(vector_shape), std::runtime_error);
    EXPECT_THROW(tensor.slice(0, 0, 1), std::runtime_error);
    EXPECT_THROW(tensor.print("test"), std::runtime_error);
    
    // Verify that is_valid returns false
    EXPECT_FALSE(tensor.is_valid());
}

// Test tensor creation for different shapes and types
TEST_F(TensorTest, TensorCreation) {
    // Test vector creation
    Tensor vector = test_helpers::createMockZeros(vector_shape, DataType::F32);
    EXPECT_TRUE(vector.is_valid());
    EXPECT_EQ(vector.ndim(), 1);
    EXPECT_EQ(vector.shape(0), 10);
    EXPECT_EQ(vector.size(), 10);
    
    // Test matrix creation
    Tensor matrix = test_helpers::createMockZeros(matrix_shape, DataType::F16);
    EXPECT_TRUE(matrix.is_valid());
    EXPECT_EQ(matrix.ndim(), 2);
    EXPECT_EQ(matrix.shape(0), 5);
    EXPECT_EQ(matrix.shape(1), 10);
    EXPECT_EQ(matrix.size(), 50);
    EXPECT_EQ(matrix.dtype(), DataType::F16);
    EXPECT_EQ(matrix.dtype_str(), "F16");
    
    // Test 3D tensor creation
    Tensor tensor3d = test_helpers::createMockZeros(tensor3d_shape, DataType::BF16);
    EXPECT_TRUE(tensor3d.is_valid());
    EXPECT_EQ(tensor3d.ndim(), 3);
    EXPECT_EQ(tensor3d.shape(0), 3);
    EXPECT_EQ(tensor3d.shape(1), 5);
    EXPECT_EQ(tensor3d.shape(2), 10);
    EXPECT_EQ(tensor3d.size(), 150);
    EXPECT_EQ(tensor3d.dtype(), DataType::BF16);
    EXPECT_EQ(tensor3d.dtype_str(), "BF16");
    
    // Test ones tensor
    Tensor ones = test_helpers::createMockOnes(vector_shape, DataType::I32);
    EXPECT_TRUE(ones.is_valid());
    EXPECT_EQ(ones.dtype(), DataType::I32);
    EXPECT_EQ(ones.dtype_str(), "I32");
    
    // Test from_data creation
    float data[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    Tensor from_data = test_helpers::createMockFromData(data, vector_shape, DataType::F32);
    EXPECT_TRUE(from_data.is_valid());
    EXPECT_EQ(from_data.ndim(), 1);
    EXPECT_EQ(from_data.shape(0), 10);
    
    // Check that data pointers are accessible
    EXPECT_NE(from_data.data(), nullptr);
    EXPECT_NE(static_cast<const Tensor&>(from_data).data(), nullptr);
}

// Test tensor reshape and view operations
TEST_F(TensorTest, TensorReshapeView) {
    // Create a 3D tensor
    Tensor original = test_helpers::createMockZeros(tensor3d_shape, default_dtype);
    EXPECT_TRUE(original.is_valid());
    EXPECT_EQ(original.ndim(), 3);
    EXPECT_EQ(original.size(), 150);
    
    // Reshape to 2D
    std::vector<size_t> new_shape = {15, 10};
    Tensor reshaped = original.reshape(new_shape);
    EXPECT_TRUE(reshaped.is_valid());
    EXPECT_EQ(reshaped.ndim(), 2);
    EXPECT_EQ(reshaped.shape(0), 15);
    EXPECT_EQ(reshaped.shape(1), 10);
    EXPECT_EQ(reshaped.size(), 150);
    
    // Reshape to 1D
    new_shape = {150};
    Tensor reshaped_flat = original.reshape(new_shape);
    EXPECT_TRUE(reshaped_flat.is_valid());
    EXPECT_EQ(reshaped_flat.ndim(), 1);
    EXPECT_EQ(reshaped_flat.shape(0), 150);
    EXPECT_EQ(reshaped_flat.size(), 150);
    
    // View as 2D
    new_shape = {15, 10};
    Tensor viewed = original.view(new_shape);
    EXPECT_TRUE(viewed.is_valid());
    EXPECT_EQ(viewed.ndim(), 2);
    EXPECT_EQ(viewed.shape(0), 15);
    EXPECT_EQ(viewed.shape(1), 10);
    EXPECT_EQ(viewed.size(), 150);
    
    // Invalid reshape (size mismatch) should throw
    new_shape = {10, 10};
    EXPECT_THROW(original.reshape(new_shape), std::runtime_error);
}

// Test tensor slice operations
TEST_F(TensorTest, TensorSlice) {
    // Create a 3D tensor
    Tensor original = test_helpers::createMockZeros(tensor3d_shape, default_dtype);
    EXPECT_TRUE(original.is_valid());
    
    // Slice along dim 0
    Tensor slice0 = original.slice(0, 1, 2);
    EXPECT_TRUE(slice0.is_valid());
    EXPECT_EQ(slice0.ndim(), 3);
    EXPECT_EQ(slice0.shape(0), 1);  // sliced dimension
    EXPECT_EQ(slice0.shape(1), 5);  // unchanged
    EXPECT_EQ(slice0.shape(2), 10); // unchanged
    
    // Invalid slice operations should throw
    EXPECT_THROW(original.slice(3, 0, 1), std::out_of_range); // dim out of range
    EXPECT_THROW(original.slice(0, 5, 1), std::runtime_error); // end before start
}

// Test context operations (basic arithmetic)
TEST_F(TensorTest, ContextOperations) {
    // Create tensors
    Tensor a = test_helpers::createMockOnes(vector_shape, default_dtype);
    Tensor b = test_helpers::createMockOnes(vector_shape, default_dtype);
    
    // Test add
    Tensor c = context->add(a, b);
    EXPECT_TRUE(c.is_valid());
    EXPECT_EQ(c.shape(), vector_shape);
    
    // Test subtract
    Tensor d = context->subtract(a, b);
    EXPECT_TRUE(d.is_valid());
    EXPECT_EQ(d.shape(), vector_shape);
    
    // Test multiply
    Tensor e = context->multiply(a, b);
    EXPECT_TRUE(e.is_valid());
    EXPECT_EQ(e.shape(), vector_shape);
    
    // Test divide
    Tensor f = context->divide(a, b);
    EXPECT_TRUE(f.is_valid());
    EXPECT_EQ(f.shape(), vector_shape);
    
    // Test matmul with compatible dimensions
    Tensor g = test_helpers::createMockOnes({10, 5}, default_dtype);
    Tensor h = test_helpers::createMockOnes({5, 3}, default_dtype);
    Tensor i = context->matmul(g, h);
    EXPECT_TRUE(i.is_valid());
    EXPECT_EQ(i.ndim(), 2);
    EXPECT_EQ(i.shape(0), 10);
    EXPECT_EQ(i.shape(1), 3);
    
    // Test activations
    Tensor j = context->relu(a);
    EXPECT_TRUE(j.is_valid());
    
    Tensor k = context->gelu(a);
    EXPECT_TRUE(k.is_valid());
    
    Tensor l = context->silu(a);
    EXPECT_TRUE(l.is_valid());
    
    Tensor m = context->softmax(a, 0);
    EXPECT_TRUE(m.is_valid());
    
    // Test reductions
    Tensor n = context->sum(a, 0);
    EXPECT_TRUE(n.is_valid());
    
    Tensor o = context->mean(a, 0);
    EXPECT_TRUE(o.is_valid());
    
    // Check backend name
    EXPECT_EQ(context->backend(), "mock");
}

// Test tensor broadcasting operations
TEST_F(TensorTest, TensorBroadcasting) {
    // Create tensors with different shapes
    Tensor scalar = test_helpers::createMockOnes({}, default_dtype);         // scalar
    Tensor vector = test_helpers::createMockOnes({5}, default_dtype);        // 1D
    Tensor matrix = test_helpers::createMockOnes({3, 4}, default_dtype);     // 2D
    Tensor tensor3d = test_helpers::createMockOnes({2, 3, 4}, default_dtype); // 3D
    
    // Test scalar broadcasting
    Tensor result1 = context->add(scalar, vector);
    EXPECT_TRUE(result1.is_valid());
    EXPECT_EQ(result1.shape(), vector.shape());
    
    Tensor result2 = context->multiply(scalar, matrix);
    EXPECT_TRUE(result2.is_valid());
    EXPECT_EQ(result2.shape(), matrix.shape());
    
    Tensor result3 = context->subtract(tensor3d, scalar);
    EXPECT_TRUE(result3.is_valid());
    EXPECT_EQ(result3.shape(), tensor3d.shape());
    
    // Test broadcasting 1D with 2D
    Tensor row_vector = test_helpers::createMockOnes({1, 4}, default_dtype);   // shape [1, 4]
    Tensor result4 = context->add(row_vector, matrix);  // should broadcast to [3, 4]
    EXPECT_TRUE(result4.is_valid());
    EXPECT_EQ(result4.shape(), matrix.shape());
    
    Tensor col_vector = test_helpers::createMockOnes({3, 1}, default_dtype);   // shape [3, 1]
    Tensor result5 = context->multiply(col_vector, matrix);  // should broadcast to [3, 4]
    EXPECT_TRUE(result5.is_valid());
    EXPECT_EQ(result5.shape(), matrix.shape());
    
    // Test broadcasting 2D with 3D
    Tensor result6 = context->add(matrix, tensor3d);  // should broadcast to [2, 3, 4]
    EXPECT_TRUE(result6.is_valid());
    EXPECT_EQ(result6.shape(), tensor3d.shape());
    
    // Test incompatible shapes (should trigger exception)
    Tensor incompatible = test_helpers::createMockOnes({5, 5}, default_dtype);  // shape [5, 5]
    EXPECT_THROW(context->add(matrix, incompatible), std::runtime_error);  // shapes [3, 4] and [5, 5] aren't compatible
}

// Test tensor backend conversion
TEST_F(TensorTest, TensorConversion) {
    // Create a tensor
    Tensor original = test_helpers::createMockZeros(vector_shape, default_dtype);
    EXPECT_TRUE(original.is_valid());
    
    // Convert to a different backend (should be a no-op if same backend)
    Tensor converted = test_helpers::convertMockTensor(original, "ggml");
    EXPECT_TRUE(converted.is_valid());
    EXPECT_EQ(converted.shape(), original.shape());
    EXPECT_EQ(converted.dtype(), original.dtype());
    
    // Test conversion to an invalid backend should throw
    EXPECT_THROW(test_helpers::convertMockTensor(original, "invalid_backend"), std::runtime_error);
}