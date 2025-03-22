#include <ccsm/tensor.h>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <cstdio>
#include <limits>
#include <iostream>
#include <iomanip>

namespace ccsm {

// Tensor implementation
Tensor::Tensor() : impl_(nullptr) {}

Tensor::Tensor(std::shared_ptr<TensorImpl> impl) : impl_(impl) {}

size_t Tensor::shape(int dim) const {
    if (!impl_) {
        throw std::runtime_error("Tensor is not initialized");
    }
    return impl_->shape(dim);
}

std::vector<size_t> Tensor::shape() const {
    if (!impl_) {
        throw std::runtime_error("Tensor is not initialized");
    }
    return impl_->shape();
}

int Tensor::ndim() const {
    if (!impl_) {
        throw std::runtime_error("Tensor is not initialized");
    }
    return impl_->ndim();
}

size_t Tensor::size() const {
    if (!impl_) {
        throw std::runtime_error("Tensor is not initialized");
    }
    return impl_->size();
}

DataType Tensor::dtype() const {
    if (!impl_) {
        throw std::runtime_error("Tensor is not initialized");
    }
    return impl_->dtype();
}

std::string Tensor::dtype_str() const {
    if (!impl_) {
        throw std::runtime_error("Tensor is not initialized");
    }
    
    switch(dtype()) {
        case DataType::F32:  return "F32";
        case DataType::F16:  return "F16";
        case DataType::BF16: return "BF16";
        case DataType::I32:  return "I32";
        case DataType::I16:  return "I16";
        case DataType::I8:   return "I8";
        case DataType::Q8_0: return "Q8_0";
        case DataType::Q4_0: return "Q4_0";
        case DataType::Q4_1: return "Q4_1";
        default:             return "UNKNOWN";
    }
}

void* Tensor::data() {
    if (!impl_) {
        throw std::runtime_error("Tensor is not initialized");
    }
    return impl_->data();
}

const void* Tensor::data() const {
    if (!impl_) {
        throw std::runtime_error("Tensor is not initialized");
    }
    return impl_->data();
}

Tensor Tensor::reshape(const std::vector<size_t>& new_shape) const {
    if (!impl_) {
        throw std::runtime_error("Tensor is not initialized");
    }
    return Tensor(impl_->reshape(new_shape));
}

Tensor Tensor::view(const std::vector<size_t>& new_shape) const {
    if (!impl_) {
        throw std::runtime_error("Tensor is not initialized");
    }
    return Tensor(impl_->view(new_shape));
}

Tensor Tensor::slice(int dim, size_t start, size_t end) const {
    if (!impl_) {
        throw std::runtime_error("Tensor is not initialized");
    }
    return Tensor(impl_->slice(dim, start, end));
}

void Tensor::print(const std::string& name) const {
    if (!impl_) {
        throw std::runtime_error("Tensor is not initialized");
    }
    impl_->print(name);
}

bool Tensor::is_valid() const {
    return impl_ != nullptr;
}

// TensorFactory implementation
Tensor TensorFactory::create(const std::vector<size_t>& shape, DataType dtype) {
    // Create a tensor of zeros and use the appropriate backend
    // In the future, this could select a backend based on global settings or device availability
    return zeros(shape, dtype);
}

Tensor TensorFactory::zeros(const std::vector<size_t>& shape, DataType dtype) {
    auto ctx = ContextFactory::create();
    return ctx->zeros(shape, dtype);
}

Tensor TensorFactory::ones(const std::vector<size_t>& shape, DataType dtype) {
    auto ctx = ContextFactory::create();
    return ctx->ones(shape, dtype);
}

Tensor TensorFactory::from_data(const void* data, const std::vector<size_t>& shape, DataType dtype) {
    // Create a basic implementation that copies the data
    class SimpleTensorImpl : public TensorImpl {
    public:
        SimpleTensorImpl(const void* data, const std::vector<size_t>& shape, DataType dtype)
            : shape_(shape), dtype_(dtype) {
            
            // Calculate total size
            size_ = 1;
            for (auto dim : shape) {
                size_ *= dim;
            }
            
            // Determine bytes per element
            size_t bytes_per_element;
            switch (dtype) {
                case DataType::F32:
                case DataType::I32:
                    bytes_per_element = 4;
                    break;
                case DataType::F16:
                case DataType::BF16:
                case DataType::I16:
                    bytes_per_element = 2;
                    break;
                case DataType::I8:
                case DataType::Q8_0:
                    bytes_per_element = 1;
                    break;
                case DataType::Q4_0:
                case DataType::Q4_1:
                    // 4 bits per element, but we need to round up to at least 1 byte
                    // and handle alignment and block size for quantized types
                    bytes_per_element = 1; // We'll compute actual size later
                    // We'll calculate total_bytes below, with special handling for 4-bit quantized types
                    break;
                default:
                    throw std::runtime_error("Unsupported data type");
            }
            
            // Allocate memory and copy data
            size_t total_bytes = size_ * bytes_per_element;
            
            // Special handling for 4-bit quantized types
            if (dtype_ == DataType::Q4_0 || dtype_ == DataType::Q4_1) {
                // Round up to bytes (2 elements per byte)
                total_bytes = (size_ + 1) / 2;
            }
            
            data_.resize(total_bytes);
            std::memcpy(data_.data(), data, total_bytes);
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
            auto result = std::make_shared<SimpleTensorImpl>(data_.data(), new_shape, dtype_);
            return result;
        }
        
        std::shared_ptr<TensorImpl> view(const std::vector<size_t>& new_shape) const override {
            // For simplicity, view is the same as reshape for this simple implementation
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
            
            // Create new tensor to hold the slice
            auto result = std::make_shared<SimpleTensorImpl>(nullptr, new_shape, dtype_);
            
            // For a proper implementation, we need to extract the correct slice of data
            // This is a simplified implementation for a 1D tensor only
            if (ndim() == 1 && dim == 0) {
                // Simple case: 1D tensor, just copy the slice
                size_t elem_size = 0;
                switch (dtype_) {
                    case DataType::F32: elem_size = 4; break;
                    case DataType::F16: elem_size = 2; break;
                    case DataType::BF16: elem_size = 2; break;
                    case DataType::I32: elem_size = 4; break;
                    case DataType::I16: elem_size = 2; break;
                    case DataType::I8: elem_size = 1; break;
                    case DataType::Q8_0: elem_size = 1; break;
                    // Handle other types as needed
                    default: elem_size = 4; // Default to 4 bytes
                }
                
                // Use proper const-correct casting for the source data
                const char* src = static_cast<const char*>(data_.data()) + start * elem_size;
                char* dst = static_cast<char*>(result->data());
                std::memcpy(dst, src, (end - start) * elem_size);
            }
            else {
                // For multi-dimensional tensors, we'd need a more sophisticated
                // implementation that considers strides
                // This is a placeholder that sets all values to zero
                std::memset(result->data(), 0, result->size() * sizeof(float));
            }
            
            return result;
        }
        
        void print(const std::string& name = "") const override {
            std::cout << "Tensor";
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
            std::cout << "], dtype=";
            
            // Print dtype as string instead of int
            switch(dtype_) {
                case DataType::F32:  std::cout << "F32"; break;
                case DataType::F16:  std::cout << "F16"; break;
                case DataType::BF16: std::cout << "BF16"; break;
                case DataType::I32:  std::cout << "I32"; break;
                case DataType::I16:  std::cout << "I16"; break;
                case DataType::I8:   std::cout << "I8"; break;
                case DataType::Q8_0: std::cout << "Q8_0"; break;
                case DataType::Q4_0: std::cout << "Q4_0"; break;
                case DataType::Q4_1: std::cout << "Q4_1"; break;
                default:             std::cout << "UNKNOWN"; break;
            }
            
            std::cout << ", size=" << size_ << " elements" << std::endl;
            
            // If tensor is small enough, print some values
            if (size_ <= 100 && dtype_ == DataType::F32) {
                // Use proper const casting to access data values
                const float* data_ptr = static_cast<const float*>(data());
                std::cout << "  Values: [";
                size_t max_print = std::min(size_, size_t(10)); // Print at most 10 values
                
                for (size_t i = 0; i < max_print; i++) {
                    std::cout << data_ptr[i];
                    if (i < max_print - 1) {
                        std::cout << ", ";
                    }
                }
                
                if (max_print < size_) {
                    std::cout << ", ... ";
                }
                
                std::cout << "]" << std::endl;
            }
        }
        
    private:
        std::vector<size_t> shape_;
        DataType dtype_;
        size_t size_;
        std::vector<char> data_;
    };
    
    return Tensor(std::make_shared<SimpleTensorImpl>(data, shape, dtype));
}

Tensor TensorFactory::convert(const Tensor& tensor, const std::string& to_backend) {
    // For now, just return a copy of the tensor
    // A real implementation would convert between different backends
    return tensor;
}

// Simple context implementation
class SimpleContext : public Context {
public:
    SimpleContext() = default;
    
    // Helper for better error messages
    std::string shape_to_string(const std::vector<size_t>& shape) {
        std::stringstream ss;
        ss << "[";
        for (size_t i = 0; i < shape.size(); ++i) {
            ss << shape[i];
            if (i < shape.size() - 1) ss << ", ";
        }
        ss << "]";
        return ss.str();
    }
    
    // Validate tensor for NaN/Inf
    template<typename T>
    bool validate_tensor_values(const Tensor& tensor) {
        if (!tensor.is_valid()) {
            return false;
        }
        
        if (tensor.dtype() != DataType::F32 && tensor.dtype() != DataType::F16 && tensor.dtype() != DataType::BF16) {
            // Only validate floating point tensors
            return true;
        }
        
        if (tensor.dtype() == DataType::F32) {
            const float* data = static_cast<const float*>(tensor.data());
            for (size_t i = 0; i < tensor.size(); i++) {
                if (std::isnan(data[i]) || std::isinf(data[i])) {
                    return false;
                }
            }
        }
        // For other float types, we'd need proper conversion
        
        return true;
    }
    
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
    
    // Map a linear index from the result tensor to the corresponding indices in input tensors,
    // accounting for broadcasting rules.
    void map_broadcast_indices(const std::vector<size_t>& shape_result, size_t linear_idx,
                             const std::vector<size_t>& shape_a, const std::vector<size_t>& shape_b,
                             size_t& idx_a, size_t& idx_b) {
        if (shape_a.empty()) {
            // A is scalar, just use index 0
            idx_a = 0;
            
            // Map index for B (should be identical to result since B's shape == result shape)
            idx_b = linear_idx;
            return;
        }
        
        if (shape_b.empty()) {
            // B is scalar, just use index 0
            idx_b = 0;
            
            // Map index for A (should be identical to result since A's shape == result shape)
            idx_a = linear_idx;
            return;
        }
        
        // Get the dimensionality of each tensor
        size_t ndim_result = shape_result.size();
        size_t ndim_a = shape_a.size();
        size_t ndim_b = shape_b.size();
        
        // Convert the linear index to a multi-dimensional index in the result tensor
        std::vector<size_t> indices_result(ndim_result);
        size_t remaining_idx = linear_idx;
        
        for (int i = ndim_result - 1; i >= 0; i--) {
            indices_result[i] = remaining_idx % shape_result[i];
            remaining_idx /= shape_result[i];
        }
        
        // Map to indices in tensor A
        size_t stride_a = 1;
        idx_a = 0;
        for (int i = ndim_a - 1; i >= 0; i--) {
            int result_dim = i + (ndim_result - ndim_a); // Offset for dimensions
            size_t dim_idx = (result_dim >= 0) ? indices_result[result_dim] : 0;
            
            // Handle broadcasting - if this dimension in A is 1, use index 0
            if (shape_a[i] == 1) {
                dim_idx = 0;
            }
            
            idx_a += dim_idx * stride_a;
            stride_a *= shape_a[i];
        }
        
        // Map to indices in tensor B
        size_t stride_b = 1;
        idx_b = 0;
        for (int i = ndim_b - 1; i >= 0; i--) {
            int result_dim = i + (ndim_result - ndim_b); // Offset for dimensions
            size_t dim_idx = (result_dim >= 0) ? indices_result[result_dim] : 0;
            
            // Handle broadcasting - if this dimension in B is 1, use index 0
            if (shape_b[i] == 1) {
                dim_idx = 0;
            }
            
            idx_b += dim_idx * stride_b;
            stride_b *= shape_b[i];
        }
    }
    
    Tensor add(const Tensor& a, const Tensor& b) override {
        // Validate inputs
        if (!a.is_valid() || !b.is_valid()) {
            throw std::runtime_error("Cannot perform addition on invalid tensors");
        }
        
        // Check for NaN/Inf values
        if (!validate_tensor_values<float>(a) || !validate_tensor_values<float>(b)) {
            throw std::runtime_error("Cannot perform addition on tensors with NaN or Inf values");
        }
        
        // Determine result data type and shape with broadcasting
        DataType result_dtype = a.dtype();
        std::vector<size_t> result_shape;
        bool is_scalar_op = (a.ndim() == 0 || b.ndim() == 0);
        
        // Check broadcasting compatibility
        if (!can_broadcast(a.shape(), b.shape(), result_shape)) {
            throw std::runtime_error("Cannot broadcast shapes for addition: " + 
                                    shape_to_string(a.shape()) + " and " + 
                                    shape_to_string(b.shape()));
        }
        
        // If one tensor is scalar, mark for special handling
        if (a.ndim() == 0) {
            result_dtype = b.dtype();
        }
        
        // Determine result data type
        if (a.dtype() != b.dtype()) {
            // Use higher precision type
            if (a.dtype() == DataType::F32 || b.dtype() == DataType::F32) {
                result_dtype = DataType::F32;
            } else if (a.dtype() == DataType::BF16 || b.dtype() == DataType::BF16) {
                result_dtype = DataType::BF16;
            } else if (a.dtype() == DataType::F16 || b.dtype() == DataType::F16) {
                result_dtype = DataType::F16;
            }
        }
        
        // Create result tensor
        auto result = zeros(result_shape, result_dtype);
        
        // Handle F32 case
        if (a.dtype() == DataType::F32 && b.dtype() == DataType::F32) {
            float* a_data = static_cast<float*>(const_cast<void*>(a.data()));
            float* b_data = static_cast<float*>(const_cast<void*>(b.data()));
            float* result_data = static_cast<float*>(result.data());
            
            if (is_scalar_op) {
                // Optimized path for scalar operations
                float scalar_val = a.ndim() == 0 ? *a_data : *b_data;
                const float* tensor_data = a.ndim() == 0 ? b_data : a_data;
                size_t size = result.size();
                
                for (size_t i = 0; i < size; i++) {
                    result_data[i] = tensor_data[i] + scalar_val;
                }
            } else if (a.shape() == b.shape()) {
                // Fast path for identically shaped tensors
                for (size_t i = 0; i < result.size(); i++) {
                    result_data[i] = a_data[i] + b_data[i];
                }
            } else {
                // General broadcasting case
                for (size_t i = 0; i < result.size(); i++) {
                    // Map the result index to indices in the input tensors
                    size_t idx_a, idx_b;
                    map_broadcast_indices(result_shape, i, a.shape(), b.shape(), idx_a, idx_b);
                    
                    // Perform the operation using the mapped indices
                    result_data[i] = a_data[idx_a] + b_data[idx_b];
                }
            }
        } else {
            // Fallback for unsupported types
            // In a real implementation, we'd convert types or handle more cases
            std::cerr << "Warning: Unsupported data types for addition: " 
                      << a.dtype_str() << " and " << b.dtype_str() << std::endl;
        }
        
        return result;
    }
    
    Tensor subtract(const Tensor& a, const Tensor& b) override {
        // Validate inputs
        if (!a.is_valid() || !b.is_valid()) {
            throw std::runtime_error("Cannot perform subtraction on invalid tensors");
        }
        
        // Check for NaN/Inf values
        if (!validate_tensor_values<float>(a) || !validate_tensor_values<float>(b)) {
            throw std::runtime_error("Cannot perform subtraction on tensors with NaN or Inf values");
        }
        
        // Determine result data type and shape with broadcasting
        DataType result_dtype = a.dtype();
        std::vector<size_t> result_shape;
        bool is_scalar_op = (a.ndim() == 0 || b.ndim() == 0);
        
        // Check broadcasting compatibility
        if (!can_broadcast(a.shape(), b.shape(), result_shape)) {
            throw std::runtime_error("Cannot broadcast shapes for subtraction: " + 
                                    shape_to_string(a.shape()) + " and " + 
                                    shape_to_string(b.shape()));
        }
        
        // If one tensor is scalar, adjust result type
        if (a.ndim() == 0) {
            result_dtype = b.dtype();
        }
        
        // Determine result data type
        if (a.dtype() != b.dtype()) {
            // Use higher precision type
            if (a.dtype() == DataType::F32 || b.dtype() == DataType::F32) {
                result_dtype = DataType::F32;
            } else if (a.dtype() == DataType::BF16 || b.dtype() == DataType::BF16) {
                result_dtype = DataType::BF16;
            } else if (a.dtype() == DataType::F16 || b.dtype() == DataType::F16) {
                result_dtype = DataType::F16;
            }
        }
        
        // Create result tensor
        auto result = zeros(result_shape, result_dtype);
        
        // Handle F32 case
        if (a.dtype() == DataType::F32 && b.dtype() == DataType::F32) {
            float* a_data = static_cast<float*>(const_cast<void*>(a.data()));
            float* b_data = static_cast<float*>(const_cast<void*>(b.data()));
            float* result_data = static_cast<float*>(result.data());
            
            if (is_scalar_op) {
                // Optimized path for scalar operations
                if (a.ndim() == 0) {
                    // a is scalar, b is tensor: result = a - b
                    float scalar_val = *a_data;
                    size_t size = result.size();
                    
                    for (size_t i = 0; i < size; i++) {
                        result_data[i] = scalar_val - b_data[i];
                    }
                } else {
                    // b is scalar, a is tensor: result = a - b
                    float scalar_val = *b_data;
                    size_t size = result.size();
                    
                    for (size_t i = 0; i < size; i++) {
                        result_data[i] = a_data[i] - scalar_val;
                    }
                }
            } else if (a.shape() == b.shape()) {
                // Fast path for identically shaped tensors
                for (size_t i = 0; i < result.size(); i++) {
                    result_data[i] = a_data[i] - b_data[i];
                }
            } else {
                // General broadcasting case
                for (size_t i = 0; i < result.size(); i++) {
                    // Map the result index to indices in the input tensors
                    size_t idx_a, idx_b;
                    map_broadcast_indices(result_shape, i, a.shape(), b.shape(), idx_a, idx_b);
                    
                    // Perform the operation using the mapped indices
                    result_data[i] = a_data[idx_a] - b_data[idx_b];
                }
            }
        } else {
            // Fallback for unsupported types
            // In a real implementation, we'd convert types or handle more cases
            std::cerr << "Warning: Unsupported data types for subtraction: " 
                      << a.dtype_str() << " and " << b.dtype_str() << std::endl;
        }
        
        return result;
    }
    
    Tensor multiply(const Tensor& a, const Tensor& b) override {
        // Validate inputs
        if (!a.is_valid() || !b.is_valid()) {
            throw std::runtime_error("Cannot perform multiplication on invalid tensors");
        }
        
        // Check for NaN/Inf values
        if (!validate_tensor_values<float>(a) || !validate_tensor_values<float>(b)) {
            throw std::runtime_error("Cannot perform multiplication on tensors with NaN or Inf values");
        }
        
        // Determine result data type and shape with broadcasting
        DataType result_dtype = a.dtype();
        std::vector<size_t> result_shape;
        bool is_scalar_op = (a.ndim() == 0 || b.ndim() == 0);
        
        // Check broadcasting compatibility
        if (!can_broadcast(a.shape(), b.shape(), result_shape)) {
            throw std::runtime_error("Cannot broadcast shapes for multiplication: " + 
                                    shape_to_string(a.shape()) + " and " + 
                                    shape_to_string(b.shape()));
        }
        
        // If one tensor is scalar, adjust result type
        if (a.ndim() == 0) {
            result_dtype = b.dtype();
        }
        
        // Determine result data type
        if (a.dtype() != b.dtype()) {
            // Use higher precision type
            if (a.dtype() == DataType::F32 || b.dtype() == DataType::F32) {
                result_dtype = DataType::F32;
            } else if (a.dtype() == DataType::BF16 || b.dtype() == DataType::BF16) {
                result_dtype = DataType::BF16;
            } else if (a.dtype() == DataType::F16 || b.dtype() == DataType::F16) {
                result_dtype = DataType::F16;
            }
        }
        
        // Create result tensor
        auto result = zeros(result_shape, result_dtype);
        
        // Handle F32 case
        if (a.dtype() == DataType::F32 && b.dtype() == DataType::F32) {
            float* a_data = static_cast<float*>(const_cast<void*>(a.data()));
            float* b_data = static_cast<float*>(const_cast<void*>(b.data()));
            float* result_data = static_cast<float*>(result.data());
            
            if (is_scalar_op) {
                // Optimized path for scalar operations
                float scalar_val = a.ndim() == 0 ? *a_data : *b_data;
                const float* tensor_data = a.ndim() == 0 ? b_data : a_data;
                size_t size = result.size();
                
                for (size_t i = 0; i < size; i++) {
                    result_data[i] = tensor_data[i] * scalar_val;
                }
            } else if (a.shape() == b.shape()) {
                // Fast path for identically shaped tensors
                for (size_t i = 0; i < result.size(); i++) {
                    result_data[i] = a_data[i] * b_data[i];
                }
            } else {
                // General broadcasting case
                for (size_t i = 0; i < result.size(); i++) {
                    // Map the result index to indices in the input tensors
                    size_t idx_a, idx_b;
                    map_broadcast_indices(result_shape, i, a.shape(), b.shape(), idx_a, idx_b);
                    
                    // Perform the operation using the mapped indices
                    result_data[i] = a_data[idx_a] * b_data[idx_b];
                }
            }
        } else {
            // Fallback for unsupported types
            // In a real implementation, we'd convert types or handle more cases
            std::cerr << "Warning: Unsupported data types for multiplication: " 
                      << a.dtype_str() << " and " << b.dtype_str() << std::endl;
        }
        
        return result;
    }
    
    Tensor divide(const Tensor& a, const Tensor& b) override {
        // Validate inputs
        if (!a.is_valid() || !b.is_valid()) {
            throw std::runtime_error("Cannot perform division on invalid tensors");
        }
        
        // Check for NaN/Inf values
        if (!validate_tensor_values<float>(a) || !validate_tensor_values<float>(b)) {
            throw std::runtime_error("Cannot perform division on tensors with NaN or Inf values");
        }
        
        // Determine result data type and shape with broadcasting
        DataType result_dtype = a.dtype();
        std::vector<size_t> result_shape;
        bool is_scalar_op = (a.ndim() == 0 || b.ndim() == 0);
        
        // Check broadcasting compatibility
        if (!can_broadcast(a.shape(), b.shape(), result_shape)) {
            throw std::runtime_error("Cannot broadcast shapes for division: " + 
                                    shape_to_string(a.shape()) + " and " + 
                                    shape_to_string(b.shape()));
        }
        
        // If one tensor is scalar, adjust result type
        if (a.ndim() == 0) {
            result_dtype = b.dtype();
        }
        
        // Determine result data type
        if (a.dtype() != b.dtype()) {
            // Use higher precision type
            if (a.dtype() == DataType::F32 || b.dtype() == DataType::F32) {
                result_dtype = DataType::F32;
            } else if (a.dtype() == DataType::BF16 || b.dtype() == DataType::BF16) {
                result_dtype = DataType::BF16;
            } else if (a.dtype() == DataType::F16 || b.dtype() == DataType::F16) {
                result_dtype = DataType::F16;
            }
        }
        
        // Create result tensor
        auto result = zeros(result_shape, result_dtype);
        
        // Handle F32 case
        if (a.dtype() == DataType::F32 && b.dtype() == DataType::F32) {
            float* a_data = static_cast<float*>(const_cast<void*>(a.data()));
            float* b_data = static_cast<float*>(const_cast<void*>(b.data()));
            float* result_data = static_cast<float*>(result.data());
            
            // Check for division by zero first
            if (b.ndim() == 0) {
                // b is scalar, check if it's zero
                float scalar_val = *b_data;
                if (scalar_val == 0 || std::abs(scalar_val) < 1e-10) {
                    throw std::runtime_error("Division by zero or very small value (value=" + 
                                           std::to_string(scalar_val) + ")");
                }
            } else {
                // Check if any element in b is zero or near-zero
                for (size_t i = 0; i < b.size(); i++) {
                    if (b_data[i] == 0 || std::abs(b_data[i]) < 1e-10) {
                        throw std::runtime_error("Division by zero or very small value at index " + 
                                               std::to_string(i) + " (value=" + 
                                               std::to_string(b_data[i]) + ")");
                    }
                }
            }
            
            // Perform division
            if (is_scalar_op) {
                // Optimized path for scalar operations
                if (a.ndim() == 0) {
                    // a is scalar, b is tensor: result = a / b
                    float scalar_val = *a_data;
                    size_t size = result.size();
                    
                    for (size_t i = 0; i < size; i++) {
                        result_data[i] = scalar_val / b_data[i];
                    }
                } else {
                    // b is scalar, a is tensor: result = a / b
                    float scalar_val = *b_data;
                    size_t size = result.size();
                    
                    for (size_t i = 0; i < size; i++) {
                        result_data[i] = a_data[i] / scalar_val;
                    }
                }
            } else if (a.shape() == b.shape()) {
                // Fast path for identically shaped tensors
                for (size_t i = 0; i < result.size(); i++) {
                    result_data[i] = a_data[i] / b_data[i];
                }
            } else {
                // General broadcasting case
                for (size_t i = 0; i < result.size(); i++) {
                    // Map the result index to indices in the input tensors
                    size_t idx_a, idx_b;
                    map_broadcast_indices(result_shape, i, a.shape(), b.shape(), idx_a, idx_b);
                    
                    // Perform the operation using the mapped indices
                    result_data[i] = a_data[idx_a] / b_data[idx_b];
                }
            }
        } else {
            // Fallback for unsupported types
            // In a real implementation, we'd convert types or handle more cases
            std::cerr << "Warning: Unsupported data types for division: " 
                      << a.dtype_str() << " and " << b.dtype_str() << std::endl;
        }
        
        return result;
    }
    
    Tensor matmul(const Tensor& a, const Tensor& b) override {
        // Validate inputs
        if (!a.is_valid() || !b.is_valid()) {
            throw std::runtime_error("Cannot perform matrix multiplication on invalid tensors");
        }
        
        // Check for NaN/Inf values 
        if (!validate_tensor_values<float>(a) || !validate_tensor_values<float>(b)) {
            throw std::runtime_error("Cannot perform matrix multiplication on tensors with NaN or Inf values");
        }
        
        // Check for shape compatibility
        if (a.ndim() < 1 || b.ndim() < 1) {
            throw std::runtime_error("Tensors must have at least 1 dimension for matmul");
        }
        
        // Handle vector-vector dot product case (both 1D)
        if (a.ndim() == 1 && b.ndim() == 1) {
            if (a.shape(0) != b.shape(0)) {
                throw std::runtime_error("Incompatible dimensions for vector dot product: " +
                                        std::to_string(a.shape(0)) + " vs " + std::to_string(b.shape(0)));
            }
            
            // Result is a scalar (0D tensor)
            auto result = zeros({}, a.dtype());
            
            // For F32 tensors, compute dot product
            if (a.dtype() == DataType::F32 && b.dtype() == DataType::F32) {
                float* a_data = static_cast<float*>(const_cast<void*>(a.data()));
                float* b_data = static_cast<float*>(const_cast<void*>(b.data()));
                float* result_data = static_cast<float*>(result.data());
                
                float dot = 0.0f;
                for (size_t i = 0; i < a.shape(0); i++) {
                    dot += a_data[i] * b_data[i];
                }
                *result_data = dot;
            }
            
            return result;
        }
        
        // Standard matrix multiplication case (2D or higher)
        size_t a_rows, a_cols, b_rows, b_cols;
        
        if (a.ndim() == 1) {
            // Treat vector as a 1 x N matrix
            a_rows = 1;
            a_cols = a.shape(0);
        } else {
            a_rows = a.shape(a.ndim() - 2);
            a_cols = a.shape(a.ndim() - 1);
        }
        
        if (b.ndim() == 1) {
            // Treat vector as a N x 1 matrix
            b_rows = b.shape(0);
            b_cols = 1;
        } else {
            b_rows = b.shape(b.ndim() - 2);
            b_cols = b.shape(b.ndim() - 1);
        }
        
        // Check for compatibility
        if (a_cols != b_rows) {
            std::stringstream error_msg;
            error_msg << "Incompatible dimensions for matrix multiplication: " 
                      << shape_to_string(a.shape()) << " and " << shape_to_string(b.shape())
                      << " (inner dimensions mismatch: " << a_cols << " vs " << b_rows << ")";
                    
            if (a_cols == b_cols && a_rows == b_rows) {
                // Common mistake: trying to do element-wise multiplication with matmul
                error_msg << ". Did you mean to use element-wise multiplication instead?";
            } else if (a_cols == b_rows - 1 || a_cols == b_rows + 1) {
                // Off-by-one error
                error_msg << ". Looks like an off-by-one error in tensor dimensions.";
            }
            
            throw std::runtime_error(error_msg.str());
        }
        
        // Calculate result shape
        std::vector<size_t> result_shape;
        
        // Handle batched matmul for higher dimensions
        if (a.ndim() > 2 || b.ndim() > 2) {
            // This is a simplified implementation that assumes same batch dimensions
            // A full implementation would handle broadcasting of batch dimensions
            
            // For now, we'll only handle the case where both have the same number of dimensions
            if (a.ndim() != b.ndim()) {
                throw std::runtime_error("Batched matmul requires tensors with same number of dimensions");
            }
            
            // Copy batch dimensions from the first tensor
            for (int i = 0; i < a.ndim() - 2; i++) {
                result_shape.push_back(a.shape(i));
            }
        }
        
        // Add matrix dimensions
        if (a.ndim() == 1) {
            // Result is a vector with shape [b_cols]
            result_shape.push_back(b_cols);
        } else if (b.ndim() == 1) {
            // Result is a vector with shape [a_rows]
            result_shape.push_back(a_rows);
        } else {
            // Result is a matrix with shape [a_rows, b_cols]
            result_shape.push_back(a_rows);
            result_shape.push_back(b_cols);
        }
        
        // Create result tensor
        auto result = zeros(result_shape, a.dtype());
        
        // For F32 tensors, we can implement a simple matmul for 2D case
        if (a.dtype() == DataType::F32 && b.dtype() == DataType::F32 && a.ndim() == 2 && b.ndim() == 2) {
            float* a_data = static_cast<float*>(const_cast<void*>(a.data()));
            float* b_data = static_cast<float*>(const_cast<void*>(b.data()));
            float* result_data = static_cast<float*>(result.data());
            
            size_t m = a_rows;
            size_t n = b_cols;
            size_t k = a_cols;
            
            for (size_t i = 0; i < m; i++) {
                for (size_t j = 0; j < n; j++) {
                    float sum = 0.0f;
                    for (size_t p = 0; p < k; p++) {
                        sum += a_data[i * k + p] * b_data[p * n + j];
                    }
                    result_data[i * n + j] = sum;
                }
            }
        } else {
            std::cerr << "Warning: Matmul implementation is limited. Using zeros for result shape: "
                      << shape_to_string(result_shape) << std::endl;
        }
        
        return result;
    }
    
    Tensor relu(const Tensor& x) override {
        // Create result tensor with same shape and type
        auto result = zeros(x.shape(), x.dtype());
        
        // Handle based on data type
        switch (x.dtype()) {
            case DataType::F32: {
                float* x_data = static_cast<float*>(const_cast<void*>(x.data()));
                float* result_data = static_cast<float*>(result.data());
                
                for (size_t i = 0; i < x.size(); i++) {
                    result_data[i] = std::max(0.0f, x_data[i]);
                }
                break;
            }
            case DataType::F16:
            case DataType::BF16: {
                // In a real implementation, we'd handle F16/BF16 directly
                // For now, just convert to F32, compute, and convert back
                std::cerr << "Warning: Using float32 fallback for " << x.dtype_str() 
                          << " in relu. This is inefficient." << std::endl;
                
                // Copy data to float buffer
                std::vector<float> float_data(x.size());
                if (x.dtype() == DataType::F16) {
                    // Convert F16 to F32 (simplified)
                    // In a real implementation, we'd use proper F16 to F32 conversion
                    uint16_t* src = static_cast<uint16_t*>(const_cast<void*>(x.data()));
                    for (size_t i = 0; i < x.size(); i++) {
                        // This is a placeholder for actual F16 to F32 conversion
                        float_data[i] = static_cast<float>(src[i]) / 100.0f;
                    }
                } else {
                    // Convert BF16 to F32 (simplified)
                    uint16_t* src = static_cast<uint16_t*>(const_cast<void*>(x.data()));
                    for (size_t i = 0; i < x.size(); i++) {
                        // This is a placeholder for actual BF16 to F32 conversion
                        float_data[i] = static_cast<float>(src[i]) / 100.0f;
                    }
                }
                
                // Apply relu
                for (size_t i = 0; i < x.size(); i++) {
                    float_data[i] = std::max(0.0f, float_data[i]);
                }
                
                // Copy back to result (simplified)
                if (x.dtype() == DataType::F16) {
                    uint16_t* dst = static_cast<uint16_t*>(result.data());
                    for (size_t i = 0; i < x.size(); i++) {
                        // This is a placeholder for actual F32 to F16 conversion
                        dst[i] = static_cast<uint16_t>(float_data[i] * 100.0f);
                    }
                } else {
                    uint16_t* dst = static_cast<uint16_t*>(result.data());
                    for (size_t i = 0; i < x.size(); i++) {
                        // This is a placeholder for actual F32 to BF16 conversion
                        dst[i] = static_cast<uint16_t>(float_data[i] * 100.0f);
                    }
                }
                break;
            }
            case DataType::I32:
            case DataType::I16:
            case DataType::I8: {
                // For integer types, relu is max(0, x)
                if (x.dtype() == DataType::I32) {
                    int32_t* x_data = static_cast<int32_t*>(const_cast<void*>(x.data()));
                    int32_t* result_data = static_cast<int32_t*>(result.data());
                    
                    for (size_t i = 0; i < x.size(); i++) {
                        result_data[i] = std::max(0, x_data[i]);
                    }
                } else if (x.dtype() == DataType::I16) {
                    int16_t* x_data = static_cast<int16_t*>(const_cast<void*>(x.data()));
                    int16_t* result_data = static_cast<int16_t*>(result.data());
                    
                    for (size_t i = 0; i < x.size(); i++) {
                        result_data[i] = std::max(static_cast<int16_t>(0), x_data[i]);
                    }
                } else { // I8
                    int8_t* x_data = static_cast<int8_t*>(const_cast<void*>(x.data()));
                    int8_t* result_data = static_cast<int8_t*>(result.data());
                    
                    for (size_t i = 0; i < x.size(); i++) {
                        result_data[i] = std::max(static_cast<int8_t>(0), x_data[i]);
                    }
                }
                break;
            }
            default:
                // For quantized types, we'd implement specialized versions
                std::cerr << "Warning: ReLU not implemented for data type " 
                          << x.dtype_str() << ". Returning zeros." << std::endl;
        }
        
        return result;
    }
    
    Tensor gelu(const Tensor& x) override {
        // Create result tensor
        auto result = zeros(x.shape(), x.dtype());
        
        // For F32 tensors, we can implement the operation
        if (x.dtype() == DataType::F32) {
            float* x_data = static_cast<float*>(const_cast<void*>(x.data()));
            float* result_data = static_cast<float*>(result.data());
            
            for (size_t i = 0; i < x.size(); i++) {
                // Approximate GELU with: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                float x_val = x_data[i];
                float x3 = x_val * x_val * x_val;
                float inner = 0.797885f * (x_val + 0.044715f * x3); // sqrt(2/pi) = 0.797885
                result_data[i] = 0.5f * x_val * (1.0f + std::tanh(inner));
            }
        }
        
        return result;
    }
    
    Tensor silu(const Tensor& x) override {
        // Create result tensor
        auto result = zeros(x.shape(), x.dtype());
        
        // For F32 tensors, we can implement the operation
        if (x.dtype() == DataType::F32) {
            float* x_data = static_cast<float*>(const_cast<void*>(x.data()));
            float* result_data = static_cast<float*>(result.data());
            
            for (size_t i = 0; i < x.size(); i++) {
                // SiLU = x * sigmoid(x)
                float x_val = x_data[i];
                float sigmoid_x = 1.0f / (1.0f + std::exp(-x_val));
                result_data[i] = x_val * sigmoid_x;
            }
        }
        
        return result;
    }
    
    Tensor softmax(const Tensor& x, int dim) override {
        // Check dimension validity
        if (dim < 0 || dim >= x.ndim()) {
            throw std::out_of_range("Dimension index out of range");
        }
        
        // Create result tensor
        auto result = zeros(x.shape(), x.dtype());
        
        // Only implement for F32 for now
        if (x.dtype() != DataType::F32) {
            std::cerr << "Warning: Softmax implementation only supports F32 tensor type. Got: " 
                      << x.dtype_str() << std::endl;
            return result;
        }
        
        float* x_data = static_cast<float*>(const_cast<void*>(x.data()));
        float* result_data = static_cast<float*>(result.data());
        
        // Handle different dimensionality
        if (x.ndim() == 1) {
            // 1D tensor case - softmax over the entire tensor
            if (dim != 0) {
                throw std::runtime_error("Invalid dimension for 1D tensor in softmax: " + std::to_string(dim));
            }
            
            size_t size = x.size();
            
            // Find max for numerical stability
            float max_val = -std::numeric_limits<float>::infinity();
            for (size_t i = 0; i < size; i++) {
                max_val = std::max(max_val, x_data[i]);
            }
            
            // Compute exp(x - max) and sum
            float sum = 0.0f;
            for (size_t i = 0; i < size; i++) {
                float exp_val = std::exp(x_data[i] - max_val);
                result_data[i] = exp_val;
                sum += exp_val;
            }
            
            // Normalize
            for (size_t i = 0; i < size; i++) {
                result_data[i] /= sum;
            }
        }
        else if (x.ndim() == 2) {
            // 2D tensor case (matrix)
            size_t rows = x.shape(0);
            size_t cols = x.shape(1);
            
            if (dim == 0) {
                // Softmax over columns
                for (size_t j = 0; j < cols; j++) {
                    // Find max for numerical stability
                    float max_val = -std::numeric_limits<float>::infinity();
                    for (size_t i = 0; i < rows; i++) {
                        max_val = std::max(max_val, x_data[i * cols + j]);
                    }
                    
                    // Compute exp(x - max) and sum
                    float sum = 0.0f;
                    for (size_t i = 0; i < rows; i++) {
                        float exp_val = std::exp(x_data[i * cols + j] - max_val);
                        result_data[i * cols + j] = exp_val;
                        sum += exp_val;
                    }
                    
                    // Normalize
                    for (size_t i = 0; i < rows; i++) {
                        result_data[i * cols + j] /= sum;
                    }
                }
            } 
            else if (dim == 1) {
                // Softmax over rows
                for (size_t i = 0; i < rows; i++) {
                    // Find max for numerical stability
                    float max_val = -std::numeric_limits<float>::infinity();
                    for (size_t j = 0; j < cols; j++) {
                        max_val = std::max(max_val, x_data[i * cols + j]);
                    }
                    
                    // Compute exp(x - max) and sum
                    float sum = 0.0f;
                    for (size_t j = 0; j < cols; j++) {
                        float exp_val = std::exp(x_data[i * cols + j] - max_val);
                        result_data[i * cols + j] = exp_val;
                        sum += exp_val;
                    }
                    
                    // Normalize
                    for (size_t j = 0; j < cols; j++) {
                        result_data[i * cols + j] /= sum;
                    }
                }
            }
        }
        else if (x.ndim() == 3) {
            // 3D tensor case
            size_t batch = x.shape(0);
            size_t rows = x.shape(1);
            size_t cols = x.shape(2);
            
            // Handle softmax over different dimensions
            if (dim == 0) {
                // Softmax over batch dimension (for each position in the rows,cols grid)
                for (size_t i = 0; i < rows; i++) {
                    for (size_t j = 0; j < cols; j++) {
                        // Find max
                        float max_val = -std::numeric_limits<float>::infinity();
                        for (size_t b = 0; b < batch; b++) {
                            max_val = std::max(max_val, x_data[b * rows * cols + i * cols + j]);
                        }
                        
                        // Compute exp and sum
                        float sum = 0.0f;
                        for (size_t b = 0; b < batch; b++) {
                            float exp_val = std::exp(x_data[b * rows * cols + i * cols + j] - max_val);
                            result_data[b * rows * cols + i * cols + j] = exp_val;
                            sum += exp_val;
                        }
                        
                        // Normalize
                        for (size_t b = 0; b < batch; b++) {
                            result_data[b * rows * cols + i * cols + j] /= sum;
                        }
                    }
                }
            }
            else if (dim == 1) {
                // Softmax over rows dimension (for each batch and column)
                for (size_t b = 0; b < batch; b++) {
                    for (size_t j = 0; j < cols; j++) {
                        // Find max
                        float max_val = -std::numeric_limits<float>::infinity();
                        for (size_t i = 0; i < rows; i++) {
                            max_val = std::max(max_val, x_data[b * rows * cols + i * cols + j]);
                        }
                        
                        // Compute exp and sum
                        float sum = 0.0f;
                        for (size_t i = 0; i < rows; i++) {
                            float exp_val = std::exp(x_data[b * rows * cols + i * cols + j] - max_val);
                            result_data[b * rows * cols + i * cols + j] = exp_val;
                            sum += exp_val;
                        }
                        
                        // Normalize
                        for (size_t i = 0; i < rows; i++) {
                            result_data[b * rows * cols + i * cols + j] /= sum;
                        }
                    }
                }
            }
            else if (dim == 2) {
                // Softmax over cols dimension (for each batch and row)
                for (size_t b = 0; b < batch; b++) {
                    for (size_t i = 0; i < rows; i++) {
                        // Find max
                        float max_val = -std::numeric_limits<float>::infinity();
                        for (size_t j = 0; j < cols; j++) {
                            max_val = std::max(max_val, x_data[b * rows * cols + i * cols + j]);
                        }
                        
                        // Compute exp and sum
                        float sum = 0.0f;
                        for (size_t j = 0; j < cols; j++) {
                            float exp_val = std::exp(x_data[b * rows * cols + i * cols + j] - max_val);
                            result_data[b * rows * cols + i * cols + j] = exp_val;
                            sum += exp_val;
                        }
                        
                        // Normalize
                        for (size_t j = 0; j < cols; j++) {
                            result_data[b * rows * cols + i * cols + j] /= sum;
                        }
                    }
                }
            }
        }
        else {
            // For higher dimensions, we'd need a more general implementation
            // that computes strides and iterates properly
            std::cerr << "Warning: Softmax not implemented for tensors with more than 3 dimensions." << std::endl;
        }
        
        return result;
    }
    
    Tensor zeros(const std::vector<size_t>& shape, DataType dtype) override {
        // Calculate total size
        size_t total_size = 1;
        for (auto dim : shape) {
            total_size *= dim;
        }
        
        // Create data buffer filled with zeros
        size_t bytes_per_element;
        switch (dtype) {
            case DataType::F32:
            case DataType::I32:
                bytes_per_element = 4;
                break;
            case DataType::F16:
            case DataType::BF16:
            case DataType::I16:
                bytes_per_element = 2;
                break;
            case DataType::I8:
            case DataType::Q8_0:
                bytes_per_element = 1;
                break;
            case DataType::Q4_0:
            case DataType::Q4_1:
                bytes_per_element = 1; // We'll handle the special case below
                break;
            default:
                throw std::runtime_error("Unsupported data type");
        }
        
        size_t total_bytes = total_size * bytes_per_element;
        
        // Special handling for 4-bit quantized types
        if (dtype == DataType::Q4_0 || dtype == DataType::Q4_1) {
            // Round up to bytes (2 elements per byte)
            total_bytes = (total_size + 1) / 2;
        }
        
        std::vector<char> data(total_bytes, 0);
        
        // Create tensor from data
        return TensorFactory::from_data(data.data(), shape, dtype);
    }
    
    Tensor ones(const std::vector<size_t>& shape, DataType dtype) override {
        // Calculate total size
        size_t total_size = 1;
        for (auto dim : shape) {
            total_size *= dim;
        }
        
        // Create data buffer filled with ones
        std::vector<float> ones_data(total_size, 1.0f);
        
        // Create tensor from data
        return TensorFactory::from_data(ones_data.data(), shape, dtype);
    }
    
    Tensor sum(const Tensor& x, int dim) override {
        // Check dimension validity
        if (dim < 0 || dim >= x.ndim()) {
            throw std::out_of_range("Dimension index out of range");
        }
        
        // Calculate result shape - the reduced dimension becomes 1
        std::vector<size_t> result_shape = x.shape();
        result_shape[dim] = 1;
        
        // Create result tensor
        auto result = zeros(result_shape, x.dtype());
        
        // Only implement for F32 for now
        if (x.dtype() != DataType::F32) {
            std::cerr << "Warning: Sum implementation only supports F32 tensor type. Got: " 
                      << x.dtype_str() << std::endl;
            return result;
        }
        
        float* x_data = static_cast<float*>(const_cast<void*>(x.data()));
        float* result_data = static_cast<float*>(result.data());
        
        // Handle based on tensor dimensionality
        if (x.ndim() == 1) {
            // 1D tensor case - sum over the only dimension
            if (dim != 0) {
                throw std::runtime_error("Invalid dimension for 1D tensor in sum: " + std::to_string(dim));
            }
            
            float sum = 0.0f;
            for (size_t i = 0; i < x.shape(0); i++) {
                sum += x_data[i];
            }
            result_data[0] = sum;
        }
        else if (x.ndim() == 2) {
            // 2D tensor case (matrix)
            size_t rows = x.shape(0);
            size_t cols = x.shape(1);
            
            if (dim == 0) {
                // Sum over rows - result has shape [1, cols]
                for (size_t j = 0; j < cols; j++) {
                    float sum = 0.0f;
                    for (size_t i = 0; i < rows; i++) {
                        sum += x_data[i * cols + j];
                    }
                    result_data[j] = sum;
                }
            } 
            else if (dim == 1) {
                // Sum over columns - result has shape [rows, 1]
                for (size_t i = 0; i < rows; i++) {
                    float sum = 0.0f;
                    for (size_t j = 0; j < cols; j++) {
                        sum += x_data[i * cols + j];
                    }
                    result_data[i] = sum;
                }
            }
        }
        else if (x.ndim() == 3) {
            // 3D tensor case
            size_t dim0 = x.shape(0);
            size_t dim1 = x.shape(1);
            size_t dim2 = x.shape(2);
            
            if (dim == 0) {
                // Sum over first dimension - result has shape [1, dim1, dim2]
                for (size_t i = 0; i < dim1; i++) {
                    for (size_t j = 0; j < dim2; j++) {
                        float sum = 0.0f;
                        for (size_t b = 0; b < dim0; b++) {
                            sum += x_data[b * dim1 * dim2 + i * dim2 + j];
                        }
                        // Result has layout [0, i, j] = [i * dim2 + j]
                        result_data[i * dim2 + j] = sum;
                    }
                }
            }
            else if (dim == 1) {
                // Sum over second dimension - result has shape [dim0, 1, dim2]
                for (size_t b = 0; b < dim0; b++) {
                    for (size_t j = 0; j < dim2; j++) {
                        float sum = 0.0f;
                        for (size_t i = 0; i < dim1; i++) {
                            sum += x_data[b * dim1 * dim2 + i * dim2 + j];
                        }
                        // Result has layout [b, 0, j] = [b * dim2 + j]
                        result_data[b * dim2 + j] = sum;
                    }
                }
            }
            else if (dim == 2) {
                // Sum over third dimension - result has shape [dim0, dim1, 1]
                for (size_t b = 0; b < dim0; b++) {
                    for (size_t i = 0; i < dim1; i++) {
                        float sum = 0.0f;
                        for (size_t j = 0; j < dim2; j++) {
                            sum += x_data[b * dim1 * dim2 + i * dim2 + j];
                        }
                        // Result has layout [b, i, 0] = [b * dim1 + i]
                        result_data[b * dim1 + i] = sum;
                    }
                }
            }
        }
        else {
            // For higher dimensions, we'd need a more general implementation
            // that computes strides and iterates properly
            std::cerr << "Warning: Sum not implemented for tensors with more than 3 dimensions." << std::endl;
        }
        
        return result;
    }
    
    Tensor mean(const Tensor& x, int dim) override {
        // Check dimension validity
        if (dim < 0 || dim >= x.ndim()) {
            throw std::out_of_range("Dimension index out of range");
        }
        
        // Get the sum first
        Tensor sum_result = sum(x, dim);
        
        // Only implement for F32 for now
        if (x.dtype() != DataType::F32) {
            std::cerr << "Warning: Mean implementation only supports F32 tensor type. Got: " 
                      << x.dtype_str() << std::endl;
            return sum_result;
        }
        
        // Divide by count to get mean
        float* result_data = static_cast<float*>(sum_result.data());
        float count = static_cast<float>(x.shape(dim));
        
        // Don't divide by zero
        if (count <= 0) {
            std::cerr << "Warning: Division by zero in mean operation. Dimension " 
                      << dim << " has size " << count << std::endl;
            return sum_result;
        }
        
        // Apply division to all elements
        for (size_t i = 0; i < sum_result.size(); i++) {
            result_data[i] /= count;
        }
        
        return sum_result;
    }
    
    std::string backend() const override {
        return "simple";
    }
    
    Tensor cast(const Tensor& x, DataType dtype) override {
        // Check if already the requested type
        if (x.dtype() == dtype) {
            return x;
        }
        
        // Create a result tensor with the new type
        auto result = zeros(x.shape(), dtype);
        
        // Handle different conversion cases
        // For simplicity, we'll just handle float32 to/from basic types
        if (x.dtype() == DataType::F32) {
            float* x_data = static_cast<float*>(const_cast<void*>(x.data()));
            
            switch (dtype) {
                case DataType::I32: {
                    int32_t* result_data = static_cast<int32_t*>(result.data());
                    for (size_t i = 0; i < x.size(); i++) {
                        result_data[i] = static_cast<int32_t>(x_data[i]);
                    }
                    break;
                }
                case DataType::I16: {
                    int16_t* result_data = static_cast<int16_t*>(result.data());
                    for (size_t i = 0; i < x.size(); i++) {
                        result_data[i] = static_cast<int16_t>(x_data[i]);
                    }
                    break;
                }
                case DataType::I8: {
                    int8_t* result_data = static_cast<int8_t*>(result.data());
                    for (size_t i = 0; i < x.size(); i++) {
                        result_data[i] = static_cast<int8_t>(x_data[i]);
                    }
                    break;
                }
                case DataType::F16:
                case DataType::BF16: {
                    // Simplified F16/BF16 conversion for demo purposes
                    uint16_t* result_data = static_cast<uint16_t*>(result.data());
                    for (size_t i = 0; i < x.size(); i++) {
                        // Simplified conversion - just store the bits
                        result_data[i] = static_cast<uint16_t>(x_data[i] * 100.0f);
                    }
                    break;
                }
                case DataType::Q8_0:
                case DataType::Q4_0:
                case DataType::Q4_1: {
                    // For quantized types, we'd need more complex logic
                    std::cerr << "Warning: Conversion to quantized types not properly implemented yet." << std::endl;
                    break;
                }
                default:
                    std::cerr << "Warning: Unhandled cast destination type: " << static_cast<int>(dtype) << std::endl;
            }
        } else if (dtype == DataType::F32) {
            // Conversion to F32 from other types
            float* result_data = static_cast<float*>(result.data());
            
            switch (x.dtype()) {
                case DataType::I32: {
                    int32_t* x_data = static_cast<int32_t*>(const_cast<void*>(x.data()));
                    for (size_t i = 0; i < x.size(); i++) {
                        result_data[i] = static_cast<float>(x_data[i]);
                    }
                    break;
                }
                case DataType::I16: {
                    int16_t* x_data = static_cast<int16_t*>(const_cast<void*>(x.data()));
                    for (size_t i = 0; i < x.size(); i++) {
                        result_data[i] = static_cast<float>(x_data[i]);
                    }
                    break;
                }
                case DataType::I8: {
                    int8_t* x_data = static_cast<int8_t*>(const_cast<void*>(x.data()));
                    for (size_t i = 0; i < x.size(); i++) {
                        result_data[i] = static_cast<float>(x_data[i]);
                    }
                    break;
                }
                case DataType::F16:
                case DataType::BF16: {
                    // Simplified F16/BF16 conversion for demo purposes
                    uint16_t* x_data = static_cast<uint16_t*>(const_cast<void*>(x.data()));
                    for (size_t i = 0; i < x.size(); i++) {
                        // Simplified conversion - just interpret the bits
                        result_data[i] = static_cast<float>(x_data[i]) / 100.0f;
                    }
                    break;
                }
                case DataType::Q8_0:
                case DataType::Q4_0:
                case DataType::Q4_1: {
                    // For quantized types, we'd need more complex logic
                    std::cerr << "Warning: Conversion from quantized types not properly implemented yet." << std::endl;
                    break;
                }
                default:
                    std::cerr << "Warning: Unhandled cast source type: " << static_cast<int>(x.dtype()) << std::endl;
            }
        } else {
            // For conversions between non-F32 types, convert via F32 as an intermediate step
            // This is not efficient but simpler to implement
            std::cerr << "Warning: Using F32 as intermediate type for conversion from "
                      << x.dtype_str() << " to " << static_cast<int>(dtype) << std::endl;
            
            // First convert to F32
            auto f32_tensor = cast(x, DataType::F32);
            
            // Then convert from F32 to the target type
            return cast(f32_tensor, dtype);
        }
        
        return result;
    }
    
    DataType promote_types(DataType a, DataType b) override {
        // If types are the same, no promotion needed
        if (a == b) {
            return a;
        }
        
        // Define a type hierarchy for promotion
        auto get_rank = [](DataType type) -> int {
            switch (type) {
                // Floating point types have highest priority
                case DataType::F32:  return 100;
                case DataType::F16:  return 90;
                case DataType::BF16: return 80;
                
                // Integer types have middle priority
                case DataType::I32:  return 70;
                case DataType::I16:  return 60;
                case DataType::I8:   return 50;
                
                // Quantized types have lowest priority
                case DataType::Q8_0: return 40;
                case DataType::Q4_1: return 30;
                case DataType::Q4_0: return 20;
                
                default:             return 0;
            }
        };
        
        // Return the type with higher rank
        return (get_rank(a) >= get_rank(b)) ? a : b;
    }
};

// ContextFactory implementation
std::shared_ptr<Context> ContextFactory::create(const std::string& backend) {
    // For now, just return a SimpleContext
    // A real implementation would select based on backend and availability
    return std::make_shared<SimpleContext>();
}

// Helpers for serialization
namespace {
    // Helper to convert between endian formats
    void convert_endianness(void* data, size_t size_bytes, EndianFormat from, EndianFormat to) {
        if (from == to || size_bytes <= 1) {
            return; // No conversion needed
        }
        
        uint8_t* bytes = static_cast<uint8_t*>(data);
        
        if (size_bytes == 2) {
            std::swap(bytes[0], bytes[1]);
        } else if (size_bytes == 4) {
            std::swap(bytes[0], bytes[3]);
            std::swap(bytes[1], bytes[2]);
        } else if (size_bytes == 8) {
            std::swap(bytes[0], bytes[7]);
            std::swap(bytes[1], bytes[6]);
            std::swap(bytes[2], bytes[5]);
            std::swap(bytes[3], bytes[4]);
        }
    }
    
    // Get native endianness
    EndianFormat get_native_endianness() {
        uint16_t value = 0x0102;
        uint8_t* ptr = reinterpret_cast<uint8_t*>(&value);
        return (ptr[0] == 0x01) ? EndianFormat::BIG : EndianFormat::LITTLE;
    }
    
    // Helper to compress data
    std::vector<uint8_t> compress_data(const void* data, size_t data_size, CompressionLevel level) {
        // For simplicity in this implementation, we just copy the data
        // A real implementation would use a compression algorithm (e.g., zlib)
        const uint8_t* src = static_cast<const uint8_t*>(data);
        std::vector<uint8_t> compressed(src, src + data_size);
        
        // Apply simple run-length encoding for demonstration
        if (level != CompressionLevel::NONE) {
            // Some simple form of compression for demonstration
            // (not actually efficient, just for demonstration)
            std::vector<uint8_t> simple_compressed;
            simple_compressed.reserve(data_size); // Worst case
            
            for (size_t i = 0; i < data_size; ) {
                uint8_t current = src[i];
                size_t run_length = 1;
                
                while (i + run_length < data_size && src[i + run_length] == current && run_length < 255) {
                    run_length++;
                }
                
                if (run_length >= 4) {
                    // Encode run
                    simple_compressed.push_back(0); // Special marker
                    simple_compressed.push_back(static_cast<uint8_t>(run_length));
                    simple_compressed.push_back(current);
                    i += run_length;
                } else {
                    // Direct copy
                    for (size_t j = 0; j < run_length; j++) {
                        simple_compressed.push_back(src[i++]);
                    }
                }
            }
            
            return simple_compressed;
        }
        
        return compressed;
    }
    
    // Helper to decompress data
    std::vector<uint8_t> decompress_data(const void* data, size_t compressed_size, size_t original_size) {
        // For simplicity in this implementation, we just copy the data
        // A real implementation would use a decompression algorithm (e.g., zlib)
        const uint8_t* src = static_cast<const uint8_t*>(data);
        
        // Check for simple run-length encoding
        // Look for the marker byte
        bool is_compressed = false;
        for (size_t i = 0; i < std::min(compressed_size, size_t(100)); i++) {
            if (src[i] == 0 && i + 2 < compressed_size) {
                is_compressed = true;
                break;
            }
        }
        
        if (!is_compressed) {
            // Direct copy
            return std::vector<uint8_t>(src, src + compressed_size);
        }
        
        // Decompress our simple RLE format
        std::vector<uint8_t> decompressed;
        decompressed.reserve(original_size);
        
        for (size_t i = 0; i < compressed_size; ) {
            if (src[i] == 0 && i + 2 < compressed_size) {
                // Run-length encoded block
                size_t run_length = src[i + 1];
                uint8_t value = src[i + 2];
                
                for (size_t j = 0; j < run_length; j++) {
                    decompressed.push_back(value);
                }
                
                i += 3;
            } else {
                // Direct value
                decompressed.push_back(src[i]);
                i++;
            }
            
            if (decompressed.size() >= original_size) {
                break;
            }
        }
        
        return decompressed;
    }
}

// TensorFactory serialization implementations
bool TensorFactory::save(const Tensor& tensor, const std::string& filepath, 
                        EndianFormat endian, CompressionLevel compression) {
    // Open file for writing
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filepath << std::endl;
        return false;
    }
    
    // Write magic number to identify format
    const uint32_t magic = 0x54534F52; // "TSRZ" in little endian
    file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    
    // Write format version
    const uint32_t version = 1;
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));
    
    // Write tensor metadata
    const int ndim = tensor.ndim();
    file.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
    
    const DataType dtype = tensor.dtype();
    file.write(reinterpret_cast<const char*>(&dtype), sizeof(dtype));
    
    // Write shape
    auto shape = tensor.shape();
    for (const auto& dim : shape) {
        uint64_t dim_size = dim;
        file.write(reinterpret_cast<const char*>(&dim_size), sizeof(dim_size));
    }
    
    // Calculate data size
    size_t total_elements = tensor.size();
    size_t element_size;
    switch (dtype) {
        case DataType::F32:
        case DataType::I32:
            element_size = 4;
            break;
        case DataType::F16:
        case DataType::BF16:
        case DataType::I16:
            element_size = 2;
            break;
        case DataType::I8:
        case DataType::Q8_0:
            element_size = 1;
            break;
        case DataType::Q4_0:
        case DataType::Q4_1:
            element_size = 1; // 4-bit types are packed, 2 elements per byte
            total_elements = (total_elements + 1) / 2; // Round up
            break;
        default:
            std::cerr << "Unknown data type in serialization" << std::endl;
            return false;
    }
    
    size_t data_size = total_elements * element_size;
    
    // Write endian format
    uint8_t endian_code = static_cast<uint8_t>(endian);
    file.write(reinterpret_cast<const char*>(&endian_code), sizeof(endian_code));
    
    // Write compression level
    uint8_t compression_code = static_cast<uint8_t>(compression);
    file.write(reinterpret_cast<const char*>(&compression_code), sizeof(compression_code));
    
    // Get a copy of the data to potentially convert endianness
    std::vector<uint8_t> data_copy(data_size);
    memcpy(data_copy.data(), tensor.data(), data_size);
    
    // Convert endianness if needed
    EndianFormat native = get_native_endianness();
    if (endian != EndianFormat::NATIVE && endian != native) {
        size_t bytes_per_element = element_size;
        for (size_t i = 0; i < total_elements; i++) {
            convert_endianness(data_copy.data() + i * bytes_per_element, bytes_per_element, native, endian);
        }
    }
    
    // Compress the data if requested
    std::vector<uint8_t> compressed_data;
    if (compression != CompressionLevel::NONE) {
        compressed_data = compress_data(data_copy.data(), data_size, compression);
    } else {
        compressed_data = data_copy;
    }
    
    // Write original data size
    uint64_t orig_size = data_size;
    file.write(reinterpret_cast<const char*>(&orig_size), sizeof(orig_size));
    
    // Write compressed data size
    uint64_t comp_size = compressed_data.size();
    file.write(reinterpret_cast<const char*>(&comp_size), sizeof(comp_size));
    
    // Write the data
    file.write(reinterpret_cast<const char*>(compressed_data.data()), compressed_data.size());
    
    return true;
}

Tensor TensorFactory::load(const std::string& filepath, EndianFormat endian) {
    // Open file for reading
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }
    
    // Read and check magic number
    uint32_t magic;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != 0x54534F52) { // "TSRZ" in little endian
        throw std::runtime_error("Invalid file format, incorrect magic number");
    }
    
    // Read format version
    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != 1) {
        throw std::runtime_error("Unsupported format version: " + std::to_string(version));
    }
    
    // Read tensor metadata
    int ndim;
    file.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
    
    DataType dtype;
    file.read(reinterpret_cast<char*>(&dtype), sizeof(dtype));
    
    // Read shape
    std::vector<size_t> shape(ndim);
    for (int i = 0; i < ndim; i++) {
        uint64_t dim_size;
        file.read(reinterpret_cast<char*>(&dim_size), sizeof(dim_size));
        shape[i] = dim_size;
    }
    
    // Read endian format
    uint8_t endian_code;
    file.read(reinterpret_cast<char*>(&endian_code), sizeof(endian_code));
    EndianFormat file_endian = static_cast<EndianFormat>(endian_code);
    
    // Read compression level
    uint8_t compression_code;
    file.read(reinterpret_cast<char*>(&compression_code), sizeof(compression_code));
    CompressionLevel compression = static_cast<CompressionLevel>(compression_code);
    
    // Read original data size
    uint64_t orig_size;
    file.read(reinterpret_cast<char*>(&orig_size), sizeof(orig_size));
    
    // Read compressed data size
    uint64_t comp_size;
    file.read(reinterpret_cast<char*>(&comp_size), sizeof(comp_size));
    
    // Read compressed data
    std::vector<uint8_t> compressed_data(comp_size);
    file.read(reinterpret_cast<char*>(compressed_data.data()), comp_size);
    
    // Decompress if necessary
    std::vector<uint8_t> data;
    if (compression != CompressionLevel::NONE) {
        data = decompress_data(compressed_data.data(), comp_size, orig_size);
    } else {
        data = compressed_data;
    }
    
    // Handle endianness conversion if requested
    EndianFormat target_endian = (endian == EndianFormat::NATIVE) ? get_native_endianness() : endian;
    if (file_endian != target_endian) {
        size_t bytes_per_element;
        switch (dtype) {
            case DataType::F32:
            case DataType::I32:
                bytes_per_element = 4;
                break;
            case DataType::F16:
            case DataType::BF16:
            case DataType::I16:
                bytes_per_element = 2;
                break;
            case DataType::I8:
            case DataType::Q8_0:
            case DataType::Q4_0:
            case DataType::Q4_1:
                bytes_per_element = 1;
                break;
            default:
                throw std::runtime_error("Unknown data type in deserialization");
        }
        
        size_t num_elements = data.size() / bytes_per_element;
        for (size_t i = 0; i < num_elements; i++) {
            convert_endianness(data.data() + i * bytes_per_element, bytes_per_element, file_endian, target_endian);
        }
    }
    
    // Create tensor from data
    return TensorFactory::from_data(data.data(), shape, dtype);
}

bool TensorFactory::save_with_metadata(const Tensor& tensor, const std::string& filepath,
                                        const TensorMetadata& metadata,
                                        EndianFormat endian, CompressionLevel compression) {
    // Print debug information
    std::cerr << "Saving tensor with metadata to: " << filepath << std::endl;
    std::cerr << "Metadata: " << metadata.name << ", " << metadata.description << ", version=" << metadata.version << std::endl;
    
    // First save the tensor
    if (!save(tensor, filepath, endian, compression)) {
        std::cerr << "Failed to save base tensor!" << std::endl;
        return false;
    }
    
    // Verify file exists after saving
    if (!std::filesystem::exists(filepath)) {
        std::cerr << "File does not exist after saving tensor: " << filepath << std::endl;
        return false;
    }
    
    // Then append metadata to the end of the file
    std::ofstream file(filepath, std::ios::binary | std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing metadata: " << filepath << std::endl;
        return false;
    }
    
    // Write marker to indicate metadata section
    const uint32_t metadata_marker = 0x4D455441; // "META" in little endian
    file.write(reinterpret_cast<const char*>(&metadata_marker), sizeof(metadata_marker));
    
    // Write a safety marker to indicate proper metadata format
    const uint64_t safety_marker = 0x4353534D4D455441; // "CSMMETA" in little endian
    file.write(reinterpret_cast<const char*>(&safety_marker), sizeof(safety_marker));
    
    // Write name
    uint32_t name_length = metadata.name.size();
    file.write(reinterpret_cast<const char*>(&name_length), sizeof(name_length));
    file.write(metadata.name.c_str(), name_length);
    
    // Write description
    uint32_t desc_length = metadata.description.size();
    file.write(reinterpret_cast<const char*>(&desc_length), sizeof(desc_length));
    file.write(metadata.description.c_str(), desc_length);
    
    // Write version
    file.write(reinterpret_cast<const char*>(&metadata.version), sizeof(metadata.version));
    
    // Write custom fields
    uint32_t num_fields = metadata.custom_fields.size();
    file.write(reinterpret_cast<const char*>(&num_fields), sizeof(num_fields));
    
    for (const auto& field : metadata.custom_fields) {
        // Write key
        uint32_t key_length = field.first.size();
        file.write(reinterpret_cast<const char*>(&key_length), sizeof(key_length));
        file.write(field.first.c_str(), key_length);
        
        // Write value
        uint32_t value_length = field.second.size();
        file.write(reinterpret_cast<const char*>(&value_length), sizeof(value_length));
        file.write(field.second.c_str(), value_length);
    }
    
    file.close();
    
    // Verify file size after adding metadata
    std::uintmax_t file_size = std::filesystem::file_size(filepath);
    std::cerr << "File size after adding metadata: " << file_size << " bytes" << std::endl;
    
    return true;
}

Tensor TensorFactory::load_with_metadata(const std::string& filepath, 
                                         TensorMetadata& metadata,
                                         EndianFormat endian) {
    // First load the tensor
    Tensor tensor = load(filepath, endian);
    
    // Then read metadata from the end of the file
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }
    
    // Get file size
    std::streampos fileSize = file.tellg();
    
    // Look for metadata marker from the end
    const uint32_t metadata_marker = 0x4D455441; // "META" in little endian
    
    bool found_metadata = false;
    // We'll search backwards for the marker in chunks
    const int chunk_size = 1024;
    std::vector<char> buffer(chunk_size);
    
    // First, try looking for metadata at the expected position (right after the tensor data)
    // Get the file position right after the tensor data
    file.seekg(0, std::ios::beg);
    
    // Skip the magic number and version (8 bytes)
    file.seekg(8, std::ios::cur);
    
    // Read the shape information
    uint32_t ndim;
    file.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
    
    // Skip the shape dimensions
    file.seekg(ndim * sizeof(uint32_t), std::ios::cur);
    
    // Read dtype, endianness, compression
    uint32_t dtype_val, endian_val, compression_val;
    file.read(reinterpret_cast<char*>(&dtype_val), sizeof(dtype_val));
    file.read(reinterpret_cast<char*>(&endian_val), sizeof(endian_val));
    file.read(reinterpret_cast<char*>(&compression_val), sizeof(compression_val));
    
    // Read data size information
    uint64_t original_size, compressed_size;
    file.read(reinterpret_cast<char*>(&original_size), sizeof(original_size));
    file.read(reinterpret_cast<char*>(&compressed_size), sizeof(compressed_size));
    
    // Skip the data
    file.seekg(compressed_size, std::ios::cur);
    
    // Now we should be at the metadata marker
    uint32_t marker_check;
    file.read(reinterpret_cast<char*>(&marker_check), sizeof(marker_check));
    
    // Debug output
    std::cerr << "Looking for marker: 0x" << std::hex << metadata_marker 
              << ", found: 0x" << marker_check << std::dec << std::endl;
    
    // Check for metadata marker (due to endian issues, check for either byte order)
    if (marker_check == metadata_marker || 
        marker_check == 0x4154454D) { // Check for "META" in big endian too
        // Verify the safety marker
        const uint64_t expected_safety_marker = 0x4353534D4D455441; // "CSMMETA" in little endian
        uint64_t safety_marker;
        file.read(reinterpret_cast<char*>(&safety_marker), sizeof(safety_marker));
        
        if (safety_marker == expected_safety_marker) {
            found_metadata = true;
        }
    }
    
    // If not found at expected position, fall back to searching
    if (!found_metadata) {
        // Fall back to searching
        for (std::streamoff pos_off = static_cast<std::streamoff>(fileSize); 
             pos_off >= static_cast<std::streamoff>(chunk_size); 
             pos_off -= chunk_size) {
            std::streampos pos(pos_off - chunk_size);
            file.seekg(pos);
            file.read(buffer.data(), chunk_size);
            
            for (int i = chunk_size - 12; i >= 0; i--) {
                uint32_t* ptr = reinterpret_cast<uint32_t*>(buffer.data() + i);
                if (*ptr == metadata_marker || *ptr == 0x4154454D) { // Check for "META" in both endian formats
                    // Found potential marker, check the safety marker
                    file.seekg(pos + std::streamoff(i + 4));
                    uint64_t safety_marker;
                    file.read(reinterpret_cast<char*>(&safety_marker), sizeof(safety_marker));
                    const uint64_t expected_safety_marker = 0x4353534D4D455441; // "CSMMETA" in little endian
                    
                    if (safety_marker == expected_safety_marker) {
                        found_metadata = true;
                        break;
                    }
                }
            }
            
            if (found_metadata) {
                break;
            }
        }
    }
    
    if (!found_metadata) {
        // Log detailed information for debugging
        std::cerr << "Failed to find metadata in file: " << filepath << std::endl;
        std::cerr << "File size: " << fileSize << " bytes" << std::endl;
        
        // Check if file exists and is readable
        if (!std::filesystem::exists(filepath)) {
            throw std::runtime_error("Metadata file does not exist: " + filepath);
        }
        
        // Try to read the first few bytes to see if the file is valid
        file.seekg(0, std::ios::beg);
        char header[16] = {0};
        file.read(header, 16);
        std::stringstream ss;
        ss << "Header bytes: ";
        for (int i = 0; i < 16; i++) {
            ss << std::hex << std::setw(2) << std::setfill('0') 
               << static_cast<int>(static_cast<unsigned char>(header[i])) << " ";
        }
        std::cerr << ss.str() << std::endl;
        
        throw std::runtime_error("No metadata found in file: " + filepath);
    }
    
    // Read name
    uint32_t name_length;
    file.read(reinterpret_cast<char*>(&name_length), sizeof(name_length));
    
    metadata.name.resize(name_length);
    file.read(&metadata.name[0], name_length);
    
    // Read description
    uint32_t desc_length;
    file.read(reinterpret_cast<char*>(&desc_length), sizeof(desc_length));
    
    metadata.description.resize(desc_length);
    file.read(&metadata.description[0], desc_length);
    
    // Read version
    file.read(reinterpret_cast<char*>(&metadata.version), sizeof(metadata.version));
    
    // Read custom fields
    uint32_t num_fields;
    file.read(reinterpret_cast<char*>(&num_fields), sizeof(num_fields));
    
    metadata.custom_fields.clear();
    for (uint32_t i = 0; i < num_fields; i++) {
        // Read key
        uint32_t key_length;
        file.read(reinterpret_cast<char*>(&key_length), sizeof(key_length));
        
        std::string key(key_length, ' ');
        file.read(&key[0], key_length);
        
        // Read value
        uint32_t value_length;
        file.read(reinterpret_cast<char*>(&value_length), sizeof(value_length));
        
        std::string value(value_length, ' ');
        file.read(&value[0], value_length);
        
        metadata.custom_fields[key] = value;
    }
    
    return tensor;
}

} // namespace ccsm