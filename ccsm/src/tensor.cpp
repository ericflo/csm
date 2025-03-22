#include <ccsm/tensor.h>
#include <sstream>
#include <stdexcept>
#include <algorithm>

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
    
    Tensor add(const Tensor& a, const Tensor& b) override {
        // Handle scalar broadcasting for tensor + scalar operations
        bool is_scalar_op = false;
        DataType result_dtype = a.dtype();
        std::vector<size_t> result_shape = a.shape();
        
        if (a.ndim() == 0) {
            // a is scalar, b is tensor
            result_shape = b.shape();
            result_dtype = b.dtype();
            is_scalar_op = true;
        } else if (b.ndim() == 0) {
            // b is scalar, a is tensor
            is_scalar_op = true;
        } else if (a.shape() != b.shape()) {
            // Check for broadcasting compatibility between tensors
            // For now, just handle exact shape matches
            throw std::runtime_error("Shape mismatch for addition: " + 
                                    shape_to_string(a.shape()) + " vs " + 
                                    shape_to_string(b.shape()));
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
                float scalar_val = a.ndim() == 0 ? *a_data : *b_data;
                const float* tensor_data = a.ndim() == 0 ? b_data : a_data;
                size_t size = result.size();
                
                for (size_t i = 0; i < size; i++) {
                    result_data[i] = tensor_data[i] + scalar_val;
                }
            } else {
                for (size_t i = 0; i < a.size(); i++) {
                    result_data[i] = a_data[i] + b_data[i];
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
        // Handle scalar broadcasting for tensor - scalar operations
        bool is_scalar_op = false;
        DataType result_dtype = a.dtype();
        std::vector<size_t> result_shape = a.shape();
        
        if (a.ndim() == 0) {
            // a is scalar, b is tensor
            result_shape = b.shape();
            result_dtype = b.dtype();
            is_scalar_op = true;
        } else if (b.ndim() == 0) {
            // b is scalar, a is tensor
            is_scalar_op = true;
        } else if (a.shape() != b.shape()) {
            // Check for broadcasting compatibility between tensors
            // For now, just handle exact shape matches
            throw std::runtime_error("Shape mismatch for subtraction: " + 
                                    shape_to_string(a.shape()) + " vs " + 
                                    shape_to_string(b.shape()));
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
            } else {
                // Element-wise subtraction
                for (size_t i = 0; i < a.size(); i++) {
                    result_data[i] = a_data[i] - b_data[i];
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
        // Handle scalar broadcasting for tensor * scalar operations
        bool is_scalar_op = false;
        DataType result_dtype = a.dtype();
        std::vector<size_t> result_shape = a.shape();
        
        if (a.ndim() == 0) {
            // a is scalar, b is tensor
            result_shape = b.shape();
            result_dtype = b.dtype();
            is_scalar_op = true;
        } else if (b.ndim() == 0) {
            // b is scalar, a is tensor
            is_scalar_op = true;
        } else if (a.shape() != b.shape()) {
            // Check for broadcasting compatibility between tensors
            // For now, just handle exact shape matches
            throw std::runtime_error("Shape mismatch for multiplication: " + 
                                    shape_to_string(a.shape()) + " vs " + 
                                    shape_to_string(b.shape()));
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
                float scalar_val = a.ndim() == 0 ? *a_data : *b_data;
                const float* tensor_data = a.ndim() == 0 ? b_data : a_data;
                size_t size = result.size();
                
                for (size_t i = 0; i < size; i++) {
                    result_data[i] = tensor_data[i] * scalar_val;
                }
            } else {
                // Element-wise multiplication
                for (size_t i = 0; i < a.size(); i++) {
                    result_data[i] = a_data[i] * b_data[i];
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
        // Handle scalar broadcasting for tensor / scalar operations
        bool is_scalar_op = false;
        DataType result_dtype = a.dtype();
        std::vector<size_t> result_shape = a.shape();
        
        if (a.ndim() == 0) {
            // a is scalar, b is tensor
            result_shape = b.shape();
            result_dtype = b.dtype();
            is_scalar_op = true;
        } else if (b.ndim() == 0) {
            // b is scalar, a is tensor
            is_scalar_op = true;
        } else if (a.shape() != b.shape()) {
            // Check for broadcasting compatibility between tensors
            // For now, just handle exact shape matches
            throw std::runtime_error("Shape mismatch for division: " + 
                                    shape_to_string(a.shape()) + " vs " + 
                                    shape_to_string(b.shape()));
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
                if (a.ndim() == 0) {
                    // a is scalar, b is tensor: result = a / b
                    float scalar_val = *a_data;
                    size_t size = result.size();
                    
                    for (size_t i = 0; i < size; i++) {
                        if (b_data[i] == 0) {
                            throw std::runtime_error("Division by zero");
                        }
                        result_data[i] = scalar_val / b_data[i];
                    }
                } else {
                    // b is scalar, a is tensor: result = a / b
                    float scalar_val = *b_data;
                    if (scalar_val == 0) {
                        throw std::runtime_error("Division by zero");
                    }
                    
                    size_t size = result.size();
                    for (size_t i = 0; i < size; i++) {
                        result_data[i] = a_data[i] / scalar_val;
                    }
                }
            } else {
                // Element-wise division
                for (size_t i = 0; i < a.size(); i++) {
                    if (b_data[i] == 0) {
                        throw std::runtime_error("Division by zero");
                    }
                    result_data[i] = a_data[i] / b_data[i];
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
            throw std::runtime_error("Incompatible dimensions for matrix multiplication: " +
                                    shape_to_string(a.shape()) + " and " + shape_to_string(b.shape()) +
                                    " (inner dims: " + std::to_string(a_cols) + " vs " + std::to_string(b_rows) + ")");
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
};

// ContextFactory implementation
std::shared_ptr<Context> ContextFactory::create(const std::string& backend) {
    // For now, just return a SimpleContext
    // A real implementation would select based on backend and availability
    return std::make_shared<SimpleContext>();
}

} // namespace ccsm