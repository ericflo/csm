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
    // TODO: Select backend based on global settings
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
                    bytes_per_element = 0.5; // 4 bits per element
                    break;
                default:
                    throw std::runtime_error("Unsupported data type");
            }
            
            // Allocate memory and copy data
            size_t total_bytes = size_ * bytes_per_element;
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
            
            // Create new tensor
            // For this simple implementation, we'll just create a zeroed tensor
            // A real implementation would extract the appropriate slice
            auto result = std::make_shared<SimpleTensorImpl>(nullptr, new_shape, dtype_);
            
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
            std::cout << "], dtype=" << static_cast<int>(dtype_) << std::endl;
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
    
    Tensor add(const Tensor& a, const Tensor& b) override {
        // Check for shape compatibility
        if (a.shape() != b.shape()) {
            throw std::runtime_error("Shape mismatch for addition");
        }
        
        // Create result tensor
        auto result = zeros(a.shape(), a.dtype());
        
        // For F32 tensors, we can implement the operation
        if (a.dtype() == DataType::F32 && b.dtype() == DataType::F32) {
            float* a_data = static_cast<float*>(const_cast<void*>(a.data()));
            float* b_data = static_cast<float*>(const_cast<void*>(b.data()));
            float* result_data = static_cast<float*>(result.data());
            
            for (size_t i = 0; i < a.size(); i++) {
                result_data[i] = a_data[i] + b_data[i];
            }
        }
        
        return result;
    }
    
    Tensor subtract(const Tensor& a, const Tensor& b) override {
        // Check for shape compatibility
        if (a.shape() != b.shape()) {
            throw std::runtime_error("Shape mismatch for subtraction");
        }
        
        // Create result tensor
        auto result = zeros(a.shape(), a.dtype());
        
        // For F32 tensors, we can implement the operation
        if (a.dtype() == DataType::F32 && b.dtype() == DataType::F32) {
            float* a_data = static_cast<float*>(const_cast<void*>(a.data()));
            float* b_data = static_cast<float*>(const_cast<void*>(b.data()));
            float* result_data = static_cast<float*>(result.data());
            
            for (size_t i = 0; i < a.size(); i++) {
                result_data[i] = a_data[i] - b_data[i];
            }
        }
        
        return result;
    }
    
    Tensor multiply(const Tensor& a, const Tensor& b) override {
        // Check for shape compatibility
        if (a.shape() != b.shape()) {
            throw std::runtime_error("Shape mismatch for multiplication");
        }
        
        // Create result tensor
        auto result = zeros(a.shape(), a.dtype());
        
        // For F32 tensors, we can implement the operation
        if (a.dtype() == DataType::F32 && b.dtype() == DataType::F32) {
            float* a_data = static_cast<float*>(const_cast<void*>(a.data()));
            float* b_data = static_cast<float*>(const_cast<void*>(b.data()));
            float* result_data = static_cast<float*>(result.data());
            
            for (size_t i = 0; i < a.size(); i++) {
                result_data[i] = a_data[i] * b_data[i];
            }
        }
        
        return result;
    }
    
    Tensor divide(const Tensor& a, const Tensor& b) override {
        // Check for shape compatibility
        if (a.shape() != b.shape()) {
            throw std::runtime_error("Shape mismatch for division");
        }
        
        // Create result tensor
        auto result = zeros(a.shape(), a.dtype());
        
        // For F32 tensors, we can implement the operation
        if (a.dtype() == DataType::F32 && b.dtype() == DataType::F32) {
            float* a_data = static_cast<float*>(const_cast<void*>(a.data()));
            float* b_data = static_cast<float*>(const_cast<void*>(b.data()));
            float* result_data = static_cast<float*>(result.data());
            
            for (size_t i = 0; i < a.size(); i++) {
                if (b_data[i] == 0) {
                    throw std::runtime_error("Division by zero");
                }
                result_data[i] = a_data[i] / b_data[i];
            }
        }
        
        return result;
    }
    
    Tensor matmul(const Tensor& a, const Tensor& b) override {
        // Check for shape compatibility
        if (a.ndim() < 2 || b.ndim() < 2 || a.shape(a.ndim() - 1) != b.shape(b.ndim() - 2)) {
            throw std::runtime_error("Incompatible dimensions for matrix multiplication");
        }
        
        // Calculate result shape
        std::vector<size_t> result_shape = a.shape();
        result_shape[result_shape.size() - 1] = b.shape(b.ndim() - 1);
        
        // Create result tensor
        auto result = zeros(result_shape, a.dtype());
        
        // For F32 tensors, we can implement a simple matmul
        if (a.dtype() == DataType::F32 && b.dtype() == DataType::F32 && a.ndim() == 2 && b.ndim() == 2) {
            float* a_data = static_cast<float*>(const_cast<void*>(a.data()));
            float* b_data = static_cast<float*>(const_cast<void*>(b.data()));
            float* result_data = static_cast<float*>(result.data());
            
            size_t m = a.shape(0);
            size_t n = b.shape(1);
            size_t k = a.shape(1);
            
            for (size_t i = 0; i < m; i++) {
                for (size_t j = 0; j < n; j++) {
                    float sum = 0.0f;
                    for (size_t p = 0; p < k; p++) {
                        sum += a_data[i * k + p] * b_data[p * n + j];
                    }
                    result_data[i * n + j] = sum;
                }
            }
        }
        
        return result;
    }
    
    Tensor relu(const Tensor& x) override {
        // Create result tensor
        auto result = zeros(x.shape(), x.dtype());
        
        // For F32 tensors, we can implement the operation
        if (x.dtype() == DataType::F32) {
            float* x_data = static_cast<float*>(const_cast<void*>(x.data()));
            float* result_data = static_cast<float*>(result.data());
            
            for (size_t i = 0; i < x.size(); i++) {
                result_data[i] = std::max(0.0f, x_data[i]);
            }
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
        
        // For F32 tensors, we can implement the operation for 2D case
        if (x.dtype() == DataType::F32 && x.ndim() == 2 && dim == 1) {
            float* x_data = static_cast<float*>(const_cast<void*>(x.data()));
            float* result_data = static_cast<float*>(result.data());
            
            size_t rows = x.shape(0);
            size_t cols = x.shape(1);
            
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
                bytes_per_element = 0.5; // 4 bits per element
                break;
            default:
                throw std::runtime_error("Unsupported data type");
        }
        
        size_t total_bytes = total_size * bytes_per_element;
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
        
        // Calculate result shape
        std::vector<size_t> result_shape = x.shape();
        result_shape[dim] = 1;
        
        // Create result tensor
        auto result = zeros(result_shape, x.dtype());
        
        // For F32 tensors, we can implement the operation for simple cases
        if (x.dtype() == DataType::F32 && x.ndim() == 2) {
            float* x_data = static_cast<float*>(const_cast<void*>(x.data()));
            float* result_data = static_cast<float*>(result.data());
            
            size_t rows = x.shape(0);
            size_t cols = x.shape(1);
            
            if (dim == 0) {
                // Sum over rows
                for (size_t j = 0; j < cols; j++) {
                    float sum = 0.0f;
                    for (size_t i = 0; i < rows; i++) {
                        sum += x_data[i * cols + j];
                    }
                    result_data[j] = sum;
                }
            } else if (dim == 1) {
                // Sum over columns
                for (size_t i = 0; i < rows; i++) {
                    float sum = 0.0f;
                    for (size_t j = 0; j < cols; j++) {
                        sum += x_data[i * cols + j];
                    }
                    result_data[i] = sum;
                }
            }
        }
        
        return result;
    }
    
    Tensor mean(const Tensor& x, int dim) override {
        // Get sum
        Tensor sum_result = sum(x, dim);
        
        // Divide by count
        if (x.dtype() == DataType::F32) {
            float* result_data = static_cast<float*>(sum_result.data());
            float count = static_cast<float>(x.shape(dim));
            
            for (size_t i = 0; i < sum_result.size(); i++) {
                result_data[i] /= count;
            }
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