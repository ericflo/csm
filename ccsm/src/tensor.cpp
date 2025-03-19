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
    // TODO: Implement this
    throw std::runtime_error("TensorFactory::from_data not implemented yet");
}

Tensor TensorFactory::convert(const Tensor& tensor, const std::string& to_backend) {
    // TODO: Implement this
    throw std::runtime_error("TensorFactory::convert not implemented yet");
}

// ContextFactory implementation
std::shared_ptr<Context> ContextFactory::create(const std::string& backend) {
    // TODO: Select backend based on input and availability
    throw std::runtime_error("ContextFactory::create not implemented yet");
}

} // namespace ccsm