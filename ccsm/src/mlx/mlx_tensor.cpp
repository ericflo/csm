#include <ccsm/mlx/mlx_tensor.h>
#include <ccsm/utils.h>
#include <stdexcept>
#include <iostream>
#include <sstream>

// Helper function to convert shape to string for debugging
namespace {
    std::string shape_to_string(const std::vector<size_t>& shape) {
        std::stringstream ss;
        ss << "[";
        for (size_t i = 0; i < shape.size(); ++i) {
            ss << shape[i];
            if (i < shape.size() - 1) {
                ss << ", ";
            }
        }
        ss << "]";
        return ss.str();
    }
}

#ifdef CCSM_WITH_MLX
// For real implementation, use MLX headers
// #include "mlx/c/array.h"
// #include "mlx/c/ops.h"
// #include "mlx/c/device.h"
// #include "mlx/c/stream.h"
#endif

namespace ccsm {

#ifdef CCSM_WITH_MLX

// MLXDevice implementation
MLXDevice::MLXDevice() {
    CCSM_DEBUG("Creating MLXDevice with default settings");
    mlx_default_device(&device_);
}

MLXDevice::MLXDevice(mlx_device_type type, int index) {
    CCSM_DEBUG("Creating MLXDevice with type " + std::to_string(type) + " and index " + std::to_string(index));
    mlx_device_t device;
    mlx_device_init(&device, type, index);
    device_ = device;
}

MLXDevice::~MLXDevice() {
    CCSM_DEBUG("Destroying MLXDevice");
    mlx_device_free(device_);
}

mlx_device_type MLXDevice::type() const {
    mlx_device_type type;
    mlx_device_get_type(device_, &type);
    return type;
}

int MLXDevice::index() const {
    int idx;
    mlx_device_get_index(device_, &idx);
    return idx;
}

std::string MLXDevice::name() const {
    char* name;
    mlx_device_get_name(device_, &name);
    std::string device_name(name);
    free(name); // Free the string allocated by mlx_device_get_name
    return device_name;
}

bool MLXDevice::is_available() {
    CCSM_INFO("Checking MLX availability on this system");
    bool has_cpu = false;
    bool has_gpu = false;
    
    mlx_has_cpu(&has_cpu);
    mlx_has_gpu(&has_gpu);
    
    CCSM_INFO("MLX CPU available: " + std::string(has_cpu ? "true" : "false"));
    CCSM_INFO("MLX GPU available: " + std::string(has_gpu ? "true" : "false"));
    
    // MLX is available if either CPU or GPU device is available
    return has_cpu || has_gpu;
}

MLXDevice MLXDevice::default_device() {
    mlx_device_t device;
    mlx_default_device(&device);
    
    // Create device with the properties of the default device
    mlx_device_type type;
    int index;
    mlx_device_get_type(device, &type);
    mlx_device_get_index(device, &index);
    
    // Free the temporary device and return a new one with the same properties
    mlx_device_free(device);
    return MLXDevice(type, index);
}

void MLXDevice::set_default_device(const MLXDevice& device) {
    CCSM_DEBUG("Setting default MLX device");
    mlx_set_default_device(device.device());
}

void MLXDevice::synchronize() {
    CCSM_DEBUG("Synchronizing MLX device");
    mlx_device_synchronize(device_);
}

// Convert between CCSM and MLX data types
mlx_dtype MLXTensorImpl::to_mlx_dtype(DataType dtype) {
    switch (dtype) {
        case DataType::F32:  return MLX_FLOAT32;
        case DataType::F16:  return MLX_FLOAT16;
        case DataType::BF16: return MLX_BFLOAT16;
        case DataType::I32:  return MLX_INT32;
        case DataType::I16:  return MLX_INT16;
        case DataType::I8:   return MLX_INT8;
        default:
            throw std::runtime_error("Unsupported data type for MLX conversion");
    }
}

DataType MLXTensorImpl::from_mlx_dtype(mlx_dtype type) {
    switch (type) {
        case MLX_FLOAT32: return DataType::F32;
        case MLX_FLOAT16: return DataType::F16;
        case MLX_BFLOAT16: return DataType::BF16;
        case MLX_INT32: return DataType::I32;
        case MLX_INT16: return DataType::I16;
        case MLX_INT8: return DataType::I8;
        default:
            throw std::runtime_error("Unsupported MLX type for conversion");
    }
}

// MLXTensorImpl implementation
MLXTensorImpl::MLXTensorImpl(mlx_array array) {
    CCSM_DEBUG("Creating MLXTensorImpl from MLX array");
    array_ = array;
}

MLXTensorImpl::~MLXTensorImpl() {
    CCSM_DEBUG("Destroying MLXTensorImpl");
    mlx_array_free(array_);
}

size_t MLXTensorImpl::shape(int dim) const {
    uint32_t ndim;
    mlx_array_ndim(array_, &ndim);
    
    if (dim < 0 || dim >= static_cast<int>(ndim)) {
        throw std::out_of_range("Dimension index out of range");
    }
    
    std::vector<int64_t> shape(ndim);
    mlx_array_shape(array_, shape.data());
    return static_cast<size_t>(shape[dim]);
}

std::vector<size_t> MLXTensorImpl::shape() const {
    uint32_t ndim;
    mlx_array_ndim(array_, &ndim);
    
    std::vector<int64_t> mlx_shape(ndim);
    mlx_array_shape(array_, mlx_shape.data());
    
    std::vector<size_t> result(ndim);
    for (uint32_t i = 0; i < ndim; ++i) {
        result[i] = static_cast<size_t>(mlx_shape[i]);
    }
    
    return result;
}

int MLXTensorImpl::ndim() const {
    uint32_t ndim;
    mlx_array_ndim(array_, &ndim);
    return static_cast<int>(ndim);
}

size_t MLXTensorImpl::size() const {
    uint32_t ndim;
    mlx_array_ndim(array_, &ndim);
    
    std::vector<int64_t> mlx_shape(ndim);
    mlx_array_shape(array_, mlx_shape.data());
    
    size_t result = 1;
    for (uint32_t i = 0; i < ndim; ++i) {
        result *= static_cast<size_t>(mlx_shape[i]);
    }
    
    return result;
}

DataType MLXTensorImpl::dtype() const {
    mlx_dtype dtype;
    mlx_array_dtype(array_, &dtype);
    return from_mlx_dtype(dtype);
}

void MLXTensorImpl::print(const std::string& name) const {
    if (!name.empty()) {
        std::cout << name << " = ";
    }
    
    std::string array_string;
    mlx_array_to_string(array_, &array_string);
    std::cout << array_string << std::endl;
    free((void*)array_string.c_str());
}

void* MLXTensorImpl::data() {
    // MLX arrays don't directly expose their data
    throw std::runtime_error("Direct data access not supported for MLX tensors");
}

const void* MLXTensorImpl::data() const {
    // MLX arrays don't directly expose their data
    throw std::runtime_error("Direct data access not supported for MLX tensors");
}

std::shared_ptr<TensorImpl> MLXTensorImpl::reshape(const std::vector<size_t>& new_shape) const {
    CCSM_DEBUG("Reshaping MLX tensor");
    
    // Convert size_t shape to int64_t shape
    std::vector<int64_t> mlx_shape(new_shape.size());
    for (size_t i = 0; i < new_shape.size(); ++i) {
        mlx_shape[i] = static_cast<int64_t>(new_shape[i]);
    }
    
    // Create reshaped array
    mlx_array reshaped;
    mlx_array_reshape(array_, mlx_shape.data(), mlx_shape.size(), &reshaped);
    
    return std::make_shared<MLXTensorImpl>(reshaped);
}

std::shared_ptr<TensorImpl> MLXTensorImpl::view(const std::vector<size_t>& new_shape) const {
    CCSM_DEBUG("Creating view of MLX tensor");
    
    // In MLX, reshape is a view operation when possible,
    // so this is similar to reshape but we'll check sizes match
    
    // Ensure the total size doesn't change
    size_t old_size = size();
    size_t new_size = 1;
    for (auto dim : new_shape) {
        new_size *= dim;
    }
    
    if (old_size != new_size) {
        throw std::runtime_error("View operation cannot change the total number of elements");
    }
    
    // Create the view using reshape
    std::vector<int64_t> mlx_shape(new_shape.size());
    for (size_t i = 0; i < new_shape.size(); ++i) {
        mlx_shape[i] = static_cast<int64_t>(new_shape[i]);
    }
    
    mlx_array viewed;
    mlx_array_reshape(array_, mlx_shape.data(), mlx_shape.size(), &viewed);
    
    return std::make_shared<MLXTensorImpl>(viewed);
}

std::shared_ptr<TensorImpl> MLXTensorImpl::slice(int dim, size_t start, size_t end) const {
    CCSM_DEBUG("Slicing MLX tensor in dimension " + std::to_string(dim) + 
              " from " + std::to_string(start) + " to " + std::to_string(end));
    
    uint32_t ndim;
    mlx_array_ndim(array_, &ndim);
    
    if (dim < 0 || dim >= static_cast<int>(ndim)) {
        throw std::out_of_range("Dimension index out of range for slice");
    }
    
    // Get the current shape
    std::vector<int64_t> shape(ndim);
    mlx_array_shape(array_, shape.data());
    
    // Check bounds
    if (start >= static_cast<size_t>(shape[dim]) || end > static_cast<size_t>(shape[dim]) || start >= end) {
        throw std::out_of_range("Slice range invalid");
    }
    
    // Create a slice array
    mlx_array result;
    mlx_array_slice(array_, dim, static_cast<int64_t>(start), static_cast<int64_t>(end), &result);
    
    return std::make_shared<MLXTensorImpl>(result);
}

// MLXContext implementation
MLXContext::MLXContext() {
    CCSM_DEBUG("Creating MLXContext");
    mlx_stream_init(&stream_);
}

MLXContext::~MLXContext() {
    CCSM_DEBUG("Destroying MLXContext");
    mlx_stream_free(stream_);
}

bool MLXContext::is_available() {
    // Check if MLX is available through the device
    return MLXDevice::is_available();
}

mlx_array MLXContext::get_mlx_array(const Tensor& tensor) {
    CCSM_DEBUG("Getting MLX array from tensor");
    
    // If this is already an MLX tensor, extract the array directly
    if (auto mlx_impl = std::dynamic_pointer_cast<MLXTensorImpl>(tensor.impl())) {
        return mlx_impl->mlx_array_handle();
    }
    
    // Otherwise, need to convert from raw data
    // Note: This implementation might be limited since direct data access
    // is not always available for all tensor types
    
    // Get shape and data type
    auto shape = tensor.shape();
    auto dtype = tensor.dtype();
    
    // Convert shape to MLX format
    std::vector<int64_t> mlx_shape(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        mlx_shape[i] = static_cast<int64_t>(shape[i]);
    }
    
    // Convert data type to MLX format
    mlx_dtype mlx_type = MLXTensorImpl::to_mlx_dtype(dtype);
    
    // Create MLX array
    mlx_array result;
    
    if (tensor.data() == nullptr) {
        // Create empty array if data is null
        mlx_array_zeros(mlx_shape.data(), mlx_shape.size(), mlx_type, &result);
        return result;
    }
    
    // Create from raw data
    mlx_array_from_data(tensor.data(), mlx_shape.data(), mlx_shape.size(), 
                       mlx_type, &result);
    
    return result;
}

Tensor MLXContext::add(const Tensor& a, const Tensor& b) {
    CCSM_DEBUG("MLX add operation");
    
    // Get MLX arrays from tensors
    mlx_array a_array = get_mlx_array(a);
    mlx_array b_array = get_mlx_array(b);
    
    // Perform addition
    mlx_array result;
    mlx_add(a_array, b_array, &result);
    
    // Create tensor from MLX array
    return Tensor(std::make_shared<MLXTensorImpl>(result));
}

Tensor MLXContext::subtract(const Tensor& a, const Tensor& b) {
    CCSM_DEBUG("MLX subtract operation");
    
    // Get MLX arrays from tensors
    mlx_array a_array = get_mlx_array(a);
    mlx_array b_array = get_mlx_array(b);
    
    // Perform subtraction
    mlx_array result;
    mlx_subtract(a_array, b_array, &result);
    
    // Create tensor from MLX array
    return Tensor(std::make_shared<MLXTensorImpl>(result));
}

Tensor MLXContext::multiply(const Tensor& a, const Tensor& b) {
    CCSM_DEBUG("MLX multiply operation");
    
    // Get MLX arrays from tensors
    mlx_array a_array = get_mlx_array(a);
    mlx_array b_array = get_mlx_array(b);
    
    // Perform multiplication
    mlx_array result;
    mlx_multiply(a_array, b_array, &result);
    
    // Create tensor from MLX array
    return Tensor(std::make_shared<MLXTensorImpl>(result));
}

Tensor MLXContext::divide(const Tensor& a, const Tensor& b) {
    CCSM_DEBUG("MLX divide operation");
    
    // Get MLX arrays from tensors
    mlx_array a_array = get_mlx_array(a);
    mlx_array b_array = get_mlx_array(b);
    
    // Perform division
    mlx_array result;
    mlx_divide(a_array, b_array, &result);
    
    // Create tensor from MLX array
    return Tensor(std::make_shared<MLXTensorImpl>(result));
}

Tensor MLXContext::matmul(const Tensor& a, const Tensor& b) {
    CCSM_DEBUG("MLX matrix multiplication");
    
    // Get MLX arrays from tensors
    mlx_array a_array = get_mlx_array(a);
    mlx_array b_array = get_mlx_array(b);
    
    // Perform matrix multiplication
    mlx_array result;
    mlx_matmul(a_array, b_array, &result);
    
    // Create tensor from MLX array
    return Tensor(std::make_shared<MLXTensorImpl>(result));
}

Tensor MLXContext::relu(const Tensor& x) {
    CCSM_DEBUG("MLX ReLU activation");
    
    // Get MLX array from tensor
    mlx_array x_array = get_mlx_array(x);
    
    // Apply ReLU activation
    mlx_array result;
    mlx_relu(x_array, &result);
    
    // Create tensor from MLX array
    return Tensor(std::make_shared<MLXTensorImpl>(result));
}

Tensor MLXContext::gelu(const Tensor& x) {
    CCSM_DEBUG("MLX GELU activation");
    
    // Get MLX array from tensor
    mlx_array x_array = get_mlx_array(x);
    
    // Apply GELU activation
    mlx_array result;
    mlx_gelu(x_array, &result);
    
    // Create tensor from MLX array
    return Tensor(std::make_shared<MLXTensorImpl>(result));
}

Tensor MLXContext::silu(const Tensor& x) {
    CCSM_DEBUG("MLX SiLU activation");
    
    // Get MLX array from tensor
    mlx_array x_array = get_mlx_array(x);
    
    // Apply SiLU activation
    mlx_array result;
    mlx_silu(x_array, &result);
    
    // Create tensor from MLX array
    return Tensor(std::make_shared<MLXTensorImpl>(result));
}

Tensor MLXContext::softmax(const Tensor& x, int dim) {
    CCSM_DEBUG("MLX softmax along dimension " + std::to_string(dim));
    
    // Get MLX array from tensor
    mlx_array x_array = get_mlx_array(x);
    
    // Apply softmax along the specified dimension
    mlx_array result;
    mlx_softmax(x_array, dim, &result);
    
    // Create tensor from MLX array
    return Tensor(std::make_shared<MLXTensorImpl>(result));
}

Tensor MLXContext::zeros(const std::vector<size_t>& shape, DataType dtype) {
    CCSM_DEBUG("MLX zeros with shape " + shape_to_string(shape));
    
    // Convert shape to MLX format
    std::vector<int64_t> mlx_shape(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        mlx_shape[i] = static_cast<int64_t>(shape[i]);
    }
    
    // Convert data type to MLX format
    mlx_dtype mlx_type = MLXTensorImpl::to_mlx_dtype(dtype);
    
    // Create zeros array
    mlx_array result;
    mlx_array_zeros(mlx_shape.data(), mlx_shape.size(), mlx_type, &result);
    
    // Create tensor from MLX array
    return Tensor(std::make_shared<MLXTensorImpl>(result));
}

Tensor MLXContext::ones(const std::vector<size_t>& shape, DataType dtype) {
    CCSM_DEBUG("MLX ones with shape " + shape_to_string(shape));
    
    // Convert shape to MLX format
    std::vector<int64_t> mlx_shape(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        mlx_shape[i] = static_cast<int64_t>(shape[i]);
    }
    
    // Convert data type to MLX format
    mlx_dtype mlx_type = MLXTensorImpl::to_mlx_dtype(dtype);
    
    // Create ones array
    mlx_array result;
    mlx_array_ones(mlx_shape.data(), mlx_shape.size(), mlx_type, &result);
    
    // Create tensor from MLX array
    return Tensor(std::make_shared<MLXTensorImpl>(result));
}

Tensor MLXContext::sum(const Tensor& x, int dim) {
    CCSM_DEBUG("MLX sum along dimension " + std::to_string(dim));
    
    // Get MLX array from tensor
    mlx_array x_array = get_mlx_array(x);
    
    // Compute sum along the specified dimension
    mlx_array result;
    mlx_array_sum(x_array, dim, true, &result); // keepdims=true to preserve dimensions
    
    // Create tensor from MLX array
    return Tensor(std::make_shared<MLXTensorImpl>(result));
}

Tensor MLXContext::mean(const Tensor& x, int dim) {
    CCSM_DEBUG("MLX mean along dimension " + std::to_string(dim));
    
    // Get MLX array from tensor
    mlx_array x_array = get_mlx_array(x);
    
    // Compute mean along the specified dimension
    mlx_array result;
    mlx_array_mean(x_array, dim, true, &result); // keepdims=true to preserve dimensions
    
    // Create tensor from MLX array
    return Tensor(std::make_shared<MLXTensorImpl>(result));
}

#else // CCSM_WITH_MLX

// Empty implementations for when MLX is not available
bool MLXContext::is_available() {
    return false;
}

#endif // CCSM_WITH_MLX

} // namespace ccsm