#include <ccsm/mlx/mlx_tensor.h>
#include <ccsm/utils.h>
#include <stdexcept>
#include <iostream>
#include <sstream>

#ifdef CCSM_WITH_MLX
#include <mlx/c/array.h>
#include <mlx/c/ops.h>
#include <mlx/c/device.h>
#include <mlx/c/stream.h>
#endif

namespace ccsm {

#ifdef CCSM_WITH_MLX

// Utility function to check MLX errors
static void check_mlx_result(int result, const char* operation) {
    if (result != 0) {
        throw std::runtime_error(std::string("MLX operation failed: ") + operation);
    }
}

// MLXDevice implementation
MLXDevice::MLXDevice() {
    // Create a default MLX device
    mlx_device dev;
    check_mlx_result(mlx_get_default_device(&dev), "get_default_device");
    device_ = dev;
}

MLXDevice::MLXDevice(mlx_device_type type, int index) {
    // Create a device with specified type and index
    device_ = mlx_device_new_type(type, index);
}

MLXDevice::~MLXDevice() {
    if (device_.ctx != nullptr) {
        mlx_device_free(device_);
    }
}

mlx_device_type MLXDevice::type() const {
    mlx_device_type type;
    check_mlx_result(mlx_device_get_type(&type, device_), "device_get_type");
    return type;
}

int MLXDevice::index() const {
    int index;
    check_mlx_result(mlx_device_get_index(&index, device_), "device_get_index");
    return index;
}

std::string MLXDevice::name() const {
    mlx_string str;
    check_mlx_result(mlx_device_tostring(&str, device_), "device_tostring");
    std::string result(str.ptr);
    mlx_string_free(str);
    return result;
}

bool MLXDevice::is_available() {
    mlx_device dev;
    if (mlx_get_default_device(&dev) != 0) {
        return false;
    }
    
    mlx_device_type type;
    if (mlx_device_get_type(&type, dev) != 0) {
        mlx_device_free(dev);
        return false;
    }
    
    // On Apple Silicon, we should have MLX_GPU device type
    bool result = (type == MLX_GPU);
    
    mlx_device_free(dev);
    return result;
}

MLXDevice MLXDevice::default_device() {
    return MLXDevice();
}

void MLXDevice::set_default_device(const MLXDevice& device) {
    check_mlx_result(mlx_set_default_device(device.device()), "set_default_device");
}

void MLXDevice::synchronize() {
    // MLX-C doesn't have a direct synchronize function, but we can
    // achieve the same effect by evaluating an empty array
    mlx_array empty = mlx_array_new();
    check_mlx_result(mlx_array_eval(empty), "array_eval (synchronize)");
    mlx_array_free(empty);
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
MLXTensorImpl::MLXTensorImpl(mlx_array array) : array_(array) {
    if (array_.ctx == nullptr) {
        throw std::runtime_error("Null array passed to MLXTensorImpl");
    }
}

MLXTensorImpl::~MLXTensorImpl() {
    if (array_.ctx != nullptr) {
        mlx_array_free(array_);
    }
}

size_t MLXTensorImpl::shape(int dim) const {
    if (dim < 0 || dim >= ndim()) {
        throw std::out_of_range("Dimension index out of range");
    }
    return mlx_array_dim(array_, dim);
}

std::vector<size_t> MLXTensorImpl::shape() const {
    std::vector<size_t> result;
    int dims = ndim();
    const int* shape_ptr = mlx_array_shape(array_);
    
    for (int i = 0; i < dims; i++) {
        result.push_back(shape_ptr[i]);
    }
    
    return result;
}

int MLXTensorImpl::ndim() const {
    return mlx_array_ndim(array_);
}

size_t MLXTensorImpl::size() const {
    return mlx_array_size(array_);
}

DataType MLXTensorImpl::dtype() const {
    return from_mlx_dtype(mlx_array_dtype(array_));
}

void MLXTensorImpl::print(const std::string& name) const {
    mlx_string str;
    check_mlx_result(mlx_array_tostring(&str, array_), "array_tostring");
    
    std::cout << "Tensor: " << name << std::endl;
    std::cout << "  Shape: [";
    
    auto shape_vec = shape();
    for (size_t i = 0; i < shape_vec.size(); i++) {
        std::cout << shape_vec[i];
        if (i < shape_vec.size() - 1) {
            std::cout << ", ";
        }
    }
    
    std::cout << "]" << std::endl;
    std::cout << "  Type: " << str.ptr << std::endl;
    
    mlx_string_free(str);
}

void* MLXTensorImpl::data() {
    // MLX arrays don't directly expose their data
    // We'd need to evaluate the array and copy to CPU
    // This is not ideal for performance but necessary for the interface
    throw std::runtime_error("Direct data access not supported for MLX tensors");
}

const void* MLXTensorImpl::data() const {
    // MLX arrays don't directly expose their data
    // We'd need to evaluate the array and copy to CPU
    // This is not ideal for performance but necessary for the interface
    throw std::runtime_error("Direct data access not supported for MLX tensors");
}

std::shared_ptr<TensorImpl> MLXTensorImpl::reshape(const std::vector<size_t>& new_shape) const {
    if (new_shape.empty()) {
        throw std::invalid_argument("Empty shape in reshape");
    }
    
    // Calculate total size
    size_t new_size = 1;
    for (size_t dim : new_shape) {
        new_size *= dim;
    }
    
    // Check if total size matches
    if (new_size != size()) {
        throw std::invalid_argument("Total size mismatch in reshape");
    }
    
    // Convert to int array for MLX
    std::vector<int> shape_int(new_shape.begin(), new_shape.end());
    
    // Create new array with new shape
    mlx_array result;
    check_mlx_result(mlx_reshape(&result, array_, shape_int.data(), shape_int.size(), mlx_stream_null), "reshape");
    
    return std::make_shared<MLXTensorImpl>(result);
}

std::shared_ptr<TensorImpl> MLXTensorImpl::view(const std::vector<size_t>& new_shape) const {
    // MLX doesn't distinguish between reshape and view
    return reshape(new_shape);
}

std::shared_ptr<TensorImpl> MLXTensorImpl::slice(int dim, size_t start, size_t end) const {
    if (dim < 0 || dim >= ndim()) {
        throw std::invalid_argument("Invalid dimension for slice");
    }
    
    if (start >= end || end > shape(dim)) {
        throw std::invalid_argument("Invalid slice range");
    }
    
    // Setup slice parameters
    std::vector<int> start_indices(ndim(), 0);
    start_indices[dim] = static_cast<int>(start);
    
    std::vector<int> stop_indices = shape();  // Will be automatically converted to int
    stop_indices[dim] = static_cast<int>(end);
    
    std::vector<int> strides(ndim(), 1);
    
    // Create sliced array
    mlx_array result;
    check_mlx_result(mlx_slice(
        &result, 
        array_, 
        start_indices.data(), start_indices.size(),
        stop_indices.data(), stop_indices.size(),
        strides.data(), strides.size(),
        mlx_stream_null
    ), "slice");
    
    return std::make_shared<MLXTensorImpl>(result);
}

// MLXContext implementation
MLXContext::MLXContext() {
    // Create a default MLX stream
    stream_ = mlx_stream_null;
}

MLXContext::~MLXContext() {
    // No need to free the null stream
}

bool MLXContext::is_available() {
    return MLXDevice::is_available();
}

mlx_array MLXContext::get_mlx_array(const Tensor& tensor) {
    if (!tensor.is_valid()) {
        throw std::runtime_error("Invalid tensor in get_mlx_array");
    }
    
    // Try to cast the implementation to MLXTensorImpl
    auto impl = std::dynamic_pointer_cast<MLXTensorImpl>(tensor.impl());
    if (impl) {
        // If it's already an MLX tensor, return the mlx_array
        return impl->mlx_array_handle();
    }
    
    // Otherwise, we need to convert from another backend
    // Get data and shape from the source tensor
    auto shape_vec = tensor.shape();
    std::vector<int> shape_int(shape_vec.begin(), shape_vec.end());
    
    // This is a complex operation because MLX doesn't directly support
    // setting an array's data from a pointer that isn't managed by MLX.
    // We need to make a copy of the data and create a new MLX array.
    
    // Check if we can access the tensor data
    const void* data_ptr = nullptr;
    try {
        data_ptr = tensor.data();
    } catch (const std::exception& e) {
        std::ostringstream ss;
        ss << "Cannot get data from non-MLX tensor: " << e.what();
        throw std::runtime_error(ss.str());
    }
    
    if (!data_ptr) {
        throw std::runtime_error("Null data pointer in tensor");
    }
    
    // Create a new MLX array with the data from the source tensor
    mlx_array result;
    mlx_dtype dtype = MLXTensorImpl::to_mlx_dtype(tensor.dtype());
    
    check_mlx_result(mlx_array_new_data(
        &result,
        data_ptr,
        shape_int.data(),
        shape_int.size(),
        dtype
    ), "array_new_data");
    
    return result;
}

Tensor MLXContext::add(const Tensor& a, const Tensor& b) {
    mlx_array a_mlx = get_mlx_array(a);
    mlx_array b_mlx = get_mlx_array(b);
    
    mlx_array result;
    check_mlx_result(mlx_add(&result, a_mlx, b_mlx, stream_), "add");
    
    return Tensor(std::make_shared<MLXTensorImpl>(result));
}

Tensor MLXContext::subtract(const Tensor& a, const Tensor& b) {
    mlx_array a_mlx = get_mlx_array(a);
    mlx_array b_mlx = get_mlx_array(b);
    
    mlx_array result;
    check_mlx_result(mlx_subtract(&result, a_mlx, b_mlx, stream_), "subtract");
    
    return Tensor(std::make_shared<MLXTensorImpl>(result));
}

Tensor MLXContext::multiply(const Tensor& a, const Tensor& b) {
    mlx_array a_mlx = get_mlx_array(a);
    mlx_array b_mlx = get_mlx_array(b);
    
    mlx_array result;
    check_mlx_result(mlx_multiply(&result, a_mlx, b_mlx, stream_), "multiply");
    
    return Tensor(std::make_shared<MLXTensorImpl>(result));
}

Tensor MLXContext::divide(const Tensor& a, const Tensor& b) {
    mlx_array a_mlx = get_mlx_array(a);
    mlx_array b_mlx = get_mlx_array(b);
    
    mlx_array result;
    check_mlx_result(mlx_divide(&result, a_mlx, b_mlx, stream_), "divide");
    
    return Tensor(std::make_shared<MLXTensorImpl>(result));
}

Tensor MLXContext::matmul(const Tensor& a, const Tensor& b) {
    mlx_array a_mlx = get_mlx_array(a);
    mlx_array b_mlx = get_mlx_array(b);
    
    mlx_array result;
    check_mlx_result(mlx_matmul(&result, a_mlx, b_mlx, stream_), "matmul");
    
    return Tensor(std::make_shared<MLXTensorImpl>(result));
}

Tensor MLXContext::relu(const Tensor& x) {
    mlx_array x_mlx = get_mlx_array(x);
    
    // MLX doesn't have a dedicated ReLU function, so we use maximum with zero
    mlx_array zero = mlx_array_new_float(0.0f);
    
    mlx_array result;
    check_mlx_result(mlx_maximum(&result, x_mlx, zero, stream_), "maximum (relu)");
    
    // Free temporary array
    mlx_array_free(zero);
    
    return Tensor(std::make_shared<MLXTensorImpl>(result));
}

Tensor MLXContext::gelu(const Tensor& x) {
    // GELU is not directly available in MLX-C API
    // Approximate using: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    mlx_array x_mlx = get_mlx_array(x);
    
    // Compute x^3
    mlx_array x3;
    check_mlx_result(mlx_multiply(&x3, x_mlx, x_mlx, stream_), "multiply (gelu)");
    check_mlx_result(mlx_multiply(&x3, x3, x_mlx, stream_), "multiply (gelu)");
    
    // Compute 0.044715 * x^3
    mlx_array scale1 = mlx_array_new_float(0.044715f);
    mlx_array term1;
    check_mlx_result(mlx_multiply(&term1, x3, scale1, stream_), "multiply (gelu)");
    
    // Compute x + 0.044715 * x^3
    mlx_array sum;
    check_mlx_result(mlx_add(&sum, x_mlx, term1, stream_), "add (gelu)");
    
    // Compute sqrt(2/π) * (x + 0.044715 * x^3)
    mlx_array scale2 = mlx_array_new_float(0.7978845608f);  // sqrt(2/π)
    mlx_array term2;
    check_mlx_result(mlx_multiply(&term2, sum, scale2, stream_), "multiply (gelu)");
    
    // Compute tanh(sqrt(2/π) * (x + 0.044715 * x^3))
    mlx_array tanh_term;
    check_mlx_result(mlx_tanh(&tanh_term, term2, stream_), "tanh (gelu)");
    
    // Compute 1 + tanh(...)
    mlx_array one = mlx_array_new_float(1.0f);
    mlx_array term3;
    check_mlx_result(mlx_add(&term3, one, tanh_term, stream_), "add (gelu)");
    
    // Compute 0.5 * (1 + tanh(...))
    mlx_array half = mlx_array_new_float(0.5f);
    mlx_array term4;
    check_mlx_result(mlx_multiply(&term4, half, term3, stream_), "multiply (gelu)");
    
    // Compute x * 0.5 * (1 + tanh(...))
    mlx_array result;
    check_mlx_result(mlx_multiply(&result, x_mlx, term4, stream_), "multiply (gelu)");
    
    // Free temporary arrays
    mlx_array_free(x3);
    mlx_array_free(scale1);
    mlx_array_free(term1);
    mlx_array_free(sum);
    mlx_array_free(scale2);
    mlx_array_free(term2);
    mlx_array_free(tanh_term);
    mlx_array_free(one);
    mlx_array_free(term3);
    mlx_array_free(half);
    mlx_array_free(term4);
    
    return Tensor(std::make_shared<MLXTensorImpl>(result));
}

Tensor MLXContext::silu(const Tensor& x) {
    mlx_array x_mlx = get_mlx_array(x);
    
    // SiLU is x * sigmoid(x)
    // First compute sigmoid(x)
    mlx_array sigmoid_x;
    check_mlx_result(mlx_sigmoid(&sigmoid_x, x_mlx, stream_), "sigmoid (silu)");
    
    // Then multiply by x
    mlx_array result;
    check_mlx_result(mlx_multiply(&result, x_mlx, sigmoid_x, stream_), "multiply (silu)");
    
    // Free temporary array
    mlx_array_free(sigmoid_x);
    
    return Tensor(std::make_shared<MLXTensorImpl>(result));
}

Tensor MLXContext::softmax(const Tensor& x, int dim) {
    mlx_array x_mlx = get_mlx_array(x);
    
    int dims = x.ndim();
    if (dim < 0) {
        dim += dims;
    }
    
    if (dim < 0 || dim >= dims) {
        throw std::invalid_argument("Invalid dimension for softmax");
    }
    
    // MLX needs an array of axes
    std::vector<int> axes = {dim};
    
    mlx_array result;
    check_mlx_result(mlx_softmax(&result, x_mlx, axes.data(), axes.size(), true, stream_), "softmax");
    
    return Tensor(std::make_shared<MLXTensorImpl>(result));
}

Tensor MLXContext::zeros(const std::vector<size_t>& shape, DataType dtype) {
    std::vector<int> shape_int(shape.begin(), shape.end());
    mlx_dtype mlx_dtype = MLXTensorImpl::to_mlx_dtype(dtype);
    
    mlx_array result;
    check_mlx_result(mlx_zeros(&result, shape_int.data(), shape_int.size(), mlx_dtype, stream_), "zeros");
    
    return Tensor(std::make_shared<MLXTensorImpl>(result));
}

Tensor MLXContext::ones(const std::vector<size_t>& shape, DataType dtype) {
    std::vector<int> shape_int(shape.begin(), shape.end());
    mlx_dtype mlx_dtype = MLXTensorImpl::to_mlx_dtype(dtype);
    
    mlx_array result;
    check_mlx_result(mlx_ones(&result, shape_int.data(), shape_int.size(), mlx_dtype, stream_), "ones");
    
    return Tensor(std::make_shared<MLXTensorImpl>(result));
}

Tensor MLXContext::sum(const Tensor& x, int dim) {
    mlx_array x_mlx = get_mlx_array(x);
    
    int dims = x.ndim();
    if (dim < 0) {
        dim += dims;
    }
    
    if (dim < 0 || dim >= dims) {
        throw std::invalid_argument("Invalid dimension for sum");
    }
    
    // MLX needs an array of axes
    std::vector<int> axes = {dim};
    
    mlx_array result;
    check_mlx_result(mlx_sum(&result, x_mlx, axes.data(), axes.size(), false, stream_), "sum");
    
    return Tensor(std::make_shared<MLXTensorImpl>(result));
}

Tensor MLXContext::mean(const Tensor& x, int dim) {
    mlx_array x_mlx = get_mlx_array(x);
    
    int dims = x.ndim();
    if (dim < 0) {
        dim += dims;
    }
    
    if (dim < 0 || dim >= dims) {
        throw std::invalid_argument("Invalid dimension for mean");
    }
    
    // MLX needs an array of axes
    std::vector<int> axes = {dim};
    
    mlx_array result;
    check_mlx_result(mlx_mean(&result, x_mlx, axes.data(), axes.size(), false, stream_), "mean");
    
    return Tensor(std::make_shared<MLXTensorImpl>(result));
}

#else // CCSM_WITH_MLX

// Empty implementations for when MLX is not available
bool MLXContext::is_available() {
    return false;
}

#endif // CCSM_WITH_MLX

} // namespace ccsm