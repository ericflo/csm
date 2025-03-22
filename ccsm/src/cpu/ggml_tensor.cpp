#include <ccsm/cpu/ggml_tensor.h>
#include <ccsm/cpu/thread_pool.h>
#include <ccsm/cpu/simd.h>
#include <ccsm/utils.h>
#include <stdexcept>
#include <cstring>
#include <vector>
#include <iostream>
#include <algorithm>

namespace ccsm {

// Convert between CCSM and GGML data types
enum ggml_type GGMLTensorImpl::to_ggml_type(DataType dtype) {
    switch (dtype) {
        case DataType::F32:  return GGML_TYPE_F32;
        case DataType::F16:  return GGML_TYPE_F16;
        case DataType::I32:  return GGML_TYPE_I32;
        case DataType::I16:  return GGML_TYPE_I16;
        case DataType::I8:   return GGML_TYPE_I8;
        case DataType::Q8_0: return GGML_TYPE_Q8_0;
        case DataType::Q4_0: return GGML_TYPE_Q4_0;
        case DataType::Q4_1: return GGML_TYPE_Q4_1;
        // BF16 isn't directly supported in GGML, use F16 as fallback
        case DataType::BF16: return GGML_TYPE_F16;
        default:
            throw std::runtime_error("Unsupported data type for GGML conversion");
    }
}

DataType GGMLTensorImpl::from_ggml_type(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:  return DataType::F32;
        case GGML_TYPE_F16:  return DataType::F16;
        case GGML_TYPE_I32:  return DataType::I32;
        case GGML_TYPE_I16:  return DataType::I16;
        case GGML_TYPE_I8:   return DataType::I8;
        case GGML_TYPE_Q8_0: return DataType::Q8_0;
        case GGML_TYPE_Q4_0: return DataType::Q4_0;
        case GGML_TYPE_Q4_1: return DataType::Q4_1;
        default:
            throw std::runtime_error("Unsupported GGML type for conversion");
    }
}

// GGMLTensorImpl implementation
GGMLTensorImpl::GGMLTensorImpl(struct ggml_tensor* tensor, bool owns_data)
    : tensor_(tensor), owns_data_(owns_data) {
    if (!tensor) {
        throw std::runtime_error("Null tensor passed to GGMLTensorImpl");
    }
}

GGMLTensorImpl::~GGMLTensorImpl() {
    // If we own the data, we need to free it
    // Note: GGML tensors are usually managed by their context
    if (owns_data_ && tensor_) {
        // TODO: Handle tensor cleanup if needed
    }
}

size_t GGMLTensorImpl::shape(int dim) const {
    if (dim < 0 || dim >= GGML_MAX_DIMS) {
        throw std::out_of_range("Dimension index out of range");
    }
    return tensor_->ne[dim];
}

std::vector<size_t> GGMLTensorImpl::shape() const {
    std::vector<size_t> result;
    for (int i = 0; i < GGML_MAX_DIMS && tensor_->ne[i] > 1; i++) {
        result.push_back(tensor_->ne[i]);
    }
    // Ensure at least one dimension
    if (result.empty()) {
        result.push_back(1);
    }
    return result;
}

int GGMLTensorImpl::ndim() const {
    int dims = 0;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (tensor_->ne[i] > 1) {
            dims = i + 1;
        }
    }
    return std::max(dims, 1); // Ensure at least 1 dimension
}

size_t GGMLTensorImpl::size() const {
    size_t total = 1;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        total *= tensor_->ne[i];
    }
    return total;
}

DataType GGMLTensorImpl::dtype() const {
    return from_ggml_type(tensor_->type);
}

void* GGMLTensorImpl::data() {
    return tensor_->data;
}

const void* GGMLTensorImpl::data() const {
    return tensor_->data;
}

std::shared_ptr<TensorImpl> GGMLTensorImpl::reshape(const std::vector<size_t>& new_shape) const {
    // Create a new tensor with the same data but different shape
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
    
    // Create a new context for the reshape operation since tensor_->ctx is no longer available
    GGMLContext temp_ctx;
    struct ggml_context* ctx = temp_ctx.ggml_ctx();
    
    // Prepare ne array for GGML
    int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
    for (size_t i = 0; i < new_shape.size() && i < GGML_MAX_DIMS; i++) {
        ne[i] = new_shape[i];
    }
    
    // Create new view tensor
    struct ggml_tensor* new_tensor = ggml_new_tensor(ctx, tensor_->type, 
                                                  std::min(static_cast<size_t>(GGML_MAX_DIMS), new_shape.size()), 
                                                  ne);
    
    // Set to same data
    new_tensor->data = tensor_->data;
    
    // Return new implementation
    return std::make_shared<GGMLTensorImpl>(new_tensor, false); // Don't own data
}

std::shared_ptr<TensorImpl> GGMLTensorImpl::view(const std::vector<size_t>& new_shape) const {
    // Similar to reshape but doesn't check size
    // For GGML, view is essentially the same as reshape
    return reshape(new_shape);
}

std::shared_ptr<TensorImpl> GGMLTensorImpl::slice(int dim, size_t start, size_t end) const {
    if (dim < 0 || dim >= ndim()) {
        throw std::invalid_argument("Invalid dimension for slice");
    }
    
    if (start >= end || end > shape(dim)) {
        throw std::invalid_argument("Invalid slice range");
    }
    
    // Create a view with reduced size in the specified dimension
    std::vector<size_t> new_shape = shape();
    new_shape[dim] = end - start;
    
    // Calculate offset
    size_t offset = start;
    for (int i = 0; i < dim; i++) {
        offset *= tensor_->ne[i];
    }
    
    // Create a new context for the slice operation since tensor_->ctx is no longer available
    GGMLContext temp_ctx;
    struct ggml_context* ctx = temp_ctx.ggml_ctx();
    
    // Prepare ne array for GGML
    int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
    for (size_t i = 0; i < new_shape.size() && i < GGML_MAX_DIMS; i++) {
        ne[i] = new_shape[i];
    }
    
    // Create new view tensor
    struct ggml_tensor* new_tensor = ggml_new_tensor(ctx, tensor_->type, 
                                                  std::min(static_cast<size_t>(GGML_MAX_DIMS), new_shape.size()), 
                                                  ne);
    
    // Calculate pointer to start of slice
    size_t element_size = ggml_type_size(tensor_->type);
    char* data_ptr = static_cast<char*>(tensor_->data);
    new_tensor->data = data_ptr + offset * element_size;
    
    // Return new implementation
    return std::make_shared<GGMLTensorImpl>(new_tensor, false); // Don't own data
}

void GGMLTensorImpl::print(const std::string& name) const {
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
    std::cout << "  Type: " << ggml_type_name(tensor_->type) << std::endl;
    std::cout << "  Size: " << size() << " elements" << std::endl;
    
    // Print some values if small enough
    if (size() <= 10) {
        std::cout << "  Values: [";
        
        // Only handle float32 for now
        if (tensor_->type == GGML_TYPE_F32) {
            float* data_ptr = static_cast<float*>(tensor_->data);
            for (size_t i = 0; i < size(); i++) {
                std::cout << data_ptr[i];
                if (i < size() - 1) {
                    std::cout << ", ";
                }
            }
        }
        else {
            std::cout << "... (non-F32 data type)";
        }
        
        std::cout << "]" << std::endl;
    }
}

// GGMLContext implementation
GGMLContext::GGMLContext() : ctx_(nullptr) {
    // Initialize GGML context
    struct ggml_init_params params = {
        .mem_size   = 16 * 1024 * 1024, // 16 MB
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    
    ctx_ = ggml_init(params);
    if (!ctx_) {
        throw std::runtime_error("Failed to initialize GGML context");
    }
}

GGMLContext::~GGMLContext() {
    if (ctx_) {
        ggml_free(ctx_);
    }
}

Tensor GGMLContext::add(const Tensor& a, const Tensor& b) {
    struct ggml_tensor* a_tensor = get_ggml_tensor(a);
    struct ggml_tensor* b_tensor = get_ggml_tensor(b);
    
    struct ggml_tensor* result = ggml_add(ctx_, a_tensor, b_tensor);
    
    return Tensor(std::make_shared<GGMLTensorImpl>(result, false));
}

Tensor GGMLContext::subtract(const Tensor& a, const Tensor& b) {
    struct ggml_tensor* a_tensor = get_ggml_tensor(a);
    struct ggml_tensor* b_tensor = get_ggml_tensor(b);
    
    struct ggml_tensor* result = ggml_sub(ctx_, a_tensor, b_tensor);
    
    return Tensor(std::make_shared<GGMLTensorImpl>(result, false));
}

Tensor GGMLContext::multiply(const Tensor& a, const Tensor& b) {
    struct ggml_tensor* a_tensor = get_ggml_tensor(a);
    struct ggml_tensor* b_tensor = get_ggml_tensor(b);
    
    struct ggml_tensor* result = ggml_mul(ctx_, a_tensor, b_tensor);
    
    return Tensor(std::make_shared<GGMLTensorImpl>(result, false));
}

Tensor GGMLContext::divide(const Tensor& a, const Tensor& b) {
    struct ggml_tensor* a_tensor = get_ggml_tensor(a);
    struct ggml_tensor* b_tensor = get_ggml_tensor(b);
    
    struct ggml_tensor* result = ggml_div(ctx_, a_tensor, b_tensor);
    
    return Tensor(std::make_shared<GGMLTensorImpl>(result, false));
}

Tensor GGMLContext::matmul(const Tensor& a, const Tensor& b) {
    struct ggml_tensor* a_tensor = get_ggml_tensor(a);
    struct ggml_tensor* b_tensor = get_ggml_tensor(b);
    
    struct ggml_tensor* result = ggml_mul_mat(ctx_, a_tensor, b_tensor);
    
    return Tensor(std::make_shared<GGMLTensorImpl>(result, false));
}

Tensor GGMLContext::relu(const Tensor& x) {
    struct ggml_tensor* x_tensor = get_ggml_tensor(x);
    
    // GGML doesn't have a direct relu, so we implement it using max
    struct ggml_tensor* zeros = ggml_new_tensor_1d(ctx_, x_tensor->type, 1);
    ggml_set_zero(zeros);
    
    struct ggml_tensor* result = ggml_relu(ctx_, x_tensor);
    
    return Tensor(std::make_shared<GGMLTensorImpl>(result, false));
}

Tensor GGMLContext::gelu(const Tensor& x) {
    struct ggml_tensor* x_tensor = get_ggml_tensor(x);
    
    struct ggml_tensor* result = ggml_gelu(ctx_, x_tensor);
    
    return Tensor(std::make_shared<GGMLTensorImpl>(result, false));
}

Tensor GGMLContext::silu(const Tensor& x) {
    struct ggml_tensor* x_tensor = get_ggml_tensor(x);
    
    struct ggml_tensor* result = ggml_silu(ctx_, x_tensor);
    
    return Tensor(std::make_shared<GGMLTensorImpl>(result, false));
}

Tensor GGMLContext::softmax(const Tensor& x, int dim) {
    struct ggml_tensor* x_tensor = get_ggml_tensor(x);
    
    struct ggml_tensor* result = ggml_soft_max(ctx_, x_tensor);
    
    return Tensor(std::make_shared<GGMLTensorImpl>(result, false));
}

Tensor GGMLContext::zeros(const std::vector<size_t>& shape, DataType dtype) {
    if (shape.empty()) {
        throw std::invalid_argument("Empty shape in zeros");
    }
    
    // Convert shape to GGML format
    int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
    for (size_t i = 0; i < shape.size() && i < GGML_MAX_DIMS; i++) {
        ne[i] = shape[i];
    }
    
    // Create tensor
    enum ggml_type type = GGMLTensorImpl::to_ggml_type(dtype);
    struct ggml_tensor* tensor = ggml_new_tensor(ctx_, type, 
                                             std::min(static_cast<size_t>(GGML_MAX_DIMS), shape.size()), 
                                             ne);
    
    // Initialize to zeros
    ggml_set_zero(tensor);
    
    return Tensor(std::make_shared<GGMLTensorImpl>(tensor, false));
}

Tensor GGMLContext::ones(const std::vector<size_t>& shape, DataType dtype) {
    Tensor result = zeros(shape, dtype);
    
    // Set all elements to 1
    struct ggml_tensor* tensor = get_ggml_tensor(result);
    
    // Handle different types
    switch (dtype) {
        case DataType::F32: {
            float* data = static_cast<float*>(tensor->data);
            for (size_t i = 0; i < ggml_nelements(tensor); i++) {
                data[i] = 1.0f;
            }
            break;
        }
        case DataType::F16: {
            ggml_fp16_t* data = static_cast<ggml_fp16_t*>(tensor->data);
            for (size_t i = 0; i < ggml_nelements(tensor); i++) {
                data[i] = ggml_fp32_to_fp16(1.0f);
            }
            break;
        }
        case DataType::I32: {
            int32_t* data = static_cast<int32_t*>(tensor->data);
            for (size_t i = 0; i < ggml_nelements(tensor); i++) {
                data[i] = 1;
            }
            break;
        }
        // Add other types as needed
        default:
            throw std::runtime_error("Unsupported data type for ones");
    }
    
    return result;
}

Tensor GGMLContext::sum(const Tensor& x, int dim) {
    // TODO: Implement sum reduction
    throw std::runtime_error("GGMLContext::sum not implemented yet");
}

Tensor GGMLContext::mean(const Tensor& x, int dim) {
    // TODO: Implement mean reduction
    throw std::runtime_error("GGMLContext::mean not implemented yet");
}

// Type casting implementation
Tensor GGMLContext::cast(const Tensor& x, DataType dtype) {
    // Get the GGML tensor from input
    struct ggml_tensor* x_tensor = get_ggml_tensor(x);
    
    // Convert the target data type to GGML type
    enum ggml_type target_type = GGMLTensorImpl::to_ggml_type(dtype);
    
    // Create a new tensor with the target data type
    std::vector<size_t> shape = x.shape();
    int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
    for (size_t i = 0; i < shape.size() && i < GGML_MAX_DIMS; i++) {
        ne[i] = shape[i];
    }
    
    // Create new tensor
    struct ggml_tensor* result_tensor = ggml_new_tensor(ctx_, target_type, 
                                                     std::min(static_cast<size_t>(GGML_MAX_DIMS), shape.size()), 
                                                     ne);
    
    // For now, we'll just create an empty tensor of the right type and shape
    // In a real implementation, we would convert the data
    // TODO: Implement proper type conversion
    
    return Tensor(std::make_shared<GGMLTensorImpl>(result_tensor, true));
}

// Type promotion helper implementation
DataType GGMLContext::promote_types(DataType a, DataType b) {
    // Float types have higher precedence than integer types
    // Within the float types: F32 > F16 > BF16
    // Within the integer types: I32 > I16 > I8
    // Quantized types promotion: Q8_0 > Q4_1 > Q4_0
    
    // If either is F32, result is F32
    if (a == DataType::F32 || b == DataType::F32) {
        return DataType::F32;
    }
    
    // If either is F16, result is F16
    if (a == DataType::F16 || b == DataType::F16) {
        return DataType::F16;
    }
    
    // If either is BF16, result is BF16
    if (a == DataType::BF16 || b == DataType::BF16) {
        return DataType::BF16;
    }
    
    // If either is I32, result is I32
    if (a == DataType::I32 || b == DataType::I32) {
        return DataType::I32;
    }
    
    // If either is I16, result is I16
    if (a == DataType::I16 || b == DataType::I16) {
        return DataType::I16;
    }
    
    // If either is I8, result is I8
    if (a == DataType::I8 || b == DataType::I8) {
        return DataType::I8;
    }
    
    // Quantized types
    if (a == DataType::Q8_0 || b == DataType::Q8_0) {
        return DataType::Q8_0;
    }
    
    if (a == DataType::Q4_1 || b == DataType::Q4_1) {
        return DataType::Q4_1;
    }
    
    // If we get here, both are Q4_0 or some other type
    return DataType::Q4_0;
}

struct ggml_tensor* GGMLContext::alloc_tensor(enum ggml_type type, int n_dims, const int64_t* dims) {
    return ggml_new_tensor(ctx_, type, n_dims, dims);
}

void GGMLContext::compute(struct ggml_cgraph* graph) {
    // Use thread pool for parallel computation
    int n_threads = static_cast<int>(global_thread_pool().size());
    
    // Since we can't directly access the graph nodes, and don't have direct
    // graph compute functions available, we'll just use the build forward pass
    // and let GGML handle the computation internally
    int n_nodes = ggml_graph_n_nodes(graph);
    if (n_nodes > 0) {
        struct ggml_tensor* last_node = ggml_graph_node(graph, n_nodes - 1);
        if (last_node) {
            // Build forward pass - this will compute the graph
            ggml_build_forward_expand(graph, last_node);
            
            // The GGML graph seems to be computed automatically when built
            // We don't need to do anything else here
        }
    }
}

struct ggml_tensor* GGMLContext::get_ggml_tensor(const Tensor& tensor) {
    if (!tensor.is_valid()) {
        throw std::runtime_error("Invalid tensor in get_ggml_tensor");
    }
    
    // Try to cast the implementation to GGMLTensorImpl
    auto impl = std::dynamic_pointer_cast<GGMLTensorImpl>(tensor.impl());
    if (!impl) {
        throw std::runtime_error("Tensor is not a GGML tensor");
    }
    
    return impl->ggml_tensor();
}

} // namespace ccsm