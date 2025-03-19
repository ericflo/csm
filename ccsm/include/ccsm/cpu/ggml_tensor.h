#ifndef CCSM_GGML_TENSOR_H
#define CCSM_GGML_TENSOR_H

#include <ccsm/tensor.h>
#include "ggml.h"

namespace ccsm {

// GGML-specific tensor implementation
class GGMLTensorImpl : public TensorImpl {
public:
    GGMLTensorImpl(struct ggml_tensor* tensor, bool owns_data = false);
    ~GGMLTensorImpl();
    
    // Shape and metadata
    size_t shape(int dim) const override;
    std::vector<size_t> shape() const override;
    int ndim() const override;
    size_t size() const override;
    DataType dtype() const override;
    
    // Data access
    void* data() override;
    const void* data() const override;
    
    // Basic operations
    std::shared_ptr<TensorImpl> reshape(const std::vector<size_t>& new_shape) const override;
    std::shared_ptr<TensorImpl> view(const std::vector<size_t>& new_shape) const override;
    std::shared_ptr<TensorImpl> slice(int dim, size_t start, size_t end) const override;
    
    // Debug
    void print(const std::string& name = "") const override;
    
    // GGML-specific methods
    struct ggml_tensor* ggml_tensor() const { return tensor_; }
    
    // Convert data type
    static enum ggml_type to_ggml_type(DataType dtype);
    static DataType from_ggml_type(enum ggml_type type);
    
private:
    struct ggml_tensor* tensor_;
    bool owns_data_;
};

// GGML-specific context implementation
class GGMLContext : public Context {
public:
    GGMLContext();
    ~GGMLContext();
    
    // Basic operations
    Tensor add(const Tensor& a, const Tensor& b) override;
    Tensor subtract(const Tensor& a, const Tensor& b) override;
    Tensor multiply(const Tensor& a, const Tensor& b) override;
    Tensor divide(const Tensor& a, const Tensor& b) override;
    
    // Matrix operations
    Tensor matmul(const Tensor& a, const Tensor& b) override;
    
    // Activations
    Tensor relu(const Tensor& x) override;
    Tensor gelu(const Tensor& x) override;
    Tensor silu(const Tensor& x) override;
    Tensor softmax(const Tensor& x, int dim) override;
    
    // Creation
    Tensor zeros(const std::vector<size_t>& shape, DataType dtype) override;
    Tensor ones(const std::vector<size_t>& shape, DataType dtype) override;
    
    // Reductions
    Tensor sum(const Tensor& x, int dim) override;
    Tensor mean(const Tensor& x, int dim) override;
    
    // Get the backend name
    std::string backend() const override { return "ggml"; }
    
    // GGML-specific methods
    struct ggml_context* ggml_ctx() const { return ctx_; }
    
    // Allocate a new tensor
    struct ggml_tensor* alloc_tensor(enum ggml_type type, int n_dims, const int64_t* dims);
    
    // Compute the graph
    void compute(struct ggml_cgraph* graph);
    
private:
    struct ggml_context* ctx_;
    
    // Helper to convert our tensors to GGML tensors
    struct ggml_tensor* get_ggml_tensor(const Tensor& tensor);
};

} // namespace ccsm

#endif // CCSM_GGML_TENSOR_H