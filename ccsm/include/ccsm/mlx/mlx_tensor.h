#ifndef CCSM_MLX_TENSOR_H
#define CCSM_MLX_TENSOR_H

#include <ccsm/tensor.h>

#ifdef CCSM_WITH_MLX
#include "mlx/c/array.h"
#include "mlx/c/ops.h"
#endif

namespace ccsm {

// MLX-specific tensor implementation
class MLXTensorImpl : public TensorImpl {
public:
#ifdef CCSM_WITH_MLX
    MLXTensorImpl(mlx_array array);
    ~MLXTensorImpl();
    
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
    
    // MLX-specific methods
    mlx_array mlx_array_handle() const { return array_; }
    
    // Convert data type
    static mlx_array_dtype to_mlx_dtype(DataType dtype);
    static DataType from_mlx_dtype(mlx_array_dtype dtype);
    
private:
    mlx_array array_;
#else
    MLXTensorImpl() = delete;
    ~MLXTensorImpl() = default;
    
    // Required overrides
    size_t shape(int dim) const override { throw std::runtime_error("MLX not supported"); }
    std::vector<size_t> shape() const override { throw std::runtime_error("MLX not supported"); }
    int ndim() const override { throw std::runtime_error("MLX not supported"); }
    size_t size() const override { throw std::runtime_error("MLX not supported"); }
    DataType dtype() const override { throw std::runtime_error("MLX not supported"); }
    void* data() override { throw std::runtime_error("MLX not supported"); }
    const void* data() const override { throw std::runtime_error("MLX not supported"); }
    std::shared_ptr<TensorImpl> reshape(const std::vector<size_t>& new_shape) const override { throw std::runtime_error("MLX not supported"); }
    std::shared_ptr<TensorImpl> view(const std::vector<size_t>& new_shape) const override { throw std::runtime_error("MLX not supported"); }
    std::shared_ptr<TensorImpl> slice(int dim, size_t start, size_t end) const override { throw std::runtime_error("MLX not supported"); }
    void print(const std::string& name = "") const override { throw std::runtime_error("MLX not supported"); }
#endif
};

// MLX-specific context implementation
class MLXContext : public Context {
public:
#ifdef CCSM_WITH_MLX
    MLXContext();
    ~MLXContext();
    
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
    std::string backend() const override { return "mlx"; }
    
    // Check if MLX is available on this system
    static bool is_available();
    
private:
    // Helper to convert our tensors to MLX arrays
    mlx_array get_mlx_array(const Tensor& tensor);
#else
    MLXContext() = delete;
    ~MLXContext() = default;
    
    // Required overrides
    Tensor add(const Tensor& a, const Tensor& b) override { throw std::runtime_error("MLX not supported"); }
    Tensor subtract(const Tensor& a, const Tensor& b) override { throw std::runtime_error("MLX not supported"); }
    Tensor multiply(const Tensor& a, const Tensor& b) override { throw std::runtime_error("MLX not supported"); }
    Tensor divide(const Tensor& a, const Tensor& b) override { throw std::runtime_error("MLX not supported"); }
    Tensor matmul(const Tensor& a, const Tensor& b) override { throw std::runtime_error("MLX not supported"); }
    Tensor relu(const Tensor& x) override { throw std::runtime_error("MLX not supported"); }
    Tensor gelu(const Tensor& x) override { throw std::runtime_error("MLX not supported"); }
    Tensor silu(const Tensor& x) override { throw std::runtime_error("MLX not supported"); }
    Tensor softmax(const Tensor& x, int dim) override { throw std::runtime_error("MLX not supported"); }
    Tensor zeros(const std::vector<size_t>& shape, DataType dtype) override { throw std::runtime_error("MLX not supported"); }
    Tensor ones(const std::vector<size_t>& shape, DataType dtype) override { throw std::runtime_error("MLX not supported"); }
    Tensor sum(const Tensor& x, int dim) override { throw std::runtime_error("MLX not supported"); }
    Tensor mean(const Tensor& x, int dim) override { throw std::runtime_error("MLX not supported"); }
    std::string backend() const override { return "mlx_unsupported"; }
    
    static bool is_available() { return false; }
#endif
};

} // namespace ccsm

#endif // CCSM_MLX_TENSOR_H