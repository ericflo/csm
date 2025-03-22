#ifndef CCSM_TENSOR_H
#define CCSM_TENSOR_H

#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <iostream>
#include <map>

namespace ccsm {

// Endian format for serialization
enum class EndianFormat {
    NATIVE, // Use native system endianness
    LITTLE, // Force little endian
    BIG     // Force big endian
};

// Compression level for serialization
enum class CompressionLevel {
    NONE,    // No compression
    FAST,    // Fast compression (less effective)
    DEFAULT, // Default compression
    BEST     // Best compression (slower)
};

// Metadata for tensor serialization
struct TensorMetadata {
    std::string name;
    std::string description;
    int version = 0;
    std::map<std::string, std::string> custom_fields;
};

// Forward declarations
class TensorImpl;
class Context;

// Tensor data types
enum class DataType {
    F32,    // 32-bit float
    F16,    // 16-bit float
    BF16,   // 16-bit brain float
    I32,    // 32-bit signed integer
    I16,    // 16-bit signed integer
    I8,     // 8-bit signed integer
    Q8_0,   // 8-bit quantized (0-bias)
    Q4_0,   // 4-bit quantized (0-bias)
    Q4_1    // 4-bit quantized (non-0-bias)
};

// Tensor class - the main abstraction for different backends
class Tensor {
public:
    // Constructors and destructor
    Tensor();
    Tensor(std::shared_ptr<TensorImpl> impl);
    ~Tensor() = default;
    
    // Shape and metadata
    size_t shape(int dim) const;
    std::vector<size_t> shape() const;
    int ndim() const;
    size_t size() const;
    DataType dtype() const;
    std::string dtype_str() const;
    
    // Data access
    void* data();
    const void* data() const;
    
    // Basic operations
    Tensor reshape(const std::vector<size_t>& new_shape) const;
    Tensor view(const std::vector<size_t>& new_shape) const;
    Tensor slice(int dim, size_t start, size_t end) const;
    
    // Debug
    void print(const std::string& name = "") const;
    
    // Check if tensor is valid
    bool is_valid() const;
    
    // Implementation access
    std::shared_ptr<TensorImpl> impl() const { return impl_; }
    
private:
    std::shared_ptr<TensorImpl> impl_;
};

// Abstract tensor implementation interface
class TensorImpl {
public:
    virtual ~TensorImpl() = default;
    
    // Shape and metadata
    virtual size_t shape(int dim) const = 0;
    virtual std::vector<size_t> shape() const = 0;
    virtual int ndim() const = 0;
    virtual size_t size() const = 0;
    virtual DataType dtype() const = 0;
    
    // Data access
    virtual void* data() = 0;
    virtual const void* data() const = 0;
    
    // Basic operations
    virtual std::shared_ptr<TensorImpl> reshape(const std::vector<size_t>& new_shape) const = 0;
    virtual std::shared_ptr<TensorImpl> view(const std::vector<size_t>& new_shape) const = 0;
    virtual std::shared_ptr<TensorImpl> slice(int dim, size_t start, size_t end) const = 0;
    
    // Debug
    virtual void print(const std::string& name = "") const = 0;
};

// Tensor factory for creating tensors with the appropriate backend
class TensorFactory {
public:
    static Tensor create(const std::vector<size_t>& shape, DataType dtype);
    static Tensor zeros(const std::vector<size_t>& shape, DataType dtype);
    static Tensor ones(const std::vector<size_t>& shape, DataType dtype);
    static Tensor from_data(const void* data, const std::vector<size_t>& shape, DataType dtype);
    
    // Conversion between backends
    static Tensor convert(const Tensor& tensor, const std::string& to_backend);
    
    // Serialization functions
    static bool save(const Tensor& tensor, const std::string& filepath, 
                    EndianFormat endian = EndianFormat::NATIVE,
                    CompressionLevel compression = CompressionLevel::NONE);
    
    static Tensor load(const std::string& filepath, 
                     EndianFormat endian = EndianFormat::NATIVE);
    
    // Serialization with metadata
    static bool save_with_metadata(const Tensor& tensor, const std::string& filepath,
                                  const TensorMetadata& metadata,
                                  EndianFormat endian = EndianFormat::NATIVE,
                                  CompressionLevel compression = CompressionLevel::NONE);
    
    static Tensor load_with_metadata(const std::string& filepath, 
                                   TensorMetadata& metadata,
                                   EndianFormat endian = EndianFormat::NATIVE);
};

// Context for tensor operations
class Context {
public:
    virtual ~Context() = default;
    
    // Basic operations
    virtual Tensor add(const Tensor& a, const Tensor& b) = 0;
    virtual Tensor subtract(const Tensor& a, const Tensor& b) = 0;
    virtual Tensor multiply(const Tensor& a, const Tensor& b) = 0;
    virtual Tensor divide(const Tensor& a, const Tensor& b) = 0;
    
    // Matrix operations
    virtual Tensor matmul(const Tensor& a, const Tensor& b) = 0;
    
    // Activations
    virtual Tensor relu(const Tensor& x) = 0;
    virtual Tensor gelu(const Tensor& x) = 0;
    virtual Tensor silu(const Tensor& x) = 0;
    virtual Tensor softmax(const Tensor& x, int dim) = 0;
    
    // Creation
    virtual Tensor zeros(const std::vector<size_t>& shape, DataType dtype) = 0;
    virtual Tensor ones(const std::vector<size_t>& shape, DataType dtype) = 0;
    
    // Reductions
    virtual Tensor sum(const Tensor& x, int dim) = 0;
    virtual Tensor mean(const Tensor& x, int dim) = 0;
    
    // Type casting
    virtual Tensor cast(const Tensor& x, DataType dtype) = 0;
    
    // Type promotion helpers
    virtual DataType promote_types(DataType a, DataType b) = 0;
    
    // Get the backend name
    virtual std::string backend() const = 0;
};

// Factory for creating contexts
class ContextFactory {
public:
    static std::shared_ptr<Context> create(const std::string& backend = "default");
};

} // namespace ccsm

#endif // CCSM_TENSOR_H