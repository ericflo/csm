#include <ccsm/tensor.h>
#include <fstream>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <cstdint>
#include <sstream>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <memory>
#include <map>
#include <cmath>
#include <filesystem>

namespace ccsm {

// Internal data type size mapping
size_t get_dtype_size(DataType dtype) {
    switch (dtype) {
        case DataType::F32:
            return 4;
        case DataType::F16:
        case DataType::BF16:
            return 2;
        case DataType::I32:
            return 4;
        case DataType::I16:
            return 2;
        case DataType::I8:
            return 1;
        case DataType::Q8_0:
            return 1;
        case DataType::Q4_0:
        case DataType::Q4_1:
            return 1; // Note: Q4 types are packed 2 values per byte
        default:
            throw std::runtime_error("Unknown data type");
    }
}

// Helper to get string representation of data type
std::string get_dtype_str(DataType dtype) {
    switch (dtype) {
        case DataType::F32:
            return "F32";
        case DataType::F16:
            return "F16";
        case DataType::BF16:
            return "BF16";
        case DataType::I32:
            return "I32";
        case DataType::I16:
            return "I16";
        case DataType::I8:
            return "I8";
        case DataType::Q8_0:
            return "Q8_0";
        case DataType::Q4_0:
            return "Q4_0";
        case DataType::Q4_1:
            return "Q4_1";
        default:
            return "Unknown";
    }
}

// Helper for compression level code
uint8_t get_compression_code(CompressionLevel level) {
    switch (level) {
        case CompressionLevel::NONE:
            return 0;
        case CompressionLevel::FAST:
            return 1;
        case CompressionLevel::DEFAULT:
            return 2;
        case CompressionLevel::BEST:
            return 3;
        default:
            return 0;
    }
}

// Helper for endian format code
uint8_t get_endian_code(EndianFormat format) {
    switch (format) {
        case EndianFormat::NATIVE:
            return 0;
        case EndianFormat::LITTLE:
            return 1;
        case EndianFormat::BIG:
            return 2;
        default:
            return 0;
    }
}

// Detect native endianness
bool is_little_endian() {
    union {
        uint32_t i;
        char c[4];
    } bint = {0x01020304};
    
    return bint.c[0] == 4; 
}

// Convert endianness if needed
void convert_endianness(void* data, size_t size, size_t element_size, EndianFormat from, EndianFormat to) {
    // If source and target are the same, do nothing
    if (from == to) {
        return;
    }
    
    // If either source or target is NATIVE, we need to determine what NATIVE means
    bool is_system_little = is_little_endian();
    bool is_from_little = (from == EndianFormat::LITTLE) || 
                          (from == EndianFormat::NATIVE && is_system_little);
    bool is_to_little = (to == EndianFormat::LITTLE) || 
                        (to == EndianFormat::NATIVE && is_system_little);
    
    // If from and to have the same actual endianness, no conversion needed
    if (is_from_little == is_to_little) {
        return;
    }
    
    // Perform byte swap for each element
    uint8_t* bytes = static_cast<uint8_t*>(data);
    size_t num_elements = size / element_size;
    
    for (size_t i = 0; i < num_elements; ++i) {
        uint8_t* element = bytes + i * element_size;
        for (size_t j = 0; j < element_size / 2; ++j) {
            std::swap(element[j], element[element_size - j - 1]);
        }
    }
}

// Generic byte-level compression using LZ4 or similar scheme (placeholder for real implementation)
std::vector<uint8_t> compress(const void* data, size_t size, CompressionLevel level) {
    // This is a placeholder for a real compression implementation
    // In a real implementation, switch on level for different levels of compression
    
    // For now, just return the data as is
    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    return std::vector<uint8_t>(bytes, bytes + size);
}

// Decompress using LZ4 or similar scheme (placeholder for real implementation)
std::vector<uint8_t> decompress(const void* data, size_t compressed_size, size_t decompressed_size) {
    // This is a placeholder for a real decompression implementation
    
    // For now, just return the data as is
    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    return std::vector<uint8_t>(bytes, bytes + compressed_size);
}

// Forward declarations
class MemoryTensorImpl;
class TensorSliceView;

// Basic Tensor implementation for holding data in memory
class MemoryTensorImpl : public TensorImpl, public std::enable_shared_from_this<MemoryTensorImpl> {
public:
    MemoryTensorImpl(const std::vector<size_t>& shape, DataType dtype) 
        : shape_(shape), dtype_(dtype) {
        
        // Calculate size in elements
        size_t num_elements = 1;
        for (size_t dim : shape) {
            num_elements *= dim;
        }
        
        // Calculate default row-major strides
        strides_.resize(shape.size());
        size_t stride = 1;
        for (int i = shape.size() - 1; i >= 0; i--) {
            strides_[i] = stride;
            stride *= shape[i];
        }
        
        // Allocate memory for data
        size_t element_size = get_dtype_size(dtype);
        size_t bytes = num_elements * element_size;
        
        // Handle special case for Q4 types which are packed
        if (dtype == DataType::Q4_0 || dtype == DataType::Q4_1) {
            // Q4 types store 2 values per byte
            bytes = (num_elements + 1) / 2;
            
            // Plus additional space for scale factors
            bytes += num_elements * sizeof(float) / 32; // One scale per 32 values
            
            // For Q4_1, we also need bias values
            if (dtype == DataType::Q4_1) {
                bytes += num_elements * sizeof(float) / 32; // One bias per 32 values
            }
        }
        
        data_.resize(bytes, 0);
    }
    
    // Constructor with data copy
    MemoryTensorImpl(const void* data, const std::vector<size_t>& shape, DataType dtype)
        : MemoryTensorImpl(shape, dtype) {
        
        if (data) {
            // Calculate size in bytes
            size_t num_elements = 1;
            for (size_t dim : shape) {
                num_elements *= dim;
            }
            
            size_t element_size = get_dtype_size(dtype);
            size_t bytes = num_elements * element_size;
            
            // Handle special case for Q4 types which are packed
            if (dtype == DataType::Q4_0 || dtype == DataType::Q4_1) {
                // Q4 types store 2 values per byte
                bytes = (num_elements + 1) / 2;
                
                // Plus additional space for scale factors
                bytes += num_elements * sizeof(float) / 32; // One scale per 32 values
                
                // For Q4_1, we also need bias values
                if (dtype == DataType::Q4_1) {
                    bytes += num_elements * sizeof(float) / 32; // One bias per 32 values
                }
            }
            
            // Copy the data
            std::memcpy(data_.data(), data, bytes);
        }
    }
    
    // Constructor with custom strides
    MemoryTensorImpl(const std::vector<size_t>& shape, DataType dtype, const std::vector<size_t>& strides)
        : MemoryTensorImpl(shape, dtype) {
        if (strides.size() == shape.size()) {
            strides_ = strides;
        }
    }
    
    // Implement TensorImpl interface
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
        size_t s = 1;
        for (size_t dim : shape_) {
            s *= dim;
        }
        return s;
    }
    
    DataType dtype() const override {
        return dtype_;
    }
    
    std::vector<size_t> strides() const override {
        return strides_;
    }
    
    bool has_strides() const override {
        return !strides_.empty();
    }
    
    void* data() override {
        return data_.data();
    }
    
    const void* data() const override {
        return data_.data();
    }
    
    std::shared_ptr<TensorImpl> reshape(const std::vector<size_t>& new_shape) const override {
        // Check if new shape has the same number of elements
        size_t new_size = 1;
        for (size_t dim : new_shape) {
            new_size *= dim;
        }
        
        if (new_size != size()) {
            throw std::invalid_argument("Reshape: new shape must have the same total size");
        }
        
        // Create a new tensor with the same data but different shape
        auto result = std::make_shared<MemoryTensorImpl>(new_shape, dtype_);
        std::memcpy(result->data_.data(), data_.data(), data_.size());
        return result;
    }
    
    std::shared_ptr<TensorImpl> view(const std::vector<size_t>& new_shape) const override;  // Implemented after TensorView class definition
    
    std::shared_ptr<TensorImpl> slice(int dim, size_t start, size_t end) const override;  // Implemented after TensorSliceView class definition
    
    void print(const std::string& name) const override {
        std::cout << "Tensor" << (name.empty() ? "" : " ") << name << " [";
        for (size_t i = 0; i < shape_.size(); ++i) {
            std::cout << shape_[i];
            if (i < shape_.size() - 1) {
                std::cout << " x ";
            }
        }
        std::cout << "] " << get_dtype_str(dtype_) << std::endl;
        
        // For a real implementation, this would print the actual tensor data
    }
    
    // Helper function to cast the data pointer to a specific type
    template<typename T>
    T* data_as() {
        return static_cast<T*>(data());
    }
    
    template<typename T>
    const T* data_as() const {
        return static_cast<const T*>(data());
    }
    
private:
    std::vector<uint8_t> data_;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    DataType dtype_;
};

// TensorView implementation - for efficient memory sharing between tensors
class TensorView : public TensorImpl {
public:
    TensorView(std::shared_ptr<const MemoryTensorImpl> parent, const std::vector<size_t>& new_shape)
        : parent_(parent), shape_(new_shape) {
        // Verify that the total size matches the parent tensor
        size_t new_size = 1;
        for (size_t dim : new_shape) {
            new_size *= dim;
        }
        
        if (new_size != parent->size()) {
            throw std::invalid_argument("View: new shape must have the same total size");
        }
        
        // Inherit strides from parent
        if (parent->has_strides()) {
            // Since we're reshaping, we can't directly use parent strides
            // We just calculate new default strides for the new shape
            strides_.resize(new_shape.size());
            size_t stride = 1;
            for (int i = new_shape.size() - 1; i >= 0; i--) {
                strides_[i] = stride;
                stride *= new_shape[i];
            }
        }
    }

    // Method to get default strides for a shape
    static std::vector<size_t> default_strides(const std::vector<size_t>& shape) {
        std::vector<size_t> strides(shape.size());
        size_t stride = 1;
        for (int i = shape.size() - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }
    
    // Implement TensorImpl interface
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
        size_t s = 1;
        for (size_t dim : shape_) {
            s *= dim;
        }
        return s;
    }
    
    DataType dtype() const override {
        return parent_->dtype();
    }
    
    std::vector<size_t> strides() const override {
        return strides_;
    }
    
    bool has_strides() const override {
        return !strides_.empty();
    }
    
    void* data() override {
        // This is safe because the parent tensor's memory isn't modified,
        // we're just providing non-const access to a memory block that's
        // shared between tensors
        return const_cast<void*>(parent_->data());
    }
    
    const void* data() const override {
        return parent_->data();
    }
    
    std::shared_ptr<TensorImpl> reshape(const std::vector<size_t>& new_shape) const override {
        // Check if new shape has the same number of elements
        size_t new_size = 1;
        for (size_t dim : new_shape) {
            new_size *= dim;
        }
        
        if (new_size != size()) {
            throw std::invalid_argument("Reshape: new shape must have the same total size");
        }
        
        // Create a new view with the same parent but different shape
        return std::shared_ptr<TensorImpl>(new TensorView(parent_, new_shape));
    }
    
    std::shared_ptr<TensorImpl> view(const std::vector<size_t>& new_shape) const override {
        // Check if new shape has the same number of elements
        size_t new_size = 1;
        for (size_t dim : new_shape) {
            new_size *= dim;
        }
        
        if (new_size != size()) {
            throw std::invalid_argument("View: new shape must have the same total size");
        }
        
        // Create a new view with the same parent but different shape
        return std::shared_ptr<TensorImpl>(new TensorView(parent_, new_shape));
    }
    
    std::shared_ptr<TensorImpl> slice(int dim, size_t start, size_t end) const override {
        // For now, delegate to parent's slice implementation
        // In a more advanced implementation, we could create a slice view
        // that keeps reference to the parent and applies offsets
        
        // First create a tensor with the original shape
        Tensor temp(parent_->reshape(shape_));
        
        // Then create a slice
        return temp.slice(dim, start, end).impl();
    }
    
    void print(const std::string& name = "") const override {
        std::cout << "TensorView" << (name.empty() ? "" : " ") << name << " [";
        for (size_t i = 0; i < shape_.size(); ++i) {
            std::cout << shape_[i];
            if (i < shape_.size() - 1) {
                std::cout << " x ";
            }
        }
        std::cout << "] " << get_dtype_str(dtype()) << " (shares memory with parent)" << std::endl;
    }
    
private:
    std::shared_ptr<const MemoryTensorImpl> parent_; // Original tensor we're viewing
    std::vector<size_t> shape_;                // New shape for this view
    std::vector<size_t> strides_;              // Strides for this view
};

// TensorSliceView implementation - for memory sharing between sliced tensors
class TensorSliceView : public TensorImpl {
public:
    TensorSliceView(std::shared_ptr<const MemoryTensorImpl> parent, int dim, size_t start, size_t end)
        : parent_(parent), offset_(0) {
        // Validate input
        if (dim < 0 || dim >= parent->ndim()) {
            throw std::invalid_argument("Slice: dimension out of range");
        }
        
        if (start >= parent->shape(dim) || end > parent->shape(dim) || start >= end) {
            throw std::invalid_argument("Slice: invalid start or end indices");
        }
        
        // Copy the parent's shape
        shape_ = parent->shape();
        
        // Modify the sliced dimension
        shape_[dim] = end - start;
        
        // Calculate total size
        size_t total_size = 1;
        for (size_t d : shape_) {
            total_size *= d;
        }
        
        // Get or compute parent strides
        std::vector<size_t> parent_strides = parent->has_strides() 
                                          ? parent->strides() 
                                          : TensorView::default_strides(parent->shape());
        
        // Copy the parent's strides
        strides_ = parent_strides;
        
        // Calculate offset based on start index and dimension stride
        offset_ = start * parent_strides[dim];
    }
    
    // Implement TensorImpl interface
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
        size_t s = 1;
        for (size_t dim : shape_) {
            s *= dim;
        }
        return s;
    }
    
    DataType dtype() const override {
        return parent_->dtype();
    }
    
    std::vector<size_t> strides() const override {
        return strides_;
    }
    
    bool has_strides() const override {
        return !strides_.empty();
    }
    
    void* data() override {
        // Calculate the memory offset
        size_t element_size = get_dtype_size(dtype());
        uint8_t* base_ptr = static_cast<uint8_t*>(const_cast<void*>(parent_->data()));
        
        // Return pointer with appropriate offset
        return base_ptr + (offset_ * element_size);
    }
    
    const void* data() const override {
        // Calculate the memory offset
        size_t element_size = get_dtype_size(dtype());
        const uint8_t* base_ptr = static_cast<const uint8_t*>(parent_->data());
        
        // Return pointer with appropriate offset
        return base_ptr + (offset_ * element_size);
    }
    
    std::shared_ptr<TensorImpl> reshape(const std::vector<size_t>& new_shape) const override {
        // Check if new shape has the same number of elements
        size_t new_size = 1;
        for (size_t dim : new_shape) {
            new_size *= dim;
        }
        
        if (new_size != size()) {
            throw std::invalid_argument("Reshape: new shape must have the same total size");
        }
        
        // First create a TensorView from this slice
        Tensor temp(std::shared_ptr<TensorImpl>(new TensorView(parent_, shape_)));
        
        // Then reshape it
        return temp.reshape(new_shape).impl();
    }
    
    std::shared_ptr<TensorImpl> view(const std::vector<size_t>& new_shape) const override {
        // Check if new shape has the same number of elements
        size_t new_size = 1;
        for (size_t dim : new_shape) {
            new_size *= dim;
        }
        
        if (new_size != size()) {
            throw std::invalid_argument("View: new shape must have the same total size");
        }
        
        // Create a TensorView from our parent but with the new shape
        // and pass along our offset
        auto view = std::make_shared<TensorSliceView>(parent_, 0, 0, parent_->shape(0));
        view->shape_ = new_shape;
        view->offset_ = offset_;
        
        // Calculate new default strides for the new shape
        view->strides_.resize(new_shape.size());
        size_t stride = 1;
        for (int i = new_shape.size() - 1; i >= 0; i--) {
            view->strides_[i] = stride;
            stride *= new_shape[i];
        }
        
        return view;
    }
    
    std::shared_ptr<TensorImpl> slice(int dim, size_t start, size_t end) const override {
        // Validate dimension
        if (dim < 0 || dim >= ndim()) {
            throw std::invalid_argument("Slice: dimension out of range");
        }
        
        // Validate start and end
        if (start >= shape_[dim] || end > shape_[dim] || start >= end) {
            throw std::invalid_argument("Slice: invalid start or end indices");
        }
        
        // Create a new slice with adjusted shape and updated offset
        auto new_slice = std::make_shared<TensorSliceView>(parent_, 0, 0, parent_->shape(0));
        new_slice->shape_ = shape_;
        new_slice->shape_[dim] = end - start;
        new_slice->strides_ = strides_;
        
        // Update offset to account for the new slice
        new_slice->offset_ = offset_ + (start * strides_[dim]);
        
        return new_slice;
    }
    
    void print(const std::string& name = "") const override {
        std::cout << "TensorSliceView" << (name.empty() ? "" : " ") << name << " [";
        for (size_t i = 0; i < shape_.size(); ++i) {
            std::cout << shape_[i];
            if (i < shape_.size() - 1) {
                std::cout << " x ";
            }
        }
        std::cout << "] " << get_dtype_str(dtype()) << " (shares memory with offset " << offset_ << ")" << std::endl;
    }
    
private:
    std::shared_ptr<const MemoryTensorImpl> parent_; // Original tensor we're viewing
    std::vector<size_t> shape_;                // Shape of this slice
    std::vector<size_t> strides_;              // Strides for this slice
    size_t offset_;                            // Offset in elements from the start of parent's data
    
    // Allow the MemoryTensorImpl::slice method to access our internals
    friend std::shared_ptr<TensorImpl> MemoryTensorImpl::slice(int, size_t, size_t) const;
};

// Implementation of MemoryTensorImpl::view now that TensorView is defined
std::shared_ptr<TensorImpl> MemoryTensorImpl::view(const std::vector<size_t>& new_shape) const {
    // Check if new shape has the same number of elements
    size_t new_size = 1;
    for (size_t dim : new_shape) {
        new_size *= dim;
    }
    
    if (new_size != size()) {
        throw std::invalid_argument("View: new shape must have the same total size");
    }
    
    // Create a shared view of the tensor that uses the same underlying memory
    return std::shared_ptr<TensorImpl>(new TensorView(shared_from_this(), new_shape));
}

// Implementation of MemoryTensorImpl::slice now that TensorSliceView is defined
std::shared_ptr<TensorImpl> MemoryTensorImpl::slice(int dim, size_t start, size_t end) const {
    // Check dimension is valid
    if (dim < 0 || dim >= ndim()) {
        throw std::invalid_argument("Slice: dimension out of range");
    }
    
    // Check start and end are valid
    if (start >= shape_[dim] || end > shape_[dim] || start >= end) {
        throw std::invalid_argument("Slice: invalid start or end");
    }
    
    // Create a TensorSliceView that shares memory with this tensor
    return std::shared_ptr<TensorImpl>(new TensorSliceView(shared_from_this(), dim, start, end));
}

// Tensor implementation

Tensor::Tensor() : impl_(nullptr) {}

Tensor::Tensor(std::shared_ptr<TensorImpl> impl) : impl_(impl) {}

size_t Tensor::shape(int dim) const {
    if (!impl_) {
        throw std::runtime_error("Tensor not initialized");
    }
    return impl_->shape(dim);
}

std::vector<size_t> Tensor::shape() const {
    if (!impl_) {
        throw std::runtime_error("Tensor not initialized");
    }
    return impl_->shape();
}

int Tensor::ndim() const {
    if (!impl_) {
        throw std::runtime_error("Tensor not initialized");
    }
    return impl_->ndim();
}

size_t Tensor::size() const {
    if (!impl_) {
        throw std::runtime_error("Tensor not initialized");
    }
    return impl_->size();
}

DataType Tensor::dtype() const {
    if (!impl_) {
        throw std::runtime_error("Tensor not initialized");
    }
    return impl_->dtype();
}

std::string Tensor::dtype_str() const {
    if (!impl_) {
        throw std::runtime_error("Tensor not initialized");
    }
    return get_dtype_str(impl_->dtype());
}

std::vector<size_t> Tensor::strides() const {
    if (!impl_) {
        throw std::runtime_error("Tensor not initialized");
    }
    return impl_->strides();
}

bool Tensor::has_strides() const {
    if (!impl_) {
        throw std::runtime_error("Tensor not initialized");
    }
    return impl_->has_strides();
}

void* Tensor::data() {
    if (!impl_) {
        throw std::runtime_error("Tensor not initialized");
    }
    return impl_->data();
}

const void* Tensor::data() const {
    if (!impl_) {
        throw std::runtime_error("Tensor not initialized");
    }
    return impl_->data();
}

Tensor Tensor::reshape(const std::vector<size_t>& new_shape) const {
    if (!impl_) {
        throw std::runtime_error("Tensor not initialized");
    }
    return Tensor(impl_->reshape(new_shape));
}

Tensor Tensor::view(const std::vector<size_t>& new_shape) const {
    if (!impl_) {
        throw std::runtime_error("Tensor not initialized");
    }
    return Tensor(impl_->view(new_shape));
}

Tensor Tensor::slice(int dim, size_t start, size_t end) const {
    if (!impl_) {
        throw std::runtime_error("Tensor not initialized");
    }
    return Tensor(impl_->slice(dim, start, end));
}

void Tensor::print(const std::string& name) const {
    if (!impl_) {
        throw std::runtime_error("Tensor not initialized");
    }
    impl_->print(name);
}

bool Tensor::is_valid() const {
    return impl_ != nullptr;
}

// TensorFactory implementation

Tensor TensorFactory::create(const std::vector<size_t>& shape, DataType dtype) {
    return Tensor(std::make_shared<MemoryTensorImpl>(shape, dtype));
}

Tensor TensorFactory::zeros(const std::vector<size_t>& shape, DataType dtype) {
    // Create tensor with all zeros (already handled by constructor)
    return Tensor(std::make_shared<MemoryTensorImpl>(shape, dtype));
}

Tensor TensorFactory::ones(const std::vector<size_t>& shape, DataType dtype) {
    // Create tensor
    auto tensor = std::make_shared<MemoryTensorImpl>(shape, dtype);
    
    // Calculate number of elements
    size_t num_elements = 1;
    for (size_t dim : shape) {
        num_elements *= dim;
    }
    
    // Fill with ones based on data type
    switch (dtype) {
        case DataType::F32: {
            float* data = static_cast<float*>(tensor->data());
            std::fill_n(data, num_elements, 1.0f);
            break;
        }
        case DataType::I32: {
            int32_t* data = static_cast<int32_t*>(tensor->data());
            std::fill_n(data, num_elements, 1);
            break;
        }
        // Other data types would be handled similarly
        default:
            // For other types, just return zeros for now as a placeholder
            break;
    }
    
    return Tensor(tensor);
}

Tensor TensorFactory::from_data(const void* data, const std::vector<size_t>& shape, DataType dtype) {
    return Tensor(std::make_shared<MemoryTensorImpl>(data, shape, dtype));
}

Tensor TensorFactory::convert(const Tensor& tensor, const std::string& to_backend) {
    // This is a placeholder - in a real implementation, this would convert to different backends
    return tensor;
}

bool TensorFactory::save(const Tensor& tensor, const std::string& filepath, 
                          EndianFormat endian, CompressionLevel compression) {
    // Make sure tensor is valid
    if (!tensor.is_valid()) {
        return false;
    }
    
    // Create directory if it doesn't exist
    std::filesystem::path path(filepath);
    auto parent_path = path.parent_path();
    if (!parent_path.empty() && !std::filesystem::exists(parent_path)) {
        std::error_code ec;
        if (!std::filesystem::create_directories(parent_path, ec) && ec) {
            return false;
        }
    }
    
    // Open file for writing
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    // File format:
    // - Magic number (4 bytes): "TSRZ" (Tensor Serialized)
    // - Format version (4 bytes): uint32_t
    // - Number of dimensions (4 bytes): int32_t
    // - Data type (4 bytes): DataType enum
    // - Shape dimensions (8 bytes each): uint64_t[]
    // - Endian format (1 byte): uint8_t
    // - Compression level (1 byte): uint8_t
    // - Compressed data size (8 bytes): uint64_t
    // - Uncompressed data size (8 bytes): uint64_t
    // - Data: bytes[]
    
    // Write magic number: "TSRZ" (Tensor Serialized)
    const uint32_t magic = 0x5A525354; // "TSRZ" in hex
    file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    
    // Write format version
    const uint32_t version = 1;
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));
    
    // Write number of dimensions
    int32_t ndim = tensor.ndim();
    file.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
    
    // Write data type
    DataType dtype = tensor.dtype();
    file.write(reinterpret_cast<const char*>(&dtype), sizeof(dtype));
    
    // Write shape dimensions
    auto shape = tensor.shape();
    for (size_t dim : shape) {
        uint64_t dim_size = static_cast<uint64_t>(dim);
        file.write(reinterpret_cast<const char*>(&dim_size), sizeof(dim_size));
    }
    
    // Write endian format code
    uint8_t endian_code = get_endian_code(endian);
    file.write(reinterpret_cast<const char*>(&endian_code), sizeof(endian_code));
    
    // Write compression level code
    uint8_t compression_code = get_compression_code(compression);
    file.write(reinterpret_cast<const char*>(&compression_code), sizeof(compression_code));
    
    // Calculate data size
    size_t element_size = get_dtype_size(dtype);
    size_t element_count = tensor.size();
    
    // Handle special case for Q4 types which are packed
    size_t data_size;
    if (dtype == DataType::Q4_0 || dtype == DataType::Q4_1) {
        // Q4 types store 2 values per byte
        data_size = (element_count + 1) / 2;
        
        // Plus additional space for scale factors
        data_size += element_count * sizeof(float) / 32; // One scale per 32 values
        
        // For Q4_1, we also need bias values
        if (dtype == DataType::Q4_1) {
            data_size += element_count * sizeof(float) / 32; // One bias per 32 values
        }
    } else {
        data_size = element_count * element_size;
    }
    
    // Get raw data pointer
    const void* data = tensor.data();
    
    // Handle endianness conversion if needed
    std::vector<uint8_t> endian_buffer;
    if (endian != EndianFormat::NATIVE) {
        endian_buffer.resize(data_size);
        std::memcpy(endian_buffer.data(), data, data_size);
        convert_endianness(endian_buffer.data(), data_size, element_size, 
                           EndianFormat::NATIVE, endian);
        data = endian_buffer.data();
    }
    
    // Compress data if needed
    std::vector<uint8_t> compressed_data;
    size_t compressed_size = data_size;
    
    if (compression != CompressionLevel::NONE) {
        compressed_data = compress(data, data_size, compression);
        compressed_size = compressed_data.size();
        data = compressed_data.data();
    }
    
    // Write data sizes
    uint64_t compressed_size_u64 = static_cast<uint64_t>(compressed_size);
    uint64_t data_size_u64 = static_cast<uint64_t>(data_size);
    file.write(reinterpret_cast<const char*>(&compressed_size_u64), sizeof(compressed_size_u64));
    file.write(reinterpret_cast<const char*>(&data_size_u64), sizeof(data_size_u64));
    
    // Write data
    file.write(static_cast<const char*>(data), compressed_size);
    
    return file.good();
}

Tensor TensorFactory::load(const std::string& filepath, EndianFormat endian) {
    // Open file for reading
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }
    
    // Read magic number
    uint32_t magic;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != 0x5A525354) {
        throw std::runtime_error("Invalid tensor file format");
    }
    
    // Read format version
    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != 1) {
        throw std::runtime_error("Unsupported tensor file version");
    }
    
    // Read number of dimensions
    int32_t ndim;
    file.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
    
    // Read data type
    DataType dtype;
    file.read(reinterpret_cast<char*>(&dtype), sizeof(dtype));
    
    // Read shape dimensions
    std::vector<size_t> shape(ndim);
    for (int i = 0; i < ndim; ++i) {
        uint64_t dim_size;
        file.read(reinterpret_cast<char*>(&dim_size), sizeof(dim_size));
        shape[i] = static_cast<size_t>(dim_size);
    }
    
    // Read endian format code
    uint8_t endian_code;
    file.read(reinterpret_cast<char*>(&endian_code), sizeof(endian_code));
    
    // Convert endian code to format
    EndianFormat file_endian;
    switch (endian_code) {
        case 0:
            file_endian = EndianFormat::NATIVE;
            break;
        case 1:
            file_endian = EndianFormat::LITTLE;
            break;
        case 2:
            file_endian = EndianFormat::BIG;
            break;
        default:
            throw std::runtime_error("Invalid endian format code");
    }
    
    // Read compression level code
    uint8_t compression_code;
    file.read(reinterpret_cast<char*>(&compression_code), sizeof(compression_code));
    
    // Convert compression code to level
    CompressionLevel compression;
    switch (compression_code) {
        case 0:
            compression = CompressionLevel::NONE;
            break;
        case 1:
            compression = CompressionLevel::FAST;
            break;
        case 2:
            compression = CompressionLevel::DEFAULT;
            break;
        case 3:
            compression = CompressionLevel::BEST;
            break;
        default:
            throw std::runtime_error("Invalid compression level code");
    }
    
    // Read data sizes
    uint64_t compressed_size, data_size;
    file.read(reinterpret_cast<char*>(&compressed_size), sizeof(compressed_size));
    file.read(reinterpret_cast<char*>(&data_size), sizeof(data_size));
    
    // Read compressed data
    std::vector<uint8_t> compressed_data(compressed_size);
    file.read(reinterpret_cast<char*>(compressed_data.data()), compressed_size);
    
    // Decompress if needed
    std::vector<uint8_t> data;
    if (compression != CompressionLevel::NONE) {
        data = decompress(compressed_data.data(), compressed_size, data_size);
    } else {
        data = compressed_data;
    }
    
    // Convert endianness if needed
    size_t element_size = get_dtype_size(dtype);
    if (file_endian != EndianFormat::NATIVE && endian != file_endian) {
        convert_endianness(data.data(), data_size, element_size, file_endian, endian);
    }
    
    // Create tensor from data
    return TensorFactory::from_data(data.data(), shape, dtype);
}

bool TensorFactory::save_with_metadata(const Tensor& tensor, const std::string& filepath,
                                        const TensorMetadata& metadata,
                                        EndianFormat endian, CompressionLevel compression) {
    // First save the tensor
    if (!save(tensor, filepath, endian, compression)) {
        return false;
    }
    
    // Open file for appending
    std::ofstream file(filepath, std::ios::binary | std::ios::app);
    if (!file.is_open()) {
        return false;
    }
    
    // Get the current file size
    file.seekp(0, std::ios::end);
    std::streampos file_size = file.tellp();
    
    // Add metadata marker and safety signature
    const uint32_t metadata_marker = 0x4D455441; // "META" in little endian
    const uint64_t safety_marker = 0x4353534D4D455441; // "CSMMETA" in little endian
    
    file.write(reinterpret_cast<const char*>(&metadata_marker), sizeof(metadata_marker));
    file.write(reinterpret_cast<const char*>(&safety_marker), sizeof(safety_marker));
    
    // Write metadata name
    uint32_t name_length = metadata.name.length();
    file.write(reinterpret_cast<const char*>(&name_length), sizeof(name_length));
    file.write(metadata.name.c_str(), name_length);
    
    // Write metadata description
    uint32_t desc_length = metadata.description.length();
    file.write(reinterpret_cast<const char*>(&desc_length), sizeof(desc_length));
    file.write(metadata.description.c_str(), desc_length);
    
    // Write version
    int32_t version = metadata.version;
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));
    
    // Write custom fields
    uint32_t num_fields = metadata.custom_fields.size();
    file.write(reinterpret_cast<const char*>(&num_fields), sizeof(num_fields));
    
    for (const auto& field : metadata.custom_fields) {
        // Write key
        uint32_t key_length = field.first.length();
        file.write(reinterpret_cast<const char*>(&key_length), sizeof(key_length));
        file.write(field.first.c_str(), key_length);
        
        // Write value
        uint32_t value_length = field.second.length();
        file.write(reinterpret_cast<const char*>(&value_length), sizeof(value_length));
        file.write(field.second.c_str(), value_length);
    }
    
    return true;
}

Tensor TensorFactory::load_with_metadata(const std::string& filepath, 
                                         TensorMetadata& metadata,
                                         EndianFormat endian) {
    // First load the tensor
    Tensor tensor = load(filepath, endian);
    
    // Open file to read metadata from the end
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }
    
    // Get file size
    std::streampos fileSize = file.tellg();
    
    // Look for metadata marker
    const uint32_t metadata_marker = 0x4D455441; // "META" in little endian
    const uint64_t safety_marker = 0x4353534D4D455441; // "CSMMETA" in little endian
    
    // Minimum size for metadata: marker + safety + minimal content
    const std::streampos min_size = sizeof(metadata_marker) + sizeof(safety_marker) + 20;
    
    // Check if file is large enough to contain metadata
    if (fileSize < min_size) {
        throw std::runtime_error("File too small to contain metadata");
    }
    
    // Position the file pointer at the potential location of the metadata marker
    // We're expecting it at the end of the base tensor data
    bool found_metadata = false;
    
    // Starting from the beginning, read the tensor header to find where data ends
    file.seekg(0, std::ios::beg);
    
    // Skip magic, version, ndim, dtype
    file.seekg(4 + 4 + 4 + 4, std::ios::cur);
    
    // Read number of dimensions
    int32_t ndim;
    file.seekg(-8, std::ios::cur); // Go back to read ndim
    file.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
    file.seekg(4, std::ios::cur); // Skip over dtype again
    
    // Skip shape dimensions
    file.seekg(ndim * sizeof(uint64_t), std::ios::cur);
    
    // Skip endian and compression codes
    file.seekg(1 + 1, std::ios::cur);
    
    // Read data sizes
    uint64_t compressed_size, data_size;
    file.read(reinterpret_cast<char*>(&compressed_size), sizeof(compressed_size));
    file.read(reinterpret_cast<char*>(&data_size), sizeof(data_size));
    
    // Calculate where data ends
    std::streampos metadata_pos = file.tellg() + static_cast<std::streampos>(compressed_size);
    
    // Go to that position
    file.seekg(metadata_pos);
    
    // Now look for metadata marker
    uint32_t marker;
    file.read(reinterpret_cast<char*>(&marker), sizeof(marker));
    
    if (marker != metadata_marker) {
        throw std::runtime_error("No metadata found in file");
    }
    
    // Check safety marker
    uint64_t safety;
    file.read(reinterpret_cast<char*>(&safety), sizeof(safety));
    
    if (safety != safety_marker) {
        throw std::runtime_error("Invalid metadata format");
    }
    
    // Read metadata name
    uint32_t name_length;
    file.read(reinterpret_cast<char*>(&name_length), sizeof(name_length));
    metadata.name.resize(name_length);
    file.read(&metadata.name[0], name_length);
    
    // Read metadata description
    uint32_t desc_length;
    file.read(reinterpret_cast<char*>(&desc_length), sizeof(desc_length));
    metadata.description.resize(desc_length);
    file.read(&metadata.description[0], desc_length);
    
    // Read version
    int32_t version;
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    metadata.version = version;
    
    // Read custom fields
    uint32_t num_fields;
    file.read(reinterpret_cast<char*>(&num_fields), sizeof(num_fields));
    
    metadata.custom_fields.clear();
    for (uint32_t i = 0; i < num_fields; ++i) {
        // Read key
        uint32_t key_length;
        file.read(reinterpret_cast<char*>(&key_length), sizeof(key_length));
        std::string key(key_length, '\0');
        file.read(&key[0], key_length);
        
        // Read value
        uint32_t value_length;
        file.read(reinterpret_cast<char*>(&value_length), sizeof(value_length));
        std::string value(value_length, '\0');
        file.read(&value[0], value_length);
        
        // Store in map
        metadata.custom_fields[key] = value;
    }
    
    return tensor;
}

// Basic CPU Context implementation
class CPUContext : public Context {
public:
    Tensor add(const Tensor& a, const Tensor& b) override {
        // Check if tensors are compatible
        if (a.ndim() != b.ndim()) {
            throw std::invalid_argument("Tensors must have the same number of dimensions");
        }
        
        // Check if shapes match
        for (int i = 0; i < a.ndim(); ++i) {
            if (a.shape(i) != b.shape(i)) {
                throw std::invalid_argument("Tensor shapes must match");
            }
        }
        
        // Determine result data type
        DataType result_dtype = promote_types(a.dtype(), b.dtype());
        
        // Create output tensor
        Tensor result = TensorFactory::zeros(a.shape(), result_dtype);
        
        // For a real implementation, perform element-wise addition here
        // This is just a placeholder that returns a tensor of the right shape
        
        return result;
    }
    
    Tensor subtract(const Tensor& a, const Tensor& b) override {
        // Similar to add, placeholder
        return add(a, b); // Just a placeholder
    }
    
    Tensor multiply(const Tensor& a, const Tensor& b) override {
        // Similar to add, placeholder
        return add(a, b); // Just a placeholder
    }
    
    Tensor divide(const Tensor& a, const Tensor& b) override {
        // Similar to add, placeholder
        return add(a, b); // Just a placeholder
    }
    
    Tensor matmul(const Tensor& a, const Tensor& b) override {
        // Check if dimensions are valid for matmul
        if (a.ndim() < 1 || b.ndim() < 1) {
            throw std::invalid_argument("Tensors must have at least 1 dimension for matmul");
        }
        
        // Determine result shape
        std::vector<size_t> result_shape;
        
        // Handle vector-vector case
        if (a.ndim() == 1 && b.ndim() == 1) {
            if (a.shape(0) != b.shape(0)) {
                throw std::invalid_argument("Inner dimensions must match for matmul");
            }
            // Result is scalar (0-dimensional)
            result_shape = {};
        }
        // Handle matrix-vector case
        else if (a.ndim() == 2 && b.ndim() == 1) {
            if (a.shape(1) != b.shape(0)) {
                throw std::invalid_argument("Inner dimensions must match for matmul");
            }
            result_shape = {a.shape(0)};
        }
        // Handle vector-matrix case
        else if (a.ndim() == 1 && b.ndim() == 2) {
            if (a.shape(0) != b.shape(0)) {
                throw std::invalid_argument("Inner dimensions must match for matmul");
            }
            result_shape = {b.shape(1)};
        }
        // Handle matrix-matrix case
        else if (a.ndim() == 2 && b.ndim() == 2) {
            if (a.shape(1) != b.shape(0)) {
                throw std::invalid_argument("Inner dimensions must match for matmul");
            }
            result_shape = {a.shape(0), b.shape(1)};
        }
        // Higher dimensional cases not fully handled in this placeholder
        else {
            throw std::invalid_argument("Higher dimensional matmul not implemented");
        }
        
        // Determine result data type
        DataType result_dtype = promote_types(a.dtype(), b.dtype());
        
        // Create output tensor
        Tensor result = TensorFactory::zeros(result_shape, result_dtype);
        
        // For a real implementation, perform matmul here
        // This is just a placeholder that returns a tensor of the right shape
        
        return result;
    }
    
    Tensor relu(const Tensor& x) override {
        // Create output tensor
        Tensor result = TensorFactory::zeros(x.shape(), x.dtype());
        
        // For a real implementation, apply ReLU here
        // This is just a placeholder that returns a tensor of the right shape
        
        return result;
    }
    
    Tensor gelu(const Tensor& x) override {
        // Similar to relu, placeholder
        return relu(x); // Just a placeholder
    }
    
    Tensor silu(const Tensor& x) override {
        // Similar to relu, placeholder
        return relu(x); // Just a placeholder
    }
    
    Tensor softmax(const Tensor& x, int dim) override {
        // Check dimension is valid
        if (dim < 0 || dim >= x.ndim()) {
            throw std::invalid_argument("Invalid dimension for softmax");
        }
        
        // Create output tensor
        Tensor result = TensorFactory::zeros(x.shape(), x.dtype());
        
        // For a real implementation, apply softmax here
        // This is just a placeholder that returns a tensor of the right shape
        
        return result;
    }
    
    Tensor zeros(const std::vector<size_t>& shape, DataType dtype) override {
        return TensorFactory::zeros(shape, dtype);
    }
    
    Tensor ones(const std::vector<size_t>& shape, DataType dtype) override {
        return TensorFactory::ones(shape, dtype);
    }
    
    Tensor sum(const Tensor& x, int dim) override {
        // Check dimension is valid
        if (dim < 0 || dim >= x.ndim()) {
            throw std::invalid_argument("Invalid dimension for sum");
        }
        
        // Create new shape with the specified dimension removed
        std::vector<size_t> result_shape;
        for (int i = 0; i < x.ndim(); ++i) {
            if (i != dim) {
                result_shape.push_back(x.shape(i));
            }
        }
        
        // Create output tensor
        Tensor result = TensorFactory::zeros(result_shape, x.dtype());
        
        // For a real implementation, compute sum along the dimension here
        // This is just a placeholder that returns a tensor of the right shape
        
        return result;
    }
    
    Tensor mean(const Tensor& x, int dim) override {
        // Similar to sum, placeholder
        return sum(x, dim); // Just a placeholder
    }
    
    Tensor cast(const Tensor& x, DataType dtype) override {
        // Create output tensor of the requested type
        Tensor result = TensorFactory::zeros(x.shape(), dtype);
        
        // For a real implementation, perform type conversion here
        // This is just a placeholder that returns a tensor of the right shape
        
        return result;
    }
    
    DataType promote_types(DataType a, DataType b) override {
        // Simple type promotion rules:
        // - F32 is higher precision than F16/BF16
        // - F16/BF16 are higher precision than integer types
        // - I32 is higher precision than I16/I8
        // - Regular types are higher precision than quantized types
        
        // If types are the same, return that type
        if (a == b) {
            return a;
        }
        
        // F32 is highest precision
        if (a == DataType::F32 || b == DataType::F32) {
            return DataType::F32;
        }
        
        // F16 and BF16 are next
        if (a == DataType::F16 || b == DataType::F16) {
            return DataType::F16;
        }
        if (a == DataType::BF16 || b == DataType::BF16) {
            return DataType::BF16;
        }
        
        // I32 is next
        if (a == DataType::I32 || b == DataType::I32) {
            return DataType::I32;
        }
        
        // I16 is next
        if (a == DataType::I16 || b == DataType::I16) {
            return DataType::I16;
        }
        
        // I8 is next
        if (a == DataType::I8 || b == DataType::I8) {
            return DataType::I8;
        }
        
        // Q8_0 is next
        if (a == DataType::Q8_0 || b == DataType::Q8_0) {
            return DataType::Q8_0;
        }
        
        // Q4_1 is next
        if (a == DataType::Q4_1 || b == DataType::Q4_1) {
            return DataType::Q4_1;
        }
        
        // Q4_0 is lowest
        return DataType::Q4_0;
    }
    
    std::string backend() const override {
        return "cpu";
    }
};

// ContextFactory implementation
std::shared_ptr<Context> ContextFactory::create(const std::string& backend) {
    // For now, just return a CPU context
    return std::make_shared<CPUContext>();
}

} // namespace ccsm