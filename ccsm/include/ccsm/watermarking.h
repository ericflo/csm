#ifndef CCSM_WATERMARKING_H
#define CCSM_WATERMARKING_H

#include <string>
#include <vector>
#include <memory>

namespace ccsm {

// Watermarker interface
class Watermarker {
public:
    virtual ~Watermarker() = default;
    
    // Create a new watermarker with a specific key
    static std::shared_ptr<Watermarker> create(const std::string& key = "");
    
    // Apply watermark to audio
    virtual std::vector<float> apply_watermark(const std::vector<float>& audio) = 0;
    
    // Detect watermark
    virtual bool detect_watermark(const std::vector<float>& audio) = 0;
    
    // Get the watermark strength
    virtual float get_strength() const = 0;
    
    // Set the watermark strength
    virtual void set_strength(float strength) = 0;
    
    // Get the watermark key
    virtual std::string get_key() const = 0;
};

// SilentCipher watermarker implementation
class SilentCipherWatermarker : public Watermarker {
public:
    // Create a new SilentCipher watermarker
    static std::shared_ptr<SilentCipherWatermarker> create(const std::string& key = "");
    
    // Apply watermark
    std::vector<float> apply_watermark(const std::vector<float>& audio) override = 0;
    
    // Detect watermark
    bool detect_watermark(const std::vector<float>& audio) override = 0;
    
    // Get the watermark strength
    float get_strength() const override = 0;
    
    // Set the watermark strength
    void set_strength(float strength) override = 0;
    
    // Get the watermark key
    std::string get_key() const override = 0;
    
    // SilentCipher-specific methods
    virtual void set_frame_size(int frame_size) = 0;
    virtual void set_hop_size(int hop_size) = 0;
    virtual bool verify_watermark(const std::vector<float>& audio, const std::string& key) = 0;
};

} // namespace ccsm

#endif // CCSM_WATERMARKING_H