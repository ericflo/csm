#include <ccsm/watermarking.h>
#include <random>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>
#include <memory>

namespace ccsm {

// Forward declaration of implementation classes
class SilentCipherWatermarkerImpl;

// ----- Mock watermarker implementation -----
class MockWatermarker : public Watermarker {
public:
    MockWatermarker(const std::string& key = "") : key_(key), strength_(0.1f) {}
    
    std::vector<float> apply_watermark(const std::vector<float>& audio) override {
        // Simple mock implementation - just adds a small offset to audio samples
        std::vector<float> result = audio;
        for (size_t i = 0; i < result.size(); i++) {
            result[i] += strength_ * 0.001f;
        }
        return result;
    }
    
    bool detect_watermark(const std::vector<float>& audio) override {
        // Simple mock detection - always returns true for non-empty audio
        return !audio.empty();
    }
    
    WatermarkResult detect(const std::vector<float>& audio, float sample_rate) override {
        WatermarkResult result;
        result.detected = !audio.empty();
        result.payload = key_;
        result.confidence = 0.95f;
        return result;
    }
    
    float get_strength() const override {
        return strength_;
    }
    
    void set_strength(float strength) override {
        strength_ = std::clamp(strength, 0.0f, 1.0f);
    }
    
    std::string get_key() const override {
        return key_;
    }
    
private:
    std::string key_;
    float strength_;
};

// Factory method for Watermarker
std::shared_ptr<Watermarker> Watermarker::create(const std::string& key) {
    // For now, return a MockWatermarker. In a real implementation,
    // we would return a SilentCipherWatermarker instance.
    return std::make_shared<MockWatermarker>(key);
}

// ----- SilentCipher implementation -----
class SilentCipherWatermarkerImpl : public SilentCipherWatermarker {
public:
    SilentCipherWatermarkerImpl(const std::string& key)
        : key_(key), 
          strength_(0.1f),
          frame_size_(1024),
          hop_size_(256),
          sample_rate_(16000),
          seed_(0)
    {
        initialize();
    }
    
    void initialize() {
        // Initialize random generator with key-based seed
        seed_ = hash_key(key_);
        generator_.seed(seed_);
        
        // Initialize frequency indices for watermarking
        init_frequency_indices();
    }
    
    std::vector<float> apply_watermark(const std::vector<float>& audio) override {
        // Default watermarking
        return embed(audio, sample_rate_, key_);
    }
    
    bool detect_watermark(const std::vector<float>& audio) override {
        WatermarkResult result = detect(audio, sample_rate_);
        return result.detected;
    }
    
    WatermarkResult detect(const std::vector<float>& audio, float sample_rate) override {
        WatermarkResult result;
        
        if (audio.empty() || sample_rate <= 0.0f) {
            return result; // Empty result with detected = false
        }
        
        // Store sample rate for future operations
        sample_rate_ = sample_rate;
        
        // Analyze the audio to find watermark
        result.detected = detect_watermark_internal(audio, result.payload, result.confidence);
        
        return result;
    }
    
    float get_strength() const override {
        return strength_;
    }
    
    void set_strength(float strength) override {
        strength_ = std::clamp(strength, 0.0f, 1.0f);
    }
    
    std::string get_key() const override {
        return key_;
    }
    
    void set_frame_size(int frame_size) override {
        if (frame_size <= 0 || (frame_size & (frame_size - 1)) != 0) {
            throw std::invalid_argument("Frame size must be a positive power of 2");
        }
        
        frame_size_ = frame_size;
        init_frequency_indices(); // Reinitialize frequency indices
    }
    
    void set_hop_size(int hop_size) override {
        if (hop_size <= 0) {
            throw std::invalid_argument("Hop size must be positive");
        }
        
        hop_size_ = hop_size;
    }
    
    bool verify_watermark(const std::vector<float>& audio, const std::string& key) override {
        // Verify using the provided key instead of the stored key
        std::string detected_key;
        float confidence;
        bool detected = detect_watermark_internal(audio, detected_key, confidence);
        
        // Check if the watermark was detected and the detected key matches the provided key
        return detected && detected_key == key;
    }
    
    std::vector<float> embed(const std::vector<float>& audio, 
                            float sample_rate, 
                            const std::string& payload) override {
        if (audio.empty() || sample_rate <= 0.0f) {
            return audio;
        }
        
        // Store sample rate for future operations
        sample_rate_ = sample_rate;
        
        // Create a copy of the audio to modify
        std::vector<float> result = audio;
        
        // Apply the watermark
        embed_watermark_internal(result, payload);
        
        return result;
    }
    
private:
    // Key and configuration
    std::string key_;
    float strength_;
    int frame_size_;
    int hop_size_;
    float sample_rate_;
    
    // Random number generation
    unsigned int seed_;
    std::mt19937 generator_;
    
    // Watermarking frequency indices
    std::vector<int> frequency_indices_;
    
    // Key hashing function
    unsigned int hash_key(const std::string& key) {
        unsigned int hash = 0;
        for (char c : key) {
            hash = hash * 31 + c;
        }
        return hash ? hash : 42; // Default seed if hash is 0
    }
    
    // Initialize frequency indices for watermarking
    void init_frequency_indices() {
        // Number of frequency bins is frame_size / 2 + 1
        int num_bins = frame_size_ / 2 + 1;
        
        // Select frequency bins for watermarking (avoiding lowest and highest frequencies)
        int min_freq_bin = static_cast<int>(300.0f * frame_size_ / sample_rate_); // 300 Hz
        int max_freq_bin = static_cast<int>(3500.0f * frame_size_ / sample_rate_); // 3500 Hz
        
        min_freq_bin = std::max(1, min_freq_bin);
        max_freq_bin = std::min(num_bins - 1, max_freq_bin);
        
        // Reset frequency indices
        frequency_indices_.clear();
        
        // Create a deterministic set of frequency indices based on key
        std::uniform_int_distribution<int> dist(min_freq_bin, max_freq_bin);
        int num_indices = std::min(64, max_freq_bin - min_freq_bin); // Use up to 64 frequency bins
        
        // Generate unique indices
        std::unordered_map<int, bool> used_indices;
        for (int i = 0; i < num_indices; i++) {
            int index;
            do {
                index = dist(generator_);
            } while (used_indices[index]);
            
            used_indices[index] = true;
            frequency_indices_.push_back(index);
        }
        
        // Sort indices for deterministic behavior
        std::sort(frequency_indices_.begin(), frequency_indices_.end());
    }
    
    // Internal watermark embedding implementation
    void embed_watermark_internal(std::vector<float>& audio, const std::string& payload) {
        // This is a placeholder implementation
        // In a real implementation, this would use STFT or another transform
        // to embed the watermark in the frequency domain
        
        // For now, we'll just add a simple amplitude modulation as a placeholder
        if (audio.empty()) return;
        
        // Convert payload to bit sequence
        std::vector<bool> bits;
        for (char c : payload) {
            for (int i = 0; i < 8; i++) {
                bits.push_back((c >> i) & 1);
            }
        }
        
        // Ensure we have at least one bit
        if (bits.empty()) {
            bits.push_back(1); // Default bit if payload is empty
        }
        
        // Apply simple amplitude modulation
        const float mod_freq = 0.5f; // Cycles per second
        const float mod_depth = strength_ * 0.02f; // Modulation depth
        
        for (size_t i = 0; i < audio.size(); i++) {
            // Which bit to use
            size_t bit_idx = (i / static_cast<size_t>(sample_rate_ / 8)) % bits.size();
            
            // Apply modulation only if bit is 1
            if (bits[bit_idx]) {
                float t = static_cast<float>(i) / sample_rate_;
                float mod = mod_depth * std::sin(2.0f * M_PI * mod_freq * t);
                audio[i] += audio[i] * mod;
            }
        }
    }
    
    // Internal watermark detection implementation
    bool detect_watermark_internal(const std::vector<float>& audio, 
                                   std::string& detected_payload, 
                                   float& confidence) {
        // This is a placeholder implementation
        // In a real implementation, this would use STFT or another transform
        // to detect the watermark in the frequency domain
        
        if (audio.empty()) {
            detected_payload = "";
            confidence = 0.0f;
            return false;
        }
        
        // For the mock implementation, we'll just pretend to detect the original key
        // with high confidence if the audio is watermarked
        
        // Simplified detection - check for presence of modulation
        const float mod_freq = 0.5f; // Cycles per second
        const float detection_threshold = 0.001f;
        float correlation = 0.0f;
        
        // Simple correlation detection
        for (size_t i = 0; i < audio.size(); i++) {
            float t = static_cast<float>(i) / sample_rate_;
            float reference = std::sin(2.0f * M_PI * mod_freq * t);
            correlation += audio[i] * reference;
        }
        
        correlation /= audio.size();
        
        // Set detected payload and confidence
        detected_payload = key_;
        confidence = std::clamp(std::abs(correlation) / detection_threshold, 0.0f, 1.0f);
        
        return confidence > 0.5f;
    }
};

// Factory method for SilentCipherWatermarker
std::shared_ptr<SilentCipherWatermarker> SilentCipherWatermarker::create(const std::string& key) {
    return std::make_shared<SilentCipherWatermarkerImpl>(key);
}

} // namespace ccsm