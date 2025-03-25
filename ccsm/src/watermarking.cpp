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
#include <complex>
#include <bitset>

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
    // Return a SilentCipherWatermarker instance
    return SilentCipherWatermarker::create(key);
}

// Utility functions for FFT
// Simple implementation of FFT (not optimized, but sufficient for watermarking)
// A production implementation would use a more optimized library
namespace {

using Complex = std::complex<float>;

// Complex FFT implementation (radix-2, Cooley-Tukey algorithm)
void fft(std::vector<Complex>& x) {
    const size_t N = x.size();
    if (N <= 1) return;
    
    // Split into even and odd
    std::vector<Complex> even(N/2), odd(N/2);
    for (size_t i = 0; i < N/2; i++) {
        even[i] = x[2*i];
        odd[i] = x[2*i + 1];
    }
    
    // Recursive FFT on even and odd
    fft(even);
    fft(odd);
    
    // Combine
    for (size_t k = 0; k < N/2; k++) {
        float angle = -2.0f * M_PI * k / static_cast<float>(N);
        Complex t = Complex(std::cos(angle), std::sin(angle)) * odd[k];
        x[k] = even[k] + t;
        x[k + N/2] = even[k] - t;
    }
}

// Inverse FFT implementation
void ifft(std::vector<Complex>& x) {
    // Conjugate the complex numbers
    for (auto& i : x) {
        i = std::conj(i);
    }
    
    // Forward FFT
    fft(x);
    
    // Conjugate again and scale
    for (auto& i : x) {
        i = std::conj(i) / static_cast<float>(x.size());
    }
}

// Compute the magnitude spectrum
std::vector<float> magnitude_spectrum(const std::vector<Complex>& spectrum) {
    std::vector<float> result(spectrum.size());
    for (size_t i = 0; i < spectrum.size(); i++) {
        result[i] = std::abs(spectrum[i]);
    }
    return result;
}

// Compute the phase spectrum
std::vector<float> phase_spectrum(const std::vector<Complex>& spectrum) {
    std::vector<float> result(spectrum.size());
    for (size_t i = 0; i < spectrum.size(); i++) {
        result[i] = std::arg(spectrum[i]);
    }
    return result;
}

// Apply Hann window to reduce spectral leakage
void apply_hann_window(std::vector<float>& frame) {
    const size_t N = frame.size();
    for (size_t i = 0; i < N; i++) {
        float window = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (N - 1)));
        frame[i] *= window;
    }
}

// Split a string into a bit array
std::vector<bool> string_to_bits(const std::string& str) {
    std::vector<bool> bits;
    for (char c : str) {
        std::bitset<8> b(c);
        for (int i = 0; i < 8; i++) {
            bits.push_back(b[i]);
        }
    }
    return bits;
}

// Convert bit array back to string
std::string bits_to_string(const std::vector<bool>& bits) {
    std::string result;
    for (size_t i = 0; i < bits.size(); i += 8) {
        std::bitset<8> b;
        for (int j = 0; j < 8 && i + j < bits.size(); j++) {
            b[j] = bits[i + j];
        }
        if (i + 8 <= bits.size()) { // Only add complete bytes
            result.push_back(static_cast<char>(b.to_ulong()));
        }
    }
    return result;
}

// Spread spectrum sequence for watermarking
std::vector<int> generate_spread_sequence(unsigned int seed, int length) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(0, 1);
    
    std::vector<int> sequence(length);
    for (int i = 0; i < length; i++) {
        sequence[i] = dist(gen) ? 1 : -1;
    }
    return sequence;
}

// Apply error correction coding (simple repetition code)
std::vector<bool> apply_error_correction(const std::vector<bool>& bits, int repetition = 3) {
    std::vector<bool> result;
    for (bool bit : bits) {
        for (int i = 0; i < repetition; i++) {
            result.push_back(bit);
        }
    }
    return result;
}

// Decode error-corrected bits
std::vector<bool> decode_error_correction(const std::vector<bool>& coded_bits, int repetition = 3) {
    std::vector<bool> result;
    for (size_t i = 0; i < coded_bits.size(); i += repetition) {
        int count = 0;
        for (int j = 0; j < repetition && i + j < coded_bits.size(); j++) {
            if (coded_bits[i + j]) count++;
        }
        result.push_back(count > repetition / 2);
    }
    return result;
}

} // anonymous namespace

// ----- SilentCipher implementation -----
class SilentCipherWatermarkerImpl : public SilentCipherWatermarker {
public:
    SilentCipherWatermarkerImpl(const std::string& key)
        : key_(key), 
          strength_(0.1f),
          frame_size_(1024),
          hop_size_(256),
          sample_rate_(16000),
          seed_(0),
          error_correction_rate_(3), // Triple redundancy by default
          min_freq_(300.0f),         // 300 Hz minimum
          max_freq_(3500.0f)         // 3500 Hz maximum
    {
        initialize();
    }
    
    void initialize() {
        // Initialize random generator with key-based seed
        seed_ = hash_key(key_);
        generator_.seed(seed_);
        
        // Initialize frequency indices for watermarking
        init_frequency_indices();
        
        // Generate spread spectrum sequence for each frequency bin
        spread_sequences_.clear();
        for (int freq_idx : frequency_indices_) {
            spread_sequences_[freq_idx] = generate_spread_sequence(seed_ + freq_idx, 64);
        }
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
        // Save current key
        std::string original_key = key_;
        unsigned int original_seed = seed_;
        
        // Set up detector with provided key
        key_ = key;
        seed_ = hash_key(key_);
        generator_.seed(seed_);
        init_frequency_indices();
        
        // Detect watermark with the provided key
        std::string detected_payload;
        float confidence;
        bool detected = detect_watermark_internal(audio, detected_payload, confidence);
        
        // Restore original key
        key_ = original_key;
        seed_ = original_seed;
        generator_.seed(seed_);
        init_frequency_indices();
        
        // Check if the watermark was detected with the provided key
        return detected;
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
    int error_correction_rate_;
    float min_freq_;
    float max_freq_;
    
    // Random number generation
    unsigned int seed_;
    std::mt19937 generator_;
    
    // Watermarking frequency indices
    std::vector<int> frequency_indices_;
    
    // Spread spectrum sequences for each frequency bin
    std::unordered_map<int, std::vector<int>> spread_sequences_;
    
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
        int min_freq_bin = static_cast<int>(min_freq_ * frame_size_ / sample_rate_);
        int max_freq_bin = static_cast<int>(max_freq_ * frame_size_ / sample_rate_);
        
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
    
    // Apply watermark to a single frame in the frequency domain
    void watermark_frame(std::vector<Complex>& spectrum, const std::vector<bool>& payload_bits, size_t bit_pos) {
        // Get the frequency magnitude and phase
        std::vector<float> magnitudes = magnitude_spectrum(spectrum);
        std::vector<float> phases = phase_spectrum(spectrum);
        
        // Apply watermark to selected frequency bins
        for (size_t i = 0; i < frequency_indices_.size() && bit_pos < payload_bits.size(); i++) {
            int freq_idx = frequency_indices_[i];
            
            // Get spread sequence for this frequency bin
            const auto& spread_seq = spread_sequences_[freq_idx];
            
            // Get current bit from payload
            bool bit = payload_bits[bit_pos++];
            
            // Modify magnitude based on bit value and spread sequence
            float scale_factor = 1.0f + strength_ * 0.1f * (bit ? 1.0f : -1.0f) * spread_seq[i % spread_seq.size()];
            magnitudes[freq_idx] *= scale_factor;
            
            // Reconstruct complex spectrum
            spectrum[freq_idx] = std::polar(magnitudes[freq_idx], phases[freq_idx]);
            
            // Also modify conjugate for real input
            if (freq_idx != 0 && freq_idx != spectrum.size() / 2) {
                int conj_idx = spectrum.size() - freq_idx;
                spectrum[conj_idx] = std::conj(spectrum[freq_idx]);
            }
        }
    }
    
    // Extract watermark bit from a single frame in the frequency domain
    void extract_frame(const std::vector<Complex>& spectrum, std::vector<float>& bit_strengths) {
        // Get the frequency magnitudes
        std::vector<float> magnitudes = magnitude_spectrum(spectrum);
        
        // Extract watermark from selected frequency bins
        for (size_t i = 0; i < frequency_indices_.size() && i < bit_strengths.size(); i++) {
            int freq_idx = frequency_indices_[i];
            
            // Get spread sequence for this frequency bin
            const auto& spread_seq = spread_sequences_[freq_idx];
            
            // Calculate correlation with expected pattern
            float correlation = magnitudes[freq_idx] * spread_seq[i % spread_seq.size()];
            
            // Accumulate bit strength
            bit_strengths[i] += correlation;
        }
    }
    
    // Apply short-time Fourier transform (STFT) to the audio
    std::vector<std::vector<Complex>> stft(const std::vector<float>& audio) {
        std::vector<std::vector<Complex>> frames;
        
        // Process audio in overlapping frames
        for (size_t start = 0; start < audio.size(); start += hop_size_) {
            // Extract frame
            std::vector<float> frame(frame_size_, 0.0f);
            for (size_t i = 0; i < frame_size_ && start + i < audio.size(); i++) {
                frame[i] = audio[start + i];
            }
            
            // Apply window function
            apply_hann_window(frame);
            
            // Convert to complex numbers for FFT
            std::vector<Complex> complex_frame(frame_size_);
            for (size_t i = 0; i < frame_size_; i++) {
                complex_frame[i] = Complex(frame[i], 0.0f);
            }
            
            // Apply FFT
            fft(complex_frame);
            
            // Add to frames
            frames.push_back(complex_frame);
        }
        
        return frames;
    }
    
    // Apply inverse STFT to reconstruct the audio
    std::vector<float> istft(const std::vector<std::vector<Complex>>& frames) {
        // Calculate the length of the output audio
        size_t output_length = (frames.size() - 1) * hop_size_ + frame_size_;
        
        // Initialize the output audio
        std::vector<float> output(output_length, 0.0f);
        std::vector<float> window_sum(output_length, 0.0f);
        
        // Process each frame
        for (size_t frame_idx = 0; frame_idx < frames.size(); frame_idx++) {
            // Copy the frame
            std::vector<Complex> frame = frames[frame_idx];
            
            // Apply inverse FFT
            ifft(frame);
            
            // Create window function
            std::vector<float> window(frame_size_);
            for (size_t i = 0; i < frame_size_; i++) {
                window[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (frame_size_ - 1)));
            }
            
            // Add to output with overlap-add
            size_t start = frame_idx * hop_size_;
            for (size_t i = 0; i < frame_size_ && start + i < output.size(); i++) {
                output[start + i] += window[i] * frame[i].real();
                window_sum[start + i] += window[i];
            }
        }
        
        // Normalize by window sum to avoid artifacts at frame boundaries
        for (size_t i = 0; i < output.size(); i++) {
            if (window_sum[i] > 0.0f) {
                output[i] /= window_sum[i];
            }
        }
        
        return output;
    }
    
    // Internal watermark embedding implementation using STFT
    void embed_watermark_internal(std::vector<float>& audio, const std::string& payload) {
        if (audio.empty()) return;
        
        // Convert payload to bit sequence
        std::vector<bool> payload_bits = string_to_bits(payload);
        
        // Apply error correction
        std::vector<bool> encoded_bits = apply_error_correction(payload_bits, error_correction_rate_);
        
        // Apply STFT to get frequency domain representation
        std::vector<std::vector<Complex>> frames = stft(audio);
        
        // Apply watermark to each frame
        size_t bit_pos = 0;
        for (auto& frame : frames) {
            if (bit_pos >= encoded_bits.size()) break;
            
            // Apply watermark to this frame
            watermark_frame(frame, encoded_bits, bit_pos);
            
            // Advance bit position
            bit_pos += frequency_indices_.size();
        }
        
        // Apply inverse STFT to get time domain signal
        std::vector<float> watermarked = istft(frames);
        
        // Ensure the output has the same length as the input
        watermarked.resize(audio.size());
        
        // Copy back to the audio buffer
        audio = watermarked;
    }
    
    // Internal watermark detection implementation using STFT
    bool detect_watermark_internal(const std::vector<float>& audio, 
                                   std::string& detected_payload, 
                                   float& confidence) {
        if (audio.empty()) {
            detected_payload = "";
            confidence = 0.0f;
            return false;
        }
        
        // Apply STFT to get frequency domain representation
        std::vector<std::vector<Complex>> frames = stft(audio);
        
        // Calculate number of bits we can extract
        size_t max_bits = frames.size() * frequency_indices_.size();
        
        // Initialize bit strengths
        std::vector<float> bit_strengths(max_bits, 0.0f);
        
        // Extract watermark from each frame
        for (size_t frame_idx = 0; frame_idx < frames.size(); frame_idx++) {
            // Extract watermark bits from this frame
            size_t start_pos = frame_idx * frequency_indices_.size();
            size_t end_pos = std::min(bit_strengths.size(), (frame_idx + 1) * frequency_indices_.size());
            
            if (start_pos < bit_strengths.size()) {
                // Create a subvector for this frame's bit strengths
                std::vector<float> frame_bits(bit_strengths.begin() + start_pos,
                                             bit_strengths.begin() + end_pos);
                
                // Extract bits from this frame
                extract_frame(frames[frame_idx], frame_bits);
                
                // Copy back the results
                for (size_t i = 0; i < frame_bits.size(); i++) {
                    if (start_pos + i < bit_strengths.size()) {
                        bit_strengths[start_pos + i] = frame_bits[i];
                    }
                }
            }
        }
        
        // Convert bit strengths to bits
        std::vector<bool> extracted_bits;
        for (float strength : bit_strengths) {
            extracted_bits.push_back(strength > 0.0f);
        }
        
        // Apply error correction decoding
        std::vector<bool> decoded_bits = decode_error_correction(extracted_bits, error_correction_rate_);
        
        // Convert to string
        detected_payload = bits_to_string(decoded_bits);
        
        // Calculate confidence based on bit strength consistency
        float total_strength = 0.0f;
        float avg_strength = 0.0f;
        for (float strength : bit_strengths) {
            total_strength += std::abs(strength);
        }
        avg_strength = total_strength / bit_strengths.size();
        
        // Calculate confidence (higher average strength = higher confidence)
        confidence = std::clamp(avg_strength / 0.1f, 0.0f, 1.0f);
        
        // Detect if watermark is present (confidence threshold)
        bool detected = confidence > 0.3f;
        
        return detected;
    }
};

// Factory method for SilentCipherWatermarker
std::shared_ptr<SilentCipherWatermarker> SilentCipherWatermarker::create(const std::string& key) {
    return std::make_shared<SilentCipherWatermarkerImpl>(key);
}

} // namespace ccsm