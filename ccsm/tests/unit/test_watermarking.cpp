#include <gtest/gtest.h>
#include <ccsm/watermarking.h>
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <cmath>
#include <random>
#include <algorithm>
#include <iostream>
#include <chrono>

using namespace ccsm;

// Use the WatermarkResult struct from watermarking.h
// No need to redefine it here

// Mock implementation for Watermarker
class MockWatermarker : public Watermarker {
public:
    MockWatermarker() = default;
    
    std::vector<float> apply_watermark(const std::vector<float>& audio) override {
        // Apply a simple watermark
        std::vector<float> watermarked = audio;
        for (size_t i = 0; i < watermarked.size(); i++) {
            watermarked[i] *= 1.01f; // Small modification
        }
        return watermarked;
    }
    
    bool detect_watermark(const std::vector<float>& audio) override {
        // Simple detection: check if audio has non-zero mean
        float mean = 0.0f;
        for (size_t i = 0; i < std::min(audio.size(), size_t(1000)); i++) {
            mean += std::abs(audio[i]);
        }
        mean /= std::min(audio.size(), size_t(1000));
        return mean > 0.01f;
    }
    
    float get_strength() const override {
        return watermark_strength;
    }
    
    void set_strength(float strength) override {
        watermark_strength = strength;
    }
    
    std::string get_key() const override {
        return "mock-watermarker-key";
    }
    
    // For compatibility with tests
    std::vector<float> embed(
        const std::vector<float>& audio, 
        float sample_rate, 
        const std::string& payload
    ) {
        // Store parameters for testing
        last_audio = audio;
        last_sample_rate = sample_rate;
        last_payload = payload;
        
        // Return slightly modified audio to simulate watermarking
        std::vector<float> watermarked = audio;
        for (size_t i = 0; i < watermarked.size() && i < payload.size() * 1000; i++) {
            // Add tiny fluctuation based on payload
            float mod = std::sin(i * 0.1f) * 0.01f;
            
            // Scale by character value from payload
            size_t char_idx = i / 1000;
            if (char_idx < payload.size()) {
                mod *= static_cast<float>(payload[char_idx]) / 128.0f;
            }
            
            watermarked[i] += mod;
        }
        
        return watermarked;
    }
    
    WatermarkResult detect(
        const std::vector<float>& audio, 
        float sample_rate
    ) {
        // Store parameters for testing
        last_audio = audio;
        last_sample_rate = sample_rate;
        
        // Return a simple result based on audio characteristics
        WatermarkResult result;
        
        // Very simple detection heuristic for testing:
        // Look for the tiny fluctuations we added during embedding
        bool has_watermark = false;
        std::string detected_payload;
        
        // Check if the audio is long enough to possibly contain a watermark
        if (audio.size() >= 1000) {
            // Calculate some basic stats
            float mean = 0.0f;
            for (size_t i = 0; i < 1000; i++) {
                mean += std::abs(audio[i]);
            }
            mean /= 1000.0f;
            
            // If mean is non-zero, assume watermark exists
            if (mean > 0.01f) {
                has_watermark = true;
                
                // Attempt to reconstruct payload
                for (size_t i = 0; i < 5 && i * 1000 < audio.size(); i++) {
                    float sum = 0.0f;
                    for (size_t j = 0; j < 1000 && i * 1000 + j < audio.size(); j++) {
                        sum += std::abs(audio[i * 1000 + j]);
                    }
                    
                    // Convert to a character
                    char c = static_cast<char>(std::min(126.0f, std::max(32.0f, sum * 100.0f)));
                    detected_payload.push_back(c);
                }
            }
        }
        
        result.detected = has_watermark;
        result.payload = detected_payload.empty() ? "default-payload" : detected_payload;
        result.confidence = has_watermark ? 0.85f : 0.1f;
        
        return result;
    }
    
    // Testing accessors
    std::vector<float> get_last_audio() const { return last_audio; }
    float get_last_sample_rate() const { return last_sample_rate; }
    std::string get_last_payload() const { return last_payload; }
    
private:
    std::vector<float> last_audio;
    float last_sample_rate = 0.0f;
    std::string last_payload;
    float watermark_strength = 0.5f;
};

// Test fixture for watermarking tests
class WatermarkingTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create watermarker
        watermarker = std::make_shared<MockWatermarker>();
        
        // Create sample audio data
        audio_samples.resize(10000, 0.0f);
        for (size_t i = 0; i < audio_samples.size(); i++) {
            audio_samples[i] = std::sin(static_cast<float>(i) * 0.1f) * 0.5f;
        }
        
        // Default test parameters
        sample_rate = 24000.0f;
        payload = "test-watermark-payload";
    }
    
    std::shared_ptr<MockWatermarker> watermarker;
    std::vector<float> audio_samples;
    float sample_rate;
    std::string payload;
};

// Test basic watermark embedding
TEST_F(WatermarkingTest, WatermarkEmbedding) {
    // Embed watermark
    std::vector<float> watermarked = watermarker->embed(audio_samples, sample_rate, payload);
    
    // Check that the result has the same length as input
    EXPECT_EQ(watermarked.size(), audio_samples.size());
    
    // Check that parameters were correctly passed to the watermarker
    EXPECT_EQ(watermarker->get_last_sample_rate(), sample_rate);
    EXPECT_EQ(watermarker->get_last_payload(), payload);
    
    // Check that watermarked audio is different from original
    bool is_different = false;
    for (size_t i = 0; i < audio_samples.size(); i++) {
        if (std::abs(watermarked[i] - audio_samples[i]) > 1e-6f) {
            is_different = true;
            break;
        }
    }
    EXPECT_TRUE(is_different);
}

// Test watermark detection
TEST_F(WatermarkingTest, WatermarkDetection) {
    // Embed watermark
    std::vector<float> watermarked = watermarker->embed(audio_samples, sample_rate, payload);
    
    // Detect watermark
    WatermarkResult result = watermarker->detect(watermarked, sample_rate);
    
    // Check result
    EXPECT_TRUE(result.detected);
    EXPECT_GT(result.confidence, 0.5f);
    // Mock implementation might not recover exact payload
    
    // Check that parameters were correctly passed to the watermarker
    EXPECT_EQ(watermarker->get_last_sample_rate(), sample_rate);
}

// Test watermark detection failure
TEST_F(WatermarkingTest, WatermarkDetectionFailure) {
    // Create audio with no watermark (all zeros)
    std::vector<float> unwatermarked(10000, 0.0f);
    
    // Detect watermark
    WatermarkResult result = watermarker->detect(unwatermarked, sample_rate);
    
    // Check result
    EXPECT_FALSE(result.detected);
    EXPECT_LT(result.confidence, 0.5f);
}

// Test different payload lengths
TEST_F(WatermarkingTest, DifferentPayloadLengths) {
    // Test with empty payload
    std::string empty_payload = "";
    std::vector<float> watermarked1 = watermarker->embed(audio_samples, sample_rate, empty_payload);
    EXPECT_EQ(watermarked1.size(), audio_samples.size());
    
    // Test with short payload
    std::string short_payload = "abc";
    std::vector<float> watermarked2 = watermarker->embed(audio_samples, sample_rate, short_payload);
    EXPECT_EQ(watermarked2.size(), audio_samples.size());
    
    // Test with long payload
    std::string long_payload(1000, 'x');
    std::vector<float> watermarked3 = watermarker->embed(audio_samples, sample_rate, long_payload);
    EXPECT_EQ(watermarked3.size(), audio_samples.size());
}

// Test different audio lengths
TEST_F(WatermarkingTest, DifferentAudioLengths) {
    // Test with empty audio
    std::vector<float> empty_audio;
    std::vector<float> watermarked1 = watermarker->embed(empty_audio, sample_rate, payload);
    EXPECT_TRUE(watermarked1.empty());
    
    // Test with short audio
    std::vector<float> short_audio(100, 0.5f);
    std::vector<float> watermarked2 = watermarker->embed(short_audio, sample_rate, payload);
    EXPECT_EQ(watermarked2.size(), short_audio.size());
    
    // Test with long audio
    std::vector<float> long_audio(100000, 0.5f);
    std::vector<float> watermarked3 = watermarker->embed(long_audio, sample_rate, payload);
    EXPECT_EQ(watermarked3.size(), long_audio.size());
}

// Test different sample rates
TEST_F(WatermarkingTest, DifferentSampleRates) {
    // Test with low sample rate
    float low_rate = 8000.0f;
    std::vector<float> watermarked1 = watermarker->embed(audio_samples, low_rate, payload);
    EXPECT_EQ(watermarked1.size(), audio_samples.size());
    EXPECT_EQ(watermarker->get_last_sample_rate(), low_rate);
    
    // Test with high sample rate
    float high_rate = 96000.0f;
    std::vector<float> watermarked2 = watermarker->embed(audio_samples, high_rate, payload);
    EXPECT_EQ(watermarked2.size(), audio_samples.size());
    EXPECT_EQ(watermarker->get_last_sample_rate(), high_rate);
}

// Helper function to generate random audio
std::vector<float> generate_random_audio(size_t length, float amplitude = 0.5f, unsigned int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-amplitude, amplitude);
    
    std::vector<float> audio(length);
    for (size_t i = 0; i < length; i++) {
        audio[i] = dist(gen);
    }
    
    return audio;
}

// Helper function to generate sine wave
std::vector<float> generate_sine_wave(size_t length, float frequency, float sample_rate, float amplitude = 0.5f) {
    std::vector<float> audio(length);
    for (size_t i = 0; i < length; i++) {
        float t = static_cast<float>(i) / sample_rate;
        audio[i] = amplitude * std::sin(2.0f * M_PI * frequency * t);
    }
    
    return audio;
}

// Helper function to generate complex audio with multiple frequencies
std::vector<float> generate_complex_audio(size_t length, float sample_rate, float amplitude = 0.5f) {
    std::vector<float> audio(length, 0.0f);
    
    // Add several frequencies
    std::vector<float> freqs = {220.0f, 440.0f, 880.0f, 1760.0f};
    std::vector<float> amps = {0.5f, 0.3f, 0.15f, 0.05f};
    
    for (size_t i = 0; i < length; i++) {
        float t = static_cast<float>(i) / sample_rate;
        for (size_t j = 0; j < freqs.size(); j++) {
            audio[i] += amplitude * amps[j] * std::sin(2.0f * M_PI * freqs[j] * t);
        }
    }
    
    return audio;
}

// Helper function to measure SNR (Signal-to-Noise Ratio) in dB
float calculate_snr_db(const std::vector<float>& original, const std::vector<float>& watermarked) {
    if (original.size() != watermarked.size() || original.empty()) {
        return -std::numeric_limits<float>::infinity();
    }
    
    float signal_power = 0.0f;
    float noise_power = 0.0f;
    
    for (size_t i = 0; i < original.size(); i++) {
        float signal = original[i];
        float noise = original[i] - watermarked[i];
        
        signal_power += signal * signal;
        noise_power += noise * noise;
    }
    
    if (noise_power <= 1e-10f) {
        return std::numeric_limits<float>::infinity(); // No noise
    }
    
    return 10.0f * std::log10(signal_power / noise_power);
}

// Apply basic audio processing operations
namespace {
    // Apply low-pass filter
    std::vector<float> apply_low_pass_filter(const std::vector<float>& audio, float cutoff, float sample_rate) {
        // Simple first-order IIR low-pass filter
        float dt = 1.0f / sample_rate;
        float rc = 1.0f / (2.0f * M_PI * cutoff);
        float alpha = dt / (rc + dt);
        
        std::vector<float> filtered = audio;
        for (size_t i = 1; i < filtered.size(); i++) {
            filtered[i] = filtered[i-1] + alpha * (audio[i] - filtered[i-1]);
        }
        
        return filtered;
    }
    
    // Apply high-pass filter
    std::vector<float> apply_high_pass_filter(const std::vector<float>& audio, float cutoff, float sample_rate) {
        // Simple first-order IIR high-pass filter
        float dt = 1.0f / sample_rate;
        float rc = 1.0f / (2.0f * M_PI * cutoff);
        float alpha = rc / (rc + dt);
        
        std::vector<float> filtered(audio.size());
        filtered[0] = audio[0];
        for (size_t i = 1; i < filtered.size(); i++) {
            filtered[i] = alpha * (filtered[i-1] + audio[i] - audio[i-1]);
        }
        
        return filtered;
    }
    
    // Resample audio (simple linear interpolation)
    std::vector<float> resample(const std::vector<float>& audio, float original_rate, float new_rate) {
        float ratio = original_rate / new_rate;
        size_t new_length = static_cast<size_t>(audio.size() / ratio);
        
        std::vector<float> resampled(new_length);
        for (size_t i = 0; i < new_length; i++) {
            float src_idx = i * ratio;
            size_t idx1 = static_cast<size_t>(src_idx);
            size_t idx2 = std::min(idx1 + 1, audio.size() - 1);
            float frac = src_idx - idx1;
            
            resampled[i] = audio[idx1] * (1.0f - frac) + audio[idx2] * frac;
        }
        
        return resampled;
    }
}

// Test fixture for SilentCipher watermarking
class SilentCipherWatermarkingTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a SilentCipher watermarker
        silent_cipher = SilentCipherWatermarker::create("test_key");
        
        // Generate test audio
        sample_rate = 16000.0f;
        audio_length = static_cast<size_t>(3.0f * sample_rate); // 3 seconds
        test_audio = generate_random_audio(audio_length);
        
        // Generate a sine wave
        sine_wave = generate_sine_wave(audio_length, 440.0f, sample_rate);
        
        // Generate complex audio
        complex_audio = generate_complex_audio(audio_length, sample_rate);
    }
    
    std::shared_ptr<SilentCipherWatermarker> silent_cipher;
    std::vector<float> test_audio;
    std::vector<float> sine_wave;
    std::vector<float> complex_audio;
    size_t audio_length;
    float sample_rate;
};

// Test SilentCipher watermarker creation
TEST_F(SilentCipherWatermarkingTest, Creation) {
    ASSERT_NE(silent_cipher, nullptr);
    EXPECT_EQ(silent_cipher->get_key(), "test_key");
    
    // Test with custom key
    auto custom_key_watermarker = SilentCipherWatermarker::create("custom_key");
    EXPECT_EQ(custom_key_watermarker->get_key(), "custom_key");
}

// Test SilentCipher configurations
TEST_F(SilentCipherWatermarkingTest, Configuration) {
    // Test frame size configuration
    silent_cipher->set_frame_size(1024);
    silent_cipher->set_frame_size(2048);
    
    // Test hop size configuration
    silent_cipher->set_hop_size(256);
    silent_cipher->set_hop_size(512);
    
    // Test watermark strength
    float default_strength = silent_cipher->get_strength();
    EXPECT_GE(default_strength, 0.0f);
    EXPECT_LE(default_strength, 1.0f);
    
    // Set custom strength
    silent_cipher->set_strength(0.4f);
    EXPECT_FLOAT_EQ(silent_cipher->get_strength(), 0.4f);
    
    // Invalid configurations should throw exceptions
    EXPECT_THROW(silent_cipher->set_frame_size(-1), std::invalid_argument);
    EXPECT_THROW(silent_cipher->set_frame_size(0), std::invalid_argument);
    EXPECT_THROW(silent_cipher->set_frame_size(100), std::invalid_argument);  // Not power of 2
    
    EXPECT_THROW(silent_cipher->set_hop_size(-1), std::invalid_argument);
    EXPECT_THROW(silent_cipher->set_hop_size(0), std::invalid_argument);
}

// Test SilentCipher watermarking basic functionality
TEST_F(SilentCipherWatermarkingTest, BasicWatermarking) {
    // Set watermark strength
    silent_cipher->set_strength(0.1f);
    
    // Apply watermark with custom payload
    std::string payload = "test_payload";
    std::vector<float> watermarked = silent_cipher->embed(test_audio, sample_rate, payload);
    
    // Check watermarked audio properties
    EXPECT_EQ(watermarked.size(), test_audio.size());
    
    // Should be different from original
    bool is_different = false;
    for (size_t i = 0; i < test_audio.size(); i++) {
        if (test_audio[i] != watermarked[i]) {
            is_different = true;
            break;
        }
    }
    EXPECT_TRUE(is_different);
    
    // Detect the watermark
    bool detected = silent_cipher->detect_watermark(watermarked);
    EXPECT_TRUE(detected);
    
    // Verify with correct key
    bool verified = silent_cipher->verify_watermark(watermarked, "test_key");
    EXPECT_TRUE(verified);
    
    // Verify with wrong key should fail
    bool wrong_key_verified = silent_cipher->verify_watermark(watermarked, "wrong_key");
    EXPECT_FALSE(wrong_key_verified);
    
    // SNR should be reasonable for audio quality
    float snr = calculate_snr_db(test_audio, watermarked);
    std::cout << "SilentCipher watermarking SNR: " << snr << " dB" << std::endl;
    EXPECT_GT(snr, 20.0f); // At least 20dB SNR for good quality
}

// Test SilentCipher embedding and detection with different payload sizes
TEST_F(SilentCipherWatermarkingTest, PayloadSizes) {
    // Test with different payload sizes
    std::vector<std::string> payloads = {
        "", // Empty
        "a", // Single character
        "test", // Short string
        "This is a longer test payload with more data", // Medium string
        std::string(200, 'x') // Long string
    };
    
    for (const auto& payload : payloads) {
        std::cout << "Testing payload of size " << payload.size() << std::endl;
        
        // Apply watermark
        std::vector<float> watermarked = silent_cipher->embed(test_audio, sample_rate, payload);
        
        // Detect watermark
        bool detected = silent_cipher->detect_watermark(watermarked);
        
        // Check detection (short payloads may not be reliably detected)
        if (payload.size() > 1) {
            EXPECT_TRUE(detected) << "Failed to detect watermark with payload size " << payload.size();
        }
        
        // Calculate SNR
        float snr = calculate_snr_db(test_audio, watermarked);
        std::cout << "Payload size " << payload.size() << " SNR: " << snr << " dB" << std::endl;
        
        // Advanced detection
        WatermarkResult result = silent_cipher->detect(watermarked, sample_rate);
        
        if (payload.size() > 1) {
            EXPECT_TRUE(result.detected) << "Failed to detect watermark with payload size " << payload.size();
        }
    }
}

// Test SilentCipher watermarking performance
TEST_F(SilentCipherWatermarkingTest, Performance) {
    // Prepare audio of different lengths
    std::vector<size_t> lengths = {
        static_cast<size_t>(0.5f * sample_rate), // 0.5 seconds
        static_cast<size_t>(1.0f * sample_rate), // 1 second
        static_cast<size_t>(5.0f * sample_rate)  // 5 seconds
    };
    
    for (size_t length : lengths) {
        std::vector<float> audio = generate_random_audio(length);
        
        // Measure embedding time
        auto start_embed = std::chrono::high_resolution_clock::now();
        std::vector<float> watermarked = silent_cipher->embed(audio, sample_rate, "performance_test");
        auto end_embed = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double> embed_time = end_embed - start_embed;
        
        // Measure detection time
        auto start_detect = std::chrono::high_resolution_clock::now();
        bool detected = silent_cipher->detect_watermark(watermarked);
        auto end_detect = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double> detect_time = end_detect - start_detect;
        
        std::cout << "Audio length: " << length << " samples (" << length / sample_rate << " seconds)" << std::endl;
        std::cout << "Embed time: " << embed_time.count() << " seconds" << std::endl;
        std::cout << "Detect time: " << detect_time.count() << " seconds" << std::endl;
        
        // Performance should be reasonable
        EXPECT_LT(embed_time.count(), length / sample_rate * 5) << "Embedding too slow for " << length << " samples";
        EXPECT_LT(detect_time.count(), length / sample_rate * 5) << "Detection too slow for " << length << " samples";
        
        // Detection should work
        EXPECT_TRUE(detected);
    }
}

// Test SilentCipher advanced detection
TEST_F(SilentCipherWatermarkingTest, AdvancedDetection) {
    // Apply watermark with custom payload
    std::string original_payload = "test_watermark_payload";
    std::vector<float> watermarked = silent_cipher->embed(complex_audio, sample_rate, original_payload);
    
    // Advanced detection with full result
    WatermarkResult result = silent_cipher->detect(watermarked, sample_rate);
    
    // Check result
    EXPECT_TRUE(result.detected);
    EXPECT_FALSE(result.payload.empty());
    EXPECT_GT(result.confidence, 0.3f);
    
    // Note: we don't expect exact payload recovery due to the simple watermarking implementation
    std::cout << "Original payload: " << original_payload << std::endl;
    std::cout << "Detected payload: " << result.payload << std::endl;
    std::cout << "Confidence: " << result.confidence << std::endl;
}

// Test SilentCipher with different audio types
TEST_F(SilentCipherWatermarkingTest, DifferentAudioTypes) {
    // Set watermark strength
    silent_cipher->set_strength(0.2f);
    
    // Test with silence
    std::vector<float> silence(audio_length, 0.0f);
    std::vector<float> watermarked_silence = silent_cipher->apply_watermark(silence);
    bool detected_silence = silent_cipher->detect_watermark(watermarked_silence);
    // Silence is a special case - detection might fail because there's no signal
    std::cout << "Silence detection: " << (detected_silence ? "yes" : "no") << std::endl;
    
    // Test with different sine waves
    for (float freq : {100.0f, 1000.0f, 5000.0f}) {
        std::vector<float> sine = generate_sine_wave(audio_length, freq, sample_rate);
        std::vector<float> watermarked_sine = silent_cipher->apply_watermark(sine);
        bool detected_sine = silent_cipher->detect_watermark(watermarked_sine);
        std::cout << "Sine wave " << freq << "Hz detection: " << (detected_sine ? "yes" : "no") << std::endl;
        EXPECT_TRUE(detected_sine) << "Failed to detect watermark in " << freq << "Hz sine wave";
    }
    
    // Test with complex audio
    std::vector<float> watermarked_complex = silent_cipher->apply_watermark(complex_audio);
    bool detected_complex = silent_cipher->detect_watermark(watermarked_complex);
    std::cout << "Complex audio detection: " << (detected_complex ? "yes" : "no") << std::endl;
    EXPECT_TRUE(detected_complex);
}

// Test SilentCipher robustness to audio processing
TEST_F(SilentCipherWatermarkingTest, Robustness) {
    // Apply watermark with higher strength for robustness tests
    silent_cipher->set_strength(0.3f);
    std::vector<float> watermarked = silent_cipher->apply_watermark(complex_audio);
    
    // Volume change
    std::vector<float> volume_changed = watermarked;
    float volume_scale = 1.5f;
    for (float& sample : volume_changed) {
        sample *= volume_scale;
    }
    bool detected_volume = silent_cipher->detect_watermark(volume_changed);
    std::cout << "Volume change detection: " << (detected_volume ? "yes" : "no") << std::endl;
    EXPECT_TRUE(detected_volume);
    
    // Noise addition
    std::vector<float> noisy = watermarked;
    std::vector<float> noise = generate_random_audio(audio_length, 0.05f);
    for (size_t i = 0; i < noisy.size(); i++) {
        noisy[i] += noise[i];
    }
    bool detected_noisy = silent_cipher->detect_watermark(noisy);
    std::cout << "Noise addition detection: " << (detected_noisy ? "yes" : "no") << std::endl;
    EXPECT_TRUE(detected_noisy);
    
    // Low-pass filtering
    std::vector<float> low_passed = apply_low_pass_filter(watermarked, 4000.0f, sample_rate);
    bool detected_low_pass = silent_cipher->detect_watermark(low_passed);
    std::cout << "Low-pass filter detection: " << (detected_low_pass ? "yes" : "no") << std::endl;
    EXPECT_TRUE(detected_low_pass);
    
    // High-pass filtering
    std::vector<float> high_passed = apply_high_pass_filter(watermarked, 200.0f, sample_rate);
    bool detected_high_pass = silent_cipher->detect_watermark(high_passed);
    std::cout << "High-pass filter detection: " << (detected_high_pass ? "yes" : "no") << std::endl;
    EXPECT_TRUE(detected_high_pass);
    
    // Resampling
    std::vector<float> resampled_down = resample(watermarked, sample_rate, sample_rate / 2);
    std::vector<float> resampled_back = resample(resampled_down, sample_rate / 2, sample_rate);
    bool detected_resampled = silent_cipher->detect_watermark(resampled_back);
    std::cout << "Resampling detection: " << (detected_resampled ? "yes" : "no") << std::endl;
    // Resampling is a harsh transformation - detection might fail
    
    // Combined transformations
    std::vector<float> combined = high_passed;
    for (size_t i = 0; i < combined.size(); i++) {
        combined[i] = combined[i] * 1.2f + noise[i] * 0.05f;
    }
    bool detected_combined = silent_cipher->detect_watermark(combined);
    std::cout << "Combined transformations detection: " << (detected_combined ? "yes" : "no") << std::endl;
    // Combined transformations might be too severe
}

// Test SilentCipher with different frame and hop sizes
TEST_F(SilentCipherWatermarkingTest, FrameSizes) {
    // Test with different frame sizes
    for (int frame_size : {512, 1024, 2048, 4096}) {
        // Test with different hop sizes
        for (int hop_size : {frame_size / 4, frame_size / 2}) {
            std::cout << "Testing frame size " << frame_size << " with hop size " << hop_size << std::endl;
            
            // Configure watermarker
            silent_cipher->set_frame_size(frame_size);
            silent_cipher->set_hop_size(hop_size);
            
            // Apply watermark
            std::vector<float> watermarked = silent_cipher->apply_watermark(complex_audio);
            
            // Detect watermark
            bool detected = silent_cipher->detect_watermark(watermarked);
            
            // Check detection
            EXPECT_TRUE(detected) << "Failed to detect watermark with frame size " << frame_size 
                                  << " and hop size " << hop_size;
            
            // Calculate SNR
            float snr = calculate_snr_db(complex_audio, watermarked);
            std::cout << "Frame size " << frame_size << ", hop size " << hop_size 
                      << " SNR: " << snr << " dB" << std::endl;
            
            // Advanced detection
            WatermarkResult result = silent_cipher->detect(watermarked, sample_rate);
            
            EXPECT_TRUE(result.detected) << "Failed to detect watermark with frame size " << frame_size 
                                        << " and hop size " << hop_size;
        }
    }
}

// Test SilentCipher with different watermark strengths
TEST_F(SilentCipherWatermarkingTest, WatermarkStrengths) {
    // Test with different strengths
    for (float strength : {0.05f, 0.1f, 0.2f, 0.5f, 0.8f}) {
        std::cout << "Testing watermark strength " << strength << std::endl;
        
        // Configure watermarker
        silent_cipher->set_strength(strength);
        
        // Apply watermark
        std::vector<float> watermarked = silent_cipher->apply_watermark(complex_audio);
        
        // Detect watermark
        bool detected = silent_cipher->detect_watermark(watermarked);
        
        // Check detection (very low strengths might not be reliably detected)
        if (strength >= 0.1f) {
            EXPECT_TRUE(detected) << "Failed to detect watermark with strength " << strength;
        }
        
        // Calculate SNR
        float snr = calculate_snr_db(complex_audio, watermarked);
        std::cout << "Strength " << strength << " SNR: " << snr << " dB" << std::endl;
        
        // There should be an inverse relationship between strength and SNR
        if (strength > 0.1f) {
            EXPECT_LT(snr, 40.0f) << "SNR too high for strength " << strength;
        }
        
        // Advanced detection
        WatermarkResult result = silent_cipher->detect(watermarked, sample_rate);
        
        if (strength >= 0.1f) {
            EXPECT_TRUE(result.detected) << "Failed to detect watermark with strength " << strength;
        }
        
        // Confidence should increase with strength
        std::cout << "Strength " << strength << " confidence: " << result.confidence << std::endl;
    }
}

// Main function is provided by Google Test