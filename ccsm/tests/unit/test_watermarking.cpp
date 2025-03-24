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
    }
    
    std::shared_ptr<SilentCipherWatermarker> silent_cipher;
    std::vector<float> test_audio;
    std::vector<float> sine_wave;
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

// Test SilentCipher watermarking
TEST_F(SilentCipherWatermarkingTest, Watermarking) {
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

// Test SilentCipher advanced detection
TEST_F(SilentCipherWatermarkingTest, AdvancedDetection) {
    // Apply watermark
    std::vector<float> watermarked = silent_cipher->apply_watermark(sine_wave);
    
    // Advanced detection with full result
    WatermarkResult result = silent_cipher->detect(watermarked, sample_rate);
    
    // Check result
    EXPECT_TRUE(result.detected);
    EXPECT_FALSE(result.payload.empty());
    EXPECT_GT(result.confidence, 0.5f);
}

// Test SilentCipher with different audio types
TEST_F(SilentCipherWatermarkingTest, DifferentAudioTypes) {
    // Test with silence
    std::vector<float> silence(audio_length, 0.0f);
    std::vector<float> watermarked_silence = silent_cipher->apply_watermark(silence);
    bool detected_silence = silent_cipher->detect_watermark(watermarked_silence);
    
    // Test with different sine waves
    for (float freq : {100.0f, 1000.0f, 5000.0f}) {
        std::vector<float> sine = generate_sine_wave(audio_length, freq, sample_rate);
        std::vector<float> watermarked_sine = silent_cipher->apply_watermark(sine);
        bool detected_sine = silent_cipher->detect_watermark(watermarked_sine);
        EXPECT_TRUE(detected_sine);
    }
}

// Test SilentCipher robustness
TEST_F(SilentCipherWatermarkingTest, Robustness) {
    // Apply watermark with higher strength for robustness tests
    silent_cipher->set_strength(0.3f);
    std::vector<float> watermarked = silent_cipher->apply_watermark(sine_wave);
    
    // Volume change
    std::vector<float> volume_changed = watermarked;
    float volume_scale = 1.5f;
    for (float& sample : volume_changed) {
        sample *= volume_scale;
    }
    bool detected_volume = silent_cipher->detect_watermark(volume_changed);
    EXPECT_TRUE(detected_volume);
    
    // Noise addition
    std::vector<float> noisy = watermarked;
    std::vector<float> noise = generate_random_audio(audio_length, 0.05f);
    for (size_t i = 0; i < noisy.size(); i++) {
        noisy[i] += noise[i];
    }
    bool detected_noisy = silent_cipher->detect_watermark(noisy);
    EXPECT_TRUE(detected_noisy);
}

// Main function is provided by Google Test