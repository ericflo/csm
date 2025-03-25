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
#include <thread>

using namespace ccsm;

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

// Helper function to calculate SNR (Signal-to-Noise Ratio) in dB
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

// Test fixture for edge cases
class WatermarkingEdgeCasesTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create watermarker
        watermarker = SilentCipherWatermarker::create("test_key");
        
        // Set default parameters
        sample_rate = 16000.0f;
        payload = "edge_case_test";
    }
    
    std::shared_ptr<SilentCipherWatermarker> watermarker;
    float sample_rate;
    std::string payload;
};

// Test watermarking with extremely short audio
TEST_F(WatermarkingEdgeCasesTest, ExtremelyShortAudio) {
    // Test with very short audio segments
    for (size_t length : {10, 20, 50, 100, 200, 500}) {
        std::vector<float> short_audio = generate_random_audio(length, 0.5f);
        
        // Apply watermark
        std::vector<float> watermarked = watermarker->embed(short_audio, sample_rate, payload);
        
        // Check output size matches input size
        EXPECT_EQ(watermarked.size(), short_audio.size());
        
        // Detection might not be reliable for very short segments
        bool detected = watermarker->detect_watermark(watermarked);
        std::cout << "Detection for " << length << " samples: " << (detected ? "yes" : "no") << std::endl;
        
        // For longer segments, detection should be more reliable
        if (length >= 500) {
            EXPECT_TRUE(detected) << "Failed to detect watermark in audio with length " << length;
        }
    }
}

// Test watermarking with different sample rates
TEST_F(WatermarkingEdgeCasesTest, ExtremeSampleRates) {
    // Generate a moderate length audio sample
    size_t audio_length = 16000; // 1 second at 16kHz
    std::vector<float> audio = generate_random_audio(audio_length, 0.5f);
    
    // Test with various sample rates
    for (float rate : {8000.0f, 11025.0f, 16000.0f, 22050.0f, 44100.0f, 48000.0f, 96000.0f}) {
        // Apply watermark
        std::vector<float> watermarked = watermarker->embed(audio, rate, payload);
        
        // Check output size matches input size
        EXPECT_EQ(watermarked.size(), audio.size());
        
        // Verify watermark can be detected
        bool detected = watermarker->detect_watermark(watermarked);
        WatermarkResult result = watermarker->detect(watermarked, rate);
        
        std::cout << "Sample rate " << rate << " Hz, detected: " << (detected ? "yes" : "no") 
                  << ", confidence: " << result.confidence << std::endl;
        
        // Extreme sample rates might cause detection issues
        if (rate >= 16000.0f && rate <= 48000.0f) {
            EXPECT_TRUE(detected) << "Failed to detect watermark at sample rate " << rate;
        }
    }
}

// Test watermarking with DC offset
TEST_F(WatermarkingEdgeCasesTest, DCOffset) {
    // Generate audio with DC offset
    size_t audio_length = 16000; // 1 second
    std::vector<float> audio = generate_random_audio(audio_length, 0.3f);
    
    // Add various DC offsets
    for (float offset : {-0.5f, -0.25f, 0.0f, 0.25f, 0.5f}) {
        std::vector<float> offset_audio = audio;
        for (float& sample : offset_audio) {
            sample += offset;
        }
        
        // Apply watermark
        std::vector<float> watermarked = watermarker->embed(offset_audio, sample_rate, payload);
        
        // Check output size matches input size
        EXPECT_EQ(watermarked.size(), offset_audio.size());
        
        // Verify watermark can be detected
        bool detected = watermarker->detect_watermark(watermarked);
        
        std::cout << "DC offset " << offset << ", detected: " << (detected ? "yes" : "no") << std::endl;
        
        // DC offset should not significantly impact detection for moderate values
        EXPECT_TRUE(detected) << "Failed to detect watermark with DC offset " << offset;
    }
}

// Test watermarking with extreme amplitudes
TEST_F(WatermarkingEdgeCasesTest, ExtremeAmplitudes) {
    // Generate audio with different amplitudes
    size_t audio_length = 16000; // 1 second
    
    for (float amplitude : {0.001f, 0.01f, 0.1f, 0.5f, 0.9f, 0.99f}) {
        std::vector<float> audio = generate_random_audio(audio_length, amplitude);
        
        // Apply watermark
        std::vector<float> watermarked = watermarker->embed(audio, sample_rate, payload);
        
        // Check output size matches input size
        EXPECT_EQ(watermarked.size(), audio.size());
        
        // Verify watermark can be detected
        bool detected = watermarker->detect_watermark(watermarked);
        
        std::cout << "Amplitude " << amplitude << ", detected: " << (detected ? "yes" : "no") << std::endl;
        
        // Very low amplitudes might cause detection issues
        if (amplitude >= 0.01f) {
            EXPECT_TRUE(detected) << "Failed to detect watermark with amplitude " << amplitude;
        }
    }
}

// Test watermarking with empty payload
TEST_F(WatermarkingEdgeCasesTest, EmptyPayload) {
    // Generate audio
    size_t audio_length = 16000; // 1 second
    std::vector<float> audio = generate_random_audio(audio_length, 0.5f);
    
    // Apply watermark with empty payload
    std::string empty_payload = "";
    std::vector<float> watermarked = watermarker->embed(audio, sample_rate, empty_payload);
    
    // Check output size matches input size
    EXPECT_EQ(watermarked.size(), audio.size());
    
    // Verify watermark can be detected
    bool detected = watermarker->detect_watermark(watermarked);
    
    std::cout << "Empty payload, detected: " << (detected ? "yes" : "no") << std::endl;
    
    // Empty payload should still allow detection (using default watermarking)
    EXPECT_TRUE(detected) << "Failed to detect watermark with empty payload";
}

// Test watermarking with extremely long payload
TEST_F(WatermarkingEdgeCasesTest, VeryLongPayload) {
    // Generate audio
    size_t audio_length = 48000; // 3 seconds at 16kHz
    std::vector<float> audio = generate_random_audio(audio_length, 0.5f);
    
    // Create payloads of different lengths
    std::vector<std::string> payloads = {
        std::string(10, 'x'),
        std::string(100, 'x'),
        std::string(500, 'x'),
        std::string(1000, 'x')
    };
    
    for (const auto& test_payload : payloads) {
        // Apply watermark
        std::vector<float> watermarked = watermarker->embed(audio, sample_rate, test_payload);
        
        // Check output size matches input size
        EXPECT_EQ(watermarked.size(), audio.size());
        
        // Verify watermark can be detected
        bool detected = watermarker->detect_watermark(watermarked);
        
        std::cout << "Payload length " << test_payload.size() << ", detected: " << (detected ? "yes" : "no") << std::endl;
        
        // Extremely long payloads might not be fully encoded, but detection should still work
        EXPECT_TRUE(detected) << "Failed to detect watermark with payload length " << test_payload.size();
    }
}

// Test watermarking with binary data in payload
TEST_F(WatermarkingEdgeCasesTest, BinaryPayload) {
    // Generate audio
    size_t audio_length = 16000; // 1 second
    std::vector<float> audio = generate_random_audio(audio_length, 0.5f);
    
    // Create a binary payload with all byte values (0-255)
    std::string binary_payload;
    for (int i = 0; i < 256; i++) {
        binary_payload.push_back(static_cast<char>(i));
    }
    
    // Apply watermark
    std::vector<float> watermarked = watermarker->embed(audio, sample_rate, binary_payload);
    
    // Check output size matches input size
    EXPECT_EQ(watermarked.size(), audio.size());
    
    // Verify watermark can be detected
    bool detected = watermarker->detect_watermark(watermarked);
    
    std::cout << "Binary payload, detected: " << (detected ? "yes" : "no") << std::endl;
    
    // Binary data should be supported
    EXPECT_TRUE(detected) << "Failed to detect watermark with binary payload";
}

// Test with combination of extreme parameters
TEST_F(WatermarkingEdgeCasesTest, CombinationTest) {
    // Use different watermark strengths
    for (float strength : {0.05f, 0.3f, 0.8f}) {
        watermarker->set_strength(strength);
        
        // Use different frame sizes
        for (int frame_size : {512, 4096}) {
            watermarker->set_frame_size(frame_size);
            
            // Use different hop sizes
            for (int hop_size : {frame_size / 4, frame_size / 2}) {
                watermarker->set_hop_size(hop_size);
                
                // Generate audio with different characteristics
                size_t audio_length = 32000; // 2 seconds
                std::vector<float> audio = generate_sine_wave(audio_length, 440.0f, sample_rate, 0.3f);
                
                // Apply watermark
                std::vector<float> watermarked = watermarker->embed(audio, sample_rate, payload);
                
                // Check output size matches input size
                EXPECT_EQ(watermarked.size(), audio.size());
                
                // Verify watermark can be detected
                bool detected = watermarker->detect_watermark(watermarked);
                
                std::cout << "Strength " << strength << ", frame_size " << frame_size 
                          << ", hop_size " << hop_size << ", detected: " << (detected ? "yes" : "no") << std::endl;
                
                // SNR should decrease with increasing strength
                float snr = calculate_snr_db(audio, watermarked);
                std::cout << "SNR: " << snr << " dB" << std::endl;
                
                // For moderate to high strengths, detection should work
                if (strength >= 0.1f) {
                    EXPECT_TRUE(detected) << "Failed to detect watermark with strength " << strength
                                         << ", frame_size " << frame_size << ", hop_size " << hop_size;
                }
            }
        }
    }
}

// Test robustness to clipping
TEST_F(WatermarkingEdgeCasesTest, Clipping) {
    // Generate audio
    size_t audio_length = 16000; // 1 second
    std::vector<float> audio = generate_random_audio(audio_length, 0.5f);
    
    // Apply watermark with high strength
    watermarker->set_strength(0.3f);
    std::vector<float> watermarked = watermarker->embed(audio, sample_rate, payload);
    
    // Apply different levels of clipping
    for (float clip_level : {0.99f, 0.9f, 0.8f, 0.7f, 0.6f, 0.5f}) {
        std::vector<float> clipped = watermarked;
        for (float& sample : clipped) {
            if (sample > clip_level) sample = clip_level;
            if (sample < -clip_level) sample = -clip_level;
        }
        
        // Verify watermark can be detected
        bool detected = watermarker->detect_watermark(clipped);
        
        std::cout << "Clipping level " << clip_level << ", detected: " << (detected ? "yes" : "no") << std::endl;
        
        // For moderate clipping, detection should still work
        if (clip_level >= 0.7f) {
            EXPECT_TRUE(detected) << "Failed to detect watermark with clipping level " << clip_level;
        }
    }
}

// Test concurrent watermarking operations
TEST_F(WatermarkingEdgeCasesTest, ConcurrentOperations) {
    // Generate multiple audio samples
    size_t audio_length = 16000; // 1 second
    std::vector<std::vector<float>> audio_samples;
    for (int i = 0; i < 5; i++) {
        audio_samples.push_back(generate_random_audio(audio_length, 0.5f, 42 + i));
    }
    
    // Create multiple watermarkers with different keys
    std::vector<std::shared_ptr<SilentCipherWatermarker>> watermarkers;
    for (int i = 0; i < 5; i++) {
        watermarkers.push_back(SilentCipherWatermarker::create("key_" + std::to_string(i)));
    }
    
    // Create threads for concurrent watermarking
    std::vector<std::thread> threads;
    std::vector<std::vector<float>> results(5);
    
    for (int i = 0; i < 5; i++) {
        threads.push_back(std::thread([i, &watermarkers, &audio_samples, &results]() {
            results[i] = watermarkers[i]->apply_watermark(audio_samples[i]);
        }));
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Verify results
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(results[i].size(), audio_samples[i].size());
        bool detected = watermarkers[i]->detect_watermark(results[i]);
        EXPECT_TRUE(detected) << "Failed to detect watermark for thread " << i;
    }
}

// Test recovery from corrupted audio
TEST_F(WatermarkingEdgeCasesTest, CorruptedAudio) {
    // Generate audio
    size_t audio_length = 16000; // 1 second
    std::vector<float> audio = generate_random_audio(audio_length, 0.5f);
    
    // Apply watermark
    watermarker->set_strength(0.3f);
    std::vector<float> watermarked = watermarker->embed(audio, sample_rate, payload);
    
    // Corrupt parts of the audio
    std::vector<float> corrupted = watermarked;
    
    // Corrupt beginning
    for (size_t i = 0; i < audio_length / 10; i++) {
        corrupted[i] = 0.0f;
    }
    bool detected_begin_corrupt = watermarker->detect_watermark(corrupted);
    std::cout << "Corrupted beginning, detected: " << (detected_begin_corrupt ? "yes" : "no") << std::endl;
    
    // Corrupt middle
    corrupted = watermarked;
    for (size_t i = audio_length * 4 / 10; i < audio_length * 6 / 10; i++) {
        corrupted[i] = 0.0f;
    }
    bool detected_middle_corrupt = watermarker->detect_watermark(corrupted);
    std::cout << "Corrupted middle, detected: " << (detected_middle_corrupt ? "yes" : "no") << std::endl;
    
    // Corrupt end
    corrupted = watermarked;
    for (size_t i = audio_length * 9 / 10; i < audio_length; i++) {
        corrupted[i] = 0.0f;
    }
    bool detected_end_corrupt = watermarker->detect_watermark(corrupted);
    std::cout << "Corrupted end, detected: " << (detected_end_corrupt ? "yes" : "no") << std::endl;
    
    // For robust watermarking, at least some of these should be detected
    EXPECT_TRUE(detected_begin_corrupt || detected_middle_corrupt || detected_end_corrupt) 
        << "Failed to detect watermark in any corrupted version";
}

// Test watermarking impact on audio quality
TEST_F(WatermarkingEdgeCasesTest, AudioQuality) {
    // Generate audio with different characteristics
    size_t audio_length = 32000; // 2 seconds
    
    // Sine wave at different frequencies
    std::vector<float> sine_440 = generate_sine_wave(audio_length, 440.0f, sample_rate, 0.5f);
    std::vector<float> sine_1000 = generate_sine_wave(audio_length, 1000.0f, sample_rate, 0.5f);
    std::vector<float> sine_8000 = generate_sine_wave(audio_length, 8000.0f, sample_rate, 0.5f);
    
    // Random audio
    std::vector<float> random_audio = generate_random_audio(audio_length, 0.5f);
    
    // Test different strengths
    for (float strength : {0.05f, 0.1f, 0.2f, 0.3f}) {
        watermarker->set_strength(strength);
        
        // Apply watermarks
        std::vector<float> watermarked_440 = watermarker->embed(sine_440, sample_rate, payload);
        std::vector<float> watermarked_1000 = watermarker->embed(sine_1000, sample_rate, payload);
        std::vector<float> watermarked_8000 = watermarker->embed(sine_8000, sample_rate, payload);
        std::vector<float> watermarked_random = watermarker->embed(random_audio, sample_rate, payload);
        
        // Calculate SNR for each
        float snr_440 = calculate_snr_db(sine_440, watermarked_440);
        float snr_1000 = calculate_snr_db(sine_1000, watermarked_1000);
        float snr_8000 = calculate_snr_db(sine_8000, watermarked_8000);
        float snr_random = calculate_snr_db(random_audio, watermarked_random);
        
        std::cout << "Strength " << strength << " SNR values:" << std::endl;
        std::cout << "  440 Hz: " << snr_440 << " dB" << std::endl;
        std::cout << "  1000 Hz: " << snr_1000 << " dB" << std::endl;
        std::cout << "  8000 Hz: " << snr_8000 << " dB" << std::endl;
        std::cout << "  Random: " << snr_random << " dB" << std::endl;
        
        // Detect watermarks
        bool detected_440 = watermarker->detect_watermark(watermarked_440);
        bool detected_1000 = watermarker->detect_watermark(watermarked_1000);
        bool detected_8000 = watermarker->detect_watermark(watermarked_8000);
        bool detected_random = watermarker->detect_watermark(watermarked_random);
        
        std::cout << "  Detection rates:" << std::endl;
        std::cout << "  440 Hz: " << (detected_440 ? "yes" : "no") << std::endl;
        std::cout << "  1000 Hz: " << (detected_1000 ? "yes" : "no") << std::endl;
        std::cout << "  8000 Hz: " << (detected_8000 ? "yes" : "no") << std::endl;
        std::cout << "  Random: " << (detected_random ? "yes" : "no") << std::endl;
        
        // For perceptual quality, SNR should be reasonably high
        if (strength <= 0.1f) {
            EXPECT_GT(snr_440, 25.0f) << "SNR too low for 440 Hz sine with strength " << strength;
            EXPECT_GT(snr_1000, 25.0f) << "SNR too low for 1000 Hz sine with strength " << strength;
        }
        
        // For reasonable strengths, detection should work on most audio types
        if (strength >= 0.1f) {
            EXPECT_TRUE(detected_440) << "Failed to detect watermark in 440 Hz sine with strength " << strength;
            EXPECT_TRUE(detected_1000) << "Failed to detect watermark in 1000 Hz sine with strength " << strength;
            EXPECT_TRUE(detected_random) << "Failed to detect watermark in random audio with strength " << strength;
        }
    }
}

// Main function is provided by Google Test