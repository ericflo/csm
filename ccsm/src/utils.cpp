#include <ccsm/utils.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <string>
#include <filesystem>
#include <stdexcept>
#include <cstdint>

namespace ccsm {

// Memory utilities can be added here in the future
std::string format_size(size_t size_bytes) {
    constexpr double kb = 1024.0;
    constexpr double mb = kb * 1024.0;
    constexpr double gb = mb * 1024.0;
    
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2);
    
    if (size_bytes >= gb) {
        ss << (size_bytes / gb) << " GB";
    } else if (size_bytes >= mb) {
        ss << (size_bytes / mb) << " MB";
    } else if (size_bytes >= kb) {
        ss << (size_bytes / kb) << " KB";
    } else {
        ss << size_bytes << " bytes";
    }
    
    return ss.str();
}

// File utilities implementation
bool FileUtils::file_exists(const std::string& path) {
    std::filesystem::path fs_path(path);
    return std::filesystem::exists(fs_path) && std::filesystem::is_regular_file(fs_path);
}

std::string FileUtils::read_text_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// Binary file operations currently not implemented in the header
// Will be used by implementation when needed

bool FileUtils::write_text_file(const std::string& path, const std::string& content) {
    std::ofstream file(path);
    if (!file.is_open()) {
        return false;
    }
    
    file << content;
    return file.good();
}

// Implement file_size rather than get_file_size to match the header
size_t FileUtils::file_size(const std::string& path) {
    std::filesystem::path fs_path(path);
    return std::filesystem::file_size(fs_path);
}

// WAV file format constants
struct WAVHeader {
    // RIFF header
    char riff_header[4] = {'R', 'I', 'F', 'F'};
    uint32_t wav_size = 0;       // Initialized later
    char wave_header[4] = {'W', 'A', 'V', 'E'};
    
    // Format header
    char fmt_header[4] = {'f', 'm', 't', ' '};
    uint32_t fmt_chunk_size = 16;
    uint16_t audio_format = 1;   // PCM
    uint16_t num_channels = 1;   // Mono
    uint32_t sample_rate = 0;    // Initialized later
    uint32_t byte_rate = 0;      // Initialized later
    uint16_t sample_alignment = 2;
    uint16_t bit_depth = 16;     // 16-bit PCM
    
    // Data header
    char data_header[4] = {'d', 'a', 't', 'a'};
    uint32_t data_bytes = 0;     // Initialized later
};

bool FileUtils::save_wav(const std::string& path, const std::vector<float>& audio, int sample_rate) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        CCSM_ERROR("Failed to open WAV file for writing: ", path);
        return false;
    }
    
    // Create WAV header
    WAVHeader header;
    header.sample_rate = sample_rate;
    header.byte_rate = sample_rate * header.num_channels * (header.bit_depth / 8);
    header.data_bytes = audio.size() * (header.bit_depth / 8);
    header.wav_size = 36 + header.data_bytes;
    
    // Write WAV header
    file.write(reinterpret_cast<const char*>(&header), sizeof(WAVHeader));
    
    // Convert float audio to 16-bit PCM and write to file
    for (float sample : audio) {
        // Clamp the sample to [-1.0, 1.0]
        if (sample < -1.0f) sample = -1.0f;
        if (sample > 1.0f) sample = 1.0f;
        
        // Convert to 16-bit PCM
        int16_t pcm_sample = static_cast<int16_t>(sample * 32767.0f);
        file.write(reinterpret_cast<const char*>(&pcm_sample), sizeof(int16_t));
    }
    
    return file.good();
}

std::vector<float> FileUtils::load_wav(const std::string& path, int* sample_rate) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open WAV file: " + path);
    }
    
    // Read the WAV header
    WAVHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(WAVHeader));
    
    // Check if this is a valid WAV file
    bool is_valid = 
        header.riff_header[0] == 'R' && header.riff_header[1] == 'I' && 
        header.riff_header[2] == 'F' && header.riff_header[3] == 'F' &&
        header.wave_header[0] == 'W' && header.wave_header[1] == 'A' && 
        header.wave_header[2] == 'V' && header.wave_header[3] == 'E';
    
    if (!is_valid) {
        throw std::runtime_error("Invalid WAV file format: " + path);
    }
    
    // Check if this is PCM format
    if (header.audio_format != 1) {
        throw std::runtime_error("Only PCM WAV files are supported: " + path);
    }
    
    // Check if this is mono
    if (header.num_channels != 1) {
        throw std::runtime_error("Only mono WAV files are supported: " + path);
    }
    
    // Check if this is 16-bit
    if (header.bit_depth != 16) {
        throw std::runtime_error("Only 16-bit WAV files are supported: " + path);
    }
    
    // Return the sample rate if requested
    if (sample_rate) {
        *sample_rate = header.sample_rate;
    }
    
    // Calculate the number of samples
    int num_samples = header.data_bytes / (header.bit_depth / 8);
    
    // Read the audio data
    std::vector<float> audio(num_samples);
    for (int i = 0; i < num_samples; i++) {
        int16_t pcm_sample;
        file.read(reinterpret_cast<char*>(&pcm_sample), sizeof(int16_t));
        
        // Convert to float in range [-1.0, 1.0]
        audio[i] = static_cast<float>(pcm_sample) / 32767.0f;
    }
    
    return audio;
}

// Progress bar implementation is defined inline in the header
// No need for implementation here

} // namespace ccsm