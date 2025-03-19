#include <ccsm/utils.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <string>
#include <filesystem>
#include <stdexcept>

namespace ccsm {

// Memory utilities implementation
size_t MemoryUtils::get_memory_usage() {
    // TODO: Implement platform-specific memory usage query
    return 0;
}

std::string MemoryUtils::format_size(size_t size_bytes) {
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

bool MemoryUtils::is_out_of_memory() {
    // TODO: Implement platform-specific OOM detection
    return false;
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

std::vector<uint8_t> FileUtils::read_binary_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<uint8_t> buffer(size);
    file.read(reinterpret_cast<char*>(buffer.data()), size);
    
    return buffer;
}

bool FileUtils::write_text_file(const std::string& path, const std::string& content) {
    std::ofstream file(path);
    if (!file.is_open()) {
        return false;
    }
    
    file << content;
    return file.good();
}

bool FileUtils::write_binary_file(const std::string& path, const std::vector<uint8_t>& data) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    file.write(reinterpret_cast<const char*>(data.data()), data.size());
    return file.good();
}

size_t FileUtils::get_file_size(const std::string& path) {
    std::filesystem::path fs_path(path);
    return std::filesystem::file_size(fs_path);
}

bool FileUtils::create_directory(const std::string& path) {
    std::filesystem::path fs_path(path);
    return std::filesystem::create_directories(fs_path);
}

// Progress bar implementation
ProgressBar::ProgressBar(int total, int width) 
    : total_(total), width_(width), last_printed_percent_(-1), finished_(false) {
}

void ProgressBar::update(int current) {
    if (finished_) return;
    
    int percent = (total_ > 0) ? (current * 100 / total_) : 0;
    
    // Only update if percentage has changed
    if (percent != last_printed_percent_) {
        last_printed_percent_ = percent;
        
        int progress_width = width_ * percent / 100;
        
        std::cout << "\r[";
        for (int i = 0; i < width_; i++) {
            if (i < progress_width) {
                std::cout << "=";
            } else if (i == progress_width) {
                std::cout << ">";
            } else {
                std::cout << " ";
            }
        }
        
        std::cout << "] " << percent << "% (" << current << "/" << total_ << ")" << std::flush;
    }
}

void ProgressBar::finish() {
    if (!finished_) {
        update(total_);
        std::cout << std::endl;
        finished_ = true;
    }
}

} // namespace ccsm