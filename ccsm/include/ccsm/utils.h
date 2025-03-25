#ifndef CCSM_UTILS_H
#define CCSM_UTILS_H

#include <string>
#include <vector>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <sstream>

namespace ccsm {

// Log levels
enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR
};

// Logger class
class Logger {
public:
    static Logger& instance() {
        static Logger instance;
        return instance;
    }
    
    void set_level(LogLevel level) {
        level_ = level;
    }
    
    LogLevel get_level() const {
        return level_;
    }
    
    template<typename... Args>
    void log(LogLevel level, const Args&... args) {
        if (level >= level_) {
            std::ostringstream oss;
            oss << "[" << level_to_string(level) << "] ";
            log_impl(oss, args...);
            std::cout << oss.str() << std::endl;
        }
    }
    
private:
    Logger() : level_(LogLevel::INFO) {}
    
    std::string level_to_string(LogLevel level) {
        switch (level) {
            case LogLevel::DEBUG:   return "DEBUG";
            case LogLevel::INFO:    return "INFO";
            case LogLevel::WARNING: return "WARNING";
            case LogLevel::ERROR:   return "ERROR";
            default:                return "UNKNOWN";
        }
    }
    
    template<typename T>
    void log_impl(std::ostringstream& oss, const T& value) {
        oss << value;
    }
    
    template<typename T, typename... Args>
    void log_impl(std::ostringstream& oss, const T& value, const Args&... args) {
        oss << value;
        log_impl(oss, args...);
    }
    
    LogLevel level_;
};

// Logging macros
#define CCSM_DEBUG(...) ccsm::Logger::instance().log(ccsm::LogLevel::DEBUG, __VA_ARGS__)
#define CCSM_INFO(...) ccsm::Logger::instance().log(ccsm::LogLevel::INFO, __VA_ARGS__)
#define CCSM_WARNING(...) ccsm::Logger::instance().log(ccsm::LogLevel::WARNING, __VA_ARGS__)
#define CCSM_ERROR(...) ccsm::Logger::instance().log(ccsm::LogLevel::ERROR, __VA_ARGS__)

// Timer class for benchmarking
class Timer {
public:
    Timer() {
        reset();
    }
    
    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    
    double elapsed_ms() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }
    
    double elapsed_s() const {
        return elapsed_ms() / 1000.0;
    }
    
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

// Simple progress bar
class ProgressBar {
public:
    ProgressBar(int total, int width = 50)
        : total_(total), width_(width), current_(0) {}
    
    void update(int current) {
        current_ = current;
        int pos = width_ * current_ / total_;
        
        std::cout << "\r[";
        for (int i = 0; i < width_; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << current_ << "/" << total_ << " (" 
                 << std::fixed << std::setprecision(1) 
                 << 100.0 * current_ / total_ << "%)";
        
        std::cout.flush();
    }
    
    void finish() {
        update(total_);
        std::cout << std::endl;
    }
    
private:
    int total_;
    int width_;
    int current_;
};

// File utilities
class FileUtils {
public:
    // Load WAV file
    static std::vector<float> load_wav(const std::string& filename, int* sample_rate = nullptr);
    
    // Save WAV file
    static bool save_wav(const std::string& filename, const std::vector<float>& audio, int sample_rate);
    
    // Check if file exists
    static bool file_exists(const std::string& filename);
    
    // Get file size
    static size_t file_size(const std::string& filename);
    
    // Read text file
    static std::string read_text_file(const std::string& filename);
    
    // Write text file
    static bool write_text_file(const std::string& filename, const std::string& content);
    
    // Get temporary directory path
    static std::string get_temp_directory();
};

} // namespace ccsm

#endif // CCSM_UTILS_H