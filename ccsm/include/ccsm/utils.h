#ifndef CCSM_UTILS_H
#define CCSM_UTILS_H

#include <string>
#include <vector>
#include <chrono>
#include <functional>
#include <memory>
#include <iostream>
#include <sstream>

namespace ccsm {

// Logging levels
enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARNING = 2,
    ERROR = 3,
    NONE = 4
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
    
    template <typename... Args>
    void debug(Args&&... args) {
        if (level_ <= LogLevel::DEBUG) {
            log(LogLevel::DEBUG, std::forward<Args>(args)...);
        }
    }
    
    template <typename... Args>
    void info(Args&&... args) {
        if (level_ <= LogLevel::INFO) {
            log(LogLevel::INFO, std::forward<Args>(args)...);
        }
    }
    
    template <typename... Args>
    void warning(Args&&... args) {
        if (level_ <= LogLevel::WARNING) {
            log(LogLevel::WARNING, std::forward<Args>(args)...);
        }
    }
    
    template <typename... Args>
    void error(Args&&... args) {
        if (level_ <= LogLevel::ERROR) {
            log(LogLevel::ERROR, std::forward<Args>(args)...);
        }
    }
    
    // Set output stream
    void set_output(std::ostream& os) {
        output_ = &os;
    }
    
private:
    Logger() : level_(LogLevel::INFO), output_(&std::cerr) {}
    ~Logger() = default;
    
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    
    template <typename T, typename... Args>
    void log(LogLevel level, T&& value, Args&&... args) {
        std::stringstream ss;
        ss << value;
        log(level, ss.str(), std::forward<Args>(args)...);
    }
    
    template <typename... Args>
    void log(LogLevel level, const std::string& message, Args&&... args) {
        std::stringstream ss;
        ss << message;
        log(level, ss.str(), std::forward<Args>(args)...);
    }
    
    void log(LogLevel level, const std::string& message) {
        std::string prefix;
        switch (level) {
            case LogLevel::DEBUG:
                prefix = "[DEBUG] ";
                break;
            case LogLevel::INFO:
                prefix = "[INFO] ";
                break;
            case LogLevel::WARNING:
                prefix = "[WARNING] ";
                break;
            case LogLevel::ERROR:
                prefix = "[ERROR] ";
                break;
            case LogLevel::NONE:
                prefix = "";
                break;
        }
        
        *output_ << prefix << message << std::endl;
    }
    
    LogLevel level_;
    std::ostream* output_;
};

// Convenience macros for logging
#define CCSM_DEBUG(...) ccsm::Logger::instance().debug(__VA_ARGS__)
#define CCSM_INFO(...) ccsm::Logger::instance().info(__VA_ARGS__)
#define CCSM_WARNING(...) ccsm::Logger::instance().warning(__VA_ARGS__)
#define CCSM_ERROR(...) ccsm::Logger::instance().error(__VA_ARGS__)

// Timer utility for benchmarking
class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}
    
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

// Memory utilities
class MemoryUtils {
public:
    // Get current process memory usage in bytes
    static size_t get_memory_usage();
    
    // Format memory size to human-readable string
    static std::string format_size(size_t size_bytes);
    
    // Check if out of memory
    static bool is_out_of_memory();
};

// File utilities
class FileUtils {
public:
    // Check if file exists
    static bool file_exists(const std::string& path);
    
    // Read entire file into a string
    static std::string read_text_file(const std::string& path);
    
    // Read binary file
    static std::vector<uint8_t> read_binary_file(const std::string& path);
    
    // Write string to file
    static bool write_text_file(const std::string& path, const std::string& content);
    
    // Write binary data to file
    static bool write_binary_file(const std::string& path, const std::vector<uint8_t>& data);
    
    // Get file size
    static size_t get_file_size(const std::string& path);
    
    // Create directory if it doesn't exist
    static bool create_directory(const std::string& path);
};

// Progress callback types
using ProgressCallback = std::function<void(int current, int total)>;

// Progress bar utility
class ProgressBar {
public:
    ProgressBar(int total, int width = 40);
    
    void update(int current);
    void finish();
    
private:
    int total_;
    int width_;
    int last_printed_percent_;
    bool finished_;
};

} // namespace ccsm

#endif // CCSM_UTILS_H