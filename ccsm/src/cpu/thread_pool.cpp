#include <ccsm/cpu/thread_pool.h>

namespace ccsm {

ThreadPool::ThreadPool(size_t num_threads) {
    // Default to hardware concurrency if not specified
    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
        // Ensure at least one thread
        if (num_threads == 0) {
            num_threads = 1;
        }
    }
    
    // Create worker threads
    threads_.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
        threads_.emplace_back(&ThreadPool::worker_thread, this);
    }
}

ThreadPool::~ThreadPool() {
    // Signal all threads to stop
    task_queue_.done();
    
    // Wait for all threads to finish
    for (auto& thread : threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void ThreadPool::worker_thread() {
    while (true) {
        std::function<void()> task;
        bool got_task = task_queue_.pop(task);
        
        if (!got_task) {
            // Queue is empty and done flag is set
            break;
        }
        
        // Execute task
        active_tasks_++;
        task();
        active_tasks_--;
        
        // Notify wait_all if no active tasks
        if (active_tasks_ == 0 && task_queue_.empty()) {
            std::unique_lock<std::mutex> lock(wait_mutex_);
            wait_cond_.notify_all();
        }
    }
}

void ThreadPool::wait_all() {
    std::unique_lock<std::mutex> lock(wait_mutex_);
    wait_cond_.wait(lock, [this]() {
        return active_tasks_ == 0 && task_queue_.empty();
    });
}

// Global thread pool singleton
ThreadPool& global_thread_pool() {
    static ThreadPool instance;
    return instance;
}

} // namespace ccsm