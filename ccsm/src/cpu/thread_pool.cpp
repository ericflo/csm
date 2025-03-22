#include <ccsm/cpu/thread_pool.h>
#include <random>

namespace ccsm {

ThreadPool::ThreadPool(size_t num_threads) : running_(true) {
    // Default to hardware concurrency if not specified
    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
        // Ensure at least one thread
        if (num_threads == 0) {
            num_threads = 1;
        }
    }
    
    // Create local task queues (one per thread)
    local_queues_.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
        local_queues_.push_back(std::make_unique<TaskQueue>());
    }
    
    // Create worker threads
    threads_.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
        threads_.emplace_back(&ThreadPool::worker_thread, this, i);
    }
}

ThreadPool::~ThreadPool() {
    // Signal all threads to stop
    running_ = false;
    
    // Signal all queues to stop
    for (auto& queue : local_queues_) {
        queue->done();
    }
    global_queue_.done();
    
    // Wait for all threads to finish
    for (auto& thread : threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void ThreadPool::worker_thread(size_t thread_id) {
    // Random number generator for task stealing
    std::random_device rd;
    std::mt19937 rng(rd());
    
    while (running_) {
        std::function<void()> task;
        bool got_task = get_task(task, thread_id);
        
        if (!got_task) {
            // No tasks available, try to steal
            if (!steal_task(task, thread_id)) {
                // No tasks to steal, sleep briefly to avoid busy waiting
                std::this_thread::yield();
                std::this_thread::sleep_for(std::chrono::microseconds(100));
                continue;
            }
        }
        
        // Execute task
        active_tasks_++;
        task();
        active_tasks_--;
        
        // Notify wait_all if no active tasks
        if (active_tasks_ == 0) {
            bool all_empty = true;
            
            // Check if all queues are empty
            if (!global_queue_.empty()) {
                all_empty = false;
            } else {
                for (auto& queue : local_queues_) {
                    if (!queue->empty()) {
                        all_empty = false;
                        break;
                    }
                }
            }
            
            if (all_empty) {
                std::unique_lock<std::mutex> lock(wait_mutex_);
                wait_cond_.notify_all();
            }
        }
    }
}

bool ThreadPool::get_task(std::function<void()>& task, size_t thread_id) {
    // First check the thread's local queue
    if (local_queues_[thread_id]->try_pop(task)) {
        return true;
    }
    
    // Then check the global queue
    if (global_queue_.try_pop(task)) {
        return true;
    }
    
    return false;
}

bool ThreadPool::steal_task(std::function<void()>& task, size_t thread_id) {
    // Try to steal from other local queues in random order
    std::vector<size_t> indices(local_queues_.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
    
    for (size_t i : indices) {
        // Don't steal from ourselves
        if (i == thread_id) {
            continue;
        }
        
        if (local_queues_[i]->try_pop(task)) {
            return true;
        }
    }
    
    // Last resort: check global queue again
    if (global_queue_.try_pop(task)) {
        return true;
    }
    
    return false;
}

void ThreadPool::wait_all() {
    std::unique_lock<std::mutex> lock(wait_mutex_);
    wait_cond_.wait(lock, [this]() {
        if (active_tasks_ > 0) {
            return false;
        }
        
        // Check if all queues are empty
        if (!global_queue_.empty()) {
            return false;
        }
        
        for (auto& queue : local_queues_) {
            if (!queue->empty()) {
                return false;
            }
        }
        
        return true;
    });
}

std::vector<size_t> ThreadPool::queue_sizes() const {
    std::vector<size_t> sizes;
    sizes.reserve(local_queues_.size() + 1);
    
    // Add global queue size
    sizes.push_back(global_queue_.size());
    
    // Add local queue sizes
    for (const auto& queue : local_queues_) {
        sizes.push_back(queue->size());
    }
    
    return sizes;
}

// Global thread pool singleton
ThreadPool& global_thread_pool() {
    static ThreadPool instance;
    return instance;
}

} // namespace ccsm