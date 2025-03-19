#ifndef CCSM_THREAD_POOL_H
#define CCSM_THREAD_POOL_H

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <atomic>

namespace ccsm {

// Thread-safe queue for tasks
class TaskQueue {
public:
    TaskQueue() : done_(false) {}
    
    // Add task to queue
    template<typename F>
    bool push(F&& f) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (done_) {
            return false;
        }
        tasks_.push(std::forward<F>(f));
        lock.unlock();
        cond_.notify_one();
        return true;
    }
    
    // Get task from queue
    bool pop(std::function<void()>& task) {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_.wait(lock, [this]{ return done_ || !tasks_.empty(); });
        if (tasks_.empty()) {
            return false;
        }
        task = std::move(tasks_.front());
        tasks_.pop();
        return true;
    }
    
    // Check if queue is empty
    bool empty() const {
        std::unique_lock<std::mutex> lock(mutex_);
        return tasks_.empty();
    }
    
    // Check number of tasks in queue
    size_t size() const {
        std::unique_lock<std::mutex> lock(mutex_);
        return tasks_.size();
    }
    
    // Signal threads to finish
    void done() {
        std::unique_lock<std::mutex> lock(mutex_);
        done_ = true;
        lock.unlock();
        cond_.notify_all();
    }
    
private:
    std::queue<std::function<void()>> tasks_;
    mutable std::mutex mutex_;
    std::condition_variable cond_;
    bool done_;
};

// Thread pool for parallel task execution
class ThreadPool {
public:
    // Initialize with number of threads (default: hardware concurrency)
    ThreadPool(size_t num_threads = 0);
    
    // Destructor waits for all tasks to complete
    ~ThreadPool();
    
    // Add task to pool and get future result
    template<typename F, typename... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::invoke_result<F, Args...>::type> {
        using return_type = typename std::invoke_result<F, Args...>::type;
        
        // Create packaged task
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
        // Get future for result
        std::future<return_type> result = task->get_future();
        
        // Submit task to queue
        task_queue_.push([task](){ (*task)(); });
        
        return result;
    }
    
    // Wait for all tasks to complete
    void wait_all();
    
    // Get number of threads
    size_t size() const { return threads_.size(); }
    
private:
    std::vector<std::thread> threads_;
    TaskQueue task_queue_;
    std::atomic<size_t> active_tasks_{0};
    std::mutex wait_mutex_;
    std::condition_variable wait_cond_;
    
    // Worker thread function
    void worker_thread();
};

// Get singleton thread pool
ThreadPool& global_thread_pool();

// RAII helper for parallel for loops
class ParallelFor {
public:
    // Execute function in parallel across range [begin, end)
    template<typename F>
    static void exec(int begin, int end, F&& f) {
        if (begin >= end) {
            return;
        }
        
        // Get optimal chunk size (simple heuristic)
        int num_items = end - begin;
        int num_threads = global_thread_pool().size();
        int min_items_per_thread = std::max(1, num_items / (num_threads * 4));
        int chunk_size = std::max(1, num_items / num_threads);
        chunk_size = std::max(min_items_per_thread, chunk_size);
        
        // Submit tasks in chunks
        std::vector<std::future<void>> futures;
        for (int i = begin; i < end; i += chunk_size) {
            int chunk_begin = i;
            int chunk_end = std::min(end, i + chunk_size);
            
            futures.push_back(global_thread_pool().enqueue([f, chunk_begin, chunk_end]() {
                for (int j = chunk_begin; j < chunk_end; j++) {
                    f(j);
                }
            }));
        }
        
        // Wait for all tasks to complete
        for (auto& future : futures) {
            future.get();
        }
    }
};

} // namespace ccsm

#endif // CCSM_THREAD_POOL_H