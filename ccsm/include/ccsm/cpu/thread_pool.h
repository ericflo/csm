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

// Thread-safe queue for tasks with support for work stealing
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
    
    // Add tasks to the front of the queue (higher priority)
    template<typename F>
    bool push_front(F&& f) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (done_) {
            return false;
        }
        
        // Create a new queue with the task at the front
        std::queue<std::function<void()>> new_queue;
        new_queue.push(std::forward<F>(f));
        
        // Add the existing tasks
        while (!tasks_.empty()) {
            new_queue.push(std::move(tasks_.front()));
            tasks_.pop();
        }
        
        tasks_ = std::move(new_queue);
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
    
    // Try to get a task without waiting (for work stealing)
    bool try_pop(std::function<void()>& task) {
        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
        if (!lock || tasks_.empty()) {
            return false;
        }
        task = std::move(tasks_.front());
        tasks_.pop();
        return true;
    }
    
    // Try to steal half the tasks from another queue
    bool steal(TaskQueue& other) {
        std::function<void()> task;
        if (!other.try_pop(task)) {
            return false;
        }
        
        // Successfully stole a task, add it to our queue
        push(std::move(task));
        
        // Try to steal more tasks if available (up to half)
        size_t stolen = 1;
        size_t target = other.size() / 2;
        
        while (stolen < target && other.try_pop(task)) {
            push(std::move(task));
            stolen++;
        }
        
        return stolen > 0;
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
    
    // Check if done flag is set
    bool is_done() const {
        std::unique_lock<std::mutex> lock(mutex_);
        return done_;
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
        
        // Find the queue with the least amount of work
        size_t min_idx = 0;
        size_t min_size = local_queues_[0]->size();
        
        for (size_t i = 1; i < local_queues_.size(); i++) {
            size_t queue_size = local_queues_[i]->size();
            if (queue_size < min_size) {
                min_idx = i;
                min_size = queue_size;
            }
        }
        
        // Submit task to the least busy queue
        local_queues_[min_idx]->push([task](){ (*task)(); });
        
        return result;
    }
    
    // Add high-priority task to the front of the queue
    template<typename F, typename... Args>
    auto enqueue_priority(F&& f, Args&&... args) 
        -> std::future<typename std::invoke_result<F, Args...>::type> {
        using return_type = typename std::invoke_result<F, Args...>::type;
        
        // Create packaged task
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
        // Get future for result
        std::future<return_type> result = task->get_future();
        
        // Submit task to the global queue for high-priority tasks
        global_queue_.push_front([task](){ (*task)(); });
        
        return result;
    }
    
    // Wait for all tasks to complete
    void wait_all();
    
    // Get number of threads
    size_t size() const { return threads_.size(); }
    
    // Alias for size() that's more descriptive for tests
    size_t getThreadCount() const { return threads_.size(); }
    
    // Get active task count
    size_t active_task_count() const { return active_tasks_.load(); }
    
    // Get queue sizes for monitoring
    std::vector<size_t> queue_sizes() const;
    
private:
    std::vector<std::thread> threads_;
    std::vector<std::unique_ptr<TaskQueue>> local_queues_; // One per thread
    TaskQueue global_queue_; // Shared queue for load balancing
    std::atomic<size_t> active_tasks_{0};
    std::mutex wait_mutex_;
    std::condition_variable wait_cond_;
    std::atomic<bool> running_{true};
    
    // Worker thread function
    void worker_thread(size_t thread_id);
    
    // Check all queues for tasks
    bool get_task(std::function<void()>& task, size_t thread_id);
    
    // Try to steal tasks from other queues
    bool steal_task(std::function<void()>& task, size_t thread_id);
};

// Get singleton thread pool
ThreadPool& global_thread_pool();

// Helper for parallel for loops
class ParallelFor {
public:
    // Execute function in parallel across range [begin, end)
    template<typename F>
    static void exec(int begin, int end, F&& f, int chunk_size = 0) {
        if (begin >= end) {
            return;
        }
        
        // Calculate optimal chunk size if not provided
        int num_items = end - begin;
        if (chunk_size <= 0) {
            // Auto-determine chunk size based on work and thread count
            int num_threads = global_thread_pool().size();
            
            // For very small workloads, just use a single thread
            if (num_items <= 32) {
                for (int i = begin; i < end; i++) {
                    f(i);
                }
                return;
            }
            
            // For small workloads, use fewer chunks than threads
            if (num_items < num_threads * 4) {
                chunk_size = std::max(1, num_items / 4);
            } else {
                // Calculate balanced chunk size based on thread count
                // We want each thread to get multiple chunks for better load balancing
                int chunks_per_thread = 4;
                int total_chunks = num_threads * chunks_per_thread;
                chunk_size = std::max(1, num_items / total_chunks);
                
                // Don't make chunks too small
                int min_chunk_size = 16;
                chunk_size = std::max(min_chunk_size, chunk_size);
                
                // Don't make chunks too large
                int max_chunk_size = num_items / num_threads;
                if (max_chunk_size > 0) {
                    chunk_size = std::min(max_chunk_size, chunk_size);
                }
            }
        }
        
        // Use atomic counter for dynamic load balancing
        std::atomic<int> next_index(begin);
        
        // Exception handling
        std::atomic<bool> has_exception(false);
        std::exception_ptr exception_ptr;
        std::mutex exception_mutex;
        
        // Submit worker tasks
        int num_threads = global_thread_pool().size();
        std::vector<std::future<void>> futures;
        futures.reserve(num_threads);
        
        for (int i = 0; i < num_threads; i++) {
            futures.push_back(global_thread_pool().enqueue([&]() {
                try {
                    while (true) {
                        // Get next chunk
                        int chunk_begin = next_index.fetch_add(chunk_size);
                        if (chunk_begin >= end) {
                            break;
                        }
                        
                        // Process chunk
                        int chunk_end = std::min(end, chunk_begin + chunk_size);
                        for (int j = chunk_begin; j < chunk_end; j++) {
                            if (has_exception.load()) {
                                break;
                            }
                            f(j);
                        }
                        
                        // Exit if exception in another thread
                        if (has_exception.load()) {
                            break;
                        }
                    }
                }
                catch (...) {
                    // Capture exception
                    std::lock_guard<std::mutex> lock(exception_mutex);
                    if (!has_exception.exchange(true)) {
                        exception_ptr = std::current_exception();
                    }
                }
            }));
        }
        
        // Wait for all tasks to complete
        for (auto& future : futures) {
            future.get();
        }
        
        // Rethrow exception if any
        if (has_exception) {
            std::rethrow_exception(exception_ptr);
        }
    }
    
    // Parallel for with 2D range
    template<typename F>
    static void exec_2d(int begin_outer, int end_outer, int begin_inner, int end_inner, F&& f, int chunk_size_outer = 0) {
        if (begin_outer >= end_outer || begin_inner >= end_inner) {
            return;
        }
        
        // Execute 2D loop in parallel
        exec(begin_outer, end_outer, [&](int i) {
            for (int j = begin_inner; j < end_inner; j++) {
                f(i, j);
            }
        }, chunk_size_outer);
    }
    
    // Parallel for with custom index mapping
    template<typename Index, typename F>
    static void exec_indexed(const std::vector<Index>& indices, F&& f, int chunk_size = 0) {
        int size = static_cast<int>(indices.size());
        
        // Use parallel for on index positions
        exec(0, size, [&](int i) {
            f(indices[i]);
        }, chunk_size);
    }
};

} // namespace ccsm

#endif // CCSM_THREAD_POOL_H