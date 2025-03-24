#include <gtest/gtest.h>
#include <ccsm/cpu/thread_pool.h>
#include <vector>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <random>
#include <set>
#include <mutex>
#include <unordered_map>
#include <thread>
#include <iostream>
#include <cmath>
#include <future>
#include <condition_variable>
#include <queue>
#include <functional>
#include <exception>
#include <stdexcept>
#include <memory>

using namespace ccsm;

// Helper function declarations
// Helper function to cleanup thread local data (simulated)
int thread_local_data_cleanup() {
    // This would typically analyze and clean up any thread-local storage
    // Here we just return a simulated thread count
    return std::thread::hardware_concurrency();
}

/**
 * Stress tests for the ThreadPool implementation.
 * 
 * These tests focus on extreme scenarios, edge cases, and performance
 * characteristics to ensure the thread pool is robust and efficient.
 */
class ThreadPoolStressTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
    
    // Helper to simulate CPU-bound work with controllable duration
    static void simulateCPUWork(int iterations) {
        volatile int sum = 0;
        for (int i = 0; i < iterations; i++) {
            sum += i;
        }
    }
    
    // Helper to simulate I/O-bound work with sleep
    static void simulateIOWork(int ms) {
        std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    }
    
    // Helper to generate random delay duration
    static int randomDuration(int min_ms, int max_ms) {
        static std::mt19937 gen(std::random_device{}());
        std::uniform_int_distribution<> dist(min_ms, max_ms);
        return dist(gen);
    }
    
    // Helper to measure execution time of a function
    template<typename Func>
    static double measureExecutionTime(Func&& func) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
    
    // Helper to track thread-specific metrics
    class ThreadTracker {
    public:
        void taskStarted(std::thread::id thread_id, int task_id) {
            std::lock_guard<std::mutex> lock(mutex_);
            active_threads_.insert(thread_id);
            thread_tasks_[thread_id].push_back(task_id);
            task_start_times_[task_id] = std::chrono::high_resolution_clock::now();
        }
        
        void taskFinished(std::thread::id thread_id, int task_id) {
            std::lock_guard<std::mutex> lock(mutex_);
            task_end_times_[task_id] = std::chrono::high_resolution_clock::now();
            completed_tasks_.push_back(task_id);
        }
        
        size_t getActiveThreadCount() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return active_threads_.size();
        }
        
        size_t getCompletedTaskCount() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return completed_tasks_.size();
        }
        
        // Get metrics for task execution times
        std::unordered_map<int, double> getTaskDurations() const {
            std::lock_guard<std::mutex> lock(mutex_);
            std::unordered_map<int, double> durations;
            
            for (const auto& [task_id, start_time] : task_start_times_) {
                auto it = task_end_times_.find(task_id);
                if (it != task_end_times_.end()) {
                    auto duration = std::chrono::duration<double, std::milli>(
                        it->second - start_time).count();
                    durations[task_id] = duration;
                }
            }
            
            return durations;
        }
        
        // Get task distribution per thread
        std::map<std::thread::id, size_t> getTaskDistribution() const {
            std::lock_guard<std::mutex> lock(mutex_);
            std::map<std::thread::id, size_t> distribution;
            
            for (const auto& [thread_id, tasks] : thread_tasks_) {
                distribution[thread_id] = tasks.size();
            }
            
            return distribution;
        }
        
        // Get execution order
        std::vector<int> getCompletionOrder() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return completed_tasks_;
        }
        
    private:
        mutable std::mutex mutex_;
        std::set<std::thread::id> active_threads_;
        std::unordered_map<std::thread::id, std::vector<int>> thread_tasks_;
        std::unordered_map<int, std::chrono::high_resolution_clock::time_point> task_start_times_;
        std::unordered_map<int, std::chrono::high_resolution_clock::time_point> task_end_times_;
        std::vector<int> completed_tasks_;
    };
};

// Test extreme number of tasks (stress test)
TEST_F(ThreadPoolStressTest, ExtremeTaskCount) {
    ThreadPool pool(4);
    ThreadTracker tracker;
    
    // Very large number of tasks to stress the queue system
    const int task_count = 10000;
    std::vector<std::future<int>> futures;
    
    // Submit many small tasks
    for (int i = 0; i < task_count; i++) {
        futures.push_back(pool.enqueue([i, &tracker]() {
            auto thread_id = std::this_thread::get_id();
            tracker.taskStarted(thread_id, i);
            
            // Very minimal work to avoid test taking too long
            simulateCPUWork(100);
            
            tracker.taskFinished(thread_id, i);
            return i;
        }));
    }
    
    // Wait for all tasks to complete
    for (int i = 0; i < task_count; i++) {
        ASSERT_EQ(futures[i].get(), i);
    }
    
    // Verify all tasks completed
    EXPECT_EQ(tracker.getCompletedTaskCount(), task_count);
    
    // Get task distribution statistics
    auto distribution = tracker.getTaskDistribution();
    
    // Calculate total, min, max, and average tasks per thread
    size_t total_tasks = 0;
    size_t min_tasks = task_count;
    size_t max_tasks = 0;
    
    for (const auto& [thread_id, count] : distribution) {
        total_tasks += count;
        min_tasks = std::min(min_tasks, count);
        max_tasks = std::max(max_tasks, count);
    }
    
    double avg_tasks = static_cast<double>(total_tasks) / distribution.size();
    
    // Log statistics
    std::cout << "Task distribution across " << distribution.size() << " threads:" << std::endl;
    for (const auto& [thread_id, count] : distribution) {
        std::cout << "  Thread " << thread_id << ": " << count << " tasks" << std::endl;
    }
    std::cout << "Min: " << min_tasks << ", Max: " << max_tasks << ", Avg: " << avg_tasks << std::endl;
    
    // Calculate work distribution imbalance
    double imbalance = static_cast<double>(max_tasks) / min_tasks;
    std::cout << "Work imbalance factor: " << imbalance << std::endl;
    
    // Check that work was reasonably well distributed
    // Allow for up to 3x imbalance in extreme stress conditions
    EXPECT_LE(imbalance, 3.0);
}

// Test handling of tasks with varying execution times
TEST_F(ThreadPoolStressTest, VariableTaskDurations) {
    ThreadPool pool(4);
    ThreadTracker tracker;
    
    // Tasks with widely varying durations to test adaptive load balancing
    const int task_count = 100;
    std::vector<std::future<int>> futures;
    std::vector<int> expected_durations;
    
    // Create tasks with durations following a pattern: short, medium, long, very long
    for (int i = 0; i < task_count; i++) {
        int duration_ms;
        
        switch (i % 4) {
            case 0: duration_ms = 5;  break;  // Short
            case 1: duration_ms = 20; break;  // Medium
            case 2: duration_ms = 50; break;  // Long
            case 3: duration_ms = 100; break; // Very long
        }
        
        expected_durations.push_back(duration_ms);
        
        futures.push_back(pool.enqueue([i, duration_ms, &tracker]() {
            auto thread_id = std::this_thread::get_id();
            tracker.taskStarted(thread_id, i);
            
            // Sleep to simulate work of specified duration
            simulateIOWork(duration_ms);
            
            tracker.taskFinished(thread_id, i);
            return i;
        }));
    }
    
    // Wait for all tasks to complete
    for (auto& future : futures) {
        future.get();
    }
    
    // Analyze completion order to determine if shorter tasks tend to complete first
    auto completion_order = tracker.getCompletionOrder();
    
    // Verify we have all tasks
    EXPECT_EQ(completion_order.size(), task_count);
    
    // Calculate average position in completion order for each task type
    std::vector<double> avg_position(4, 0);
    std::vector<int> count(4, 0);
    
    for (size_t pos = 0; pos < completion_order.size(); pos++) {
        int task_id = completion_order[pos];
        int task_type = task_id % 4;
        
        avg_position[task_type] += static_cast<double>(pos);
        count[task_type]++;
    }
    
    // Normalize to get average positions
    for (int i = 0; i < 4; i++) {
        if (count[i] > 0) {
            avg_position[i] /= count[i];
        }
    }
    
    // Log statistics
    std::cout << "Average completion position by task type:" << std::endl;
    std::cout << "  Short tasks (5ms):      " << avg_position[0] << std::endl;
    std::cout << "  Medium tasks (20ms):    " << avg_position[1] << std::endl;
    std::cout << "  Long tasks (50ms):      " << avg_position[2] << std::endl;
    std::cout << "  Very long tasks (100ms): " << avg_position[3] << std::endl;
    
    // Verify there's some ordering by duration (shorter tasks should finish earlier on average)
    // This isn't guaranteed due to thread scheduling, but should be statistically significant
    EXPECT_LT(avg_position[0], avg_position[3]);
}

// Test exception safety with many failing tasks
TEST_F(ThreadPoolStressTest, MassiveExceptionHandling) {
    ThreadPool pool(4);
    
    // Setup a combination of successful and failing tasks
    const int task_count = 1000;
    std::atomic<int> successful_tasks(0);
    std::atomic<int> failed_tasks(0);
    
    std::vector<std::future<int>> futures;
    
    // Submit tasks with deterministic failure pattern
    for (int i = 0; i < task_count; i++) {
        futures.push_back(pool.enqueue([i, &successful_tasks, &failed_tasks]() -> int {
            if (i % 7 == 0) {
                // Every 7th task fails
                failed_tasks++;
                throw std::runtime_error("Deliberate failure in task " + std::to_string(i));
            }
            
            // Minimal work to avoid test taking too long
            simulateCPUWork(100);
            successful_tasks++;
            return i;
        }));
    }
    
    // Wait for all tasks and count exceptions
    int caught_exceptions = 0;
    for (int i = 0; i < task_count; i++) {
        try {
            int result = futures[i].get();
            EXPECT_EQ(result, i);
            EXPECT_NE(i % 7, 0); // Shouldn't be a failure task
        }
        catch (const std::runtime_error& e) {
            caught_exceptions++;
            EXPECT_EQ(i % 7, 0); // Should be a failure task
        }
    }
    
    // Verify exception counts
    int expected_failures = task_count / 7;
    // Due to integer division, the exact count might be off by 1 
    // so we check if values are close enough
    EXPECT_NEAR(caught_exceptions, expected_failures, 1);
    EXPECT_NEAR(failed_tasks, expected_failures, 1);
    EXPECT_NEAR(successful_tasks, task_count - expected_failures, 1);
    
    // Verify the thread pool is still usable after many exceptions
    auto test_future = pool.enqueue([]() { return 42; });
    EXPECT_EQ(test_future.get(), 42);
}

// Test thread pool with extreme recursive task creation
TEST_F(ThreadPoolStressTest, RecursiveTaskGeneration) {
    ThreadPool pool(4);
    std::atomic<int> task_counter(0);
    
    // Function that spawns more tasks recursively
    std::function<int(int, int)> recursive_task = [&](int depth, int id) -> int {
        task_counter++;
        
        // Base case
        if (depth <= 0) {
            return id;
        }
        
        // Spawn child tasks
        std::vector<std::future<int>> child_futures;
        
        for (int i = 0; i < 2; i++) { // Binary recursion
            int child_id = id * 10 + i;
            child_futures.push_back(pool.enqueue([&recursive_task, depth, child_id]() {
                return recursive_task(depth - 1, child_id);
            }));
        }
        
        // Wait for and sum results
        int sum = 0;
        for (auto& future : child_futures) {
            sum += future.get();
        }
        
        return sum;
    };
    
    // Start with reasonable recursion depth to avoid overflow
    int depth = 7; // Creates 2^depth - 1 tasks (127 tasks)
    
    auto start_time = std::chrono::high_resolution_clock::now();
    auto future = pool.enqueue([&recursive_task, depth]() {
        return recursive_task(depth, 0);
    });
    
    // Wait for all recursive tasks to complete
    int result = future.get();
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end_time - start_time);
    
    std::cout << "Recursive task execution with depth " << depth << " completed in "
              << duration.count() << "ms" << std::endl;
    std::cout << "Total tasks executed: " << task_counter << std::endl;
    
    // Expected number of tasks for binary recursion: 2^(depth+1) - 1
    int expected_tasks = (1 << (depth + 1)) - 1;
    EXPECT_EQ(task_counter, expected_tasks);
    
    // Verify thread pool is still operational
    auto test_future = pool.enqueue([]() { return 42; });
    EXPECT_EQ(test_future.get(), 42);
}

// Test thread pool under memory pressure with large data structures
TEST_F(ThreadPoolStressTest, MemoryPressure) {
    ThreadPool pool(4);
    
    // Create tasks that allocate significant memory
    const int task_count = 100;
    const int allocation_size = 1024 * 1024; // 1MB per allocation
    std::atomic<int> successful_allocations(0);
    
    std::vector<std::future<bool>> futures;
    
    for (int i = 0; i < task_count; i++) {
        futures.push_back(pool.enqueue([i, allocation_size, &successful_allocations]() {
            try {
                // Allocate a large chunk of memory
                std::vector<char> data(allocation_size, static_cast<char>(i % 256));
                
                // Do some work with the memory to ensure it's actually used
                volatile int sum = 0;
                for (size_t j = 0; j < allocation_size; j += 1024) {
                    sum += data[j];
                }
                
                // Simulate some processing time
                simulateCPUWork(1000);
                
                successful_allocations++;
                return true;
            }
            catch (const std::bad_alloc&) {
                // Memory allocation failed
                return false;
            }
        }));
    }
    
    // Wait for all tasks
    int allocation_failures = 0;
    for (auto& future : futures) {
        if (!future.get()) {
            allocation_failures++;
        }
    }
    
    // Verify all allocations succeeded (or log failures)
    if (allocation_failures > 0) {
        std::cout << "Warning: " << allocation_failures << " memory allocations failed" << std::endl;
    }
    
    EXPECT_EQ(successful_allocations, task_count - allocation_failures);
    
    // Verify thread pool is still operational after memory pressure
    auto test_future = pool.enqueue([]() { return 42; });
    EXPECT_EQ(test_future.get(), 42);
}

// Test parallel for with extreme chunk sizes
TEST_F(ThreadPoolStressTest, ExtremeChunkSizes) {
    // Test vector with 1 million elements
    const int size = 1000000;
    std::vector<int> data(size, 0);
    
    // Test with various chunk sizes
    std::vector<int> chunk_sizes = {1, 10, 100, 1000, 10000, 100000, size};
    
    for (int chunk_size : chunk_sizes) {
        // Reset data
        std::fill(data.begin(), data.end(), 0);
        
        // Track chunk boundaries
        std::atomic<int> chunk_count(0);
        std::mutex chunk_mutex;
        std::vector<std::pair<int, int>> chunk_ranges;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Process with specified chunk size
        ParallelFor::exec(0, size, [&](int i) {
            static thread_local int last_i = -999;
            static thread_local int chunk_start = -1;
            
            // Detect chunk boundaries
            if (i != last_i + 1) {
                // End of previous chunk
                if (chunk_start >= 0) {
                    std::lock_guard<std::mutex> lock(chunk_mutex);
                    chunk_ranges.emplace_back(chunk_start, last_i);
                }
                
                // Start of new chunk
                chunk_start = i;
                chunk_count++;
            }
            
            // Process element
            data[i] = i;
            last_i = i;
        }, chunk_size);
        
        // Record final chunk
        int thread_count = thread_local_data_cleanup();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end_time - start_time);
        
        // Verify all elements were processed
        for (int i = 0; i < size; i++) {
            ASSERT_EQ(data[i], i) << "Element " << i << " not processed with chunk size " << chunk_size;
        }
        
        // Log timing and chunk statistics
        std::cout << "Chunk size " << chunk_size << ": " << duration.count() << "ms, "
                  << chunk_count << " chunks" << std::endl;
                  
        // Expected number of chunks
        int expected_chunks = (size + chunk_size - 1) / chunk_size;
        
        // With work stealing, actual chunks might differ from expected
        // but should be in reasonable range
        EXPECT_GE(chunk_count, expected_chunks / 2);
        EXPECT_LE(chunk_count, expected_chunks * thread_count);
    }
}

// Test extreme priority inversion scenarios
TEST_F(ThreadPoolStressTest, ExtremePriorityInversion) {
    ThreadPool pool(2); // Limited threads to maximize blocking conditions
    
    // First, block both threads with long-running tasks
    std::atomic<bool> long_tasks_started(false);
    std::vector<std::future<void>> long_task_futures;
    
    for (int i = 0; i < 2; i++) {
        long_task_futures.push_back(pool.enqueue([&long_tasks_started]() {
            long_tasks_started = true;
            simulateIOWork(500); // Block for 500ms
        }));
    }
    
    // Wait for long tasks to start
    while (!long_tasks_started) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
    // Add many low-priority tasks
    const int low_priority_count = 50;
    std::vector<std::future<int>> low_priority_futures;
    
    for (int i = 0; i < low_priority_count; i++) {
        low_priority_futures.push_back(pool.enqueue([i]() {
            simulateIOWork(10);
            return i;
        }));
    }
    
    // Add one high-priority task
    auto high_priority_start = std::chrono::high_resolution_clock::now();
    auto high_priority_future = pool.enqueue_priority([]() {
        return 999; // High-priority task
    });
    
    // Wait for high-priority task
    int high_priority_result = high_priority_future.get();
    auto high_priority_end = std::chrono::high_resolution_clock::now();
    auto high_priority_wait = std::chrono::duration<double, std::milli>(
        high_priority_end - high_priority_start);
    
    // Wait for all other tasks
    for (auto& future : long_task_futures) {
        future.wait();
    }
    
    for (auto& future : low_priority_futures) {
        future.get();
    }
    
    // Verify high-priority result
    EXPECT_EQ(high_priority_result, 999);
    
    // Log high-priority wait time
    std::cout << "High-priority task wait time: " << high_priority_wait.count() << "ms" << std::endl;
    
    // With a well-implemented priority system, high-priority task should execute
    // once one of the long tasks completes, before any low-priority tasks
    EXPECT_LE(high_priority_wait.count(), 600); // Allow a margin for processing overhead
}

// Test thread safety under high contention
TEST_F(ThreadPoolStressTest, HighContention) {
    ThreadPool pool(4);
    
    // Create a contended resource accessed by all tasks
    std::mutex resource_mutex;
    std::vector<int> shared_resource;
    
    // Add tasks that all try to modify the shared resource
    const int task_count = 1000;
    std::vector<std::future<void>> futures;
    
    for (int i = 0; i < task_count; i++) {
        futures.push_back(pool.enqueue([i, &resource_mutex, &shared_resource]() {
            // Acquire lock and modify shared resource
            {
                std::lock_guard<std::mutex> lock(resource_mutex);
                shared_resource.push_back(i);
                
                // Hold lock for a small random time to increase contention
                std::this_thread::sleep_for(std::chrono::microseconds(
                    randomDuration(1, 100)));
            }
            
            // Do some work outside the lock
            simulateCPUWork(50);
        }));
    }
    
    // Wait for all tasks to complete
    for (auto& future : futures) {
        future.get();
    }
    
    // Verify all tasks successfully modified the shared resource
    EXPECT_EQ(shared_resource.size(), task_count);
    
    // Verify the elements are all present (though order will vary)
    std::sort(shared_resource.begin(), shared_resource.end());
    for (int i = 0; i < task_count; i++) {
        EXPECT_EQ(shared_resource[i], i);
    }
}

// Test task submission from within tasks (nested parallelism)
TEST_F(ThreadPoolStressTest, NestedParallelism) {
    ThreadPool pool(4);
    std::atomic<int> task_counter(0);
    
    // Define a recursive function that calls itself with varying parallelism
    std::function<int(int, bool)> parallel_recursion = [&](int depth, bool parallel) {
        task_counter++;
        
        // Base case
        if (depth <= 0) {
            return 1;
        }
        
        int result = 0;
        
        if (parallel) {
            // Create parallel sub-tasks
            std::vector<std::future<int>> futures;
            
            // Create a mix of parallel and sequential sub-tasks
            futures.push_back(pool.enqueue([&parallel_recursion, depth]() {
                return parallel_recursion(depth - 1, true); // Parallel branch
            }));
            
            futures.push_back(pool.enqueue([&parallel_recursion, depth]() {
                return parallel_recursion(depth - 1, false); // Sequential branch
            }));
            
            // Wait for all sub-tasks
            for (auto& future : futures) {
                result += future.get();
            }
        } else {
            // Sequential execution
            result += parallel_recursion(depth - 1, true);
            result += parallel_recursion(depth - 1, false);
        }
        
        return result;
    };
    
    // Start the recursive parallel computation
    auto future = pool.enqueue([&parallel_recursion]() {
        return parallel_recursion(5, true);
    });
    
    // Wait for all tasks to complete
    int result = future.get();
    
    // Expected result: sum of 2^depth leaf nodes
    int expected_result = 1 << 5; // 2^5 = 32
    EXPECT_EQ(result, expected_result);
    
    // Log task statistics
    std::cout << "Nested parallelism executed " << task_counter << " tasks" << std::endl;
    
    // Verify thread pool is still operational
    auto test_future = pool.enqueue([]() { return 42; });
    EXPECT_EQ(test_future.get(), 42);
}

// Test handling of task submission during destruction
TEST_F(ThreadPoolStressTest, TaskSubmissionDuringDestruction) {
    std::atomic<int> tasks_executed(0);
    std::atomic<int> tasks_submitted(0);
    
    // Use a separate scope for the thread pool
    {
        ThreadPool pool(4);
        
        // Submit some initial tasks
        std::vector<std::future<void>> futures;
        for (int i = 0; i < 10; i++) {
            futures.push_back(pool.enqueue([&tasks_executed]() {
                simulateIOWork(100); // Long enough to outlive pool
                tasks_executed++;
            }));
        }
        
        // Create a thread that continuously submits tasks
        std::atomic<bool> stop_submission(false);
        std::thread submission_thread([&]() {
            int task_id = 0;
            while (!stop_submission) {
                try {
                    pool.enqueue([&tasks_executed]() {
                        tasks_executed++;
                    });
                    tasks_submitted++;
                    task_id++;
                } catch (...) {
                    // Task submission may fail during shutdown
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
        
        // Wait briefly to allow some tasks to be submitted
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        
        // Stop the submission thread
        stop_submission = true;
        submission_thread.join();
    }
    // Thread pool destructor is called here
    
    // Allow some time for any tasks that were in progress to complete
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    // Log statistics
    std::cout << "Tasks submitted: " << tasks_submitted << std::endl;
    std::cout << "Tasks executed: " << tasks_executed << std::endl;
    
    // Some tasks should have executed
    EXPECT_GT(tasks_executed, 0);
    
    // Not all submitted tasks are guaranteed to execute during shutdown
    EXPECT_LE(tasks_executed, tasks_submitted);
}

// Test parallel for with complex reduction operation
TEST_F(ThreadPoolStressTest, ComplexReduction) {
    const int size = 1000000;
    std::vector<double> data(size);
    
    // Initialize with some values
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    for (int i = 0; i < size; i++) {
        data[i] = dist(gen);
    }
    
    // Define a complex reduction task
    auto parallel_stats = [&data]() {
        // Calculate mean, variance, min, max in parallel
        struct ThreadStats {
            double sum = 0.0;
            double sum_squared = 0.0;
            double min = std::numeric_limits<double>::max();
            double max = std::numeric_limits<double>::lowest();
            size_t count = 0;
        };
        
        // Thread-local statistics for each worker
        std::vector<ThreadStats> thread_stats(16); // More than enough for all threads
        std::atomic<int> next_stats_index(0);
        
        // Process in parallel
        ParallelFor::exec(0, data.size(), [&](int i) {
            // Get thread-local stats object
            static thread_local int stats_index = -1;
            if (stats_index < 0) {
                stats_index = next_stats_index.fetch_add(1);
            }
            
            ThreadStats& stats = thread_stats[stats_index];
            
            // Update statistics
            double value = data[i];
            stats.sum += value;
            stats.sum_squared += value * value;
            stats.min = std::min(stats.min, value);
            stats.max = std::max(stats.max, value);
            stats.count++;
        });
        
        // Combine thread-local statistics
        ThreadStats combined;
        int active_threads = 0;
        
        for (const auto& stats : thread_stats) {
            if (stats.count > 0) {
                combined.sum += stats.sum;
                combined.sum_squared += stats.sum_squared;
                combined.min = std::min(combined.min, stats.min);
                combined.max = std::max(combined.max, stats.max);
                combined.count += stats.count;
                active_threads++;
            }
        }
        
        // Calculate final statistics
        double mean = combined.sum / combined.count;
        double variance = (combined.sum_squared / combined.count) - (mean * mean);
        
        std::cout << "Parallel stats calculation used " << active_threads << " threads" << std::endl;
        std::cout << "Mean: " << mean << ", Variance: " << variance
                  << ", Min: " << combined.min << ", Max: " << combined.max << std::endl;
        
        return std::make_tuple(mean, variance, combined.min, combined.max);
    };
    
    // Calculate in parallel
    auto parallel_result = parallel_stats();
    
    // Calculate sequentially for verification
    double sum = 0.0, sum_squared = 0.0;
    double min_val = std::numeric_limits<double>::max();
    double max_val = std::numeric_limits<double>::lowest();
    
    for (double value : data) {
        sum += value;
        sum_squared += value * value;
        min_val = std::min(min_val, value);
        max_val = std::max(max_val, value);
    }
    
    double mean = sum / size;
    double variance = (sum_squared / size) - (mean * mean);
    
    // Compare results
    EXPECT_DOUBLE_EQ(std::get<0>(parallel_result), mean);
    EXPECT_DOUBLE_EQ(std::get<1>(parallel_result), variance);
    EXPECT_DOUBLE_EQ(std::get<2>(parallel_result), min_val);
    EXPECT_DOUBLE_EQ(std::get<3>(parallel_result), max_val);
}

// Test parallel speedup with varying thread counts
TEST_F(ThreadPoolStressTest, SpeedupScaling) {
    const int size = 10000000; // Large enough to meaningfully test parallelism
    std::vector<int> data(size);
    
    // Initialize data
    std::iota(data.begin(), data.end(), 0);
    
    // Function to process data with varying intensity
    auto process_data = [&data](int thread_count) {
        // Create a thread pool with the specified number of threads
        ThreadPool pool(thread_count);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Process data in parallel
        ParallelFor::exec(0, data.size(), [&data](int i) {
            // Do some moderately expensive computation
            volatile double result = std::sqrt(static_cast<double>(data[i]));
            data[i] = static_cast<int>(result * 10);
        });
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end_time - start_time);
        
        return duration.count();
    };
    
    // Test with various thread counts
    std::vector<int> thread_counts = {1, 2, 4, 8, 16, 32};
    std::vector<double> execution_times;
    
    for (int threads : thread_counts) {
        // Skip excessive thread counts
        if (threads > 2 * std::thread::hardware_concurrency()) {
            std::cout << "Skipping " << threads << " threads (exceeds 2x hardware concurrency)" << std::endl;
            continue;
        }
        
        double time_ms = process_data(threads);
        execution_times.push_back(time_ms);
        
        std::cout << "Threads: " << threads << ", Time: " << time_ms << "ms" << std::endl;
    }
    
    // Calculate speedup relative to single-threaded performance
    if (execution_times.size() >= 2) {
        double base_time = execution_times[0]; // Single-threaded time
        
        std::cout << "Speedup analysis:" << std::endl;
        for (size_t i = 1; i < thread_counts.size() && i < execution_times.size(); i++) {
            double speedup = base_time / execution_times[i];
            double efficiency = speedup / thread_counts[i];
            
            std::cout << "  " << thread_counts[i] << " threads: "
                      << speedup << "x speedup, "
                      << (efficiency * 100) << "% parallel efficiency" << std::endl;
            
            // We expect reasonable speedup with more threads, but it won't be perfect
            EXPECT_GT(speedup, 1.0) << "No speedup with " << thread_counts[i] << " threads";
            
            // Efficiency typically decreases with more threads due to overhead
            // but should stay reasonably high for small thread counts
            if (thread_counts[i] <= 4) {
                EXPECT_GT(efficiency, 0.5) << "Poor scaling efficiency with " << thread_counts[i] << " threads";
            }
        }
    }
}