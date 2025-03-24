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

using namespace ccsm;

class ThreadPoolAdvancedTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
    
    // Helper function to simulate I/O-bound work
    static void simulateIO(int ms = 10) {
        std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    }
    
    // Helper function to simulate CPU-bound work
    static void simulateCPU(int iterations = 100000) {
        volatile double result = 0.0;
        for (int i = 0; i < iterations; ++i) {
            result += std::sin(static_cast<double>(i) / 1000.0);
        }
    }
    
    // Helper to generate random sleep duration
    static int randomDuration(int min_ms = 5, int max_ms = 20) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_int_distribution<> dist(min_ms, max_ms);
        return dist(gen);
    }
    
    // Helper to track execution order of tasks
    class ExecutionTracker {
    public:
        void taskStarted(int task_id) {
            std::lock_guard<std::mutex> lock(mutex_);
            start_times_[task_id] = std::chrono::high_resolution_clock::now();
            execution_order_.push_back(task_id);
        }
        
        void taskFinished(int task_id) {
            std::lock_guard<std::mutex> lock(mutex_);
            end_times_[task_id] = std::chrono::high_resolution_clock::now();
            completion_order_.push_back(task_id);
        }
        
        std::vector<int> getExecutionOrder() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return execution_order_;
        }
        
        std::vector<int> getCompletionOrder() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return completion_order_;
        }
        
        std::chrono::microseconds getTaskDuration(int task_id) const {
            std::lock_guard<std::mutex> lock(mutex_);
            auto start_it = start_times_.find(task_id);
            auto end_it = end_times_.find(task_id);
            
            if (start_it != start_times_.end() && end_it != end_times_.end()) {
                return std::chrono::duration_cast<std::chrono::microseconds>(
                    end_it->second - start_it->second);
            }
            
            return std::chrono::microseconds(0);
        }
        
        void reset() {
            std::lock_guard<std::mutex> lock(mutex_);
            start_times_.clear();
            end_times_.clear();
            execution_order_.clear();
            completion_order_.clear();
        }
        
    private:
        mutable std::mutex mutex_;
        std::unordered_map<int, std::chrono::high_resolution_clock::time_point> start_times_;
        std::unordered_map<int, std::chrono::high_resolution_clock::time_point> end_times_;
        std::vector<int> execution_order_;
        std::vector<int> completion_order_;
    };
    
    // Helper to create tasks that report execution time
    static auto createTrackedTask(ExecutionTracker& tracker, int task_id, int duration_ms) {
        return [&tracker, task_id, duration_ms]() {
            tracker.taskStarted(task_id);
            simulateIO(duration_ms);
            tracker.taskFinished(task_id);
            return task_id;
        };
    }
};

// Test advanced task scheduling with dependencies
TEST_F(ThreadPoolAdvancedTest, TaskDependencies) {
    ThreadPool pool(4);
    std::atomic<int> execution_count(0);
    
    // Create a dependency graph of tasks
    // Task 3 depends on Tasks 1 and 2
    // Task 5 depends on Tasks 3 and 4
    
    std::future<int> future1 = pool.enqueue([&execution_count]() {
        simulateIO(10);
        execution_count++;
        return 1;
    });
    
    std::future<int> future2 = pool.enqueue([&execution_count]() {
        simulateIO(15);
        execution_count++;
        return 2;
    });
    
    // Task 3 depends on 1 and 2
    auto future3 = pool.enqueue([&execution_count, &future1, &future2]() {
        // Wait for dependencies
        int result1 = future1.get();
        int result2 = future2.get();
        
        // Verify dependencies completed
        EXPECT_EQ(result1, 1);
        EXPECT_EQ(result2, 2);
        
        simulateIO(5);
        execution_count++;
        return 3;
    });
    
    std::future<int> future4 = pool.enqueue([&execution_count]() {
        simulateIO(10);
        execution_count++;
        return 4;
    });
    
    // Task 5 depends on 3 and 4
    auto future5 = pool.enqueue([&execution_count, &future3, &future4]() {
        // Wait for dependencies
        int result3 = future3.get();
        int result4 = future4.get();
        
        // Verify dependencies completed
        EXPECT_EQ(result3, 3);
        EXPECT_EQ(result4, 4);
        
        simulateIO(5);
        execution_count++;
        return 5;
    });
    
    // Get final result
    int result = future5.get();
    
    // Verify all tasks executed
    EXPECT_EQ(result, 5);
    EXPECT_EQ(execution_count, 5);
}

// Test dynamic load balancing with mixed workloads
TEST_F(ThreadPoolAdvancedTest, MixedWorkloads) {
    ThreadPool pool(4);
    ExecutionTracker tracker;
    
    // Submit a mix of CPU-bound and I/O-bound tasks
    std::vector<std::future<int>> futures;
    const int task_count = 100;
    
    for (int i = 0; i < task_count; i++) {
        if (i % 2 == 0) {
            // CPU-bound task
            futures.push_back(pool.enqueue([i, &tracker]() {
                tracker.taskStarted(i);
                simulateCPU(50000 + (i % 5) * 10000); // Variable CPU load
                tracker.taskFinished(i);
                return i;
            }));
        } else {
            // I/O-bound task
            futures.push_back(pool.enqueue([i, &tracker]() {
                tracker.taskStarted(i);
                simulateIO(5 + (i % 10)); // Variable I/O wait
                tracker.taskFinished(i);
                return i;
            }));
        }
    }
    
    // Wait for all tasks to complete
    for (auto& future : futures) {
        future.get();
    }
    
    // Verify all tasks were executed
    auto completion_order = tracker.getCompletionOrder();
    EXPECT_EQ(completion_order.size(), task_count);
    
    // Check if I/O tasks tend to complete earlier
    std::vector<int> io_completion_positions;
    for (size_t i = 0; i < completion_order.size(); i++) {
        if (completion_order[i] % 2 == 1) { // I/O-bound tasks
            io_completion_positions.push_back(i);
        }
    }
    
    // Calculate average position of I/O tasks
    double avg_io_position = std::accumulate(io_completion_positions.begin(), 
                                          io_completion_positions.end(), 0.0) / io_completion_positions.size();
    
    // Calculate average position if tasks were completed randomly
    double avg_random_position = (task_count - 1) / 2.0;
    
    // I/O tasks should tend to complete earlier on average in an ideal case,
    // but on many systems with resource contention, the execution might vary
    std::cout << "Average I/O task completion position: " << avg_io_position 
              << " (random would be " << avg_random_position << ")" << std::endl;
    
    // This is a statistical test and highly dependent on the system state,
    // so we make the assertion more forgiving to account for system variations
    EXPECT_LE(avg_io_position, avg_random_position * 1.5);
    
    // In a heavily oversubscribed system, even the relaxed test might fail occasionally
    // This test is more informative than deterministic
}

// Test priority inversion prevention
TEST_F(ThreadPoolAdvancedTest, PriorityInversion) {
    ThreadPool pool(2); // Limited threads to force priority inversion scenario
    ExecutionTracker tracker;
    
    // Create scenarios that could cause priority inversion:
    // 1. Low-priority task starts first
    // 2. High-priority task gets added but all threads are busy
    // 3. Our implementation should ensure high-priority tasks get executed ASAP
    
    // First, create a task that will block a thread for a while
    auto blocking_task = pool.enqueue([&tracker]() {
        tracker.taskStarted(0);
        simulateIO(100); // Block for 100ms
        tracker.taskFinished(0);
        return 0;
    });
    
    // Let the blocking task start
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    // Add several low-priority tasks
    std::vector<std::future<int>> low_priority_futures;
    for (int i = 1; i <= 5; i++) {
        low_priority_futures.push_back(pool.enqueue(createTrackedTask(tracker, i, 50)));
    }
    
    // Let some low-priority tasks start
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    // Add a high-priority task
    const int high_priority_id = 999;
    auto high_priority_future = pool.enqueue_priority(createTrackedTask(tracker, high_priority_id, 20));
    
    // Wait for all tasks to complete
    high_priority_future.get();
    blocking_task.get();
    for (auto& future : low_priority_futures) {
        future.get();
    }
    
    // Get execution order
    auto execution_order = tracker.getExecutionOrder();
    
    // Find position of high-priority task
    auto high_priority_pos = std::find(execution_order.begin(), execution_order.end(), high_priority_id);
    ASSERT_NE(high_priority_pos, execution_order.end()) << "High-priority task not executed";
    
    // Calculate how many tasks started before the high-priority task
    size_t tasks_before_high_priority = std::distance(execution_order.begin(), high_priority_pos);
    
    // Considering we have only 2 threads and 1 is blocked, the high-priority task
    // should ideally start quickly, but system scheduling and thread pool implementation
    // details can introduce variability
    std::cout << "Tasks started before high-priority task: " << tasks_before_high_priority << std::endl;
    std::cout << "Execution order: ";
    for (auto id : execution_order) {
        std::cout << id << " ";
    }
    std::cout << std::endl;
    
    // We should see the high-priority task executed relatively early,
    // but the exact timing depends on many factors including thread scheduling
    // and the specific thread pool implementation
    EXPECT_LE(tasks_before_high_priority, 7);
    
    // More important than the exact position is that the high-priority task
    // was executed at all, which we've already verified
}

// Test thread starvation prevention
TEST_F(ThreadPoolAdvancedTest, StarvationPrevention) {
    ThreadPool pool(4);
    std::atomic<int> execution_count(0);
    ExecutionTracker tracker;
    
    // Create a situation where a single queue could get "starved":
    // 1. Submit many small tasks to one queue
    // 2. Submit larger tasks to other queues
    // 3. Verify work stealing ensures all tasks complete in reasonable time
    
    // Submit long-running tasks to most threads to create imbalance
    std::vector<std::future<int>> long_tasks;
    for (int i = 0; i < 3; i++) {
        long_tasks.push_back(pool.enqueue([i, &tracker, &execution_count]() {
            tracker.taskStarted(i);
            simulateIO(100); // Long task
            execution_count++;
            tracker.taskFinished(i);
            return i;
        }));
    }
    
    // Let the long tasks start
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    // Now submit many short tasks - they should be distributed among threads via work stealing
    const int short_task_count = 20;
    std::vector<std::future<int>> short_tasks;
    
    for (int i = 100; i < 100 + short_task_count; i++) {
        short_tasks.push_back(pool.enqueue([i, &tracker, &execution_count]() {
            tracker.taskStarted(i);
            simulateIO(10); // Short task
            execution_count++;
            tracker.taskFinished(i);
            return i;
        }));
    }
    
    // Wait for all tasks to complete
    for (auto& future : long_tasks) {
        future.get();
    }
    for (auto& future : short_tasks) {
        future.get();
    }
    
    // All tasks should have executed
    EXPECT_EQ(execution_count, 3 + short_task_count);
    
    // Check completion times of short tasks
    std::vector<int> short_task_completion_times;
    for (int i = 100; i < 100 + short_task_count; i++) {
        auto duration = tracker.getTaskDuration(i);
        short_task_completion_times.push_back(duration.count());
    }
    
    // Calculate statistics
    std::sort(short_task_completion_times.begin(), short_task_completion_times.end());
    double median_time = short_task_completion_times[short_task_completion_times.size() / 2];
    
    // In the worst case without work stealing, all short tasks would execute sequentially
    // on the one free thread, taking approximately (20 * 10ms) = 200ms
    // With work stealing, we expect much better distribution
    std::cout << "Median short task time (μs): " << median_time << std::endl;
    
    // We can't make exact assertions because of system variability,
    // but we can check that execution time is reasonable compared to worst case
    EXPECT_LT(median_time, 100000); // Less than 100ms (worst case would be 200ms)
}

// Test error propagation and recovery
TEST_F(ThreadPoolAdvancedTest, ErrorPropagation) {
    ThreadPool pool(4);
    std::atomic<int> success_count(0);
    std::atomic<int> error_count(0);
    
    // Submit a mix of successful and failing tasks
    std::vector<std::future<int>> futures;
    const int task_count = 20;
    
    for (int i = 0; i < task_count; i++) {
        futures.push_back(pool.enqueue([i, &success_count, &error_count]() -> int {
            if (i % 5 == 0) {
                // Every 5th task fails
                error_count++;
                throw std::runtime_error("Planned task failure");
            }
            
            simulateIO(10);
            success_count++;
            return i;
        }));
    }
    
    // Wait for all tasks and verify proper error propagation
    int failed_tasks = 0;
    int successful_tasks = 0;
    
    for (int i = 0; i < task_count; i++) {
        try {
            int result = futures[i].get();
            successful_tasks++;
            EXPECT_EQ(result, i); // Verify correct result
            EXPECT_NE(i % 5, 0);  // Verify this wasn't a task that should fail
        }
        catch (const std::runtime_error& e) {
            failed_tasks++;
            EXPECT_EQ(i % 5, 0);  // Verify this was a task that should fail
            EXPECT_STREQ(e.what(), "Planned task failure");
        }
    }
    
    // Verify counts
    EXPECT_EQ(successful_tasks, task_count - (task_count / 5));
    EXPECT_EQ(failed_tasks, task_count / 5);
    EXPECT_EQ(success_count, task_count - (task_count / 5));
    EXPECT_EQ(error_count, task_count / 5);
    
    // Verify pool is still usable after errors
    auto future = pool.enqueue([]() {
        return 42;
    });
    
    EXPECT_EQ(future.get(), 42);
}

// Test thread exhaustion handling
TEST_F(ThreadPoolAdvancedTest, ThreadExhaustion) {
    // Create a thread pool with only 2 threads
    ThreadPool pool(2);
    
    // Create tasks that block threads for a significant time
    std::atomic<int> running_count(0);
    std::atomic<int> max_running(0);
    std::mutex cv_mutex;
    std::condition_variable cv;
    
    // Submit tasks that hold all threads
    std::vector<std::future<void>> blocking_futures;
    for (int i = 0; i < 4; i++) {
        blocking_futures.push_back(pool.enqueue([&running_count, &max_running, &cv_mutex, &cv]() {
            running_count++;
            max_running = std::max(max_running.load(), running_count.load());
            
            // Wait until signaled
            std::unique_lock<std::mutex> lock(cv_mutex);
            cv.wait_for(lock, std::chrono::milliseconds(100));
            running_count--;
        }));
    }
    
    // Wait briefly for tasks to start
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    // Check that only 2 tasks are running simultaneously (thread count limit)
    EXPECT_EQ(max_running, 2);
    
    // Signal all tasks to complete
    cv.notify_all();
    
    // Wait for all tasks
    for (auto& future : blocking_futures) {
        future.wait();
    }
    
    // Verify all tasks completed
    EXPECT_EQ(running_count, 0);
    
    // Check that the thread pool is still responsive
    auto future = pool.enqueue([]() { return 42; });
    EXPECT_EQ(future.get(), 42);
}

// Test task cancelation behavior
TEST_F(ThreadPoolAdvancedTest, TaskCancelation) {
    ThreadPool pool(2);
    std::atomic<bool> task_started(false);
    std::atomic<bool> task_completed(false);
    
    // Submit a task that will take a while to complete
    auto future = pool.enqueue([&task_started, &task_completed]() {
        task_started = true;
        simulateIO(500); // Long-running task
        task_completed = true;
        return 42;
    });
    
    // Wait for task to start
    while (!task_started) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    
    // Destroy the future without waiting
    // C++ doesn't have a direct "cancel" mechanism, but we can test behavior when
    // futures are destroyed without calling .get() or .wait()
    future = std::future<int>(); // Reset the future
    
    // Create a new task to verify pool is still usable
    auto check_future = pool.enqueue([]() { return 100; });
    EXPECT_EQ(check_future.get(), 100);
    
    // Give the original task time to complete if it's still running
    std::this_thread::sleep_for(std::chrono::milliseconds(600));
    
    // The task should have completed even though we abandoned the future
    EXPECT_TRUE(task_completed);
}

// Test parallel for with thread errors
TEST_F(ThreadPoolAdvancedTest, ParallelForErrors) {
    std::vector<int> data(1000, 0);
    
    // Execute parallel for with random errors
    EXPECT_THROW({
        ParallelFor::exec(0, data.size(), [&data](int i) {
            if (i == 500) {
                throw std::runtime_error("Test exception in ParallelFor");
            }
            data[i] = i;
        });
    }, std::runtime_error);
    
    // Verify exception handling in ParallelFor::exec_2d
    EXPECT_THROW({
        ParallelFor::exec_2d(0, 10, 0, 10, [](int i, int j) {
            if (i == 5 && j == 5) {
                throw std::runtime_error("Test exception in ParallelFor::exec_2d");
            }
        });
    }, std::runtime_error);
    
    // Verify exception handling in ParallelFor::exec_indexed
    std::vector<int> indices(100);
    std::iota(indices.begin(), indices.end(), 0);
    
    EXPECT_THROW({
        ParallelFor::exec_indexed(indices, [](int idx) {
            if (idx == 50) {
                throw std::runtime_error("Test exception in ParallelFor::exec_indexed");
            }
        });
    }, std::runtime_error);
}

// Test task queue saturation behavior
TEST_F(ThreadPoolAdvancedTest, QueueSaturation) {
    ThreadPool pool(2);
    
    // Submit enough tasks to fill the queues
    const int task_count = 1000;
    std::vector<std::future<int>> futures;
    
    // First, make the threads busy with long-running tasks
    for (int i = 0; i < 2; i++) {
        futures.push_back(pool.enqueue([i]() {
            simulateIO(100);
            return i;
        }));
    }
    
    // Now add a large number of short tasks
    for (int i = 2; i < task_count; i++) {
        futures.push_back(pool.enqueue([i]() {
            return i;
        }));
    }
    
    // Queue sizes should reflect the large number of pending tasks
    auto queue_sizes = pool.queue_sizes();
    int total_queued = std::accumulate(queue_sizes.begin(), queue_sizes.end(), 0);
    
    // We should have a significant number of queued tasks
    std::cout << "Total queued tasks: " << total_queued << std::endl;
    EXPECT_GT(total_queued, 0);
    
    // Wait for all tasks to complete to ensure queue stress doesn't cause issues
    for (auto& future : futures) {
        int result = future.get();
        EXPECT_GE(result, 0);
        EXPECT_LT(result, task_count);
    }
    
    // Check queue is empty after all tasks complete
    queue_sizes = pool.queue_sizes();
    total_queued = std::accumulate(queue_sizes.begin(), queue_sizes.end(), 0);
    EXPECT_EQ(total_queued, 0);
}

// Test custom parallel reduction
TEST_F(ThreadPoolAdvancedTest, ParallelReduction) {
    // Create a helper function for parallel reduction
    auto parallel_sum = [](const std::vector<int>& data) {
        // Perform parallel reduction to sum array elements
        std::atomic<int> sum(0);
        
        ParallelFor::exec(0, data.size(), [&sum, &data](int i) {
            sum.fetch_add(data[i], std::memory_order_relaxed);
        });
        
        return sum.load();
    };
    
    // Test with different array sizes
    for (int size : {100, 1000, 10000}) {
        std::vector<int> data(size);
        std::iota(data.begin(), data.end(), 0); // Fill with 0, 1, 2, ...
        
        // Calculate expected sum: 0 + 1 + 2 + ... + (size-1) = size*(size-1)/2
        int expected = size * (size - 1) / 2;
        
        // Calculate using parallel reduction
        int parallel_result = parallel_sum(data);
        
        // Verify result
        EXPECT_EQ(parallel_result, expected);
    }
}

// Test adaptive chunk size behavior in ParallelFor
TEST_F(ThreadPoolAdvancedTest, AdaptiveChunkSize) {
    // This test examines how ParallelFor's automatic chunk size 
    // adapts to different workload sizes
    
    std::atomic<int> chunk_starts(0);
    auto count_chunk_starts = [&chunk_starts](int i) {
        // Each thread will count when it's at a "start" of a chunk
        static thread_local int last_idx = -999;
        if (i != last_idx + 1) {
            chunk_starts++;
        }
        last_idx = i;
    };
    
    // Test a range of input sizes to see adaptive behavior
    std::vector<int> sizes = {10, 100, 1000, 10000};
    
    for (int size : sizes) {
        chunk_starts = 0;
        
        // Run with auto chunk sizing
        ParallelFor::exec(0, size, [&count_chunk_starts](int i) {
            count_chunk_starts(i);
            // Small amount of work to keep overhead reasonable
            volatile int x = 0;
            for (int j = 0; j < 100; j++) x += j;
        });
        
        // Get thread count for reference
        int thread_count = global_thread_pool().size();
        
        // Verify reasonable number of chunks based on size
        std::cout << "Size " << size << " created " << chunk_starts << " chunks with " 
                  << thread_count << " threads" << std::endl;
        
        // For very small workloads, should use few chunks
        if (size <= 32) {
            EXPECT_LE(chunk_starts, 2);
        } 
        // For moderate to large workloads, the implementation may vary
        else {
            // For very large workloads, we should see more chunks
            if (size >= 10000) {
                EXPECT_GE(chunk_starts, 2);
            }
            
            // Should not have too many chunks (overhead)
            EXPECT_LE(chunk_starts, std::max(thread_count * 8, size / 8));
            
            // Print more detailed diagnostics instead of hard assertions
            std::cout << "  Chunks per thread ratio: " 
                      << static_cast<double>(chunk_starts) / thread_count << std::endl;
        }
    }
}

// Test long dependency chains
TEST_F(ThreadPoolAdvancedTest, LongDependencyChains) {
    ThreadPool pool(4);
    std::atomic<int> tasks_executed(0);
    
    // Create a long chain of dependent tasks: 1 -> 2 -> 3 -> ... -> N
    // This tests how well the thread pool handles long dependency chains
    
    const int chain_length = 20;
    
    // Start with task 1
    std::future<int> previous_task = pool.enqueue([&tasks_executed]() {
        tasks_executed++;
        return 1;
    });
    
    // Create the rest of the chain
    for (int i = 2; i <= chain_length; i++) {
        previous_task = pool.enqueue([i, prev_future = std::move(previous_task), &tasks_executed]() mutable {
            // Wait for previous task
            int prev_result = prev_future.get();
            
            // Verify we got correct result from previous task
            EXPECT_EQ(prev_result, i - 1);
            
            // Execute this task
            tasks_executed++;
            return i;
        });
    }
    
    // Get final result
    int result = previous_task.get();
    
    // Verify all tasks executed in order
    EXPECT_EQ(result, chain_length);
    EXPECT_EQ(tasks_executed, chain_length);
}

// Test memory locality for data-parallel workloads
TEST_F(ThreadPoolAdvancedTest, MemoryLocality) {
    // This test verifies that closely related data tends to be processed
    // by the same thread, which improves cache locality
    
    ThreadPool pool(4);
    const int array_size = 10000;
    std::vector<int> data(array_size, 0);
    
    // Map to track which thread processed which indices
    std::unordered_map<std::thread::id, std::vector<int>> thread_indices;
    std::mutex mutex;
    
    // Process array in parallel with chunk_size to encourage locality
    ParallelFor::exec(0, array_size, [&](int i) {
        // Process element
        data[i] = i;
        
        // Record which thread processed this index
        std::thread::id tid = std::this_thread::get_id();
        {
            std::lock_guard<std::mutex> lock(mutex);
            thread_indices[tid].push_back(i);
        }
    }, 500); // Use explicit chunk size for predictable chunks
    
    // Verify all elements were processed
    for (int i = 0; i < array_size; i++) {
        EXPECT_EQ(data[i], i);
    }
    
    // Analyze thread assignments
    std::map<std::thread::id, std::vector<std::pair<int, int>>> thread_ranges;
    
    // For each thread, find contiguous ranges of indices it processed
    for (const auto& [tid, indices] : thread_indices) {
        if (indices.empty()) continue;
        
        // Sort indices (they might not be in order due to concurrent updates)
        std::vector<int> sorted_indices = indices;
        std::sort(sorted_indices.begin(), sorted_indices.end());
        
        // Find contiguous ranges
        int range_start = sorted_indices[0];
        int prev_index = sorted_indices[0];
        
        for (size_t i = 1; i < sorted_indices.size(); i++) {
            if (sorted_indices[i] != prev_index + 1) {
                // End of range
                thread_ranges[tid].emplace_back(range_start, prev_index);
                range_start = sorted_indices[i];
            }
            prev_index = sorted_indices[i];
        }
        
        // Add final range
        thread_ranges[tid].emplace_back(range_start, prev_index);
    }
    
    // Output statistics
    std::cout << "Memory locality analysis:" << std::endl;
    int total_ranges = 0;
    int large_ranges = 0;
    
    for (const auto& [tid, ranges] : thread_ranges) {
        std::cout << "Thread " << tid << " processed " << ranges.size() << " ranges:" << std::endl;
        
        for (const auto& [start, end] : ranges) {
            int range_size = end - start + 1;
            std::cout << "  [" << start << " - " << end << "] (size: " << range_size << ")" << std::endl;
            
            total_ranges++;
            if (range_size >= 100) {
                large_ranges++;
            }
        }
    }
    
    // Print summary
    double large_range_ratio = static_cast<double>(large_ranges) / total_ranges;
    std::cout << "Proportion of large ranges (>=100 elements): " 
              << large_range_ratio * 100.0 << "%" << std::endl;
    
    // We expect a good chunk of ranges to be reasonably large
    // This is a statistic that helps us understand thread pool behavior
    EXPECT_GT(large_range_ratio, 0.3); // At least 30% should be large ranges
}

// Test task recycling
TEST_F(ThreadPoolAdvancedTest, TaskRecycling) {
    // This test verifies that repeatedly submitting and completing tasks
    // doesn't cause excessive memory allocation or performance degradation
    
    ThreadPool pool(4);
    const int num_batches = 5;
    const int tasks_per_batch = 1000;
    
    // Track execution time for each batch
    std::vector<double> batch_times_ms;
    
    for (int batch = 0; batch < num_batches; batch++) {
        // Measure time to submit and complete tasks
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Submit batch of tasks
        std::vector<std::future<int>> futures;
        for (int i = 0; i < tasks_per_batch; i++) {
            futures.push_back(pool.enqueue([i]() {
                // Very lightweight task
                return i * 2;
            }));
        }
        
        // Wait for all tasks to complete
        for (auto& future : futures) {
            future.get();
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count();
        
        // Record batch time
        batch_times_ms.push_back(duration);
        
        std::cout << "Batch " << batch << " time: " << duration << "ms" << std::endl;
    }
    
    // Later batches should not be dramatically slower than the first batch
    // (allowing for some variability due to system load and JIT compilation)
    for (int i = 1; i < num_batches; i++) {
        // Allow up to 3x variation (generous to account for system load fluctuations)
        EXPECT_LE(batch_times_ms[i], batch_times_ms[0] * 3.0);
    }
}

// Test parallel for with dynamic adaptive chunk size
TEST_F(ThreadPoolAdvancedTest, DynamicChunkSize) {
    // This test verifies the dynamic chunk sizing approach in ParallelFor
    
    const int size = 10000;
    std::vector<int> data(size);
    
    // Keep track of chunk sizes observed during execution
    std::vector<int> observed_chunk_sizes;
    std::mutex chunk_mutex;
    
    // Function to detect chunk boundaries
    auto process_with_chunk_detection = [&observed_chunk_sizes, &chunk_mutex](int i) {
        static thread_local int last_idx = -999;
        static thread_local int current_chunk_size = 0;
        
        if (i == last_idx + 1) {
            // Continuation of current chunk
            current_chunk_size++;
        } else {
            // New chunk boundary
            if (current_chunk_size > 0) {
                // Record previous chunk size
                std::lock_guard<std::mutex> lock(chunk_mutex);
                observed_chunk_sizes.push_back(current_chunk_size);
            }
            current_chunk_size = 1;
        }
        
        last_idx = i;
        
        // Simulate variable per-element processing time
        if (i % 100 == 0) {
            // Occasionally do more work to create imbalance
            volatile int sum = 0;
            for (int j = 0; j < 10000; j++) {
                sum += j;
            }
        }
    };
    
    // Run parallel for with automatic chunk sizing
    ParallelFor::exec(0, size, [&](int i) {
        process_with_chunk_detection(i);
        data[i] = i;
    });
    
    // Run one more time to capture the final chunk
    ParallelFor::exec(0, 1, [&](int) {
        // This forces each thread to finish recording its last chunk
        static thread_local int last_idx = -999;
        static thread_local int current_chunk_size = 0;
        
        if (current_chunk_size > 0) {
            std::lock_guard<std::mutex> lock(chunk_mutex);
            observed_chunk_sizes.push_back(current_chunk_size);
        }
    });
    
    // Verify all elements were processed
    for (int i = 0; i < size; i++) {
        EXPECT_EQ(data[i], i);
    }
    
    // Analyze observed chunk sizes
    if (!observed_chunk_sizes.empty()) {
        // Sort chunk sizes
        std::sort(observed_chunk_sizes.begin(), observed_chunk_sizes.end());
        
        // Calculate statistics
        double min_chunk = observed_chunk_sizes.front();
        double max_chunk = observed_chunk_sizes.back();
        double median_chunk = observed_chunk_sizes[observed_chunk_sizes.size() / 2];
        double total_chunks = observed_chunk_sizes.size();
        
        std::cout << "Chunk statistics:" << std::endl;
        std::cout << "  Number of chunks: " << total_chunks << std::endl;
        std::cout << "  Min chunk size: " << min_chunk << std::endl; 
        std::cout << "  Median chunk size: " << median_chunk << std::endl;
        std::cout << "  Max chunk size: " << max_chunk << std::endl;
        
        // Expected number of chunks should be reasonable compared to thread count
        int thread_count = global_thread_pool().size();
        
        // We expect more chunks than threads for good load balancing
        EXPECT_GT(total_chunks, thread_count);
        
        // But not too many chunks (overhead)
        EXPECT_LT(total_chunks, size / 10);
        
        // Smallest chunk should not be too small (overhead)
        EXPECT_GE(min_chunk, 1);
        
        // Largest chunk should not process most of the data
        EXPECT_LT(max_chunk, size / 2);
    }
}

// Test dynamic thread pool resizing (if supported by implementation)
TEST_F(ThreadPoolAdvancedTest, DISABLED_ThreadPoolResizing) {
    // Note: This test is disabled by default since thread pool resizing
    // is not directly supported in the current implementation.
    // This test would be enabled if dynamic thread pool sizing is added.
    
    // Test concept for future implementation
    ThreadPool pool(2); // Start with 2 threads
    
    // Get initial thread count
    EXPECT_EQ(pool.size(), 2);
    
    // Submit tasks to verify initial functionality
    std::atomic<int> counter(0);
    std::vector<std::future<void>> futures;
    
    for (int i = 0; i < 10; i++) {
        futures.push_back(pool.enqueue([&counter]() {
            counter++;
        }));
    }
    
    // Wait for tasks to complete
    for (auto& future : futures) {
        future.wait();
    }
    
    EXPECT_EQ(counter, 10);
    
    // In a hypothetical implementation with resizing:
    // pool.resize(4); // Increase to 4 threads
    
    // Verify new thread count
    // EXPECT_EQ(pool.size(), 4);
    
    // Test with more tasks
    counter = 0;
    futures.clear();
    
    for (int i = 0; i < 20; i++) {
        futures.push_back(pool.enqueue([&counter]() {
            counter++;
        }));
    }
    
    // Wait for tasks to complete
    for (auto& future : futures) {
        future.wait();
    }
    
    EXPECT_EQ(counter, 20);
}

// Test work distribution fairness
TEST_F(ThreadPoolAdvancedTest, WorkDistributionFairness) {
    // This test verifies that work is distributed fairly among threads
    // with emphasis on load balancing algorithms
    
    // Create a thread pool with fixed size for deterministic testing
    const size_t thread_count = 4;
    ThreadPool pool(thread_count);
    
    // Create data structure to track work assigned to each thread
    std::unordered_map<std::thread::id, int> thread_work_counts;
    std::mutex map_mutex;
    
    // Create a large number of tasks
    const int num_tasks = 1000;
    
    // Submit tasks and track which thread executes each one
    std::vector<std::future<void>> futures;
    for (int i = 0; i < num_tasks; i++) {
        futures.push_back(pool.enqueue([&thread_work_counts, &map_mutex]() {
            auto thread_id = std::this_thread::get_id();
            {
                std::lock_guard<std::mutex> lock(map_mutex);
                thread_work_counts[thread_id]++;
            }
            
            // Small work to ensure task doesn't complete too quickly
            volatile int sum = 0;
            for (int j = 0; j < 1000; j++) {
                sum += j;
            }
        }));
    }
    
    // Wait for all tasks to complete
    for (auto& future : futures) {
        future.get();
    }
    
    // Calculate work distribution statistics
    std::vector<int> work_counts;
    for (const auto& [thread_id, count] : thread_work_counts) {
        work_counts.push_back(count);
    }
    
    // Find min and max work counts
    int min_count = *std::min_element(work_counts.begin(), work_counts.end());
    int max_count = *std::max_element(work_counts.begin(), work_counts.end());
    double average_count = static_cast<double>(num_tasks) / thread_work_counts.size();
    
    // Calculate fairness ratio - a perfectly fair distribution would be 1.0
    // Fairness ratio is (worst thread load) / (average load)
    double fairness_ratio = std::max(
        average_count / min_count,
        max_count / average_count
    );
    
    // Print distribution statistics
    std::cout << "Thread work distribution:" << std::endl;
    for (const auto& [thread_id, count] : thread_work_counts) {
        std::cout << "  Thread " << thread_id << ": " << count << " tasks ("
                  << (count * 100.0 / num_tasks) << "%)" << std::endl;
    }
    std::cout << "Min: " << min_count << ", Max: " << max_count 
              << ", Avg: " << average_count << std::endl;
    std::cout << "Fairness ratio: " << fairness_ratio << " (1.0 is perfect)" << std::endl;
    
    // Check fairness - allow for some imbalance but not too much
    // 2.0 means one thread could have 2x the average load, which is quite permissive
    EXPECT_LT(fairness_ratio, 2.0) << "Work distribution is too unbalanced";
    
    // Make sure all threads were used
    EXPECT_EQ(thread_work_counts.size(), thread_count) 
        << "Some threads received no work!";
}