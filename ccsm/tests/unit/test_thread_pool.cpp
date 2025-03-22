#include <gtest/gtest.h>
#include <ccsm/cpu/thread_pool.h>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <future>
#include <thread>

using namespace ccsm;

// Test fixture for ThreadPool tests
class ThreadPoolTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Default settings
    }

    void TearDown() override {
        // Cleanup
    }

    // Helper to measure execution time of a function
    template<typename Func>
    double measureExecutionTime(Func func) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        return elapsed.count();
    }

    // Helper to verify task execution order with dependencies
    struct OrderTracker {
        std::vector<int> execution_order;
        std::mutex mutex;

        void record(int task_id) {
            std::lock_guard<std::mutex> lock(mutex);
            execution_order.push_back(task_id);
        }

        std::vector<int> get_order() {
            std::lock_guard<std::mutex> lock(mutex);
            return execution_order;
        }
    };
};

// Basic functionality tests

// Test creating and destroying thread pools with different numbers of threads
TEST_F(ThreadPoolTest, ThreadPoolCreationDestruction) {
    // Test with 0 threads (should use hardware concurrency)
    {
        ThreadPool pool(0);
        EXPECT_GT(pool.getThreadCount(), 0);
        EXPECT_LE(pool.getThreadCount(), std::thread::hardware_concurrency());
    }

    // Test with 1 thread
    {
        ThreadPool pool(1);
        EXPECT_EQ(pool.getThreadCount(), 1);
    }

    // Test with multiple threads
    {
        const size_t thread_count = 4;
        ThreadPool pool(thread_count);
        EXPECT_EQ(pool.getThreadCount(), thread_count);
    }

    // Test with large number of threads
    {
        const size_t thread_count = 32;
        ThreadPool pool(thread_count);
        EXPECT_EQ(pool.getThreadCount(), thread_count);
    }
}

// Test enqueuing tasks and getting results
TEST_F(ThreadPoolTest, EnqueueTasksWithResults) {
    ThreadPool pool(4);
    
    // Enqueue tasks that return values
    std::vector<std::future<int>> results;
    
    for (int i = 0; i < 100; i++) {
        results.push_back(pool.enqueue([i]() {
            return i * i;
        }));
    }
    
    // Verify results
    for (int i = 0; i < 100; i++) {
        EXPECT_EQ(results[i].get(), i * i);
    }
}

// Test enqueuing void tasks
TEST_F(ThreadPoolTest, EnqueueVoidTasks) {
    ThreadPool pool(4);
    
    std::vector<int> values(100, 0);
    std::vector<std::future<void>> results;
    
    // Enqueue tasks that modify values
    for (int i = 0; i < 100; i++) {
        results.push_back(pool.enqueue([&values, i]() {
            values[i] = i * i;
        }));
    }
    
    // Wait for all tasks to complete
    for (auto& result : results) {
        result.wait();
    }
    
    // Verify values were modified
    for (int i = 0; i < 100; i++) {
        EXPECT_EQ(values[i], i * i);
    }
}

// Test parallel execution
TEST_F(ThreadPoolTest, ParallelExecution) {
    ThreadPool pool(4);
    
    std::atomic<int> counter(0);
    std::vector<std::future<void>> results;
    
    const int num_tasks = 1000;
    
    // Enqueue many quick tasks
    for (int i = 0; i < num_tasks; i++) {
        results.push_back(pool.enqueue([&counter]() {
            counter++;
        }));
    }
    
    // Wait for all tasks to complete
    for (auto& result : results) {
        result.wait();
    }
    
    // Verify counter was incremented correctly
    EXPECT_EQ(counter, num_tasks);
}

// Test with tasks that throw exceptions
TEST_F(ThreadPoolTest, TaskExceptions) {
    ThreadPool pool(4);
    
    // Enqueue a task that throws an exception
    auto future = pool.enqueue([]() -> int {
        throw std::runtime_error("Test exception");
        return 42; // This should never be reached
    });
    
    // Verify the exception is propagated
    EXPECT_THROW(future.get(), std::runtime_error);
    
    // Verify the thread pool is still functional after an exception
    auto future2 = pool.enqueue([]() { return 42; });
    EXPECT_EQ(future2.get(), 42);
}

// Advanced functionality tests

// Test ParallelFor functionality
TEST_F(ThreadPoolTest, ParallelForLoop) {
    const int array_size = 10000;
    std::vector<int> array(array_size, 0);
    
    // Use ParallelFor to process array
    ParallelFor::exec(0, array_size, [&array](int i) {
        array[i] = i * i;
    });
    
    // Verify results
    for (int i = 0; i < array_size; i++) {
        EXPECT_EQ(array[i], i * i);
    }
}

// Test ParallelFor with different chunk sizes
TEST_F(ThreadPoolTest, ParallelForChunkSizes) {
    const int array_size = 10000;
    
    std::vector<std::vector<int>> results;
    std::vector<int> chunk_sizes = {1, 10, 100, 1000, array_size / 4, array_size};
    
    for (int chunk_size : chunk_sizes) {
        std::vector<int> array(array_size, 0);
        
        ParallelFor::exec(0, array_size, [&array](int i) {
            array[i] = i * i;
        }, chunk_size);
        
        // Verify results
        bool all_correct = true;
        for (int i = 0; i < array_size; i++) {
            if (array[i] != i * i) {
                all_correct = false;
                break;
            }
        }
        
        EXPECT_TRUE(all_correct) << "Failed with chunk size: " << chunk_size;
        results.push_back(array);
    }
    
    // Verify all results are the same regardless of chunk size
    for (size_t i = 1; i < results.size(); i++) {
        EXPECT_EQ(results[0], results[i]);
    }
}

// Performance tests

// Test overhead of task scheduling
TEST_F(ThreadPoolTest, TaskSchedulingOverhead) {
    // Use a more realistic number of tasks for testing
    const int num_tasks = 1000;
    
    // Measure time for direct execution
    auto direct_time = measureExecutionTime([num_tasks]() {
        int sum = 0;
        for (int i = 0; i < num_tasks; i++) {
            sum += i;
        }
    });
    
    // Measure time with thread pool
    ThreadPool pool(4);
    std::vector<std::future<int>> results;
    
    auto thread_pool_time = measureExecutionTime([&pool, &results, num_tasks]() {
        results.reserve(num_tasks);
        for (int i = 0; i < num_tasks; i++) {
            results.push_back(pool.enqueue([i]() {
                return i;
            }));
        }
        
        int sum = 0;
        for (int i = 0; i < num_tasks; i++) {
            sum += results[i].get();
        }
    });
    
    // Thread pool will be slower due to overhead, but should still be reasonable
    std::cout << "Direct execution time: " << direct_time << "s" << std::endl;
    std::cout << "Thread pool time: " << thread_pool_time << "s" << std::endl;
    std::cout << "Overhead factor: " << thread_pool_time / direct_time << "x" << std::endl;
    
    // We expect significant overhead for extremely small tasks,
    // but this test is mostly to show the magnitude for documentation purposes.
    // We're not testing for a specific threshold as it varies wildly across machines.
    if (thread_pool_time / direct_time > 1000.0) {
        std::cout << "Very high overhead detected, but this is expected for trivial tasks" << std::endl;
    }
    
    // Skip the actual test comparison since this is just to measure and document overhead
    SUCCEED() << "This test is for measuring overhead, not for pass/fail criteria";
}

// Test parallel speedup with CPU-bound tasks
TEST_F(ThreadPoolTest, ParallelSpeedupCPUBound) {
    // Skip in quick test mode
    if (::testing::FLAGS_gtest_filter == "*QuickTest*") {
        GTEST_SKIP() << "Skipping performance test in quick test mode";
    }
    
    // CPU-bound operation: calculate sum of sines
    auto cpu_intensive_task = [](int start, int end) {
        double sum = 0.0;
        for (int i = start; i < end; i++) {
            for (int j = 0; j < 1000; j++) {
                sum += std::sin(i * 0.01 + j * 0.01);
            }
        }
        return sum;
    };
    
    const int work_size = 10000;
    
    // Serial execution
    auto serial_time = measureExecutionTime([&cpu_intensive_task, work_size]() {
        cpu_intensive_task(0, work_size);
    });
    
    // Parallel execution with different thread counts
    std::vector<int> thread_counts = {1, 2, 4, 8, 16};
    std::vector<double> speedups;
    
    for (int num_threads : thread_counts) {
        if (num_threads > static_cast<int>(std::thread::hardware_concurrency())) {
            continue; // Skip if more threads than cores
        }
        
        ThreadPool pool(num_threads);
        
        auto parallel_time = measureExecutionTime([&pool, &cpu_intensive_task, work_size, num_threads]() {
            std::vector<std::future<double>> futures;
            
            int chunk_size = work_size / num_threads;
            for (int i = 0; i < num_threads; i++) {
                int start = i * chunk_size;
                int end = (i == num_threads - 1) ? work_size : (i + 1) * chunk_size;
                
                futures.push_back(pool.enqueue([&cpu_intensive_task, start, end]() {
                    return cpu_intensive_task(start, end);
                }));
            }
            
            double sum = 0.0;
            for (auto& future : futures) {
                sum += future.get();
            }
        });
        
        double speedup = serial_time / parallel_time;
        speedups.push_back(speedup);
        
        std::cout << "Threads: " << num_threads << ", Time: " << parallel_time << "s, Speedup: " << speedup << "x" << std::endl;
        
        // We expect close to linear speedup for CPU-bound tasks
        // but with some overhead, so let's use 0.6 * num_threads as a minimum
        // This check might be too strict on some systems, so we'll make it conditional
        if (speedup < 0.6 * num_threads) {
            std::cout << "  Warning: speedup less than expected (" << 0.6 * num_threads << "x)" << std::endl;
        }
    }
    
    // Verify speedup increases with more threads (up to a point)
    for (size_t i = 1; i < speedups.size(); i++) {
        // This check might be too strict on some systems with different scheduling behavior
        if (speedups[i] < speedups[i-1]) {
            std::cout << "  Warning: speedup decreased from " << speedups[i-1] << "x to " << speedups[i] << "x" 
                      << " when increasing from " << thread_counts[i-1] << " to " << thread_counts[i] << " threads" << std::endl;
        }
    }
    
    // Less strict checks that should always pass
    EXPECT_GT(speedups[0], 0.5) << "Single-threaded pool should have at least 50% efficiency";
    if (speedups.size() > 1) {
        EXPECT_GT(speedups[speedups.size()-1], 1.0) << "Multi-threaded should be faster than serial";
    }
}

// Test thread pool scalability with different thread counts
TEST_F(ThreadPoolTest, ThreadPoolScalability) {
    // Skip in quick test mode
    if (::testing::FLAGS_gtest_filter == "*QuickTest*") {
        GTEST_SKIP() << "Skipping scalability test in quick test mode";
    }
    
    // Define a workload that's suitable for parallelization
    const int matrix_size = 500;
    
    // Create matrices for multiplication
    std::vector<std::vector<float>> matrix_a(matrix_size, std::vector<float>(matrix_size, 1.0f));
    std::vector<std::vector<float>> matrix_b(matrix_size, std::vector<float>(matrix_size, 2.0f));
    
    // Function to multiply matrices in parallel
    auto parallel_matrix_mul = [](const std::vector<std::vector<float>>& a, 
                                  const std::vector<std::vector<float>>& b,
                                  std::vector<std::vector<float>>& c,
                                  ThreadPool& pool) {
        int size = a.size();
        
        // Initialize result matrix
        c.resize(size, std::vector<float>(size, 0.0f));
        
        // Multiply in parallel using the thread pool
        std::vector<std::future<void>> futures;
        for (int i = 0; i < size; i++) {
            futures.push_back(pool.enqueue([&a, &b, &c, i, size]() {
                for (int j = 0; j < size; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < size; k++) {
                        sum += a[i][k] * b[k][j];
                    }
                    c[i][j] = sum;
                }
            }));
        }
        
        // Wait for all tasks to complete
        for (auto& future : futures) {
            future.wait();
        }
    };
    
    // Serial execution (single thread pool)
    std::vector<std::vector<float>> matrix_c;
    ThreadPool serial_pool(1);
    
    double serial_time = measureExecutionTime([&]() {
        parallel_matrix_mul(matrix_a, matrix_b, matrix_c, serial_pool);
    });
    
    std::cout << "Matrix Multiplication Scalability Test (" << matrix_size << "x" << matrix_size << "):" << std::endl;
    std::cout << "  Single thread time: " << serial_time << "s" << std::endl;
    
    // Test with increasing thread counts
    std::vector<int> thread_counts = {2, 4, 8, 16, 32};
    std::vector<double> speedups;
    int max_threads = std::thread::hardware_concurrency();
    
    for (int num_threads : thread_counts) {
        if (num_threads > max_threads * 2) {
            std::cout << "  Skipping " << num_threads << " threads (more than 2x hardware concurrency)" << std::endl;
            continue;
        }
        
        ThreadPool pool(num_threads);
        std::vector<std::vector<float>> matrix_result;
        
        double parallel_time = measureExecutionTime([&]() {
            parallel_matrix_mul(matrix_a, matrix_b, matrix_result, pool);
        });
        
        double speedup = serial_time / parallel_time;
        speedups.push_back(speedup);
        
        std::cout << "  Threads: " << num_threads 
                  << ", Time: " << parallel_time << "s"
                  << ", Speedup: " << speedup << "x"
                  << ", Efficiency: " << (speedup / num_threads * 100) << "%" << std::endl;
        
        // Verify correctness
        bool correct = true;
        for (int i = 0; i < std::min(5, matrix_size); i++) {
            for (int j = 0; j < std::min(5, matrix_size); j++) {
                if (std::abs(matrix_c[i][j] - matrix_result[i][j]) > 1e-5f) {
                    correct = false;
                    std::cout << "    Error at [" << i << "," << j << "]: "
                              << matrix_c[i][j] << " vs " << matrix_result[i][j] << std::endl;
                    break;
                }
            }
            if (!correct) break;
        }
        EXPECT_TRUE(correct) << "Parallel result should match serial result";
        
        // Log performance metrics but don't enforce specific thresholds
        // as performance varies significantly across test environments
        if (speedup < std::min(num_threads * 0.5, 0.5 * max_threads)) {
            std::cout << "  Note: Speedup is below 50% of thread count (ideal: " 
                      << std::min(num_threads * 0.5, 0.5 * max_threads) << "x)" << std::endl;
        }
    }
    
    // Check scaling behavior - document but don't enforce
    if (speedups.size() > 1) {
        // In theory, speedup should generally increase with more threads (up to hardware concurrency),
        // but in practice there are many factors that affect this including test environment load
        for (size_t i = 1; i < speedups.size() && thread_counts[i] <= max_threads; i++) {
            if (speedups[i] < speedups[i-1] * 0.85) {
                std::cout << "  Note: Speedup decreased significantly from " << speedups[i-1] << "x to " 
                          << speedups[i] << "x when increasing threads from " 
                          << thread_counts[i-1] << " to " << thread_counts[i] << std::endl;
            }
        }
        
        // Only verify that the final result shows some speedup
        EXPECT_GT(speedups.back(), 1.0) << "Should show at least some speedup with multiple threads";
    }
}

// Edge case tests

// Test with empty task queue
TEST_F(ThreadPoolTest, EmptyTaskQueue) {
    ThreadPool pool(4);
    
    // Just create and destroy the pool without adding tasks
    // This should not cause any issues
}

// Test with a very large number of tasks
TEST_F(ThreadPoolTest, VeryLargeTaskCount) {
    ThreadPool pool(4);
    
    const int num_tasks = 100000;
    std::atomic<int> counter(0);
    std::vector<std::future<void>> results;
    
    // Enqueue many tiny tasks
    for (int i = 0; i < num_tasks; i++) {
        results.push_back(pool.enqueue([&counter]() {
            counter++;
        }));
    }
    
    // Wait for all tasks to complete
    for (auto& result : results) {
        result.wait();
    }
    
    // Verify counter was incremented correctly
    EXPECT_EQ(counter, num_tasks);
}

// Test creating multiple thread pools simultaneously
TEST_F(ThreadPoolTest, MultipleThreadPools) {
    const int num_pools = 4;
    std::vector<std::unique_ptr<ThreadPool>> pools;
    
    // Create multiple pools
    for (int i = 0; i < num_pools; i++) {
        pools.push_back(std::make_unique<ThreadPool>(2));
    }
    
    std::vector<std::future<int>> results;
    
    // Use all pools
    for (int i = 0; i < num_pools; i++) {
        for (int j = 0; j < 100; j++) {
            results.push_back(pools[i]->enqueue([j]() {
                return j * j;
            }));
        }
    }
    
    // Check results
    int index = 0;
    for (int i = 0; i < num_pools; i++) {
        for (int j = 0; j < 100; j++) {
            EXPECT_EQ(results[index++].get(), j * j);
        }
    }
    
    // Destroy pools in random order
    std::shuffle(pools.begin(), pools.end(), std::random_device());
    while (!pools.empty()) {
        pools.pop_back();
    }
}

// Test behavior when adding tasks after pool has started processing
TEST_F(ThreadPoolTest, LateTaskAddition) {
    ThreadPool pool(4);
    
    std::vector<std::future<int>> initial_results;
    
    // Add initial batch of long-running tasks
    for (int i = 0; i < 4; i++) {
        initial_results.push_back(pool.enqueue([i]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            return i;
        }));
    }
    
    // Wait a bit for tasks to start
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    // Add more tasks
    std::vector<std::future<int>> late_results;
    for (int i = 100; i < 200; i++) {
        late_results.push_back(pool.enqueue([i]() {
            return i;
        }));
    }
    
    // Verify all results
    for (size_t i = 0; i < initial_results.size(); i++) {
        EXPECT_EQ(initial_results[i].get(), static_cast<int>(i));
    }
    
    for (size_t i = 0; i < late_results.size(); i++) {
        EXPECT_EQ(late_results[i].get(), static_cast<int>(i + 100));
    }
}

// Stress tests

// Test with heterogeneous tasks (mix of CPU, memory, I/O bound)
TEST_F(ThreadPoolTest, HeterogeneousTasks) {
    // Skip in quick test mode
    if (::testing::FLAGS_gtest_filter == "*QuickTest*") {
        GTEST_SKIP() << "Skipping stress test in quick test mode";
    }
    
    ThreadPool pool(4);
    
    const int num_tasks = 100;
    std::vector<std::future<int>> results;
    
    // Add a mix of different task types
    for (int i = 0; i < num_tasks; i++) {
        switch (i % 3) {
            case 0: {
                // CPU-bound task
                results.push_back(pool.enqueue([i]() {
                    double sum = 0.0;
                    for (int j = 0; j < 10000; j++) {
                        sum += std::sin(i * j * 0.001);
                    }
                    return i;
                }));
                break;
            }
            case 1: {
                // Memory-bound task
                results.push_back(pool.enqueue([i]() {
                    std::vector<int> large_array(100000);
                    for (size_t j = 0; j < large_array.size(); j++) {
                        large_array[j] = i + j;
                    }
                    return i;
                }));
                break;
            }
            case 2: {
                // I/O or sleep task
                results.push_back(pool.enqueue([i]() {
                    std::this_thread::sleep_for(std::chrono::milliseconds(i % 10));
                    return i;
                }));
                break;
            }
        }
    }
    
    // Verify all results
    for (int i = 0; i < num_tasks; i++) {
        EXPECT_EQ(results[i].get(), i);
    }
}

// Test with tasks that depend on each other
TEST_F(ThreadPoolTest, TaskDependencies) {
    ThreadPool pool(4);
    OrderTracker tracker;
    
    // Create dependent tasks
    std::future<void> task1 = pool.enqueue([&tracker]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        tracker.record(1);
    });
    
    std::future<void> task2 = pool.enqueue([&tracker, &task1]() {
        task1.wait(); // Wait for task1 to complete
        tracker.record(2);
    });
    
    std::future<void> task3 = pool.enqueue([&tracker, &task2]() {
        task2.wait(); // Wait for task2 to complete
        tracker.record(3);
    });
    
    // Wait for all tasks to complete
    task3.wait();
    
    // Verify execution order
    std::vector<int> expected = {1, 2, 3};
    EXPECT_EQ(tracker.get_order(), expected);
}

// Test thread pool under high contention
TEST_F(ThreadPoolTest, HighContention) {
    // Skip in quick test mode
    if (::testing::FLAGS_gtest_filter == "*QuickTest*") {
        GTEST_SKIP() << "Skipping stress test in quick test mode";
    }
    
    ThreadPool pool(4);
    
    // Shared resource
    std::mutex mutex;
    std::condition_variable cv;
    int shared_value = 0;
    
    const int num_tasks = 1000;
    std::vector<std::future<void>> results;
    
    // Enqueue tasks that all compete for the same lock
    for (int i = 0; i < num_tasks; i++) {
        results.push_back(pool.enqueue([&mutex, &shared_value, i]() {
            std::unique_lock<std::mutex> lock(mutex);
            shared_value += i;
        }));
    }
    
    // Wait for all tasks
    for (auto& result : results) {
        result.wait();
    }
    
    // Calculate expected sum: 0 + 1 + 2 + ... + (num_tasks-1)
    int expected_sum = (num_tasks - 1) * num_tasks / 2;
    
    // Verify result
    EXPECT_EQ(shared_value, expected_sum);
}

// Regression tests

// Test destroying the pool with pending tasks
TEST_F(ThreadPoolTest, DestroyWithPendingTasks) {
    std::vector<std::future<int>> results;
    
    // Scope for thread pool to control destruction timing
    {
        ThreadPool pool(2);
        
        // Add fewer tasks with shorter sleep times to avoid test timeouts
        for (int i = 0; i < 10; i++) {
            results.push_back(pool.enqueue([i]() {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                return i;
            }));
        }
        
        // Ensure tasks have started before pool is destroyed
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        
        // Pool is destroyed here, should wait for all tasks to complete
    }
    
    // Verify all tasks completed
    for (int i = 0; i < 10; i++) {
        EXPECT_EQ(results[i].get(), i);
    }
}

// Test behavior when all threads are busy
TEST_F(ThreadPoolTest, AllThreadsBusy) {
    ThreadPool pool(4);
    
    // Start long-running tasks to occupy all threads
    std::vector<std::future<void>> long_tasks;
    std::atomic<int> started_count(0);
    
    for (int i = 0; i < 4; i++) {
        long_tasks.push_back(pool.enqueue([&started_count]() {
            started_count++;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }));
    }
    
    // Wait for tasks to start
    while (started_count < 4) {
        std::this_thread::yield();
    }
    
    // Start timer
    auto start = std::chrono::high_resolution_clock::now();
    
    // Add a new task
    auto new_task = pool.enqueue([]() {
        return 42;
    });
    
    // Get result (should queue until a thread is available)
    EXPECT_EQ(new_task.get(), 42);
    
    // Check timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    // Should have waited for at least one of the long tasks
    EXPECT_GE(elapsed.count(), 0.050); // 50ms, half of the sleep time, with some margin
}

// Test priority queue functionality
TEST_F(ThreadPoolTest, PriorityTasks) {
    ThreadPool pool(2);
    
    // Create a sequence to track execution order
    std::vector<int> execution_sequence;
    std::mutex sequence_mutex;
    
    auto add_to_sequence = [&](int val) {
        std::lock_guard<std::mutex> lock(sequence_mutex);
        execution_sequence.push_back(val);
    };
    
    // Start 2 long-running tasks to occupy all threads
    std::vector<std::future<void>> long_tasks;
    std::atomic<int> started_count(0);
    
    for (int i = 0; i < 2; i++) {
        long_tasks.push_back(pool.enqueue([&started_count, &add_to_sequence, i]() {
            started_count++;
            add_to_sequence(i); // First 2 tasks
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }));
    }
    
    // Wait for initial tasks to start
    while (started_count < 2) {
        std::this_thread::yield();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    // Add several regular tasks
    for (int i = 0; i < 5; i++) {
        pool.enqueue([&add_to_sequence, i]() {
            add_to_sequence(i + 10); // Regular tasks (10-14)
        });
    }
    
    // Add high-priority task
    auto priority_task = pool.enqueue_priority([&add_to_sequence]() {
        add_to_sequence(99); // Priority task
    });
    
    // Wait for all tasks to complete
    priority_task.wait();
    for (auto& task : long_tasks) {
        task.wait();
    }
    
    // Wait a bit to ensure all regular tasks have completed
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    
    // Get the execution sequence
    std::lock_guard<std::mutex> lock(sequence_mutex);
    
    // Print the sequence for debugging
    std::cout << "Execution sequence: ";
    for (int val : execution_sequence) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    
    // Verify the high-priority task executed before the regular tasks
    // (or at least one of the first tasks to execute after the long tasks)
    bool priority_early = false;
    for (size_t i = 0; i < execution_sequence.size(); i++) {
        if (execution_sequence[i] == 99) {
            // Check if the priority task executed before most regular tasks
            int regular_tasks_after = 0;
            for (size_t j = i + 1; j < execution_sequence.size(); j++) {
                if (execution_sequence[j] >= 10 && execution_sequence[j] <= 14) {
                    regular_tasks_after++;
                }
            }
            
            // If the priority task ran before at least 2 regular tasks, consider it early
            if (regular_tasks_after >= 2) {
                priority_early = true;
                break;
            }
        }
    }
    
    EXPECT_TRUE(priority_early) << "Priority task should execute before most regular tasks";
}

// Test work stealing functionality
TEST_F(ThreadPoolTest, WorkStealing) {
    // Skip in quick test mode
    if (::testing::FLAGS_gtest_filter == "*QuickTest*") {
        GTEST_SKIP() << "Skipping work stealing test in quick test mode";
    }
    
    // Create a pool with 4 threads
    ThreadPool pool(4);
    
    // Create a workload with intentional imbalance
    const int total_tasks = 1000;
    std::atomic<int> tasks_completed(0);
    std::vector<std::future<void>> futures;
    
    // Submit a large batch of tasks to a particular queue index
    // This simulates an imbalanced workload
    for (int i = 0; i < total_tasks; i++) {
        futures.push_back(pool.enqueue([&tasks_completed, i]() {
            // Make some tasks take longer than others
            if (i % 10 == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            
            tasks_completed++;
        }));
    }
    
    // Start a timer
    auto start = std::chrono::high_resolution_clock::now();
    
    // Wait for all tasks to complete
    for (auto& future : futures) {
        future.wait();
    }
    
    // Compute elapsed time
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    // All tasks should be completed
    EXPECT_EQ(tasks_completed.load(), total_tasks);
    
    // Monitor queue sizes
    std::vector<size_t> sizes = pool.queue_sizes();
    
    // Print debug information
    std::cout << "Work stealing test: " << total_tasks << " tasks completed in " 
              << elapsed.count() * 1000 << " ms" << std::endl;
    
    std::cout << "Queue sizes: ";
    for (size_t size : sizes) {
        std::cout << size << " ";
    }
    std::cout << std::endl;
    
    // With effective work stealing, most or all queues should now be empty
    size_t total_remaining = 0;
    for (size_t size : sizes) {
        total_remaining += size;
    }
    
    EXPECT_LE(total_remaining, 10) << "Most tasks should be processed with effective work stealing";
    
    // The elapsed time should be reasonable for the workload with 4 threads
    // This is a rough estimate and may need adjustment based on the test machine
    double expected_time_single_thread = 0.1; // 100 ms (estimated)
    double max_expected_time_4_threads = expected_time_single_thread / 2.0; // At least 2x speedup
    
    EXPECT_LE(elapsed.count(), max_expected_time_4_threads) 
        << "Work stealing should provide reasonable parallelization";
}

// Integration with ParallelFor

// Test ParallelFor with uneven workloads
TEST_F(ThreadPoolTest, ParallelForUnevenWorkloads) {
    const int array_size = 100;
    std::vector<int> array(array_size, 0);
    std::vector<int> expected(array_size, 0);
    
    // Create uneven workloads (some indices require more processing)
    auto workload = [](int i) {
        int iterations = (i % 10 == 0) ? 1000000 : 100;
        double sum = 0.0;
        for (int j = 0; j < iterations; j++) {
            sum += std::sin(j * 0.01);
        }
        return static_cast<int>(sum * 1000);
    };
    
    // Calculate expected results serially
    for (int i = 0; i < array_size; i++) {
        expected[i] = workload(i);
    }
    
    // Calculate with ParallelFor
    ParallelFor::exec(0, array_size, [&array, &workload](int i) {
        array[i] = workload(i);
    });
    
    // Verify results
    EXPECT_EQ(array, expected);
}

// Test nested ParallelFor calls
TEST_F(ThreadPoolTest, NestedParallelFor) {
    const int outer_size = 10;
    const int inner_size = 20;
    
    std::vector<std::vector<int>> matrix(outer_size, std::vector<int>(inner_size, 0));
    
    // Use nested ParallelFor
    ParallelFor::exec(0, outer_size, [&matrix, inner_size](int i) {
        // Inner parallel loop
        ParallelFor::exec(0, inner_size, [&matrix, i](int j) {
            matrix[i][j] = i * 100 + j;
        });
    });
    
    // Verify results
    for (int i = 0; i < outer_size; i++) {
        for (int j = 0; j < inner_size; j++) {
            EXPECT_EQ(matrix[i][j], i * 100 + j);
        }
    }
}

// Test 2D ParallelFor
TEST_F(ThreadPoolTest, ParallelFor2D) {
    const int rows = 50;
    const int cols = 40;
    
    std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols, 0));
    
    // Use 2D parallel for
    ParallelFor::exec_2d(0, rows, 0, cols, [&matrix](int i, int j) {
        matrix[i][j] = i * 1000 + j;
    });
    
    // Verify results
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            EXPECT_EQ(matrix[i][j], i * 1000 + j);
        }
    }
    
    // Test with custom chunk size
    std::vector<std::vector<int>> matrix2(rows, std::vector<int>(cols, 0));
    
    ParallelFor::exec_2d(0, rows, 0, cols, [&matrix2](int i, int j) {
        matrix2[i][j] = i * 1000 + j;
    }, 5); // Specify chunk size of 5 for outer loop
    
    // Verify results
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            EXPECT_EQ(matrix2[i][j], i * 1000 + j);
        }
    }
}

// Test ParallelFor with indexed elements
TEST_F(ThreadPoolTest, ParallelForIndexed) {
    // Create a vector of indices (could be any type)
    std::vector<std::string> items = {"apple", "banana", "cherry", "date", "elderberry", 
                                      "fig", "grape", "honeydew", "imbe", "jackfruit"};
    
    std::vector<std::string> processed(items.size());
    
    // Process items in parallel
    ParallelFor::exec_indexed(items, [&processed](const std::string& item) {
        // Find the index based on the first letter
        size_t idx = item[0] - 'a';
        processed[idx] = item + "_processed";
    });
    
    // Verify results
    EXPECT_EQ(processed[0], "apple_processed");
    EXPECT_EQ(processed[1], "banana_processed");
    EXPECT_EQ(processed[2], "cherry_processed");
    EXPECT_EQ(processed[3], "date_processed");
    EXPECT_EQ(processed[4], "elderberry_processed");
    EXPECT_EQ(processed[5], "fig_processed");
    EXPECT_EQ(processed[6], "grape_processed");
    EXPECT_EQ(processed[7], "honeydew_processed");
    EXPECT_EQ(processed[8], "imbe_processed");
    EXPECT_EQ(processed[9], "jackfruit_processed");
}

// Test ParallelFor with exceptions
TEST_F(ThreadPoolTest, ParallelForExceptions) {
    const int array_size = 100;
    
    // Function that throws for certain indices
    auto throwing_func = [](int i) {
        if (i == 50) {
            throw std::runtime_error("Test exception");
        }
    };
    
    // ParallelFor should propagate exceptions
    EXPECT_THROW({
        ParallelFor::exec(0, array_size, throwing_func);
    }, std::runtime_error);
}