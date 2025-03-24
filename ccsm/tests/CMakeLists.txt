enable_testing()

# Find Google Test
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    # Use FetchContent to download and configure GoogleTest
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG main
    )
    FetchContent_MakeAvailable(googletest)
endif()

# Unit tests
file(GLOB TEST_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/unit/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/main_test.cpp"
)

# Make sure our specific test files are included
set(ADDITIONAL_TEST_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/unit/test_tensor_type_promotion.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/unit/test_tensor_serialization.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/unit/test_tensor_quantized_operations.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/unit/test_ggml_advanced_quantization.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/unit/test_ggml_kv_cache_quantization.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/unit/test_quantized_matrix_mul.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/unit/test_ggml_model_quantization.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/unit/test_ggml_model_quantization_memory.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/unit/test_thread_pool.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/unit/test_thread_pool_advanced.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/unit/test_kv_cache_memory_optimization.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/unit/test_ggml_model_memory_optimization.cpp"
)

# Append any missing files
foreach(SRC ${ADDITIONAL_TEST_SOURCES})
    if(NOT "${TEST_SOURCES}" MATCHES "${SRC}")
        list(APPEND TEST_SOURCES "${SRC}")
    endif()
endforeach()

# Make sure test files exist
foreach(SRC ${ADDITIONAL_TEST_SOURCES})
    if(NOT EXISTS "${SRC}")
        get_filename_component(FILENAME "${SRC}" NAME)
        message(WARNING "${FILENAME} not found. Associated tests will not be included.")
    endif()
endforeach()

# Create unit test executable
add_executable(ccsm_unit_tests ${TEST_SOURCES})
target_link_libraries(ccsm_unit_tests PRIVATE ccsm_core GTest::gtest)
target_include_directories(ccsm_unit_tests PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Add tokenizer test
add_executable(test_tokenizer 
    ${CMAKE_CURRENT_SOURCE_DIR}/tokenizer/test_tokenizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/main_test.cpp
)
target_link_libraries(test_tokenizer PRIVATE ccsm_core GTest::gtest)
target_include_directories(test_tokenizer PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Add SIMD optimization tests
add_executable(test_simd 
    ${CMAKE_CURRENT_SOURCE_DIR}/simd/test_simd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/simd/test_simd_matmul.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/simd/test_simd_activation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/simd/test_simd_fused.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/simd/test_simd_quantization.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/simd/test_simd_fused_quantized.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/main_test.cpp
)
target_link_libraries(test_simd PRIVATE ccsm_core GTest::gtest)
target_include_directories(test_simd PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Add a separate target for tensor tests only (for development)
add_executable(test_tensor_only
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/test_tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/test_tensor_type_promotion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/test_tensor_serialization.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/test_tensor_quantized_operations.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/main_test.cpp
)
target_link_libraries(test_tensor_only PRIVATE ccsm_core GTest::gtest)
target_include_directories(test_tensor_only PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Add a separate target for thread pool tests
add_executable(test_thread_pool
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/test_thread_pool.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/main_test.cpp
)
target_link_libraries(test_thread_pool PRIVATE ccsm_core GTest::gtest)
target_include_directories(test_thread_pool PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Add a separate target for advanced thread pool tests
add_executable(test_thread_pool_advanced
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/test_thread_pool.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/test_thread_pool_advanced.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/main_test.cpp
)
target_link_libraries(test_thread_pool_advanced PRIVATE ccsm_core GTest::gtest)
target_include_directories(test_thread_pool_advanced PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Add a separate target for GGML KV cache quantization tests
add_executable(test_ggml_kv_cache
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/test_ggml_kv_cache_quantization.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/main_test.cpp
)
target_link_libraries(test_ggml_kv_cache PRIVATE ccsm_core GTest::gtest)
target_include_directories(test_ggml_kv_cache PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Add a separate target for Generator tests
add_executable(test_generator
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/test_generator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/test_generator_advanced.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/main_test.cpp
)
target_link_libraries(test_generator PRIVATE ccsm_core GTest::gtest)
target_include_directories(test_generator PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Add a separate target for GGML model quantization tests
add_executable(test_ggml_model_quantization
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/test_ggml_model_quantization.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/main_test.cpp
)
target_link_libraries(test_ggml_model_quantization PRIVATE ccsm_core GTest::gtest)
target_include_directories(test_ggml_model_quantization PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Add a separate target for KV cache memory optimization tests
add_executable(test_kv_cache_memory
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/test_kv_cache_memory_optimization.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/main_test.cpp
)
target_link_libraries(test_kv_cache_memory PRIVATE ccsm_core GTest::gtest)
target_include_directories(test_kv_cache_memory PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Add a separate target for GGMLModel memory optimization tests
add_executable(test_model_memory_optimization
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/test_ggml_model_memory_optimization.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/main_test.cpp
)
target_link_libraries(test_model_memory_optimization PRIVATE ccsm_core GTest::gtest)
target_include_directories(test_model_memory_optimization PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Add a separate target for GGML model quantization and memory optimization tests
add_executable(test_model_quantization_memory
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/test_ggml_model_quantization_memory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/main_test.cpp
)
target_link_libraries(test_model_quantization_memory PRIVATE ccsm_core GTest::gtest)
target_include_directories(test_model_quantization_memory PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Add a separate target for quantized matrix multiplication tests
add_executable(test_quantized_matrix_mul
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/test_quantized_matrix_mul.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/main_test.cpp
)
target_link_libraries(test_quantized_matrix_mul PRIVATE ccsm_core GTest::gtest)
target_include_directories(test_quantized_matrix_mul PRIVATE ${CMAKE_SOURCE_DIR}/include)

# We'll add integration tests after defining coverage targets

# Add MLX tests conditionally
if(WITH_MLX)
    file(GLOB MLX_TEST_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/unit/mlx_*.cpp"
    )
    # Only create MLX tests if there are MLX test sources
    if(MLX_TEST_SOURCES)
        add_executable(ccsm_mlx_tests ${MLX_TEST_SOURCES})
        target_link_libraries(ccsm_mlx_tests PRIVATE ccsm_core ccsm_mlx GTest::gtest GTest::gtest_main)
        target_include_directories(ccsm_mlx_tests PRIVATE ${CMAKE_SOURCE_DIR}/include)
    endif()
endif()

# Add the tests to CTest - specify full paths for test binaries
add_test(NAME UnitTests COMMAND ${CMAKE_CURRENT_BINARY_DIR}/ccsm_unit_tests)
add_test(NAME TokenizerTest COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test_tokenizer)
add_test(NAME SIMDTest COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test_simd)
add_test(NAME TensorTest COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test_tensor_only)
add_test(NAME ThreadPoolTest COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test_thread_pool)
add_test(NAME ThreadPoolAdvancedTest COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test_thread_pool_advanced --gtest_filter="ThreadPoolAdvancedTest.*")
add_test(NAME GGMLKVCacheTest COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test_ggml_kv_cache)
add_test(NAME GeneratorTest COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test_generator)
add_test(NAME GGMLModelQuantizationTest COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test_ggml_model_quantization)
add_test(NAME KVCacheMemoryTest COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test_kv_cache_memory)
add_test(NAME ModelMemoryOptimizationTest COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test_model_memory_optimization)
add_test(NAME ModelQuantizationMemoryTest COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test_model_quantization_memory)
add_test(NAME QuantizedMatrixMulTest COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test_quantized_matrix_mul)

if(WITH_MLX AND MLX_TEST_SOURCES)
    add_test(NAME MLXTests COMMAND ccsm_mlx_tests)
endif()

# Setup code coverage targets if coverage is enabled
if(WITH_COVERAGE)
    # Create coverage directory
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/coverage)
    
    # Setup the unit test coverage target
    setup_target_for_coverage(
        NAME coverage_unit
        EXECUTABLE ccsm_unit_tests
        DEPENDENCIES ccsm_unit_tests
        LCOV_ARGS --no-external
        GENHTML_ARGS --prefix ${CMAKE_SOURCE_DIR}
    )
    
    # Setup the tokenizer test coverage target
    setup_target_for_coverage(
        NAME coverage_tokenizer
        EXECUTABLE test_tokenizer
        DEPENDENCIES test_tokenizer
        LCOV_ARGS --no-external
        GENHTML_ARGS --prefix ${CMAKE_SOURCE_DIR}
    )
    
    # Setup the SIMD test coverage target
    setup_target_for_coverage(
        NAME coverage_simd
        EXECUTABLE test_simd
        DEPENDENCIES test_simd
        LCOV_ARGS --no-external
        GENHTML_ARGS --prefix ${CMAKE_SOURCE_DIR}
    )
    
    # Setup the thread pool test coverage target
    setup_target_for_coverage(
        NAME coverage_thread_pool
        EXECUTABLE test_thread_pool
        DEPENDENCIES test_thread_pool
        LCOV_ARGS --no-external
        GENHTML_ARGS --prefix ${CMAKE_SOURCE_DIR}
    )
    
    # Setup the advanced thread pool test coverage target
    setup_target_for_coverage(
        NAME coverage_thread_pool_advanced
        EXECUTABLE test_thread_pool_advanced
        DEPENDENCIES test_thread_pool_advanced
        LCOV_ARGS --no-external
        GENHTML_ARGS --prefix ${CMAKE_SOURCE_DIR}
    )
    
    # Setup the GGML KV cache test coverage target
    setup_target_for_coverage(
        NAME coverage_ggml_kv_cache
        EXECUTABLE test_ggml_kv_cache
        DEPENDENCIES test_ggml_kv_cache
        LCOV_ARGS --no-external
        GENHTML_ARGS --prefix ${CMAKE_SOURCE_DIR}
    )
    
    # Setup the Generator test coverage target
    setup_target_for_coverage(
        NAME coverage_generator
        EXECUTABLE test_generator
        DEPENDENCIES test_generator
        LCOV_ARGS --no-external
        GENHTML_ARGS --prefix ${CMAKE_SOURCE_DIR}
    )
    
    # Setup the GGML model quantization test coverage target
    setup_target_for_coverage(
        NAME coverage_ggml_model_quantization
        EXECUTABLE test_ggml_model_quantization
        DEPENDENCIES test_ggml_model_quantization
        LCOV_ARGS --no-external
        GENHTML_ARGS --prefix ${CMAKE_SOURCE_DIR}
    )
    
    # Setup the KV cache memory optimization test coverage target
    setup_target_for_coverage(
        NAME coverage_kv_cache_memory
        EXECUTABLE test_kv_cache_memory
        DEPENDENCIES test_kv_cache_memory
        LCOV_ARGS --no-external
        GENHTML_ARGS --prefix ${CMAKE_SOURCE_DIR}
    )
    
    # Setup the model memory optimization test coverage target
    setup_target_for_coverage(
        NAME coverage_model_memory_optimization
        EXECUTABLE test_model_memory_optimization
        DEPENDENCIES test_model_memory_optimization
        LCOV_ARGS --no-external
        GENHTML_ARGS --prefix ${CMAKE_SOURCE_DIR}
    )
    
    # Setup the model quantization with memory optimization test coverage target
    setup_target_for_coverage(
        NAME coverage_model_quantization_memory
        EXECUTABLE test_model_quantization_memory
        DEPENDENCIES test_model_quantization_memory
        LCOV_ARGS --no-external
        GENHTML_ARGS --prefix ${CMAKE_SOURCE_DIR}
    )
    
    # Setup the quantized matrix multiplication test coverage target
    setup_target_for_coverage(
        NAME coverage_quantized_matrix_mul
        EXECUTABLE test_quantized_matrix_mul
        DEPENDENCIES test_quantized_matrix_mul
        LCOV_ARGS --no-external
        GENHTML_ARGS --prefix ${CMAKE_SOURCE_DIR}
    )
    
    # Combined coverage target
    add_custom_target(coverage
        DEPENDS coverage_unit coverage_tokenizer coverage_simd coverage_thread_pool coverage_thread_pool_advanced coverage_ggml_kv_cache coverage_generator coverage_ggml_model_quantization coverage_kv_cache_memory coverage_model_memory_optimization coverage_model_quantization_memory coverage_quantized_matrix_mul
        COMMENT "Generating combined code coverage report"
    )
    
    # If MLX tests are available, add MLX coverage target
    if(WITH_MLX AND MLX_TEST_SOURCES)
        setup_target_for_coverage(
            NAME coverage_mlx
            EXECUTABLE ccsm_mlx_tests
            DEPENDENCIES ccsm_mlx_tests
            LCOV_ARGS --no-external
            GENHTML_ARGS --prefix ${CMAKE_SOURCE_DIR}
        )
        add_dependencies(coverage coverage_mlx)
    endif()
    
    # Now add integration tests after defining the coverage target
    add_subdirectory(integration)
endif()

# Add custom tests
add_subdirectory(unit/custom)