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

# Make sure test_thread_pool.cpp is included
if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit/test_thread_pool.cpp")
    message(WARNING "test_thread_pool.cpp not found. Thread pool tests will not be included.")
endif()

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
    
    # Combined coverage target
    add_custom_target(coverage
        DEPENDS coverage_unit coverage_tokenizer coverage_simd
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