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
)

# Create unit test executable
add_executable(ccsm_unit_tests ${TEST_SOURCES})
target_link_libraries(ccsm_unit_tests PRIVATE ccsm_core GTest::gtest GTest::gtest_main)

# Add tokenizer test
add_executable(test_tokenizer ${CMAKE_CURRENT_SOURCE_DIR}/tokenizer/test_tokenizer.cpp)
target_link_libraries(test_tokenizer PRIVATE ccsm_core)

# Add SIMD optimization test
add_executable(test_simd ${CMAKE_CURRENT_SOURCE_DIR}/simd/test_simd.cpp)
target_link_libraries(test_simd PRIVATE ccsm_core)

# Add MLX tests conditionally
if(WITH_MLX)
    file(GLOB MLX_TEST_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/unit/mlx_*.cpp"
    )
    add_executable(ccsm_mlx_tests ${MLX_TEST_SOURCES})
    target_link_libraries(ccsm_mlx_tests PRIVATE ccsm_core ccsm_mlx GTest::gtest GTest::gtest_main)
endif()

# Add the tests to CTest
add_test(NAME UnitTests COMMAND ccsm_unit_tests)
add_test(NAME TokenizerTest COMMAND test_tokenizer)
add_test(NAME SIMDTest COMMAND test_simd)

if(WITH_MLX)
    add_test(NAME MLXTests COMMAND ccsm_mlx_tests)
endif()