# Integration tests for CCSM

# Find Google Test if not already found
if(NOT TARGET GTest::gtest)
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG main
        )
        FetchContent_MakeAvailable(googletest)
    endif()
endif()

# Define integration test sources
file(GLOB INTEGRATION_TEST_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

# Create integration test executable
add_executable(ccsm_integration_tests ${INTEGRATION_TEST_SOURCES})
target_link_libraries(ccsm_integration_tests PRIVATE ccsm_core GTest::gtest GTest::gtest_main)
target_include_directories(ccsm_integration_tests PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Add integration tests to CTest
add_test(NAME IntegrationTests COMMAND ccsm_integration_tests)

# Setup code coverage target if coverage is enabled
if(WITH_COVERAGE)
    setup_target_for_coverage(
        NAME coverage_integration
        EXECUTABLE ccsm_integration_tests
        DEPENDENCIES ccsm_integration_tests
        LCOV_ARGS --no-external
        GENHTML_ARGS --prefix ${CMAKE_SOURCE_DIR}
    )
    add_dependencies(coverage coverage_integration)
endif()