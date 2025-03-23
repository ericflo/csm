add_executable(test_neon_vectors test_neon_vectors.cpp)
target_link_libraries(test_neon_vectors PRIVATE ccsm_core GTest::gtest)
target_include_directories(test_neon_vectors PRIVATE ${CMAKE_SOURCE_DIR})

add_test(NAME NeonVectorTest COMMAND test_neon_vectors)