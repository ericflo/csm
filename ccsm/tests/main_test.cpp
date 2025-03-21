#include <gtest/gtest.h>

// Common main function for all tests
// This avoids duplicate main() functions across test files
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}