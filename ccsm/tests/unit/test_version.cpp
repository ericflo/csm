#include <gtest/gtest.h>
#include <ccsm/version.h>
#include <string>

TEST(VersionTest, VersionMacros) {
    // Check that version macros are defined
    EXPECT_EQ(std::string(CCSM_VERSION), "0.1.0");
    EXPECT_EQ(CCSM_VERSION_MAJOR, 0);
    EXPECT_EQ(CCSM_VERSION_MINOR, 1);
    EXPECT_EQ(CCSM_VERSION_PATCH, 0);
}

TEST(VersionTest, VersionFormat) {
    // Check that version format is correct (major.minor.patch)
    std::string version(CCSM_VERSION);
    EXPECT_NE(version.find('.'), std::string::npos);
    
    // Count number of dots
    size_t dot_count = 0;
    for (char c : version) {
        if (c == '.') dot_count++;
    }
    
    // Should have 2 dots (major.minor.patch)
    EXPECT_EQ(dot_count, 2);
}