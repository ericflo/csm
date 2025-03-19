cmake_minimum_required(VERSION 3.16)
project(ccsm VERSION 0.1.0 LANGUAGES C CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option for backend selection
option(WITH_MLX "Build with MLX support for Apple Silicon" ON)
option(WITH_CUDA "Build with CUDA support" OFF)
option(WITH_VULKAN "Build with Vulkan support" OFF)

# Find required packages
find_package(SentencePiece QUIET)
if(NOT SentencePiece_FOUND)
    message(STATUS "SentencePiece not found via find_package, will use vendored version or fetch content")
    # Use FetchContent or vendored code for SentencePiece
    include(FetchContent)
    FetchContent_Declare(
        sentencepiece
        GIT_REPOSITORY https://github.com/google/sentencepiece.git
        GIT_TAG v0.1.99
    )
    FetchContent_MakeAvailable(sentencepiece)
endif()

# Platform detection
if(APPLE AND WITH_MLX)
    message(STATUS "Configuring for Apple Silicon with MLX acceleration")
    add_definitions(-DCCSM_WITH_MLX)
    set(BUILD_MLX_VERSION ON)
endif()

if(WITH_CUDA)
    message(STATUS "Configuring with CUDA support")
    add_definitions(-DCCSM_WITH_CUDA)
endif()

if(WITH_VULKAN)
    message(STATUS "Configuring with Vulkan support")
    add_definitions(-DCCSM_WITH_VULKAN)
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/reference/llama.cpp
)

# Add submodules if using vendored code
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/reference/llama.cpp/ggml.h")
    message(FATAL_ERROR "llama.cpp dependency not found. Please init and update git submodules.")
endif()

if(WITH_MLX)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/reference/mlx-c/mlx")
        message(FATAL_ERROR "mlx-c dependency not found. Please init and update git submodules.")
    endif()
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/reference/mlx-c)
endif()

# Add subdirectories
add_subdirectory(src)
add_subdirectory(tests)

# Main executable - CPU version
add_executable(ccsm-generate src/main.cpp)
target_link_libraries(ccsm-generate PRIVATE ccsm_core)

# MLX version (conditional)
if(WITH_MLX)
    add_executable(ccsm-generate-mlx src/main_mlx.cpp)
    target_link_libraries(ccsm-generate-mlx PRIVATE ccsm_core ccsm_mlx)
endif()

# Install targets
install(TARGETS ccsm-generate
    RUNTIME DESTINATION bin
)

if(WITH_MLX)
    install(TARGETS ccsm-generate-mlx
        RUNTIME DESTINATION bin
    )
endif()