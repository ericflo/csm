# Main source directory CMakeLists.txt

# Core library - common components
file(GLOB CORE_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/cpu/*.cpp"
)

add_library(ccsm_core STATIC ${CORE_SOURCES})
target_include_directories(ccsm_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
# Include SentencePiece headers
target_include_directories(ccsm_core PRIVATE 
    ${CMAKE_BINARY_DIR}/_deps/sentencepiece-src/src
    ${CMAKE_BINARY_DIR}/_deps/sentencepiece-build/src
)
# Include GGML headers
target_include_directories(ccsm_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../reference/llama.cpp/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../reference/llama.cpp/ggml/include/ggml
)

# Find GGML library or build it directly
find_library(GGML_LIB ggml PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../../reference/llama.cpp/build)
if(NOT GGML_LIB)
    # Use GGML sources directly
    set(GGML_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/../../reference/llama.cpp/ggml/src/ggml.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../reference/llama.cpp/ggml/src/ggml-alloc.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../reference/llama.cpp/ggml/src/ggml-quants.c"
    )
    
    # Add GGML backend and GGUF sources (they're .cpp files, not .c)
    set(GGML_CPP_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/../../reference/llama.cpp/ggml/src/ggml-backend.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../reference/llama.cpp/ggml/src/ggml-backend-reg.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../reference/llama.cpp/ggml/src/ggml-opt.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../reference/llama.cpp/ggml/src/ggml-threading.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../reference/llama.cpp/ggml/src/gguf.cpp"
    )
    
    # Create a separate GGML library target
    add_library(ggml_for_ccsm STATIC ${GGML_SOURCES} ${GGML_CPP_SOURCES})
    set_target_properties(ggml_for_ccsm PROPERTIES LINKER_LANGUAGE CXX)
    target_include_directories(ggml_for_ccsm PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../reference/llama.cpp/ggml/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../reference/llama.cpp/ggml/src
    )
    
    # Set C flags for GGML sources
    set_source_files_properties(${GGML_SOURCES} PROPERTIES LANGUAGE C)
    
    # Add pthread for threading support
    find_package(Threads REQUIRED)
    target_link_libraries(ggml_for_ccsm PRIVATE Threads::Threads)
    
    # For Apple devices, we need to link against Accelerate framework
    if(APPLE)
        find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
        target_link_libraries(ggml_for_ccsm PRIVATE ${ACCELERATE_FRAMEWORK})
    endif()
    
    # Link against our new GGML library
    target_link_libraries(ccsm_core PRIVATE ggml_for_ccsm)
    message(STATUS "Building with GGML sources directly")
else()
    target_link_libraries(ccsm_core PRIVATE ${GGML_LIB})
    message(STATUS "Found GGML library: ${GGML_LIB}")
endif()

target_link_libraries(ccsm_core PRIVATE sentencepiece)

# MLX library (conditional)
if(WITH_MLX)
    # Core MLX sources 
    file(GLOB MLX_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/mlx/*.cpp"
    )
    
    # Create MLX library target
    add_library(ccsm_mlx STATIC ${MLX_SOURCES})
    target_include_directories(ccsm_mlx PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
    
    # Check if we have the actual MLX library installed
    if(MLX_LIBRARY)
        message(STATUS "Using real MLX library: ${MLX_LIBRARY}")
        
        # Add the MLX-C sources if we're building them directly
        if(DEFINED MLX_C_SOURCES)
            target_sources(ccsm_mlx PRIVATE ${MLX_C_SOURCES})
            message(STATUS "Added MLX-C sources directly to ccsm_mlx")
        endif()
        
        # Link MLX library
        target_link_libraries(ccsm_mlx PRIVATE ${MLX_LIBRARY})
        
        # Link MLX-C library if found separately
        if(MLX_C_LIBRARY)
            target_link_libraries(ccsm_mlx PRIVATE ${MLX_C_LIBRARY})
        endif()
    else()
        # Use mock/stub implementation
        message(STATUS "Using stub MLX implementation (real MLX library not found)")
        add_definitions(-DCCSM_MLX_STUB)
    endif()
    
    # Link dependencies
    target_link_libraries(ccsm_mlx PRIVATE ccsm_core)
    
    # macOS specific settings for Metal framework
    if(APPLE)
        find_library(METAL_FRAMEWORK Metal REQUIRED)
        find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
        target_link_libraries(ccsm_mlx PRIVATE ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
    endif()
endif()