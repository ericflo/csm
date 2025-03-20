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
target_link_libraries(ccsm_core PRIVATE sentencepiece)

# MLX library (conditional)
if(WITH_MLX)
    file(GLOB MLX_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/mlx/*.cpp"
    )
    add_library(ccsm_mlx STATIC ${MLX_SOURCES})
    target_include_directories(ccsm_mlx PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
    target_link_libraries(ccsm_mlx PRIVATE ccsm_core)
endif()