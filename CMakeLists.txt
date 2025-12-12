cmake_minimum_required(VERSION 3.19)
project(CUDA_MATRIX LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# (I'm using my 3080, compute capability 8.6)
# see https://developer.nvidia.com/cuda/gpus for other gpus
set(CUDA_COMPUTE_CAPABILITY 86)

set(PROJECT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/src)

# cuda debug flags
add_compile_options(
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-src-in-ptx>
)

file(GLOB SRC
    "${PROJECT_INCLUDE_DIR}/*.cu"
    "${PROJECT_INCLUDE_DIR}/*.cpp"
)

file(GLOB HEADERS
    "${PROJECT_INCLUDE_DIR}/*.h"
    "${PROJECT_INCLUDE_DIR}/*.cuh"
)

add_executable(matrix ${SRC})

target_sources(matrix
    PRIVATE
        ${SRC}
    PUBLIC FILE_SET HEADERS
        BASE_DIRS ${PROJECT_INCLUDE_DIR}
        FILES ${HEADERS}
)

source_group("Header Files" FILES ${HEADERS})

target_include_directories(matrix PRIVATE ${PROJECT_INCLUDE_DIR})

set_target_properties(matrix PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
