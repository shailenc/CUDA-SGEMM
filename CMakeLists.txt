cmake_minimum_required(VERSION 3.19)
project(cuda_gemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(CUDAToolkit REQUIRED)

set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)
set(KERNEL_DIR ${SRC_DIR}/kernels)

# -------------------------
# Core (dispatcher only)
# -------------------------
set(CORE_SRC
    ${SRC_DIR}/sgemm_dispatcher.cu
)

add_library(core STATIC ${CORE_SRC})
target_include_directories(core PUBLIC ${SRC_DIR})

# -------------------------
# Kernels
# -------------------------
set(KERNEL_SRC
    ${KERNEL_DIR}/cublas.cu
    ${KERNEL_DIR}/naive.cu
)

add_library(kernels STATIC
    ${KERNEL_SRC}
)

target_include_directories(kernels PUBLIC ${SRC_DIR})

# -------------------------
# Executable
# -------------------------
add_executable(bench
    ${SRC_DIR}/main.cpp
)

target_include_directories(bench PRIVATE ${SRC_DIR})

target_link_libraries(bench PRIVATE core kernels CUDA::cudart)
target_link_libraries(core PUBLIC CUDA::cublas)