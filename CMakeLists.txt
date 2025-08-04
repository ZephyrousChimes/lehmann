cmake_minimum_required(VERSION 3.25.1)
project(Lehmann LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 86)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)


file(GLOB_RECURSE SRC_FILES
        "src/core/*.cpp"
        "src/cuda/*.cu"
        "src/main.cu"
)

add_executable(Lehmann ${SRC_FILES}
        src/cuda/Simulation.cuh)

target_include_directories(Lehmann PRIVATE src/)

target_compile_options(Lehmann PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

target_link_directories(Lehmann PRIVATE /usr/local/cuda-12/lib64)
target_link_libraries(Lehmann PRIVATE cudart)

set(CMAKE_BINARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/bin)

set_target_properties(Lehmann PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY           "${CMAKE_BINARY_DIR}/$<CONFIG>/Lehmann"
        LIBRARY_OUTPUT_DIRECTORY           "${CMAKE_BINARY_DIR}/$<CONFIG>/Lehmann"
        ARCHIVE_OUTPUT_DIRECTORY           "${CMAKE_BINARY_DIR}/$<CONFIG>/Lehmann"
        CUDA_SEPARABLE_COMPILATION         ON
        CUDA_RESOLVE_DEVICE_SYMBOLS        ON
)

set_property(TARGET Lehmann PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
set_property(TARGET Lehmann PROPERTY INTERPROCEDURAL_OPTIMIZATION_DEBUG OFF)
