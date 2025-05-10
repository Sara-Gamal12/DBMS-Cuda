cmake_minimum_required(VERSION 3.10)
project(ExecuteDuckDB LANGUAGES CXX CUDA)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Set C++ and CUDA standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



# Include DuckDB headers
include_directories(/home/nesma/CUDA-SQL/DBMS-Cuda/duckdb/src/include)

# Link to DuckDB library directory
link_directories(/home/nesma/CUDA-SQL/DBMS-Cuda/duckdb/build/release/src)

# Add the executable with both C++ and CUDA source files
add_executable(gpu ../gpu.cu ./kernels/agg.cu ./kernels/get.cu ./kernels/join.cu ./kernels/project.cu ./kernels/sort.cu ./utilities/schema_utilities.cpp ./utilities/filter_utilities.cpp ./utilities/join_utilities.cpp )
add_executable(cpu ../cpu.cu ./kernels/agg.cu ./kernels/get.cu ./kernels/join.cu ./kernels/project.cu ./kernels/sort.cu ./utilities/schema_utilities.cpp ./utilities/filter_utilities.cpp ./utilities/join_utilities.cpp )
add_executable(main ../main.cu ./kernels/agg.cu ./kernels/get.cu ./kernels/join.cu ./kernels/project.cu ./kernels/sort.cu ./utilities/schema_utilities.cpp ./utilities/filter_utilities.cpp ./utilities/join_utilities.cpp )

# Specify that the CUDA file should be compiled with NVCC
set_source_files_properties(
    ./kernels/agg.cu
    ./kernels/get.cu
    ./kernels/project.cu
    ./kernels/sort.cu
    ./kernels/join.cu
    PROPERTIES
    LANGUAGE CUDA
)

# Link necessary libraries
target_link_libraries(gpu duckdb pthread dl)
target_link_libraries(cpu duckdb pthread dl)
target_link_libraries(main duckdb pthread dl)

# Optional: CUDA separable compilation for larger projects
set_target_properties(gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)


set_target_properties(cpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
