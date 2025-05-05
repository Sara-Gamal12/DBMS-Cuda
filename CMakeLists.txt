cmake_minimum_required(VERSION 3.10)
project(ExecuteDuckDB LANGUAGES CXX CUDA)

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
add_executable(execute_duckdb ../execute_duckdb.cu ./kernels/agg.cu ./kernels/get.cu ./utilities/schema_utilities.cpp ./utilities/filter_utilities.cpp )

# Specify that the CUDA file should be compiled with NVCC
set_source_files_properties(
    ./kernels/agg.cu
    ./kernels/get.cu
    PROPERTIES
    LANGUAGE CUDA
)

# Link necessary libraries
target_link_libraries(execute_duckdb duckdb pthread dl)

# Optional: CUDA separable compilation for larger projects
set_target_properties(execute_duckdb PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)


