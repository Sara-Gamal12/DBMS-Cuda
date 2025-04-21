cmake_minimum_required(VERSION 3.10)
project(ExecuteDuckDB)

set(CMAKE_CXX_STANDARD 17)

# 👇 Replace /home/sara/duckdb with your actual duckdb path
include_directories(/home/sara/duckdb/src/include)

# 👇 Point to the built duckdb .so directory
link_directories(/home/sara/duckdb/build/release/src)

add_executable(execute_duckdb ../execute_duckdb.cpp)
target_link_libraries(execute_duckdb duckdb pthread dl)
