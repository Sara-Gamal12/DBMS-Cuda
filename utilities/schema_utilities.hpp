#ifndef UTILITIES_SCHEMA
#define UTILITIES_SCHEMA

#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>
#include <filesystem>
#include "duckdb.hpp"


// Use DuckDB's namespace
using namespace duckdb;


struct ColumnInfo {
    std::string type;
    std::string name;
    bool is_primary = false;
    std::string foreign_table;
};

void print_schema(std::unordered_map<std::string, std::vector<ColumnInfo>> schema);
void get_schema(Connection *con, std::unordered_map<std::string, std::vector<ColumnInfo>> &schema);

#endif // AGG_KERNELS_H