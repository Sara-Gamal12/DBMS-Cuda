#ifndef UTILITIES_SCHEMA
#define UTILITIES_SCHEMA

#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>


#include <fstream>
#include <iostream>
#include <sstream>
#include<string>
#include <vector>

struct ColumnInfo {
    std::string type;
    int size_in_bytes;
    std::string name;
    bool is_primary = false;
    std::string foreign_table;
};
typedef std::unordered_map<std::string, std::pair<std::shared_ptr<std::ifstream>,std::vector<ColumnInfo>>> Schema;
void print_schema(Schema schema);
void get_schema( Schema &schema);

#endif // AGG_KERNELS_H