#ifndef UTILITIES_SCHEMA
#define UTILITIES_SCHEMA

#include "duckdb.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/planner/planner.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/planner/table_filter.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/common/enums/expression_type.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "duckdb/planner/filter/null_filter.hpp"
#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/planner/operator/logical_order.hpp"
#include "duckdb/execution/executor.hpp"

#include "duckdb/common/common.hpp"
#include "duckdb/common/enums/pending_execution_result.hpp"
#include "duckdb/common/mutex.hpp"
#include "duckdb/common/pair.hpp"
#include "duckdb/common/reference_map.hpp"
#include "duckdb/main/query_result.hpp"
#include "duckdb/execution/task_error_manager.hpp"
#include "duckdb/execution/progress_data.hpp"
#include "duckdb/parallel/pipeline.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map> // Added missing include
#include <filesystem>

struct ColumnInfo
{
    std::string type;
    int size_in_bytes;
    int acc_col_size;
    std::string name;
    bool is_primary = false;
    std::string foreign_table;
};

typedef std::unordered_map<std::string, std::pair<std::shared_ptr<std::ifstream>, std::vector<ColumnInfo>>> Schema;

void get_schema(Schema &schema);
void print_chunk(std::vector<char> chunk, std::string table_name);
std::string to_duckdb_type(const std::string &type);
void create_tables_from_schema(duckdb::Connection &conn, const Schema &schema);
std::vector<char> read_csv_chunk(std::string table_name, long chunk_size_in_bytes, int &row_size);
#endif // UTILITIES_SCHEMA