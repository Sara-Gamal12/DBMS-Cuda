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
#include "duckdb/planner/table_filter.hpp"
#include "duckdb/planner/table_filter.hpp"
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
#include <cuda_runtime.h>
#include<cuda.h>

#include "./kernels/agg.cuh"
#include "./utilities/schema_utilities.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include<string>
#include <vector>

using namespace std;
using namespace duckdb;

void traverse_plan(LogicalOperator *op)
{

    if (!op)
    {
        return;
    }

    // Print the type of the logical operator
    std::cout << "Logical Operator: " << LogicalOperatorToString(op->type) << std::endl;
    auto table_indexes = op->GetTableIndex(); // This will return a vector<long unsigned int>
    if (op->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN)
    {
        auto &join_op = static_cast<duckdb::LogicalComparisonJoin &>(*op);
        std::cout << "Join Conditions:" << std::endl;
        for (auto &condition : join_op.conditions)
        {
            std::cout << "  Left: " << condition.left->ToString() << std::endl;
            std::cout << "  Right: " << condition.right->ToString() << std::endl;
            std::cout << "  Comparison: " << duckdb::ExpressionTypeToString(condition.comparison) << std::endl;
        }
    }
    // Print each table index
    if (op->type == LogicalOperatorType::LOGICAL_ORDER_BY)
    {
        auto &order_by_op = static_cast<duckdb::LogicalOrder &>(*op);
        for (auto &order : order_by_op.orders)
        {
            std::cout << "Column: " << order.expression->ToString() << ", ";
            std::cout << "Order: " << (order.type == duckdb::OrderType::ASCENDING ? "ASC" : "DESC") << std::endl;
        }
    }

    if (auto get_op = dynamic_cast<duckdb::LogicalGet *>(op))
    {
        if (get_op->GetTable())
        {
            auto &entry = *get_op->GetTable();
            std::cout << "Table Name: " << entry.name << std::endl;
            // print filters

            if (!get_op->table_filters.filters.empty())
            {
                std::cout << "Filters:" << std::endl;
                for (auto &[col_idx, filter] : get_op->table_filters.filters)
                {
                    std::cout << get_op->names[col_idx] << " ";

                    switch (filter->filter_type)
                    {
                    case duckdb::TableFilterType::CONSTANT_COMPARISON:
                    {
                        auto &constant_filter = static_cast<duckdb::ConstantFilter &>(*filter);
                        std::cout << duckdb::ExpressionTypeToString(constant_filter.comparison_type)
                                  << " " << constant_filter.constant.ToString();
                        break;
                    }
                    case duckdb::TableFilterType::IS_NULL:
                        std::cout << "IS NULL";
                        break;
                    case duckdb::TableFilterType::IS_NOT_NULL:
                        std::cout << "IS NOT NULL";
                        break;
                    case duckdb::TableFilterType::CONJUNCTION_AND:
                        std::cout << "AND";
                        break;
                    case duckdb::TableFilterType::CONJUNCTION_OR:
                        std::cout << "OR";
                        break;
                    default:
                        std::cout << "UNKNOWN FILTER TYPE";
                        break;
                    }
                    std::cout << std::endl;
                }
            }
        }
    }

    if (op->expressions.size() > 0)
    {
        std::cout << "Expressions:" << std::endl;
        for (auto &expr : op->expressions)
        {
            std::cout << "  " << expr->ToString() << std::endl;
            if (expr->ToString().find("compress") != std::string::npos)
                cout << " compress node" << endl;
        }
    }

    // Recursively traverse children
    for (auto &child : op->children)
    {
        traverse_plan(child.get());
    }
}




long RAM=4 * pow(1024, 3); // 4GB
Schema schema;


void print_chunk(std::vector<char>chunk,std::string table_name){

    std::vector<ColumnInfo> cols = schema.at(table_name).second;
    int row_size = 0;
    for (const auto& col : cols) {
        row_size += col.size_in_bytes;
    }

    size_t offset = 0;
    for (size_t i = 0; i < chunk.size() / row_size; ++i) {
        size_t row_start = i * row_size;
        offset = 0;

        for (auto& col : cols) {
            std::cout << "Column " << col.name << ": ";

            // Get pointer to this column's data within the row
            char* data_ptr = chunk.data() + row_start + offset;

            if (col.type == "Numeric") {
                double value = *reinterpret_cast<double*>(data_ptr);
                std::cout << value;
                offset += sizeof(double);
            } else if (col.type == "DateTime") {
                std::time_t time_value = *reinterpret_cast<std::time_t*>(data_ptr);
                std::tm* tm = std::localtime(&time_value);
                char buffer[20];
                std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", tm);
                std::cout << buffer;
                offset += sizeof(std::time_t);
            } else if (col.type == "Text") {
                std::string text(data_ptr, col.size_in_bytes);
                std::cout << text.c_str();  // safe printing
                offset += col.size_in_bytes;
            }

            std::cout << " | ";
        }

        std::cout << std::endl;
}
}

std::vector<char> read_csv_chunk(std::string table_name, long chunk_size_in_bytes) {
    
    if (schema.find(table_name) == schema.end()) {
        std::cout << "Table schema not found for: " << table_name << std::endl;
        return {};
    }

    std::shared_ptr<std::ifstream> file = schema.at(table_name).first;
    
    std::vector<ColumnInfo> cols = schema.at(table_name).second;
    int row_size = 0;
    for (const auto& col : cols) {
        row_size += col.size_in_bytes;
    }
    
    
    std::vector<char> chunk;
    std::string line;
    long total_size_in_bytes = 0;

   
    while (total_size_in_bytes + row_size <= chunk_size_in_bytes && std::getline(*file, line))
    {
        std::stringstream ss(line);
        std::string token;

        cout << line << std::endl;
        for (const auto &col : cols)
        {
            if (!std::getline(ss, token, ',')) break;
            if (col.type == "Numeric") {
                double value = std::stof(token);
                const char* ptr = reinterpret_cast<const char*>(&value);
                chunk.insert(chunk.end(), ptr, ptr + col.size_in_bytes);
            } else if (col.type == "DateTime") {
                std::tm tm = {};
                std::istringstream ss(token);
                ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
                std::time_t time_value = std::mktime(&tm);
                const char* ptr = reinterpret_cast<const char*>(&time_value);
                chunk.insert(chunk.end(), ptr, ptr + col.size_in_bytes);
            } else if (col.type == "Text") {
                // Truncate or pad string to fit size
                std::string fixed = token.substr(0, col.size_in_bytes);
                fixed.resize(col.size_in_bytes, '\0');
                chunk.insert(chunk.end(), fixed.begin(), fixed.end());
            } 

        total_size_in_bytes += row_size;
        }

    }

    print_chunk(chunk, table_name);
    return chunk;
}


std::string to_duckdb_type(const std::string& type) {
    if (type == "Text") return "VARCHAR(150)";
    if (type == "Numeric") return "DOUBLE"; // or FLOAT depending on your use
    if (type == "DateTime") return "TIMESTAMP";
    return "VARCHAR(150)";
}



// helper to join strings
std::string join(const std::vector<std::string>& vec, const std::string& sep) {
    std::ostringstream oss;
    for (size_t i = 0; i < vec.size(); ++i) {
        oss << vec[i];
        if (i + 1 < vec.size()) oss << sep;
    }
    return oss.str();
}



void create_tables_from_schema(duckdb::Connection& conn, const Schema& schema) {
    std::string forign_key="";
    for (const auto& [table_name, pair] : schema) {
        const auto& columns = pair.second;
        std::string sql = "CREATE TABLE " + table_name + " (";

        std::vector<std::string> col_defs;
        std::string primary_key;


        for (const auto& col : columns) {
            std::string col_def = col.name + " " + to_duckdb_type(col.type);
            col_defs.push_back(col_def);

            if (col.is_primary) {
                primary_key = col.name;
            }

            
            if (!col.foreign_table.empty()) {
                forign_key += "ALTER TABLE " + table_name + " ADD FOREIGN KEY (" + col.name + ") REFERENCES " + col.foreign_table + ";";
            }
        }

        sql += join(col_defs, ", ");

        if (!primary_key.empty()) {
            sql += ", PRIMARY KEY(" + primary_key + ")";
        }


        sql += ");";

        conn.Query(sql);

        string insert_sql = "INSERT INTO " + table_name + " SELECT * FROM read_csv_auto('../DB/" + table_name + ".csv', HEADER=true);";
        conn.Query(insert_sql);
    }

    conn.Query(forign_key);

    // print duckdb schema
    // auto result = conn.Query("select*from student;");
    // result->Print();
}


int main()
{
    using namespace duckdb;
    //    Initialize DuckDB instance
    DuckDB db(nullptr);
    Connection con(db);

    ClientContext &context = *con.context;
    

    get_schema( schema);
    create_tables_from_schema(con, schema);


    string query = "SELECT max(age) from student where id<3;";
    // Parse the query
    // Parser parser;
    // parser.ParseQuery(query);
    // auto statements = std::move(parser.statements);
    // for (size_t i = 0; i < statements.size(); i++)
    // {
    //     cout << "Statement " << i + 1 << ":\n";
    //     cout << statements[i]->ToString() << "\n";
    // }
    // // Start a transaction
    // con.BeginTransaction(); // Start transaction using Connection

    // // Create a planner and plan the query
    // Planner planner(context);
    // planner.CreatePlan(std::move(statements[0]));

    // // Now you can proceed with further processing or optimization
    // cout << "Planning successful!" << endl;
    // cout << "Unoptimized Logical Plan:\n"
    //      << planner.plan->ToString() << endl;

    // Optimizer optimizer(*planner.binder, context);
    // auto logical_plan = optimizer.Optimize(std::move(planner.plan));
    // cout << "Optimized Logical Plan:\n";
    // cout << logical_plan->ToString() << endl;

    // traverse_plan(logical_plan.get());

    // Commit the transaction after planning
    con.Commit(); // Commit transaction using Connection
}



