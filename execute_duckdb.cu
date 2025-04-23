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

// int main()
// {
//     using namespace duckdb;
//     //    Initialize DuckDB instance
//     DuckDB db(nullptr);
//     Connection con(db);
// 
//     ClientContext &context = *con.context;
//     con.Query("PRAGMA force_compression='uncompressed';");
//     con.Query("PRAGMA disable_compression;");
//     con.Query("PRAGMA disabled_compression_methods='patas,fors,pfor,bitpacking,fsst';"); // disable known ones
//     con.Query("SET disabled optimizens = fIlter pushdown, statistics propagation';");
//     con.Query("SET force_compression='uncompressed';");
//     con.Query("PRAGMA force_compression='uncompressed';");
//     con.Query("PRAGMA disabled_compression_methods='patas,fsst,bitpacking,rle,dictionary';");
// 
//     // Example SQL query
//     string sql = "CREATE TABLE student (id INTEGER, name VARCHAR, age INTEGER) ;";
//     string sql2 = "CREATE TABLE course (id INTEGER, name VARCHAR, credits INTEGER);";
// 
//     // Execute the SQL query to create a table
//     con.Query(sql);
//     con.Query(sql2);
//     // Insert some data into the table
//     sql = "INSERT INTO student VALUES (1, 'Alice', 30), (2, 'Bob', 25), (3, 'Charlie', 35);";
//     sql2 = "INSERT INTO course VALUES (1, 'Math', 3), (2, 'Science', 4), (3, 'History', 2);";
//     con.Query(sql);
//     con.Query(sql2);
// 
//     string query = "SELECT age from student,course where student.id=course.id";
//         // Parse the query
//     Parser parser;
//     parser.ParseQuery(query);
//     auto statements = std::move(parser.statements);
//     for (size_t i = 0; i < statements.size(); i++)
//     {
//         cout << "Statement " << i + 1 << ":\n";
//         cout << statements[i]->ToString() << "\n";
//     }
//     // Start a transaction
//     con.BeginTransaction(); // Start transaction using Connection
// 
//     // Create a planner and plan the query
//     Planner planner(context);
//     planner.CreatePlan(std::move(statements[0]));
// 
//     // Now you can proceed with further processing or optimization
//     cout << "Planning successful!" << endl;
//     cout << "Unoptimized Logical Plan:\n"
//          << planner.plan->ToString() << endl;
// 
//     Optimizer optimizer(*planner.binder, context);
//     auto logical_plan = optimizer.Optimize(std::move(planner.plan));
//     cout << "Optimized Logical Plan:\n";
//     cout << logical_plan->ToString() << endl;
// 
//     traverse_plan(logical_plan.get());
// 
//     // Commit the transaction after planning
//     con.Commit(); // Commit transaction using Connection
// }




std::unordered_map<std::string, std::vector<ColumnInfo>> schema;


int main(){
    
    DuckDB db(nullptr);
    Connection connect(db);

    ClientContext &context = *connect.context;
    get_schema(&connect, schema);
    print_schema(schema);

    
}
