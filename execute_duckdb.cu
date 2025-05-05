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
#include <cuda.h>

#include "./kernels/agg.cuh"
#include "./kernels/get.cuh"
#include "./utilities/schema_utilities.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;
using namespace duckdb;

long RAM = 4 * pow(1024, 3); // 4GB
Schema schema;

struct PlanNode
{
    std::string name;
    std::vector<std::string> details; // e.g., filters, expressions
    std::vector<std::shared_ptr<PlanNode>> children;
};

std::vector<char> post_order_traverse_and_launch_kernel(std::shared_ptr<PlanNode> node)
{
    if (!node)
        return {};

    // 1. Traverse children first (post-order)
    std::vector<std::vector<char>> child_results;
    for (auto &child : node->children)
    {
        std::vector<char> data =post_order_traverse_and_launch_kernel(child);
        child_results.push_back(data);
    }

    // 2. Process current node (e.g., launch kernel)
    std::cout << "Launching kernel for operator: " << node->name << std::endl;

    // You can decide which CUDA kernel to call based on node->name
    if (node->name == "GET")
    {

        string table_name = node->details[0];
        int row_size ;
        std::vector<char> chunk = read_csv_chunk(table_name, 0.5 * RAM, row_size);
        if (node->details.size() > 1)
        {
            int  *acc_sums=new int [(schema[table_name].second.size())];

            Condition *conditions=new Condition[node->details.size()-1];

            for (size_t i = 1; i < node->details.size(); ++i)
            {
                std::string condition_str = node->details[i];
                std::istringstream iss(condition_str);
                std::string column, op, value;
                iss >> column >> op >> value;  

                for (int j = 0; j < schema[table_name].second.size(); ++j)
                {
                    if (schema[table_name].second[j].name == column)
                    {
                        conditions[i - 1].col_index = j;
                        if (schema[table_name].second[j].type == "Numeric")
                        {
                            conditions[i-1].type = 0;
                            conditions[i-1].f_value = std::stof(value);
                        }
                        else if (schema[table_name].second[j].type == "Text")
                        {
                            conditions[i-1].type = 1;
                            strncpy(conditions[i-1].s_value, value.c_str(), sizeof(conditions[i-1].s_value));
                        }
                        // else if (schema[table_name].second[j].type == "DateTime")
                        // {
                        //     conditions[i-1].type = 2;
                        // }
                    }
                    acc_sums[j] = schema[table_name].second[j].acc_col_size;
                }

                if(op=="EQUAL")
                {
                    conditions[i-1].op = OP_EQ;
                }
                else if(op=="GREATERTHAN")
                {
                    conditions[i-1].op = OP_GT;
                }
                else if(op=="LESSTHAN")
                {
                    conditions[i-1].op = OP_LT;
                }
                else if(op=="NOTEQUAL")
                {
                    conditions[i-1].op = OP_NEQ;
                }
                else if(op=="GREATERTHANOREQUALTO")
                {
                    conditions[i-1].op = OP_GTE;
                }
                else if(op=="LESSTHANOREQUALTO")
                {
                    conditions[i-1].op = OP_LTE;
                }

            }
            
            int n = chunk.size() / row_size;
            int output_counter = 0;
            char*data = call_get_kernel(chunk.data(), row_size, acc_sums, conditions, node->details.size()-1,n,output_counter,schema[table_name].second.size());
            // std::cout << "Output counter: " << output_counter << std::endl;
            return std::vector<char>(data, data +  output_counter* row_size);
        }
        else{
            return chunk;
        }

        // launch_get_kernel();  // Your kernel logic here
    }
    
    else if (node->name == "JOIN")
    {
        // launch_join_kernel(); // Your kernel logic here
    }
    else if (node->name == "ORDERBY")
    {
        // launch_order_kernel(); // Your kernel logic here
    }
    else if (node->name == "AGGREGATE")
    {
        if (node->details[0].find("max") != std::string::npos)
        {
        }
        else if (node->details[0].find("sum") != std::string::npos)
        {
            // launch_sum_kernel(); // Your kernel logic here
        }
        else if (node->details[0].find("avg") != std::string::npos)
        {
            // launch_avg_kernel(); // Your kernel logic here
        }
    }
    else if (node->name == "PROJECTION"){
        return child_results[0];
    }
    else
    {
        std::cout << "No matching kernel for: " << node->name << std::endl;
    }
}

std::shared_ptr<PlanNode> build_plan_tree(LogicalOperator *op)
{
    if (!op)
        return nullptr;

    auto node = std::make_shared<PlanNode>();
    node->name = LogicalOperatorToString(op->type);

    // Handle LogicalGet
    if (auto get_op = dynamic_cast<duckdb::LogicalGet *>(op))
    {
        if (get_op->GetTable())
        {
            node->details.push_back(get_op->GetTable()->name);
        }
        for (auto &[col_idx, filter] : get_op->table_filters.filters)
        {
            std::ostringstream oss;
            oss << get_op->names[col_idx] << " ";
            switch (filter->filter_type)
            {
            case duckdb::TableFilterType::CONSTANT_COMPARISON:
            {
                auto &f = static_cast<duckdb::ConstantFilter &>(*filter);
                oss << ExpressionTypeToString(f.comparison_type) << " " << f.constant.ToString();
                break;
            }
            case duckdb::TableFilterType::IS_NULL:
                oss << "IS NULL";
                break;
            case duckdb::TableFilterType::IS_NOT_NULL:
                oss << "IS NOT NULL";
                break;
            case duckdb::TableFilterType::CONJUNCTION_AND:
                oss << "AND";
                break;
            case duckdb::TableFilterType::CONJUNCTION_OR:
                oss << "OR";
                break;
            default:
                oss << "UNKNOWN";
                break;
            }
            node->details.push_back( oss.str());
        }
    }

    // Handle LogicalComparisonJoin
    if (op->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN)
    {
        auto &join_op = static_cast<duckdb::LogicalComparisonJoin &>(*op);
        for (auto &condition : join_op.conditions)
        {
            node->details.push_back("Join: " + condition.left->ToString() + " " + ExpressionTypeToString(condition.comparison) + " " + condition.right->ToString());
        }
    }

    // Handle LogicalOrder
    if (op->type == LogicalOperatorType::LOGICAL_ORDER_BY)
    {
        auto &order_by_op = static_cast<duckdb::LogicalOrder &>(*op);
        for (auto &order : order_by_op.orders)
        {
            std::string order_str = "Order By: " + order.expression->ToString();
            order_str += (order.type == duckdb::OrderType::ASCENDING) ? " ASC" : " DESC";
            node->details.push_back(order_str);
        }
    }

    for (auto &expr : op->expressions)
    {
        node->details.push_back("Expr: " + expr->ToString());
    }

    // Recurse on children
    for (auto &child : op->children)
    {
        node->children.push_back(build_plan_tree(child.get()));
    }

    return node;
}


void print_tree(std::shared_ptr<PlanNode> node, int indent = 0){
    if (!node)
        return;
    std::cout << std::string(indent, ' ') << "- " << node->name << std::endl;
    for (const auto &detail : node->details)
    {
        std::cout << std::string(indent + 2, ' ') << "* " << detail << std::endl;
    }
    for (const auto &child : node->children)
    {
        print_tree(child, indent + 4);
    }
}

int main()
{
    using namespace duckdb;
    //    Initialize DuckDB instance
    DuckDB db(nullptr);
    Connection con(db);
    ClientContext &context = *con.context;
    get_schema(schema);
    create_tables_from_schema(con, schema);
    string query = "select * from student where age>=25 and name !='ddd' ;";
    int row_size;

    Parser parser;
    parser.ParseQuery(query);
    auto statements = std::move(parser.statements);
    // Start a transaction
    con.BeginTransaction(); // Start transaction using Connection

    // Create a planner and plan the query
    Planner planner(context);
    planner.CreatePlan(std::move(statements[0]));

    // Now you can proceed with further processing or optimization
    cout << "Planning successful!" << endl;
    cout << "Unoptimized Logical Plan:\n"
        << planner.plan->ToString() << endl;

    Optimizer optimizer(*planner.binder, context);
    auto logical_plan = optimizer.Optimize(std::move(planner.plan));
    cout << "Optimized Logical Plan:\n";
    cout << logical_plan->ToString() << endl;

    auto tree_root = build_plan_tree(logical_plan.get());
    print_tree(tree_root);

    // Traverse the plan tree and launch kernels
    std::vector<char> data_out =post_order_traverse_and_launch_kernel(tree_root);

    print_chunk(data_out, "student");
    // Commit the transaction after planning
    con.Commit(); // Commit transaction using Connection
}

