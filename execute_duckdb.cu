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

struct PlanNode
{
    std::string name;
    std::vector<std::string> details; // e.g., filters, expressions
    std::vector<std::shared_ptr<PlanNode>> children;
};

void post_order_traverse_and_launch_kernel(std::shared_ptr<PlanNode> node)
{
    if (!node)
        return;

    // 1. Traverse children first (post-order)
    for (auto &child : node->children)
    {
        post_order_traverse_and_launch_kernel(child);
    }

    // 2. Process current node (e.g., launch kernel)
    std::cout << "Launching kernel for operator: " << node->name << std::endl;

    // You can decide which CUDA kernel to call based on node->name
    if (node->name == "GET")
    {

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
            node->details.push_back("Table: " + get_op->GetTable()->name);
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
            node->details.push_back("Filter: " + oss.str());
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

long RAM = 4 * pow(1024, 3); // 4GB
Schema schema;
void print_tree(std::shared_ptr<PlanNode> node, int indent = 0)
{
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

// int main()
// {
//     using namespace duckdb;
//     //    Initialize DuckDB instance
//     DuckDB db(nullptr);
//     Connection con(db);
//     ClientContext &context = *con.context;
//     get_schema(schema);
//     create_tables_from_schema(con, schema);
//     string query = "select name, id as total_count from (select s.name,s.id from student s  where s.id > 1) order by name Asc;";
//     int row_size;
//     std::vector<char> chunk = read_csv_chunk("student", 0.5 * RAM, row_size);
//     // Example usage
//     // const int n = 5;
//     // char* d_input_data;
//     // double* d_output_data;
//     // double* h_output_data;

//     // // Allocate device memory
//     // cudaMalloc((void**)&d_input_data, n * row_size*sizeof(char));
//     // cudaMemcpy(d_input_data, chunk.data(),  n * row_size*sizeof(char), cudaMemcpyHostToDevice);

//     // // Launch kernel
//     // int blockSize =1;
//     // int numBlocks = (n + (blockSize*2) - 1) / (blockSize*2);
//     // h_output_data = (double*)malloc(numBlocks * sizeof(double));
//     // cudaMalloc((void**)&d_output_data, numBlocks * sizeof(double));
//     // sum_kernel<<<numBlocks, blockSize,2*blockSize>>>(d_input_data,row_size,schema["student"].second[2].acc_col_size,d_output_data, n);
//     // cudaMemcpy(h_output_data, d_output_data, numBlocks * sizeof(double), cudaMemcpyDeviceToHost);
//     // // Print the result
//     // for (int i = 0; i < numBlocks; i++) {
//     //     std::cout << "Max element in block " << i << ": " << h_output_data[i] << std::endl;
//     // }
//     // // Cleanup
//     // cudaFree(d_input_data);
//     // cudaFree(d_output_data);

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

//     // Create a planner and plan the query
//     Planner planner(context);
//     planner.CreatePlan(std::move(statements[0]));

//     // Now you can proceed with further processing or optimization
//     cout << "Planning successful!" << endl;
//     cout << "Unoptimized Logical Plan:\n"
//          << planner.plan->ToString() << endl;

//     Optimizer optimizer(*planner.binder, context);
//     auto logical_plan = optimizer.Optimize(std::move(planner.plan));
//     cout << "Optimized Logical Plan:\n";
//     cout << logical_plan->ToString() << endl;

//     auto tree_root = build_plan_tree(logical_plan.get());
//     print_tree(tree_root);

//     // Commit the transaction after planning
//     con.Commit(); // Commit transaction using Connection
// }

int main()
{
    get_schema(schema);
    int row_size;

    std::vector<char> chunk = read_csv_chunk("student", 0.5 * RAM, row_size);

    Condition conds_host[2];

    // WHERE age > 25
    conds_host[0].col_index = 1;
    conds_host[0].op = OP_NEQ;
    conds_host[0].type = 1;
    std::string fixed = "dds";
    fixed.resize(150, '\0');
    // conds_host[0].s_value = (char *)malloc(150*sizeof(char));
    memcpy(conds_host[0].s_value, fixed.c_str(), 150*sizeof(char));
    

    conds_host[1].col_index = 2;
    conds_host[1].op = OP_GT;
    conds_host[1].type = 0;
    conds_host[1].f_value = 25.0;
    // Copy to device
    Condition *d_conds;
    cudaMalloc(&d_conds, 2* sizeof(Condition));
    cudaMemcpy(d_conds, conds_host, 2 * sizeof(Condition), cudaMemcpyHostToDevice);

    // Launch kernel
    const int n = chunk.size() / row_size;
    char *d_input;
    cudaMalloc((void**)&d_input, n * row_size*sizeof(char));
    cudaMemcpy(d_input, chunk.data(), n * row_size*sizeof(char), cudaMemcpyHostToDevice);
    char *d_output;
    cudaMalloc((void**)&d_output, n * row_size*sizeof(char));
    int *d_output_counter;
    int *h_output_counter=(int*)malloc(sizeof(int));
    cudaMalloc((void**)&d_output_counter, sizeof(int));
    cudaMemset(d_output_counter, 0, sizeof(int));

    int * h_acc_col_size;
    h_acc_col_size = (int *)malloc(schema["student"].second.size() * sizeof(int));
    for (int i = 0; i < schema["student"].second.size(); i++)
    {
       
            h_acc_col_size[i] = schema["student"].second[i].acc_col_size;

    }
    int *d_acc_col_size;
    cudaMalloc((void**)&d_acc_col_size, schema["student"].second.size() * sizeof(int));
    cudaMemcpy(d_acc_col_size, h_acc_col_size, schema["student"].second.size() * sizeof(int), cudaMemcpyHostToDevice);
    get_kernel<<<3, 256>>>(d_input, row_size, d_acc_col_size,
                                    d_output, d_output_counter,
                                    d_conds, 2, n);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
    }
    // Copy result back to host
    char* output_data = (char*)malloc(n * row_size*sizeof(char));

    cudaMemcpy(output_data, d_output, n * row_size*sizeof(char), cudaMemcpyDeviceToHost);
    
    cudaMemcpy(h_output_counter, d_output_counter, sizeof(int), cudaMemcpyDeviceToHost);
    // Print the output data
    std::cout << "Filtered Output Data:" << std::endl;
    std::cout << "Filtered Output h_output_counter:"<<(*h_output_counter) << std::endl;
    for (int i = 0; i < (*h_output_counter) * row_size; i += row_size) {
        for (int j = 0; j < row_size; ++j) {
            std::cout << output_data[i + j];
        }
        std::cout << std::endl;
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_output_counter);
    cudaFree(d_conds);
    cudaFree(d_acc_col_size);
    free(h_acc_col_size);
}