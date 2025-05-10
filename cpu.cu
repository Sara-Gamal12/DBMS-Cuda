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
#include "duckdb/common/reference_map.hpp"
#include "duckdb/main/query_result.hpp"
#include "duckdb/execution/task_error_manager.hpp"
#include "duckdb/execution/progress_data.hpp"
#include "duckdb/parallel/pipeline.hpp"
#include "duckdb/execution/column_binding_resolver.hpp"

#include <iostream>


#include "./utilities/schema_utilities.hpp"


#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <regex>
#include <cfloat>
using namespace std;
using namespace duckdb;

Schema schema;

int main(int argc, char *argv[])
{
    // DuckDB
    using namespace duckdb;
    DuckDB db(nullptr);
    Connection con(db);
    ClientContext &context = *con.context;
    con.Query("SET disabled_optimizers='filter_pushdown,statistics_propagation';");

    con.BeginTransaction(); // Start transaction using Connection
    con.Commit(); // Commit transaction using Connection
    while (true)
    {
        cout << "\nEnter SQL query (or type 'exit' to quit): ";
        string query;
        getline(cin, query);

        if (query == "exit" || query == "quit")
        {
            cout << "Exiting CLI.\n";
            break;
        }

        auto start = std::chrono::high_resolution_clock::now();
        get_schema(schema);


        create_tables_from_schema(con, schema);
        auto result=con.Query(query);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> duration = end - start;   

        std::cout << "Query execution time on CPU : " << duration.count() << " seconds" << std::endl;

      
        if (result->HasError()) {
            std::cerr << "Query Error: " << result->GetError() << std::endl;
        } else {
            std::ofstream file("team_14_cpu.csv");
            if (!file.is_open()) {
                std::cerr << "Failed to open output.csv for writing." << std::endl;
                return 1;
            }
        
            // Write header
            auto &names = result->names;
            for (size_t i = 0; i < names.size(); ++i) {
                file << names[i];
                if (i + 1 < names.size()) file << ",";
            }
            file << "\n";
        
            // Write rows
            while (true) {
                auto chunk = result->Fetch();
                if (!chunk || chunk->size() == 0) break;
        
                for (size_t row = 0; row < chunk->size(); ++row) {
                    for (size_t col = 0; col < chunk->ColumnCount(); ++col) {
                        file << chunk->GetValue(col, row).ToString();
                        if (col + 1 < chunk->ColumnCount()) file << ",";
                    }
                    file << "\n";
                }
            }
        
            file.close();
            std::cout << "Result written to output.csv\n";
        }   
        
    }
}