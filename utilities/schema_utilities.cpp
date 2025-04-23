#include "schema_utilities.hpp"




void print_schema(std::unordered_map<std::string, std::vector<ColumnInfo>> schema) {
    for (const auto &table_pair : schema) {
        const std::string &table_name = table_pair.first;
        const std::vector<ColumnInfo> &columns = table_pair.second;

        std::cout << "Table: " << table_name << "\n";
        for (const auto &col : columns) {
            std::cout << "  - " << col.name;
            if (col.is_primary) {
                std::cout << " (Primary Key)";
            }
            if (!col.foreign_table.empty()) {
                std::cout << " (Foreign Key → " << col.foreign_table << ")";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
}

// Function to get the schema of the tables
void get_schema(Connection *con, std::unordered_map<std::string, std::vector<ColumnInfo>> &schema) {
    for (const auto& entry : std::filesystem::directory_iterator("../DB")) {
        if (entry.path().extension() == ".csv") {
            std::string file_path = entry.path().string();
            std::string table_name = entry.path().stem().string();

            std::vector<ColumnInfo> columns;

            auto relation = con->TableFunction("read_csv_auto", {file_path});
            auto result = relation->Execute();

            auto &names = result->names;
            auto &types = result->types;

            for (size_t i = 0; i < names.size(); ++i) {
                ColumnInfo col;
                col.name = names[i];
                auto type_str = types[i].ToString();
                
                // Detect primary key
                if (col.name.find("(P)") != std::string::npos) {
                    col.is_primary = true;
                    col.name.erase(col.name.find("(P)"), 3); // Remove (P) from name
                }

                // Detect foreign key
                if (col.name[0]=='#'&& col.name.rfind("_") != std::string::npos) {
                    size_t start = 1;
                    size_t end = col.name.rfind("_");
                    col.foreign_table = col.name.substr(start, end - start);
                    col.name = col.name.substr(end +1,end); // Remove foreign key prefix
                }

                columns.push_back(col);
                
            }

            schema[table_name] = columns;
        }
    }

}

