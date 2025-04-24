#include "schema_utilities.hpp"





void print_schema(Schema schema) {
    for (const auto &table_pair : schema) {
        const std::string &table_name = table_pair.first;
        const std::vector<ColumnInfo> &columns = table_pair.second.second;

        std::cout << "Table: " << table_name << "\n";
        for (const auto &col : columns) {
            std::cout << "  - " << col.name;
            if (col.is_primary) {
                std::cout << " (Primary Key)";
            }
            if (!col.foreign_table.empty()) {
                std::cout << " (Foreign Key → " << col.foreign_table << ")";
            }
            std::cout<<" Type: "<<col.type;
            std::cout<<" Size: "<<col.size_in_bytes<<" bytes";
            std::cout << "\n";
        }
        std::cout << "\n";
    }
}

// Function to get the schema of the tables
void get_schema(Schema &schema) {
    for (const auto& entry : std::filesystem::directory_iterator("../DB")) {
        if (entry.path().extension() == ".csv") {
            std::string file_path = entry.path().string();
            std::string table_name = entry.path().stem().string();

            std::shared_ptr<std::ifstream> file = std::make_shared<std::ifstream>(file_path);
            if (!file->is_open()) {
                std::cerr << "Failed to open file: " << file_path << std::endl;
                continue;
            }

            std::string header_line;
            if (!std::getline(*file, header_line)) {
                std::cerr << "Failed to read header from: " << file_path << std::endl;
                continue;
            }

            schema[table_name].first = file;
            // Reset file pointer to the beginning

            std::stringstream ss(header_line);
            std::string column;
            std::vector<ColumnInfo> columns;

            while (std::getline(ss, column, ',')) {
                ColumnInfo col;
                col.name = column;

                // Detect primary key
                if (col.name.find("(P)") != std::string::npos) {
                    col.is_primary = true;
                    col.name.erase(col.name.find("(P)"), 3);
                }

                // Detect foreign key
                if (col.name[0] == '#' && col.name.rfind("_") != std::string::npos) {
                    size_t start = 1;
                    size_t end = col.name.rfind("_");
                    col.foreign_table = col.name.substr(start, end - start);
                    col.name = col.name.substr(end + 1);
                }

                // Detect column types
                if (col.name.find("(N)") != std::string::npos) {
                    col.type = "Numeric";
                    col.size_in_bytes = 8;
                    col.name.erase(col.name.find("(N)"), 3);
                } else if (col.name.find("(T)") != std::string::npos) {
                    col.type = "Text";
                    col.size_in_bytes = 150;
                    col.name.erase(col.name.find("(T)"), 3);
                } else if (col.name.find("(D)") != std::string::npos) {
                    col.type = "DateTime";
                    col.size_in_bytes = 8;
                    col.name.erase(col.name.find("(D)"), 3);
                }

                columns.push_back(col);
            }

            schema[table_name].second = columns;
        }
    }
}