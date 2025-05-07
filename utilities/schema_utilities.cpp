#include "schema_utilities.hpp"

extern Schema schema; // External declaration of schema

void print_chunk(std::vector<char> chunk, std::vector<ColumnInfo> cols,std::unordered_map<std::string, std::string> alias_map)
{

    int row_size = 0;
    for (const auto &col : cols)
    {
        row_size += col.size_in_bytes;
    }

    size_t offset = 0;
    for (size_t i = 0; i < chunk.size() / row_size; ++i)
    {
        size_t row_start = i * row_size;
        offset = 0;

        for (auto &col : cols)
        {
            if(alias_map.find(col.name) != alias_map.end())
            {
                col.name = alias_map[col.name];
            }
            std::cout << "Column " << col.name << ": ";

            // Get pointer to this column's data within the row
            char *data_ptr = chunk.data() + row_start + offset;

            if(strcmp(data_ptr, "NULL") == 0)
            {
                std::cout << "NULL |";
                offset += col.size_in_bytes;
                continue;
            }
            if (col.type == "Numeric")
            {
                double value = *reinterpret_cast<double *>(data_ptr);
                std::cout << value;
                offset += sizeof(double);
            }
            else if (col.type == "DateTime")
            {

                std::time_t time_value = *reinterpret_cast<std::time_t *>(data_ptr);
                std::tm *tm = std::localtime(&time_value);
                char buffer[20];
                std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", tm);
                std::cout << buffer;
                offset += sizeof(std::time_t);
            }
            else if (col.type == "Text")
            {
                std::string text(data_ptr, col.size_in_bytes);
                std::cout << text.c_str(); // safe printing
                offset += col.size_in_bytes;
            }

            std::cout << " | ";
        }

        std::cout << std::endl;
    }
}

std::vector<char> read_csv_chunk(std::string table_name, long chunk_size_in_bytes, int &row_size)
{

    if (schema.find(table_name) == schema.end())
    {
        std::cout << "Table schema not found for: " << table_name << std::endl;
        return {};
    }

    std::shared_ptr<std::ifstream> file = schema.at(table_name).first;

    std::vector<ColumnInfo> cols = schema.at(table_name).second;
    row_size = 0;
    for (const auto &col : cols)
    {
        row_size += col.size_in_bytes;
    }

    std::vector<char> chunk;
    std::string line;
    long total_size_in_bytes = 0;

    while (total_size_in_bytes + row_size <= chunk_size_in_bytes && std::getline(*file, line))
    {
        std::stringstream ss(line);
        std::string token;

        // std::cout << line << std::endl;
        for (const auto &col : cols)
        {
            std::getline(ss, token, ',');
            token=clean_column(token);
            if (col.type == "Numeric")
            {
                if (token == "")
                {
                    char *null = (char *)malloc(col.size_in_bytes);
                    null = "NULL";
                    chunk.insert(chunk.end(), null, null + col.size_in_bytes);
                }
                else
                {
                    double value = std::stof(token);
                    const char *ptr = reinterpret_cast<const char *>(&value);
                    chunk.insert(chunk.end(), ptr, ptr + col.size_in_bytes);
                }
            }
            else if (col.type == "DateTime")
            {
                if (token == "")
                {
                    char *null = (char *)malloc(col.size_in_bytes);
                    null = "NULL";
                    chunk.insert(chunk.end(), null, null + col.size_in_bytes);
                }
                else
                {
                    std::tm tm = {};
                    std::istringstream ss(token);
                    ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
                    std::time_t time_value = std::mktime(&tm);
                    const char *ptr = reinterpret_cast<const char *>(&time_value);
                    chunk.insert(chunk.end(), ptr, ptr + col.size_in_bytes);
                }
            }
            else if (col.type == "Text")
            {
                if (token == "")
                {
                    char *null = (char *)malloc(col.size_in_bytes);
                    null = "NULL";
                    chunk.insert(chunk.end(), null, null + col.size_in_bytes);
                }
                else
                {
                    // Truncate or pad string to fit size
                    std::string fixed = token.substr(0, col.size_in_bytes);
                    fixed.resize(col.size_in_bytes, '\0');
                    chunk.insert(chunk.end(), fixed.begin(), fixed.end());
                }
            }
            total_size_in_bytes += row_size;
        }
    }
    // print_chunk(chunk, table_name);
    return chunk;
}

std::string to_duckdb_type(const std::string &type)
{
    if (type == "Text")
        return "VARCHAR(150)";
    if (type == "Numeric")
        return "DOUBLE"; // or FLOAT depending on your use
    if (type == "DateTime")
        return "TIMESTAMP";
    return "VARCHAR(150)";
}

// helper to join strings
std::string join(const std::vector<std::string> &vec, const std::string &sep)
{
    std::ostringstream oss;
    for (size_t i = 0; i < vec.size(); ++i)
    {
        oss << vec[i];
        if (i + 1 < vec.size())
            oss << sep;
    }
    return oss.str();
}

void create_tables_from_schema(duckdb::Connection &conn, const Schema &schema)
{
    std::string forign_key = "";
    for (const auto &[table_name, pair] : schema)
    {
        const auto &columns = pair.second;
        std::string sql = "CREATE TABLE " + table_name + " (";

        std::vector<std::string> col_defs;
        std::string primary_key;

        for (const auto &col : columns)
        {
            std::string col_def = col.name + " " + to_duckdb_type(col.type);
            col_defs.push_back(col_def);

            if (col.is_primary)
            {
                primary_key = col.name;
            }

            if (!col.foreign_table.empty())
            {
                forign_key += "ALTER TABLE " + table_name + " ADD FOREIGN KEY (" + col.name + ") REFERENCES " + col.foreign_table + ";";
            }
        }

        sql += join(col_defs, ", ");

        if (!primary_key.empty())
        {
            sql += ", PRIMARY KEY(" + primary_key + ")";
        }

        sql += ");";

        conn.Query(sql);

        std::string insert_sql = "INSERT INTO " + table_name + " SELECT * FROM read_csv_auto('../DB/" + table_name + ".csv', HEADER=true);";
        conn.Query(insert_sql);
    }

    conn.Query(forign_key);
}

std::string clean_column(const std::string &s)
{
    std::string result = s;
    result.erase(remove_if(result.begin(), result.end(), ::isspace), result.end()); // remove spaces
    result.erase(remove(result.begin(), result.end(), '\r'), result.end());         // remove \r
    result.erase(remove(result.begin(), result.end(), '\n'), result.end());         // remove \n

    if ((unsigned char)result[0] == 0xEF &&
        (unsigned char)result[1] == 0xBB &&
        (unsigned char)result[2] == 0xBF)
    {
        result = result.substr(3); // Remove BOM
    }
    return result;
}

// Function to get the schema of the tables
void get_schema(Schema &schema)
{
    for (const auto &entry : std::filesystem::directory_iterator("../DB"))
    {
        int acc_col_size = 0;
        if (entry.path().extension() == ".csv")
        {
            std::string file_path = entry.path().string();
            std::string table_name = entry.path().stem().string();

            std::shared_ptr<std::ifstream> file = std::make_shared<std::ifstream>(file_path);
            if (!file->is_open())
            {
                std::cerr << "Failed to open file: " << file_path << std::endl;
                continue;
            }

            std::string header_line;
            if (!std::getline(*file, header_line))
            {
                std::cerr << "Failed to read header from: " << file_path << std::endl;
                continue;
            }

            schema[table_name].first = file;
            // Reset file pointer to the beginning

            std::stringstream ss(header_line);
            std::string column;
            std::vector<ColumnInfo> columns;

            while (std::getline(ss, column, ','))
            {
                ColumnInfo col;
                column = clean_column(column);
                col.name = column;

                // Detect primary key
                if (col.name.find("(P)") != std::string::npos)
                {
                    col.is_primary = true;
                    col.name.erase(col.name.find("(P)"), 3);
                }

                // Detect foreign key
                if (col.name[0] == '#' && col.name.rfind("_") != std::string::npos)
                {
                    size_t start = 1;
                    size_t end = col.name.rfind("_");
                    col.foreign_table = col.name.substr(start, end - start);
                    col.name = col.name.substr(end + 1);
                }

                // Detect column types
                if (col.name.find("(N)") != std::string::npos)
                {
                    col.type = "Numeric";
                    col.size_in_bytes = 8;
                    col.name.erase(col.name.find("(N)"), 3);
                }
                else if (col.name.find("(T)") != std::string::npos)
                {
                    col.type = "Text";
                    col.size_in_bytes = 150;
                    col.name.erase(col.name.find("(T)"), 3);
                }
                else if (col.name.find("(D)") != std::string::npos)
                {
                    col.type = "DateTime";
                    col.size_in_bytes = 8;
                    col.name.erase(col.name.find("(D)"), 3);
                }

                col.name.erase(std::remove(col.name.begin(), col.name.end(), ' '), col.name.end());
                col.acc_col_size = acc_col_size;
                acc_col_size += col.size_in_bytes;
                columns.push_back(col);
            }

            schema[table_name].second = columns;
        }
    }
}