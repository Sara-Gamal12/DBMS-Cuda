
using namespace std;
#include "join_utilities.hpp"

std::vector<JoinConditionToken> join_parse_postfix(std::vector<std::string> postfix, std::vector<ColumnInfo> schema_a, std::vector<ColumnInfo> schema_b, int *acc_sums_a, int *acc_sums_b)
{
    std::vector<JoinConditionToken> stack;
    for (int i = 0; i < postfix.size(); i++)
    {
        JoinConditionToken token;
        if (is_operator(postfix[i]))
        {
            if (postfix[i] == "and" || postfix[i] == "AND")
                token.type = TOKEN_AND;
            else if (postfix[i] == "or" || postfix[i] == "OR")
                token.type = TOKEN_OR;
        }
        else
        {
            token.type = TOKEN_CONDITION;
            Join_Condition condition;
            std::string condition_str = postfix[i];
            std::istringstream iss(condition_str);
            std::string column_a, op, column_b;
            iss >> column_a >> op>> column_b;
             for (int j = 0; j < schema_a.size(); ++j)
            {
                for (int k = 0; k < schema_b.size(); ++k)
                {
                    if (schema_a[j].name == column_a && schema_b[k].name == column_b)
                    {
                        // cout << "Found schema_b[i].name: " << schema_a[j].name << " and schema_b[k].name " << schema_b[k].name << endl;
                        // cout << "Found schema_b[i].type: " << schema_a[j].type << " and schema_b[k].type " << schema_b[k].type << endl;
                        // cout<< " j = " << j << " k = " << k << endl;

                        if (schema_a[j].type == "Numeric" || schema_a[j].type == "DateTime")
                        {
                            condition.col_index_a = j;
                            condition.col_index_b = k;
                            condition.type = 0;
                        }
                        else if (schema_a[j].type == "Text")
                        {
                            condition.col_index_a = j;
                            condition.col_index_b = k;
                            condition.type = 1;
                        }
                    }
                    acc_sums_b[k] = schema_b[k].acc_col_size;
                }
                acc_sums_a[j] = schema_a[j].acc_col_size;
            }
            if (op == "EQUAL")
            {
                condition.op = OP_EQ;
            }
            else if (op == "GREATERTHAN")
            {
                condition.op = OP_GT;
            }
            else if (op == "LESSTHAN")
            {
                condition.op = OP_LT;
            }
            else if (op == "NOTEQUAL")
            {
                condition.op = OP_NEQ;
            }
            else if (op == "GREATERTHANOREQUALTO")
            {
                condition.op = OP_GTE;
            }
            else if (op == "LESSTHANOREQUALTO")
            {
                condition.op = OP_LTE;
            }
            token.joinCond = condition;
        }
        stack.push_back(token);
    }
    return stack;
}
