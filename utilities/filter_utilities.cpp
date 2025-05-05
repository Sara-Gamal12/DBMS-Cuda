#include "filter_utilities.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cctype>
#include <algorithm>



// Helper to determine if a token is an operator
bool is_operator(const std::string &s)
{
    return s == "AND" || s == "OR" || s == "or" || s == "and";
}

int precedence(const std::string &op)
{
    if (op == "OR" || op == "or")
        return 1;
    if (op == "AND" || op == "and")
        return 2;
    return 0;
}

// Convert infix to postfix (RPN)
std::vector<std::string> infix_to_postfix(const std::vector<Token> &tokens)
{
    std::vector<std::string> output;
    std::stack<std::string> ops;

    for (const auto &tok : tokens)
    {
        if (tok.type == OPERAND)
        {
            output.push_back(tok.value);
        }
        else if (tok.type == OPERATOR)
        {
            while (!ops.empty() && precedence(ops.top()) >= precedence(tok.value))
            {
                output.push_back(ops.top());
                ops.pop();
            }
            ops.push(tok.value);
        }
        else if (tok.type == LEFT_PAREN)
        {
            ops.push(tok.value);
        }
        else if (tok.type == RIGHT_PAREN)
        {
            while (!ops.empty() && ops.top() != "(")
            {
                output.push_back(ops.top());
                ops.pop();
            }
            ops.pop(); // discard "("
        }
    }

    while (!ops.empty())
    {
        output.push_back(ops.top());
        ops.pop();
    }

    return output;
}

// Very basic tokenizer
std::vector<Token> tokenize(std::vector<std::string> expr)
{
    std::vector<Token> tokens;
    std::string token;
    for (int i = 0; i < expr.size(); i++)
    {
        token = expr[i];
        if (token == "AND" || token == "OR"||token == "and" || token == "or")
        {
            tokens.push_back({token, OPERATOR});
        }
        else if (token == "(")
        {
            tokens.push_back({token, LEFT_PAREN});
        }
        else if (token == ")")
        {
            tokens.push_back({token, RIGHT_PAREN});
        }
        else
        {
            tokens.push_back({token, OPERAND});
        }
    }
    return tokens;
}

std::vector<ConditionToken> parse_postfix(std::vector<std::string> postfix, std::vector<ColumnInfo> schema,int * acc_sums)
{

    std::vector<ConditionToken> stack;
    for (int i = 0; i < postfix.size(); i++)
    {
        ConditionToken token;
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
            Condition condition;
            std::string condition_str = postfix[i];
            std::istringstream iss(condition_str);
            std::string column, op, value;
            iss >> column >> op >> value;
            
            for (int j = 0; j < schema.size(); ++j)
            {

                if (schema[j].name== column)
                {
                    condition.col_index = j;
                    if (schema[j].type == "Numeric")
                    {
                        condition.type = 0;
                        condition.f_value = std::stof(value);
                    }
                    else if (schema[j].type == "Text")
                    {
                        condition.type = 1;
                        strncpy(condition.s_value, value.c_str(), sizeof(condition.s_value));
                    }
                    // else if (schema[table_name].second[j].type == "DateTime")
                    // {
                    //     conditions[i-1].type = 2;
                    // }
                }
                acc_sums[j] = schema[j].acc_col_size;
            }

            if (op == "EQUAL" )
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

            token.condition = condition;
        }
        stack.push_back(token);
    }
    return stack;
}





std::vector<std::string> tokenizeExpression(const std::string& input) {
    std::vector<std::string> tokens;
    std::string buffer;
    std::istringstream stream(input);
    char c;

    auto flushBuffer = [&]() {
        if (!buffer.empty()) {
            std::string trimmed = buffer;
            trimmed.erase(trimmed.begin(), std::find_if(trimmed.begin(), trimmed.end(), [](unsigned char ch) {
                return !std::isspace(ch);
            }));
            trimmed.erase(std::find_if(trimmed.rbegin(), trimmed.rend(), [](unsigned char ch) {
                return !std::isspace(ch);
            }).base(), trimmed.end());

            if (trimmed == "AND" || trimmed == "OR" || trimmed == "and" || trimmed == "or") {
                tokens.push_back(trimmed);
            } else if (!trimmed.empty()) {
                tokens.push_back(trimmed);
            }
            buffer.clear();
        }
    };

    while (stream.get(c)) {
        if (c == '(' || c == ')') {
            flushBuffer();
            tokens.push_back(std::string(1, c));
        } else {
            buffer += c;
        }
    }
    flushBuffer();

    return tokens;
}



std::string  replace_operatirs(std::string &input)
{
    std::string output;
    std::istringstream iss(input);
    std::string token;

    while (iss >> token)
    {
        if (token == "=")
            output += " EQUAL ";
        else if (token == ">")
            output += " GREATERTHAN ";
        else if (token == "<")
            output += " LESSTHAN ";
        else if (token == "!=")
            output += " NOTEQUAL ";
        else if (token == ">=")
            output += " GREATERTHANOREQUALTO ";
        else if (token == "<=")
            output += " LESSTHANOREQUALTO ";
        else
            output += token ;
    }
    return output;
}
