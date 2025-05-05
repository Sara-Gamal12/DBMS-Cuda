#ifndef FILTER_UTILITIES_HPP
#define FILTER_UTILITIES_HPP
#include<iostream>
#include<string>
#include<vector>
#include<stack>
#include<sstream>
#include "../kernels/get.cuh"
#include "schema_utilities.hpp"

enum TokenType { OPERAND, OPERATOR, LEFT_PAREN, RIGHT_PAREN };

struct Token {
    std::string value;
    TokenType type;
};

bool is_operator(const std::string &s);
int precedence(const std::string &op);
std::vector<std::string> infix_to_postfix(const std::vector<Token> &tokens);
std::vector<Token> tokenize(std::vector<std::string> expr);
std::vector<ConditionToken> parse_postfix(std::vector<std::string> postfix, std::vector<ColumnInfo> schema,int * acc_sums);
std::vector<std::string> tokenizeExpression(const std::string &input);
std::string  replace_operatirs(std::string &input);

#endif // FILTER_UTILITIES_HPP