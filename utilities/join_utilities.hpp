#ifndef JOIN_UTILITIES_HPP
#define JOIN_UTILITIES_HPP
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include "../kernels/join.cuh"
#include "schema_utilities.hpp"
#include "filter_utilities.hpp"

std::vector<JoinConditionToken> join_parse_postfix(std::vector<std::string> postfix, std::vector<ColumnInfo> schema_a, std::vector<ColumnInfo> schema_b, int *acc_sums_a, int *acc_sums_b);
#endif // JOIN_UTILITIES_HPP
