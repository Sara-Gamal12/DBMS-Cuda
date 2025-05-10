// kernels/get.cuh
#ifndef GET_KERNELS_H
#define GET_KERNELS_H
#include <cuda_runtime.h>
#include <vector>

enum Operator {
    OP_GT,    // >
    OP_LT,    // <
    OP_EQ,    // ==
    OP_NEQ,   // !=
    OP_GTE,   // >=
    OP_LTE    // <=
    // IS NOT NULL
    // IS NULL
};

struct Condition {
    int col_index;    // index into acc_col_size[]
    Operator op;      // operation
    int type;         // 0 = numerical, 1 = string ,2 col-numeric,3col-text
    union {
        double f_value;
        char s_value[150];
        int sec_col_index; // index into acc_col_size[]
    };
};

enum ConditionTokenType {
    TOKEN_CONDITION,  // actual condition (like col1 > 5)
    TOKEN_AND,
    TOKEN_OR
};

struct ConditionToken {
    ConditionTokenType type;
    Condition condition; // valid if type == TOKEN_CONDITION
};

__device__ int device_strcmp(const char *s1, const char *s2);
__global__ void get_kernel(char *input_data, int row_size, int *acc_col_size,
                           char *output_data, int *output_counter,
                           ConditionToken *tokens, int token_count, int n);

__host__ char *call_get_kernel(char *input_data, int row_size, int *acc_sums, std::vector<ConditionToken> condition_tokens, int cond_count, int n, int &output_counter, int column_num,cudaStream_t stream);


#endif // GET_KERNELS_H
