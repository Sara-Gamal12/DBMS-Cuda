// kernels/get.cuh
#ifndef GET_KERNELS_H
#define GET_KERNELS_H

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
    int type;         // 0 = numerical, 1 = string
    union {
        float f_value;
        char s_value[150];
    };
};

__device__ int device_strcmp(const char *s1, const char *s2);

__device__ bool eval_condition(char *row_ptr, int *acc_col_size, const Condition &cond);

__global__ void get_kernel(char *input_data, int row_size, int *acc_col_size,
                           char *output_data, int *output_counter,
                           Condition *conditions, int cond_count, int n);

__host__ char *call_get_kernel(char *input_data, int row_size, int *acc_sums, Condition *conditions, int cond_count, int n, int &output_counter,int column_num);

#endif // GET_KERNELS_H
