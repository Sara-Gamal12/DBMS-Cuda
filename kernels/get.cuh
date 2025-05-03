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

__global__ void get_kernel(char *input_data, int row_size, int *acc_col_size,
                           char *output_data, int *output_counter,
                           Condition *conditions, int cond_count, int n);

#endif // GET_KERNELS_H
