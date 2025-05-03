#include <cstdio>
#include <algorithm>
#include "get.cuh"
__device__ int device_strcmp(const char *s1, const char *s2) {
    // printf("s1 %c s2 %c\n", *s1, *s2);
    while (*s1 == *s2) {
        if (*s1 == '\0') {
            return 0; // Strings are equal
        }
        // printf("s1 %c s2 %c\n", *s1, *s2);
        s1++;
        s2++;
    }
    return *(const unsigned char*)s1 - *(const unsigned char*)s2;
}


__device__ bool eval_condition(char *row_ptr, int *acc_col_size, const Condition &cond)
{
    char *field_ptr = row_ptr + acc_col_size[cond.col_index];

    if (cond.type == 0)
    { // int
        double *val=(double*)malloc(sizeof(double));
        memcpy(val,field_ptr, sizeof(double));
        switch (cond.op)
        {
        case OP_GT:
            return *val > cond.f_value;
        case OP_LT:
            return *val < cond.f_value;
        case OP_EQ:
            return *val == cond.f_value;
        case OP_NEQ:
            return *val != cond.f_value;
        case OP_GTE:
            return *val >= cond.f_value;
        case OP_LTE:
            return *val <= cond.f_value;
        }
    }
    else
    { // float
        char *val=(char*)malloc(150); ;
        memcpy(val,field_ptr,  150);
        printf("val %s\n", val);
        // printf("cond.s_value %s\n", cond.s_value);
        switch (cond.op)
        {
        case OP_EQ:
            return device_strcmp(val, cond.s_value) == 0;
        case OP_NEQ:
            return device_strcmp(val, cond.s_value) != 0;
        }
    }
    return false;
}

__global__ void get_kernel(char *input_data, int row_size, int *acc_col_size,
                           char *output_data, int *output_counter,
                           Condition *conditions, int cond_count, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    // char *row_ptr = input_data + idx * row_size;
    char *row_ptr = &input_data [idx * row_size];
    
    
    

    bool pass = true;
    for (int i = 0; i < cond_count; ++i)
    {
        if (!eval_condition(row_ptr, acc_col_size, conditions[i]))
        {
            pass = false;
            break;
        }
    }

    if (pass)
    {
        int out_idx = atomicAdd(output_counter, 1);
        printf("Row  idx %d passed\n", out_idx);
        char *out_ptr = output_data + out_idx * row_size;
        for (int i = 0; i < row_size; ++i)
        {
            out_ptr[i] = row_ptr[i];
        }
    }
}
