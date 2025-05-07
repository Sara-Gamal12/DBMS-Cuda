#include <cstdio>
#include <algorithm>
#include "get.cuh"

__device__ int device_strcmp(const char *s1, const char *s2)
{

    while (*s1 == *s2)
    {
        if (*s1 == '\0')
        {
            return 0; // Strings are equal
        }

        s1++;
        s2++;
    }
    return *(const unsigned char *)s1 - *(const unsigned char *)s2;
}

__device__ bool eval_condition(char *row_ptr, int *acc_col_size, const Condition &cond)
{
    char *field_ptr = row_ptr + acc_col_size[cond.col_index];
    if (device_strcmp(field_ptr, "NULL") == 0)
    {
        return false;
    }

    if (cond.type == 0)
    { // int
        double *val = (double *)malloc(sizeof(double));
        memcpy(val, field_ptr, sizeof(double));
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
    else if (cond.type == 1)
    { // float
        char *val = (char *)malloc(150);
        ;
        memcpy(val, field_ptr, 150);
        switch (cond.op)
        {
        case OP_EQ:
            return device_strcmp(val, cond.s_value) == 0;
        case OP_NEQ:
            return device_strcmp(val, cond.s_value) != 0;
        }
    }
    else if (cond.type == 2)
    { // column
        char *col_ptr = row_ptr + acc_col_size[cond.sec_col_index];
        if (device_strcmp(col_ptr, "NULL") == 0)
        {
            return false;
        }
        double *val1 = (double *)malloc(sizeof(double));
        memcpy(val1, field_ptr, sizeof(double));
        double *val2 = (double *)malloc(sizeof(double));
        memcpy(val2, col_ptr, sizeof(double));
        switch (cond.op)
        {
        case OP_GT:
            return *val1 > *val2;
        case OP_LT:
            return *val1 < *val2;
        case OP_EQ:
            return *val1 == *val2;
        case OP_NEQ:
            return *val1 != *val2;
        case OP_GTE:
            return *val1 >= *val2;
        case OP_LTE:
            return *val1 <= *val2;
        }
    }
    else if (cond.type == 3)
    { // float
        char *col_ptr = row_ptr + acc_col_size[cond.sec_col_index];
        if (device_strcmp(col_ptr, "NULL") == 0)
        {
            return false;
        }
        char *val1 = (char *)malloc(150);
        char *val2 = (char *)malloc(150);
        memcpy(val1, field_ptr, 150);
        memcpy(val2, col_ptr, 150);
        switch (cond.op)
        {
        case OP_EQ:
            return device_strcmp(val1, val2) == 0;
        case OP_NEQ:
            return device_strcmp(val1, val2) != 0;
        }
    }
    return false;
}

__device__ bool eval_condition_tokens(char *row_ptr, int *acc_col_size, ConditionToken *tokens, int token_count)
{
    bool stack[16]; // adjust size if needed
    int sp = 0;     // stack pointer

    for (int i = 0; i < token_count; ++i)
    {
        if (tokens[i].type == TOKEN_CONDITION)
        {
            bool res = eval_condition(row_ptr, acc_col_size, tokens[i].condition);
            stack[sp++] = res;
        }
        else if (tokens[i].type == TOKEN_AND)
        {
            if (sp < 2)
                return false; // stack underflow
            bool b = stack[--sp];
            bool a = stack[--sp];
            stack[sp++] = a && b;
        }
        else if (tokens[i].type == TOKEN_OR)
        {
            if (sp < 2)
                return false;
            bool b = stack[--sp];
            bool a = stack[--sp];
            stack[sp++] = a || b;
        }
    }

    return sp == 1 ? stack[0] : false;
}

__global__ void get_kernel(char *input_data, int row_size, int *acc_col_size,
                           char *output_data, int *output_counter,
                           ConditionToken *tokens, int token_count, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    // char *row_ptr = input_data + idx * row_size;
    char *row_ptr = &input_data[idx * row_size];

    bool pass = eval_condition_tokens(row_ptr, acc_col_size, tokens, token_count);

    if (pass)
    {
        int out_idx = atomicAdd(output_counter, 1);
        char *out_ptr = output_data + out_idx * row_size;
        for (int i = 0; i < row_size; ++i)
        {
            out_ptr[i] = row_ptr[i];
        }
    }
}

__host__ char *call_get_kernel(char *input_data, int row_size, int *acc_sums, std::vector<ConditionToken> condition_tokens, int cond_count, int n, int &output_counter, int column_num)
{

    int *h_output_counter = (int *)malloc(sizeof(int));
    char *d_input_data;
    char *d_output_data;
    int *d_acc_col_size;
    int *d_output_counter;
    ConditionToken *d_conditions;
    ConditionToken *conditions = condition_tokens.data();

    cudaMalloc(&d_input_data, n * row_size);
    cudaMalloc(&d_output_data, n * row_size);
    cudaMalloc(&d_acc_col_size, sizeof(int) * column_num);
    cudaMalloc(&d_output_counter, sizeof(int));
    cudaMalloc(&d_conditions, sizeof(ConditionToken) * cond_count);

    cudaMemcpy(d_input_data, input_data, n * row_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_acc_col_size, acc_sums, sizeof(int) * column_num, cudaMemcpyHostToDevice);
    cudaMemcpy(d_conditions, conditions, sizeof(ConditionToken) * cond_count, cudaMemcpyHostToDevice);

    cudaMemset(d_output_counter, 0, sizeof(int));

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    get_kernel<<<numBlocks, blockSize>>>(d_input_data, row_size,
                                         d_acc_col_size,
                                         d_output_data,
                                         d_output_counter,
                                         d_conditions,
                                         cond_count,
                                         n);

    // 1. Check for *launch* errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Kernel launch failed: %s \n", cudaGetErrorString(err));
    }

    // 2. Check for *asynchronous* errors (e.g., during execution)
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        printf("Kernel launch failed: %s \n", cudaGetErrorString(err));
    }

    cudaMemcpy(h_output_counter, d_output_counter, sizeof(int), cudaMemcpyDeviceToHost);
    char *h_output_data = (char *)malloc(*h_output_counter * row_size * sizeof(char));
    cudaMemcpy(h_output_data, d_output_data, *h_output_counter * row_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input_data);
    cudaFree(d_output_data);
    cudaFree(d_acc_col_size);
    cudaFree(d_output_counter);
    cudaFree(d_conditions);

    output_counter = *h_output_counter;
    return h_output_data;
}
