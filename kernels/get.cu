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
    else
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
    char *row_ptr = &input_data[idx * row_size];

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
        char *out_ptr = output_data + out_idx * row_size;
        for (int i = 0; i < row_size; ++i)
        {
            out_ptr[i] = row_ptr[i];
        }
    }
}

__host__ char *call_get_kernel(char *input_data, int row_size, int *acc_sums, Condition *conditions, int cond_count, int n, int &output_counter,int column_num)
{
    int *h_output_counter = (int *)malloc(sizeof(int));
    char *d_input_data;
    char *d_output_data;
    int *d_acc_col_size;
    int *d_output_counter;
    Condition *d_conditions;

    cudaMalloc(&d_input_data, n * row_size);
    cudaMalloc(&d_output_data, n * row_size);
    cudaMalloc(&d_acc_col_size, sizeof(int) * column_num);
    cudaMalloc(&d_output_counter, sizeof(int));
    cudaMalloc(&d_conditions, sizeof(Condition) * cond_count);

    

    cudaMemcpy(d_input_data, input_data, n * row_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_acc_col_size, acc_sums, sizeof(int) * column_num, cudaMemcpyHostToDevice);
    cudaMemcpy(d_conditions, conditions, sizeof(Condition) * cond_count, cudaMemcpyHostToDevice);
    

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
        printf("Kernel launch failed: %s \n" , cudaGetErrorString(err)) ;
    }

    // 2. Check for *asynchronous* errors (e.g., during execution)
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s \n" , cudaGetErrorString(err)) ;
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
