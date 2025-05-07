#include <cstdio>
#include <algorithm>
#include "agg.cuh"
#include"get.cuh"

__global__ void max_kernel(char *input_data, int row_size, int acc_col_size, double *max_element, int n)
{
    // n=num of rows in given chunk
    extern __shared__ double partial_max[];
    unsigned int t = threadIdx.x;
    unsigned int start = 2 * blockIdx.x * blockDim.x;

    // load 1st element data into shared memory
    if (start + t < n)
    {
        char *data_ptr = &input_data[(start + t) * row_size + acc_col_size];
        if (device_strcmp(data_ptr, "NULL") == 0)
            partial_max[t] = INT_MIN;
        else
            memcpy(&partial_max[t], data_ptr, sizeof(double));
    }
    else
        partial_max[t] = INT_MIN;
    // load 2nd element data into shared memory
    if (start + blockDim.x + t < n)
    {
        char *data_ptr = &input_data[(start + blockDim.x + t) * row_size + acc_col_size];
        if (device_strcmp(data_ptr, "NULL") == 0)
            partial_max[blockDim.x + t] = INT_MIN;
        else
            memcpy(&partial_max[blockDim.x + t], data_ptr, sizeof(double));

    }
    else
        partial_max[blockDim.x + t] = INT_MIN;

    
    
    
    // loop to reduce the data in shared memory
    // each thread will be responsible for 2 elements
    for (unsigned int stride = blockDim.x; stride > 0; stride /= 2)
    {
        __syncthreads();
        if (t < stride)
            partial_max[t] = max(partial_max[t + stride], partial_max[t]);
    }
    __syncthreads();
    // write the result for this block to global memory
    if (t == 0)
    {
        max_element[blockIdx.x] = partial_max[0];
    }
}

__global__ void min_kernel(char *input_data, int row_size, int acc_col_size, double *min_element, int n)
{
    extern __shared__ double partial_min[];
    unsigned int t = threadIdx.x;
    unsigned int start = 2 * blockIdx.x * blockDim.x;

    // load 1st element data into shared memory
    if (start + t < n)
    {
        char *data_ptr = &input_data[(start + t) * row_size + acc_col_size];
        if (device_strcmp(data_ptr, "NULL") == 0)
            partial_min[ t] = INT_MAX;
        else
            memcpy(&partial_min[t], data_ptr, sizeof(double));
    }
    else
        partial_min[t] = INT_MAX;
    // load 2nd element data into shared memory
    if (start + blockDim.x + t < n)
    {
        char *data_ptr = &input_data[(start + blockDim.x + t) * row_size + acc_col_size];
        if (device_strcmp(data_ptr, "NULL") == 0)
            partial_min[blockDim.x + t] = INT_MAX;
        else
            memcpy(&partial_min[blockDim.x + t], data_ptr, sizeof(double));
    }
    else
        partial_min[blockDim.x + t] = INT_MAX;
    // loop to reduce the data in shared memory
    // each thread will be responsible for 2 elements
    for (unsigned int stride = blockDim.x; stride > 0; stride /= 2)
    {
        __syncthreads();
        if (t < stride)
            if (t + stride < 2 * blockDim.x)
                partial_min[t] = min(partial_min[t + stride], partial_min[t]);
    }
    __syncthreads();
    // write the result for this block to global memory
    if (t == 0)
    {
        min_element[blockIdx.x] = partial_min[0];
    }
}

__global__ void sum_kernel(char *input_data, int row_size, int acc_col_size, double *sum_element, int n)
{
    extern __shared__ double partial_sum[];
    unsigned int t = threadIdx.x;
    unsigned int start = 2 * blockIdx.x * blockDim.x;

    // load 1st element data into shared memory
    if (start + t < n)
    {
        char *data_ptr = &input_data[(start + t) * row_size + acc_col_size];
        if (device_strcmp(data_ptr, "NULL") == 0)
            partial_sum[t] = 0;
        else
            memcpy(&partial_sum[t], data_ptr, sizeof(double));
    }
    else
        partial_sum[t] = 0;
    // load 2nd element data into shared memory
    if (start + blockDim.x + t < n)
    {
        
        char *data_ptr = &input_data[(start + blockDim.x + t) * row_size + acc_col_size];
        if (device_strcmp(data_ptr, "NULL") == 0)
            partial_sum[blockDim.x + t] = 0;
        else
            memcpy(&partial_sum[blockDim.x + t], data_ptr, sizeof(double));
        
    }
    else
        partial_sum[blockDim.x + t] = 0;
    // loop to reduce the data in shared memory
    // each thread will be responsible for 2 elements
    for (unsigned int stride = blockDim.x; stride > 0; stride /= 2)
    {
        __syncthreads();
        if (t < stride)
            if (t + stride < 2 * blockDim.x)
                partial_sum[t] = partial_sum[t + stride] + partial_sum[t];
    }
    __syncthreads();
    // write the result for this block to global memory
    if (t == 0)
    {
        sum_element[blockIdx.x] = partial_sum[0];
    }
}

__host__ double call_agg_kernel(char *input_data, int row_size, int acc_col_size, char *op, int n)
{
    char *d_input_data;
    double *d_output_data;
    double *h_output_data;

    // Allocate device memory
    cudaMalloc((void **)&d_input_data, n * row_size * sizeof(char));
    cudaMemcpy(d_input_data, input_data, n * row_size * sizeof(char), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (n + (blockSize * 2) - 1) / (blockSize * 2);
    h_output_data = (double *)malloc(numBlocks * sizeof(double));
    cudaMalloc((void **)&d_output_data, numBlocks * sizeof(double));


    size_t shared=2 * blockSize * sizeof(double); ;
    if (strcmp(op, "max") == 0)
        max_kernel<<<numBlocks, blockSize, shared>>>(d_input_data, row_size, acc_col_size, d_output_data, n);
    else if (strcmp(op, "min") == 0)
        min_kernel<<<numBlocks, blockSize,shared>>>(d_input_data, row_size, acc_col_size, d_output_data, n);
    else if (strcmp(op, "sum") == 0 || strcmp(op, "avg") == 0)
        sum_kernel<<<numBlocks, blockSize, shared>>>(d_input_data, row_size, acc_col_size, d_output_data, n);

    // copy back the data
    cudaMemcpy(h_output_data, d_output_data, numBlocks * sizeof(double), cudaMemcpyDeviceToHost);

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

    double result;
    if (strcmp(op, "max") == 0)
    {
        result = INT_MIN;
        for (int i = 0; i < numBlocks; i++)
            result = max(result, h_output_data[i]);
    }
    else if (strcmp(op, "min") == 0)
    {
        result = INT_MAX;
        for (int i = 0; i < numBlocks; i++)
            result = min(result, h_output_data[i]);
    }
    else if (strcmp(op, "sum") == 0 || strcmp(op, "avg") == 0)
    {
        result = 0;
        for (int i = 0; i < numBlocks; i++)
            result += h_output_data[i];

        if (strcmp(op, "avg") == 0)
            result /= n;
    }

    // Cleanup
    cudaFree(d_input_data);
    cudaFree(d_output_data);

    return result;
}
