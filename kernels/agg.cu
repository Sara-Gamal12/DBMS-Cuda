#include <cstdio>
#include <algorithm>
#include "agg.cuh"

__global__ void max_kernel(char* input_data,int row_size,int acc_col_size,double * max_element, int n) {
    // n=num of rows in given chunk
    extern __shared__ double partial_max[];
    unsigned int t = threadIdx.x;
    unsigned int start = 2*blockIdx.x*blockDim.x;

    // load 1st element data into shared memory
    if (start + t < n) {
    char* data_ptr = &input_data[(start + t) * row_size + acc_col_size];
    memcpy(&partial_max[t], data_ptr, sizeof(double));
    }
    else
         partial_max[t] = INT_MIN;
    // load 2nd element data into shared memory
    if (start + blockDim.x + t < n)
      {char* data_ptr = &input_data[(start + blockDim.x+t)*row_size + acc_col_size];
        memcpy(& partial_max[blockDim.x+t], data_ptr, sizeof(double));
    }
    else
        partial_max[blockDim.x+t] = INT_MIN;
    // loop to reduce the data in shared memory
    // each thread will be responsible for 2 elements
    for (unsigned int stride = blockDim.x; stride > 0;  stride /= 2) 
    {
        __syncthreads();
        if (t < stride) 
                partial_max[t]= max(partial_max[t+stride], partial_max[t]);
    }
    __syncthreads();
    // write the result for this block to global memory
    if (t == 0) {
        max_element[blockIdx.x] = partial_max[0];
    }
}


__global__ void min_kernel( char *input_data, int row_size, int acc_col_size, double *min_element, int n) {
    extern __shared__ double partial_min[];
    unsigned int t = threadIdx.x;
    unsigned int start = 2*blockIdx.x*blockDim.x;

    // load 1st element data into shared memory
    if (start + t < n) 
        {
            char* data_ptr = &input_data[(start + t) * row_size + acc_col_size];
    memcpy(&partial_min[t], data_ptr, sizeof(double));
        }
    else
        partial_min[t] = INT_MAX;
    // load 2nd element data into shared memory
    if (start + blockDim.x + t < n)
       {
        {char* data_ptr = &input_data[(start + blockDim.x+t)*row_size + acc_col_size];
            memcpy(& partial_min[blockDim.x+t], data_ptr, sizeof(double));
        }
       }
    else
        partial_min[blockDim.x+t] = INT_MAX;
    // loop to reduce the data in shared memory
    // each thread will be responsible for 2 elements
    for (unsigned int stride = blockDim.x; stride > 0;  stride /= 2) 
    {
        __syncthreads();
        if (t < stride) 
            if (t + stride < 2*blockDim.x )
                partial_min[t]= min(partial_min[t+stride], partial_min[t]);
    }
    __syncthreads();
    // write the result for this block to global memory
    if (t == 0) {
        min_element[blockIdx.x] = partial_min[0];
    }
}



__global__ void sum_kernel( char *input_data, int row_size, int acc_col_size, double *sum_element, int n) {
    extern __shared__ double partial_sum[];
    unsigned int t = threadIdx.x;
    unsigned int start = 2*blockIdx.x*blockDim.x;

    // load 1st element data into shared memory
    if (start + t < n) 
        {
            char* data_ptr = &input_data[(start + t) * row_size + acc_col_size];
    memcpy(&partial_sum[t], data_ptr, sizeof(double));
        }
    else
        partial_sum[t] = INT_MAX;
    // load 2nd element data into shared memory
    if (start + blockDim.x + t < n)
       {
        {char* data_ptr = &input_data[(start + blockDim.x+t)*row_size + acc_col_size];
            memcpy(& partial_sum[blockDim.x+t], data_ptr, sizeof(double));
        }
       }
    else
        partial_sum[blockDim.x+t] = 0;
    // loop to reduce the data in shared memory
    // each thread will be responsible for 2 elements
    for (unsigned int stride = blockDim.x; stride > 0;  stride /= 2) 
    {
        __syncthreads();
        if (t < stride) 
            if (t + stride < 2*blockDim.x )
                partial_sum[t]= partial_sum[t+stride]+ partial_sum[t];
    }
    __syncthreads();
    // write the result for this block to global memory
    if (t == 0) {
        sum_element[blockIdx.x] = partial_sum[0];
    }
}



