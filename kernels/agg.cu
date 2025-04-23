#include <cstdio>
#include <algorithm>
#include "agg.cuh"

__global__ void max_kernel(const float* input_data,float * max_element, int n) {
    extern __shared__ float partial_max[];
    unsigned int t = threadIdx.x;
    unsigned int start = 2*blockIdx.x*blockDim.x;

    // load 1st element data into shared memory
    if (start + t < n) 
        partial_max[t] = input_data[start + t];
    else
        partial_max[t] = INT_MIN;
    // load 2nd element data into shared memory
    if (start + blockDim.x + t < n)
        partial_max[blockDim.x+t] = input_data[start + blockDim.x+t];
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


__global__ void min_kernel(const float* input_data,float * min_element, int n) {
    extern __shared__ float partial_min[];
    unsigned int t = threadIdx.x;
    unsigned int start = 2*blockIdx.x*blockDim.x;

    // load 1st element data into shared memory
    if (start + t < n) 
        partial_min[t] = input_data[start + t];
    else
        partial_min[t] = INT_MAX;
    // load 2nd element data into shared memory
    if (start + blockDim.x + t < n)
        partial_min[blockDim.x+t] = input_data[start + blockDim.x+t];
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



__global__ void sum_kernel(const float* input_data,float * sum_element, int n) {
    extern __shared__ float partial_sum[];
    unsigned int t = threadIdx.x;
    unsigned int start = 2*blockIdx.x*blockDim.x;

    // load 1st element data into shared memory
    if (start + t < n) 
        partial_sum[t] = input_data[start + t];
    else
        partial_sum[t] = INT_MAX;
    // load 2nd element data into shared memory
    if (start + blockDim.x + t < n)
        partial_sum[blockDim.x+t] = input_data[start + blockDim.x+t];
    else
        partial_sum[blockDim.x+t] = INT_MAX;
    // loop to reduce the data in shared memory
    // each thread will be responsible for 2 elements
    for (unsigned int stride = blockDim.x; stride > 0;  stride /= 2) 
    {
        __syncthreads();
        if (t < stride) 
            if (t + stride < 2*blockDim.x )
                partial_sum[t]+= partial_sum[t+stride];
    }
    __syncthreads();
    // write the result for this block to global memory
    if (t == 0) {
        sum_element[blockIdx.x] = partial_sum[0];
    }
}



