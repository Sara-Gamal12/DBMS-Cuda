// kernels/agg.cuh
#ifndef AGG_KERNELS_H
#define AGG_KERNELS_H

__global__ void sum_kernel(const float* input, float* output, int n);
__global__ void min_kernel(const float* input, float* output, int n);
__global__ void max_kernel(const float* input, float* output, int n);

#endif // AGG_KERNELS_H
