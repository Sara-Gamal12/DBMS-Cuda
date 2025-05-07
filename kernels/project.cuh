// kernels/get.cuh
#ifndef PROJECT_KERNELS_H
#define PROJECT_KERNELS_H
#include <cuda_runtime.h>
#include <cstdio>

__global__ void project_kernel(char *input_data, int row_size, int new_row_size, int *col_inedx, int col_num, char *output_data, int n, int *acc_sum, int *size);
__host__ char *call_project_kernel(char *input_data, int new_row_size, int row_size, int*col_index ,int *acc_sums, int n,int column_num,int*sizes);

#endif // PROJECT_KERNELS_H