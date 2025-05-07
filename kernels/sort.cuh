// kernels/sort.cuh
#ifndef SORT_KERNELS_H
#define SORT_KERNELS_H
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>

__device__ int co_rank(int k, const char *A, int m, const char *B, int n, int row_size, int acc_col_size, bool ascending);

__global__ void co_rank_merge_batch(const char *input, int row_size, char *output, int n, int width, int acc_col_size, bool ascending);

char *call_sort_kernel(char *h_input, int row_size, int n, int acc_col_size, bool ascending);

#endif // SORT_KERNELS_H 