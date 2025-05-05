// kernels/agg.cuh
#ifndef AGG_KERNELS_H
#define AGG_KERNELS_H

__global__ void sum_kernel( char *input_data, int row_size, int acc_col_size, double *sum_element, int n);

__global__ void min_kernel( char *input_data, int row_size, int acc_col_size, double *min_element, int n);

__global__ void max_kernel(char *input_data, int row_size, int acc_col_size, double *max_element, int n);

__host__ double call_agg_kernel(char *input_data, int row_size, int acc_col_size, char *op, int n);

#endif // AGG_KERNELS_H
