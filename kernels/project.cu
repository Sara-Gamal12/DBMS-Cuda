#include "project.cuh"

__global__ void project_kernel(char *input_data, int row_size,int new_row_size, int *col_inedx, int col_num, char *output_data, int n, int *acc_sum, int *size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    // char *row_ptr = input_data + idx * row_size;
    char *row_ptr = &input_data[idx * row_size];
    int prev_sum = 0;
    for (int i = 0; i < col_num; i++)
    {
        char *field_ptr = row_ptr + acc_sum[i]; // acc_sum[col_inedx[i]];
        int field_size = size[i];
        char *out_ptr = output_data + idx*new_row_size + prev_sum;
        prev_sum += field_size;
        for (int j = 0; j < field_size; ++j)
        {
            out_ptr[j] = field_ptr[j];
        }

    }
}



__host__ char *call_project_kernel(char *input_data,int new_row_size, int row_size, int*col_index ,int *acc_sums, int n,int column_num,int*sizes)
{

    char *d_input_data;
    char *d_output_data;
    int *d_acc_col_size;
    int *d_sizes;
    int *d_col_index;
    char *h_output_data = (char *)malloc(n * new_row_size * sizeof(char));

    cudaMalloc(&d_input_data, n * row_size);
    cudaMalloc(&d_output_data, n * new_row_size);
    cudaMalloc(&d_acc_col_size, sizeof(int) * column_num);
    cudaMalloc(&d_sizes, sizeof(int) * column_num);
    cudaMalloc(&d_col_index, sizeof(int) * column_num);

    cudaMemcpy(d_col_index, col_index, sizeof(int) * column_num, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_data, input_data, n * row_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_acc_col_size, acc_sums, sizeof(int) * column_num, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sizes, sizes, sizeof(int) * column_num, cudaMemcpyHostToDevice);
    



    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    project_kernel<<<numBlocks, blockSize>>>(d_input_data, row_size,
                                             new_row_size, d_col_index, column_num,
                                             d_output_data, n, d_acc_col_size, d_sizes
                                        );

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

    cudaMemcpy(h_output_data, d_output_data, n * new_row_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input_data);
    cudaFree(d_output_data);
    cudaFree(d_acc_col_size);
    cudaFree(d_sizes);
    cudaFree(d_col_index);
    
    
    return h_output_data;
}

