#include "sort.cuh"
#include <cfloat>
__device__ int co_rank(int k, const char *A, int m, const char *B, int n, int row_size, int acc_col_size, bool ascending)
{
    int low = max(0, k - n);
    int high = min(k, m);
    int i = low;

    while (low <= high)
    {
        i = (low + high) / 2;
        int j = k - i;
        double a_i, b_j, a_i1, b_j1;
        if (device_strcmp(&A[i * row_size + acc_col_size], "NULL") == 0)
            a_i = -DBL_MAX;
        else
            memcpy(&a_i, &A[i * row_size + acc_col_size], sizeof(double));
        if (device_strcmp(&B[j * row_size + acc_col_size], "NULL") == 0)
            b_j = -DBL_MAX;
        else
            memcpy(&b_j, &B[j * row_size + acc_col_size], sizeof(double));

        if (i > 0 && j < n)
        {
            if (device_strcmp(&A[(i - 1) * row_size + acc_col_size], "NULL") == 0)
                a_i1 = -DBL_MAX;
            else
                memcpy(&a_i1, &A[(i - 1) * row_size + acc_col_size], sizeof(double));
            bool cond = ascending ? (a_i1 > b_j) : (a_i1 < b_j);
            if (cond)
            {
                high = i - 1;
                continue;
            }
        }

        if (j > 0 && i < m)
        {
            if (device_strcmp(&B[(j - 1) * row_size + acc_col_size], "NULL") == 0)
                b_j1 = -DBL_MAX;
            else
                memcpy(&b_j1, &B[(j - 1) * row_size + acc_col_size], sizeof(double));
            bool cond = ascending ? (b_j1 > a_i) : (b_j1 < a_i);
            if (cond)
            {
                low = i + 1;
                continue;
            }
        }

        return i;
    }
    return i;
}

__global__ void co_rank_merge_batch(const char *input, int row_size, char *output, int n, int width, int acc_col_size, bool ascending)
{
    int merge_id = blockIdx.x;
    int i = merge_id * 2 * width;
    if (i >= n)
        return;

    int mid = min(i + width, n);
    int end = min(i + 2 * width, n);

    if (mid >= end)
    {
        if (i < n && threadIdx.x == 0)
        {
            for (int k = i; k < end; k++)
            {
                memcpy(&output[k * row_size], &input[k * row_size], row_size);
            }
        }
        return;
    }

    const char *A = &input[i * row_size];
    const char *B = &input[mid * row_size];
    char *C = &output[i * row_size];

    int m = mid - i;
    int nn = end - mid;
    int total = m + nn;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    int k_start = tid * total / num_threads;
    int k_end = (tid + 1) * total / num_threads;
    if (k_start >= total)
        return;

    int i_start = co_rank(k_start, A, m, B, nn, row_size, acc_col_size, ascending);
    int j_start = k_start - i_start;

    int i_end = co_rank(k_end, A, m, B, nn, row_size, acc_col_size, ascending);
    int j_end = k_end - i_end;

    int a = i_start, b = j_start, k = k_start;
    while (k < k_end && k < total)
    {
        double Aa, Bb;
        if (device_strcmp(&A[a * row_size + acc_col_size], "NULL") == 0)
            Aa = -DBL_MAX;
        else
            memcpy(&Aa, &A[a * row_size + acc_col_size], sizeof(double));

        if (device_strcmp(&B[b * row_size + acc_col_size], "NULL") == 0)
            Bb = -DBL_MAX;
        else
            memcpy(&Bb, &B[b * row_size + acc_col_size], sizeof(double));
        if (b >= nn || (a < m && (ascending ? Aa <= Bb : Aa >= Bb)))
        {
            memcpy(&C[k * row_size], &A[a * row_size], row_size);
            a++;
        }
        else
        {
            memcpy(&C[k * row_size], &B[b * row_size], row_size);
            b++;
        }
        k++;
    }
}

char *call_sort_kernel(char *h_input, int row_size, int n, int acc_col_size, bool ascending)
{
    char *d_input;
    char *d_temp;
    if (n <= 1)
        return h_input;

    cudaMalloc(&d_input, n * row_size * sizeof(char));
    cudaMalloc(&d_temp, n * row_size * sizeof(char));

    cudaMemcpy(d_input, h_input, n * row_size * sizeof(char), cudaMemcpyHostToDevice);

    for (int width = 1; width < n; width *= 2)
    {
        int num_merges = (n + 2 * width - 1) / (2 * width);
        dim3 blocks(num_merges);
        co_rank_merge_batch<<<blocks, 1>>>(d_input, row_size, d_temp, n, width, acc_col_size, ascending);

        cudaDeviceSynchronize();

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

        std::swap(d_input, d_temp);
    }

    cudaMemcpy(h_input, d_input, n * row_size * sizeof(char), cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_temp);
    return h_input;
}