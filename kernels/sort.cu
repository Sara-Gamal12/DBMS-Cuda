#include <stdio.h>
#include <stdlib.h>
#include "sort.cuh"


__global__ void mergePassKernel(int *input, int *output, int width, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int start = tid * 2 * width;
    
    if (start >= size) return;

    int mid = min(start + width, size);
    int end = min(start + 2 * width, size);

    int i = start, j = mid, k = start;

    while (i < mid && j < end) {
        output[k++] = (input[i] <= input[j]) ? input[i++] : input[j++];
    }
    while (i < mid) output[k++] = input[i++];
    while (j < end) output[k++] = input[j++];
}

void gpuMergeSort(int *data, int size) {
    int *d_data1, *d_data2;
    cudaMalloc(&d_data1, size * sizeof(int));
    cudaMalloc(&d_data2, size * sizeof(int));

    cudaMemcpy(d_data1, data, size * sizeof(int), cudaMemcpyHostToDevice);

    int *in = d_data1;
    int *out = d_data2;
    int width = 1;

    while (width < size) {
        int blocks = (size + 2 * width - 1) / (2 * width);
        mergePassKernel<<<blocks, 256>>>(in, out, width, size);
        cudaDeviceSynchronize();

        int *temp = in;
        in = out;
        out = temp;

        width *= 2;
    }

    cudaMemcpy(data, in, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_data1);
    cudaFree(d_data2);
}