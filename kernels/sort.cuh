// kernels/sort.cuh
#ifndef SORT_KERNELS_H
#define SORT_KERNELS_H

__global__ void mergePassKernel(int *input, int *output, int width, int size);


#endif // SORT_KERNELS_H 