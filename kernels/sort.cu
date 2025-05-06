#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Error checking macro
#define CUDA_CHECK(err) do { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Kernel to compute histogram (count occurrences of 4-bit digits)
__global__ void histogramKernel(const unsigned int* input, int* counts, int n, int shift) {
    extern __shared__ int localCounts[];
    
    // Initialize shared memory
    int tid = threadIdx.x;
    if (tid < 16) {
        localCounts[tid] = 0;
    }
    __syncthreads();

    // Compute global index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int digit = (input[idx] >> shift) & 0xF; // Extract 4-bit digit
        atomicAdd(&localCounts[digit], 1);
    }
    __syncthreads();

    // Aggregate to global counts
    if (tid < 16 && localCounts[tid] > 0) {
        atomicAdd(&counts[tid], localCounts[tid]);
    }
}

// Kernel to compute global offsets for each digit
__global__ void computeOffsetsKernel(int* counts, int* offsets, int n) {
    int tid = threadIdx.x;
    if (tid < 16) {
        // Compute prefix sum (exclusive scan)
        int sum = 0;
        if (tid > 0) {
            for (int i = 0; i < tid; i++) {
                sum += counts[i];
            }
        }
        offsets[tid] = sum;
    }
}

// Kernel to scatter elements based on offsets
__global__ void scatterKernel(const unsigned int* input, unsigned int* output, int* offsets, int n, int shift) {
    extern __shared__ int localOffsets[];
    
    // Copy global offsets to shared memory
    int tid = threadIdx.x;
    if (tid < 16) {
        localOffsets[tid] = offsets[tid];
    }
    __syncthreads();

    // Process elements
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int digit = (input[idx] >> shift) & 0xF;
        int pos = atomicAdd(&localOffsets[digit], 1);
        output[pos] = input[idx];
    }
}

// Host function for Radix Sort
void radixSort4BitCUDA(unsigned int* h_arr, int n) {
    // Device arrays
    unsigned int *d_input, *d_output;
    int *d_counts, *d_offsets;
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_counts, 16 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_offsets, 16 * sizeof(int)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_arr, n * sizeof(unsigned int), cudaMemcpyHostToDevice));

    // Find max element (on host for simplicity)
    unsigned int max = h_arr[0];
    for (int i = 1; i < n; i++) {
        if (h_arr[i] > max) max = h_arr[i];
    }

    // CUDA configuration
    const int threadsPerBlock = 256;
    const int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    const int sharedMemSize = 16 * sizeof(int); // For 16 bins

    // Process 4 bits at a time
    for (int shift = 0; (max >> shift) > 0; shift += 4) {
        // Reset counts
        CUDA_CHECK(cudaMemset(d_counts, 0, 16 * sizeof(int)));

        // Step 1: Compute histogram
        histogramKernel<<<blocks, threadsPerBlock, sharedMemSize>>>(d_input, d_counts, n, shift);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Step 2: Compute offsets (prefix sums)
        computeOffsetsKernel<<<1, 16>>>(d_counts, d_offsets, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Step 3: Scatter elements
        scatterKernel<<<blocks, threadsPerBlock, sharedMemSize>>>(d_input, d_output, d_offsets, n, shift);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Swap input and output
        unsigned int* temp = d_input;
        d_input = d_output;
        d_output = temp;
    }

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_arr, d_input, n * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_counts));
    CUDA_CHECK(cudaFree(d_offsets));
}

// Example usage
int main() {
    // Example array
    unsigned int arr[] = {170, 45, 75, 90, 802, 24, 2, 66};
    int n = sizeof(arr) / sizeof(arr[0]);

    printf("Original array: ");
    for (int i = 0; i < n; i++) {
        printf("%u ", arr[i]);
    }
    printf("\n");

    radixSort4BitCUDA(arr, n);

    printf("Sorted array: ");
    for (int i = 0; i < n; i++) {
        printf("%u ", arr[i]);
    }
    printf("\n");

    return 0;
}