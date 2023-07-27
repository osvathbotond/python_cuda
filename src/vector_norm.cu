#include "vector_functions.cuh"
#include "exceptions.hpp"


static const int num_threads = 512;

__global__ void normKernel(const float* vec, float* res, const int n, const bool power, const float p) {
    __shared__ float partial_sum[num_threads];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        if (power) {
            partial_sum[threadIdx.x] = pow(abs(vec[tid]), p);
        }
        else {
            partial_sum[threadIdx.x] = vec[tid];
        }
    }
    else {
        partial_sum[threadIdx.x] = 0.0;
    }

    // Sync the threads to have all of the needed data in the shared memory
    __syncthreads();

    // Do the reduction (example with 8 numbers):
    // a               b       c   d   e f g h
    // a+e             b+f     c+g d+h e f g h
    // a+e+c+g         b+f+c+g c+g d+h e f g h
    // a+e+c+g+b+f+c+g b+f+c+g c+g d+h e f g h
    // And the result is just the 0-th element
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        }
        // We do need to wait for all of the threads to do the sums
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        if (gridDim.x == 1) {
            res[blockIdx.x] = pow(partial_sum[0], (float)(1.0/p));
        }
        else {
            res[blockIdx.x] = partial_sum[0];
        }
        
    }

}

float normCuda(const float* d_vec, const float p, const size_t vector_length) {
    // Host-side variables
    float res;

    size_t bytes = vector_length * sizeof(float);

    // ceil(vector_length / num_threads)
    int num_blocks = (vector_length + num_threads - 1) / num_threads;

    // Pointers to the device-side variables
    float *d_res1, *d_res2;

    // Allocate the memory on the GPU and move the vector (with error handling)
    cudaError_t err = cudaSuccess;
    err = cudaMalloc(&d_res1, bytes);
    if (err != cudaSuccess) {
        throw CudaMallocError(cudaGetErrorString(err));
    }

    err = cudaMalloc(&d_res2, bytes);
    if (err != cudaSuccess) {
        throw CudaMallocError(cudaGetErrorString(err));
    }

    // The first sum-reduction. Each block gives back a number, so the first num_blocks elements
    // of the result d_res will have the needed information for us (the partial sums).
    normKernel<<<num_blocks, num_threads>>>(d_vec, d_res1, vector_length, true, p);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw CudaKernelError(cudaGetErrorString(err));
    }

    // Since a reduction gives us back num_blocks elements, we need to do it until num_blocks == 1.
    int left;
    int num_blocks_red = num_blocks;
    int source_counter = 1;
    do {
        left = num_blocks_red;
        num_blocks_red = (left + num_threads - 1) / num_threads;
        if (source_counter == 1) {
            normKernel<<<num_blocks_red, num_threads>>>(d_res1, d_res2, left, false, p);
            source_counter = 2;
        }
        else {
            normKernel<<<num_blocks_red, num_threads>>>(d_res2, d_res1, left, false, p);
            source_counter = 1;
        }

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw CudaKernelError(cudaGetErrorString(err));
        }
    } while (num_blocks_red > 1);

    // Copying back to the host (only one number; the 0-th element of the d_res), with error handling.
    if (source_counter == 1) {
        err = cudaMemcpy(&res, d_res1, sizeof(float), cudaMemcpyDeviceToHost);
    }
    else {
        err = cudaMemcpy(&res, d_res2, sizeof(float), cudaMemcpyDeviceToHost);
    }
    if (err != cudaSuccess) {
        throw CudaCopyError(cudaGetErrorString(err));
    }

    // Freeing the memory on the device. Not doing so can cause memory-leak.
    err = cudaFree(d_res1);
    if (err != cudaSuccess) {
        throw CudaFreeError(cudaGetErrorString(err));
    }

    err = cudaFree(d_res2);
    if (err != cudaSuccess) {
        throw CudaFreeError(cudaGetErrorString(err));
    }

    return res;
}
