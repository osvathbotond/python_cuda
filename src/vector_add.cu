#include "vector_functions.cuh"
#include "exceptions.hpp"


static const int num_threads_per_block = 512;


__global__ void addKernel(const float* vec1, const float* vec2, float* res, const size_t vector_length) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < vector_length) {
        res[tid] = vec1[tid] + vec2[tid];
    }
}

void addCuda(const float* d_vec1, const float* d_vec2, float* d_res, const size_t vector_length) {
    // ceil(vector_length / num_threads_per_block)
    int num_blocks = (vector_length + num_threads_per_block - 1) / num_threads_per_block;

    addKernel<<<num_blocks, num_threads_per_block>>>(d_vec1, d_vec2, d_res, vector_length);
    cudaError_t err = cudaSuccess;
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw CudaKernelError(cudaGetErrorString(err));
    }
}
