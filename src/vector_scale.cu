#include "vector_functions.cuh"
#include "exceptions.hpp"


static const int num_threads = 512;


__global__ void scaleKernel(const float* vec, const float scalar, float* res, const size_t vector_length) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < vector_length) {
        res[tid] = vec[tid] * scalar;
    }
}

void scaleCuda(const float* d_vec, const float scalar, float* d_res, const size_t vector_length) {
    // ceil(vector_length / num_threads)
    int num_blocks = (vector_length + num_threads - 1) / num_threads;

    scaleKernel<<<num_blocks, num_threads>>>(d_vec, scalar, d_res, vector_length);
    cudaError_t err = cudaSuccess;
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw CudaKernelError(cudaGetErrorString(err));
    }
}


