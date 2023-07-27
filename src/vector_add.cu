#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <stdio.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <random>
#include <typeinfo>
#include <thread>
#include <cstdlib>
#include <algorithm>
#include <cassert>
#include <numeric>
#include <string>

#include "vector_functions.cuh"
#include "exceptions.hpp"


static const int num_threads = 512;

__global__ void addKernel(const float* vec1, const float* vec2, float* res, size_t vector_length) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < vector_length) {
        res[tid] = vec1[tid] + vec2[tid];
    }
}

void add(float* vec1, float* vec2, float* res, const size_t vector_length) {
    size_t bytes = vector_length * sizeof(float);

    // ceil(vector_length / num_threads)
    int num_blocks = (vector_length + num_threads - 1) / num_threads;

    // Pointers to the device-side variables
    float *d_vec1, *d_vec2, *d_res;

    // Allocate the memory on the GPU and move the vector (with error handling)
    cudaError_t err = cudaSuccess;
    err = cudaMalloc(&d_vec1, bytes);
    if (err != cudaSuccess) {
        throw CudaMallocError(cudaGetErrorString(err));
    }

    err = cudaMalloc(&d_vec2, bytes);
    if (err != cudaSuccess) {
        throw CudaMallocError(cudaGetErrorString(err));
    }

    err = cudaMalloc(&d_res, bytes);
    if (err != cudaSuccess) {
        throw CudaMallocError(cudaGetErrorString(err));
    }

    err = cudaMemcpy(d_vec1, vec1, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw CudaCopyError(cudaGetErrorString(err));
    }

    err = cudaMemcpy(d_vec2, vec2, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw CudaCopyError(cudaGetErrorString(err));
    }

    // const float* vec1, const float* vec2, float* res, size_t vector_length
    addKernel<<<num_blocks, num_threads>>>(d_vec1, d_vec2, d_res, vector_length);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw CudaKernelError(cudaGetErrorString(err));
    }

    // Copying back to the host
    err = cudaMemcpy(res, d_res, bytes, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) {
        throw CudaCopyError(cudaGetErrorString(err));
    }

    // Freeing the memory on the device. Not doing so can cause memory-leak.
    err = cudaFree(d_vec1);
    if (err != cudaSuccess) {
        throw CudaFreeError(cudaGetErrorString(err));
    }

    err = cudaFree(d_vec2);
    if (err != cudaSuccess) {
        throw CudaFreeError(cudaGetErrorString(err));
    }

    err = cudaFree(d_res);
    if (err != cudaSuccess) {
        throw CudaFreeError(cudaGetErrorString(err));
    }
}
