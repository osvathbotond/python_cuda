#include "memory.cuh"
#include "exceptions.hpp"


void* _cudaMalloc(size_t mySize) {
    void* ptr;
    cudaError_t err = cudaSuccess;
    cudaMalloc((void**)&ptr, mySize);
    if (err != cudaSuccess) {
        throw CudaMallocError(cudaGetErrorString(err));
    }
    return ptr;
}

void cudaFreeDeleter::operator()(float* ptr) const {
    cudaError_t err = cudaSuccess;
    cudaFree(ptr);
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

cuda_shared_ptr::cuda_shared_ptr(float* ptr) : std::shared_ptr<float>(ptr, cudaFreeDeleter()) {}

float* cuda_shared_ptr::get() const {
    return std::shared_ptr<float>::get();
}

float* cuda_shared_ptr::release() {
    float* releasedPtr = std::shared_ptr<float>::get();
    reset();
    return releasedPtr;
}
cuda_shared_ptr getDevicePointer(size_t bytes){

    cuda_shared_ptr d_data((float*)_cudaMalloc(bytes));

    return d_data;
}

void cudaDevice2Host(float* data, cuda_shared_ptr d_data, size_t bytes) {
    cudaError_t err = cudaSuccess;
    err = cudaMemcpy(data, d_data.get(), bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw CudaCopyError(cudaGetErrorString(err));
    }
}

void cudaHost2Devide(cuda_shared_ptr d_data, float* data, size_t bytes) {
    cudaError_t err = cudaSuccess;
    err = cudaMemcpy(d_data.get(), data, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw CudaCopyError(cudaGetErrorString(err));
    }
}
