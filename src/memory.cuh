#pragma once

#include <memory>


void* _cudaMalloc(size_t mySize);

struct cudaFreeDeleter {
    void operator()(float* ptr) const;
};

class cuda_shared_ptr : public std::shared_ptr<float> {
    public:
        explicit cuda_shared_ptr(float* ptr = nullptr);
        float* get() const;
        float* release();
    private:
        using std::shared_ptr<float>::reset;
};

cuda_shared_ptr getDevicePointer(size_t bytes);
void cudaDevice2Host(float* data, cuda_shared_ptr d_data, size_t bytes);
void cudaHost2Devide(cuda_shared_ptr d_data, float* data, size_t bytes);





