#include "vector_functions.hpp"
#include "vector_functions.cuh"

Vector vectorSin(const Vector& vec) {
    size_t n = vec.getSize();

    Vector res(n, 0);

    sinCuda(vec.getDeviceData(), res.getDeviceData(), n);

    return res;
}