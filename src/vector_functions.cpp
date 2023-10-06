// #include <boost/python.hpp>
// #include <boost/python/numpy.hpp>

#include "vector_functions.hpp"
#include "vector_functions.cuh"
#include "exceptions.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

Vector vectorAdd(const Vector& vec1, const Vector& vec2) {
    size_t n1 = vec1.getSize();
    size_t n2 = vec2.getSize();

    if (n1 != n2) {
        throw NumpyLengthError("The two arrays must have the same number of elements!");
    }

    Vector res(n1, 0);

    addCuda(vec1.getDeviceData(), vec2.getDeviceData(), res.getDeviceData(), n1);

    return res;
}

void vectorInplaceAdd(Vector& vec1, const Vector& vec2) {
    size_t n1 = vec1.getSize();
    size_t n2 = vec2.getSize();

    if (n1 != n2) {
        throw NumpyLengthError("The two arrays must have the same number of elements!");
    }

    addCuda(vec1.getDeviceData(), vec2.getDeviceData(), vec1.getDeviceData(), n1);
}

Vector vectorSub(const Vector& vec1, const Vector& vec2) {
    size_t n1 = vec1.getSize();
    size_t n2 = vec2.getSize();

    if (n1 != n2) {
        throw NumpyLengthError("The two arrays must have the same number of elements!");
    }

    Vector res(n1, 0);

    subCuda(vec1.getDeviceData(), vec2.getDeviceData(), res.getDeviceData(), n1);

    return res;
}

float vectorNorm(const Vector& vec, const float p) {
    size_t n = vec.getSize();

    float res = normCuda(vec.getDeviceData(), p, n);

    return res;
}
