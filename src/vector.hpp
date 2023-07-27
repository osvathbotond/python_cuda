#pragma once

#include <boost/python/numpy.hpp>
#include "memory.cuh"


class Vector {
    public:
        Vector(boost::python::numpy::ndarray& np_array);
        Vector(const Vector& vec): data(vec.data), d_data(vec.d_data), n(vec.n), bytes(vec.bytes), np_array(vec.np_array) {};
        void device2Host();
        void host2Device();
        boost::python::numpy::ndarray getArray() {return np_array;};
        float* getData() {return data;};
        cuda_shared_ptr getDeviceData() {return d_data;};
        size_t getSize() {return n;};
    private:
        float* data = nullptr;
        cuda_shared_ptr d_data;
        size_t n;
        size_t bytes;
        boost::python::numpy::ndarray np_array;
};