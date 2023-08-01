#pragma once

#include <boost/python/numpy.hpp>
#include <cudasharedptr.h>


class Vector {
    public:
        Vector(boost::python::numpy::ndarray& np_array);
        void device2Host();
        void host2Device();
        boost::python::numpy::ndarray getArray() {return np_array;};
        float* getData() {return data;};
        float* getDeviceData() {return d_data.data();};
        size_t getSize() {return n;};
    private:
        float* data = nullptr;
        fun::cuda::shared_ptr<float> d_data;
        size_t n;
        size_t bytes;
        boost::python::numpy::ndarray np_array;
};