#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

#include "vector_functions.hpp"
#include "vector_functions.cuh"


Vector vectorAdd(Vector& vec1, Vector& vec2) {
    size_t n = vec1.getSize();
    
    boost::python::tuple shape = boost::python::make_tuple(n);
    auto np_float = boost::python::numpy::dtype::get_builtin<float>();
    boost::python::numpy::ndarray res_np_array = boost::python::numpy::empty(shape, np_float);

    Vector res(res_np_array);

    addCuda(vec1.getDeviceData().get(), vec2.getDeviceData().get(), res.getDeviceData().get(), n);

     return res;
}

Vector vectorSub(Vector& vec1, Vector& vec2) {
    size_t n = vec1.getSize();
    
    boost::python::tuple shape = boost::python::make_tuple(n);
    auto np_float = boost::python::numpy::dtype::get_builtin<float>();
    boost::python::numpy::ndarray res_np_array = boost::python::numpy::empty(shape, np_float);

    Vector res(res_np_array);

    subCuda(vec1.getDeviceData().get(), vec2.getDeviceData().get(), res.getDeviceData().get(), n);

     return res;
}

float vectorNorm(Vector& vec, const float p) {
    size_t n = vec.getSize();

    float res = normCuda(vec.getDeviceData().get(), p, n);

    return res;

}