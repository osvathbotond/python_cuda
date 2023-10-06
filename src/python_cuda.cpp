#include <Python.h>
// #include <boost/python.hpp>
// #include <boost/python/numpy.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "exceptions.hpp"
#include "vector.hpp"
#include "vector_functions.cuh"
#include "vector_functions.hpp"
#include <cudasharedptr.h>


PYBIND11_MODULE(python_cuda, m) {
    pybind11::class_<Vector>(m, "Vector")
        .def(pybind11::init<pybind11::array_t<float> &>())
        .def("get_array", &Vector::getArray)
        .def("device2host", &Vector::device2Host)
        .def("host2device", &Vector::host2Device)
        .def("norm", &vectorNorm)
        .def("__add__", &vectorAdd)
        .def("__sub__", &vectorSub)
        .def("add", &Vector::add);
}