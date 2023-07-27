#include <Python.h>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

#include "exceptions.hpp"
#include "vector.hpp"
#include "vector_functions.cuh"
#include "vector_functions.hpp"


BOOST_PYTHON_MODULE(python_cuda) {
    boost::python::numpy::initialize();

    boost::python::class_<Vector>("Vector", boost::python::init<boost::python::numpy::ndarray&>())
        .def("get_array", &Vector::getArray)
        .def("device2host", &Vector::device2Host)
        .def("host2device", &Vector::host2Device)
        .def("norm", vectorNorm)
        .def("__add__", vectorAdd)
        .def("__sub__", vectorSub);
}
