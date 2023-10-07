#include <Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "exceptions.hpp"
#include "vector.hpp"
#include "vector_functions.cuh"
#include "vector_functions.hpp"
#include <cudasharedptr.h>


PYBIND11_MODULE(python_cuda, m) {
    pybind11::class_<Vector>(m, "Vector")
        .def(pybind11::init<pybind11::array_t<float> &, bool>(), pybind11::arg("array"), pybind11::arg("copy") = false)
        .def("get_array", &Vector::getArray)
        .def("device2host", &Vector::device2Host)
        .def("host2device", &Vector::host2Device)
        .def("norm", &vectorNorm)
        .def("__add__", &vectorAdd)
        .def("__sub__", &vectorSub)
        .def("add", &Vector::add);

    pybind11::register_exception<NumpyShapeError>(m, "NumpyShapeError");
    pybind11::register_exception<NumpyTypeError>(m, "NumpyTypeError");
    pybind11::register_exception<NumpyLengthError>(m, "NumpyLengthError");
    pybind11::register_exception<CudaCopyError>(m, "CudaCopyError");
    pybind11::register_exception<CudaMallocError>(m, "CudaMallocError");
    pybind11::register_exception<CudaKernelError>(m, "CudaKernelError");
    pybind11::register_exception<CudaFreeError>(m, "CudaFreeError");

}
