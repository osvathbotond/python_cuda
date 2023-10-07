#include <Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "exceptions.hpp"
#include "vector.hpp"
#include "vector_functions.cuh"
#include "vector_functions.hpp"
#include <cudasharedptr.h>


PYBIND11_MODULE(python_cuda, m) {
    pybind11::class_<Vector>(m, "Vector", "1 dimensional vector of floats with CUDA support.")
        .def(pybind11::init<pybind11::array_t<float> &, bool>(), "By default, it will use the same memory for the device-side array as the input array. You can avoid it by setting copy=True.", pybind11::arg("array"), pybind11::arg("copy") = false)
        .def("get_array", &Vector::getArray, "Returns the data in numpy.ndarray. Note that it might be out-of-sync with the actual, GPU array if you modified that.")
        .def("device2host", &Vector::device2Host, "Copy the data from the device (GPU) to the host (CPU). Equivalent to gpu2cpu.")
        .def("gpu2cpu", &Vector::device2Host, "Copy the data from the device (GPU) to the host (CPU). Equivalent to device2host.")
        .def("host2device", &Vector::host2Device, "Copy the data from the host (CPU) to the device (GPU). Equivalent to cpu2gpu.")
        .def("cpu2gpu", &Vector::host2Device, "Copy the data from the host (CPU) to the device (GPU). Equivalent to host2device.")
        .def("norm", &vectorNorm, "Calculate the p-norm of the vector.", pybind11::arg("p") = 2)
        .def("__add__", &vectorAdd)
        .def("__sub__", &vectorSub)
        .def("add", &Vector::add, "Add the vector vec inplace.", pybind11::arg("vec"));

    pybind11::register_exception<NumpyShapeError>(m, "NumpyShapeError");
    pybind11::register_exception<NumpyTypeError>(m, "NumpyTypeError");
    pybind11::register_exception<NumpyLengthError>(m, "NumpyLengthError");
    pybind11::register_exception<CudaCopyError>(m, "CudaCopyError");
    pybind11::register_exception<CudaMallocError>(m, "CudaMallocError");
    pybind11::register_exception<CudaKernelError>(m, "CudaKernelError");
    pybind11::register_exception<CudaFreeError>(m, "CudaFreeError");
}
