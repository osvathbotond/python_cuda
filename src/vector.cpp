#include "vector.hpp"
#include "exceptions.hpp"
#include "memory.cuh"


Vector::Vector(boost::python::numpy::ndarray& np_array): np_array(np_array) {
    auto np_float = boost::python::numpy::dtype::get_builtin<float>();

    if (np_array.get_dtype() != np_float) {
        throw NumpyTypeError("The arrays must be of type np.float32!");
    }

    if (np_array.get_nd() != 1) {
        throw NumpyShapeError("Only 1 dimensional arrays are supported!");
    }
    
    n = np_array.shape(0);
    bytes = n * sizeof(float);
    data = reinterpret_cast<float*>(np_array.get_data());

    d_data = getDevicePointer(bytes);
}

void Vector::device2Host() {
    cudaDevice2Host(data, d_data, bytes);
}

void Vector::host2Device() {
    cudaHost2Devide(d_data, data, bytes);
}