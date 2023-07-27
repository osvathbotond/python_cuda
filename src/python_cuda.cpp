#include <iostream>

#include <Python.h>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

#include "exceptions.hpp"
#include "vector_functions.cuh"

boost::python::numpy::ndarray vectorAdd(boost::python::numpy::ndarray& vec1, boost::python::numpy::ndarray& vec2) {
    if (vec1.get_nd() != 1 || vec2.get_nd() != 1) {
        throw NumpyShapeError("Only 1 dimensional arrays are supported!");
    }

    size_t n = vec1.shape(0);

    if (n != vec2.shape(0)) {
        throw NumpyShapeError("The arrays must have the same number of elements!");
    }

    auto np_float = boost::python::numpy::dtype::get_builtin<float>();


    if(vec1.get_dtype() != np_float || vec2.get_dtype() != np_float) {
        throw NumpyTypeError("The arrays must be of type np.float32!");
    }

    float res[n];

    boost::python::tuple shape = boost::python::make_tuple(n);

    add(reinterpret_cast<float*>(vec1.get_data()), reinterpret_cast<float*>(vec2.get_data()), res, n);

    boost::python::numpy::ndarray res_np = boost::python::numpy::zeros(shape, np_float);
    std::copy(res, res + n, reinterpret_cast<float*>(res_np.get_data()));

    return res_np;
}

boost::python::numpy::ndarray vectorSub(boost::python::numpy::ndarray& vec1, boost::python::numpy::ndarray& vec2) {
    if (vec1.get_nd() != 1 || vec2.get_nd() != 1) {
        throw NumpyShapeError("Only 1 dimensional arrays are supported!");
    }

    size_t n = vec1.shape(0);

    if (n != vec2.shape(0)) {
        throw NumpyShapeError("The arrays must have the same number of elements!");
    }

    auto np_float = boost::python::numpy::dtype::get_builtin<float>();


    if(vec1.get_dtype() != np_float || vec2.get_dtype() != np_float) {
        throw NumpyTypeError("The arrays must be of type np.float32!");
    }

    float res[n];

    boost::python::tuple shape = boost::python::make_tuple(n);

    sub(reinterpret_cast<float*>(vec1.get_data()), reinterpret_cast<float*>(vec2.get_data()), res, n);

    boost::python::numpy::ndarray res_np = boost::python::numpy::zeros(shape, np_float);
    std::copy(res, res + n, reinterpret_cast<float*>(res_np.get_data()));

    return res_np;
}

float vectorNorm(boost::python::numpy::ndarray& vec1, const float p) {
    if (vec1.get_nd() != 1) {
        throw NumpyShapeError("Only 1 dimensional arrays are supported!");
    }

    size_t n = vec1.shape(0);

    auto np_float = boost::python::numpy::dtype::get_builtin<float>();


    if(vec1.get_dtype() != np_float) {
        throw NumpyTypeError("The arrays must be of type np.float32!");
    }


    boost::python::tuple shape = boost::python::make_tuple(n);

    float res = norm(reinterpret_cast<float*>(vec1.get_data()), p, n);

    return res;
}

void cudaTest() {
    std::vector<float> vec1{1, 2, 3, 4, 5};
    std::vector<float> vec2{-1, -2, 3, 2, 1};

    float res[5] = {0};

    add(vec1.data(), vec2.data(), res, 5);

    std::cout << "Add:" << std::endl;
    for(int i = 0; i < 5; i++) {
        std::cout << res[i] << " ";
    }
    std::cout << std::endl;

    sub(vec1.data(), vec2.data(), res, 5);
    std::cout << "Sub:" << std::endl;
    for(int i = 0; i < 5; i++) {
        std::cout << res[i] << " ";
    }
    std::cout << std::endl;
}

void greet() {
    std::cout << "Hello!" << std::endl;
}

BOOST_PYTHON_MODULE(python_cuda) {
    boost::python::numpy::initialize();
    boost::python::def("vector_add", vectorAdd);
    boost::python::def("vector_sub", vectorSub);
    boost::python::def("vector_norm", vectorNorm);
    boost::python::def("greet", greet);
    boost::python::def("cuda_test", cudaTest);
}