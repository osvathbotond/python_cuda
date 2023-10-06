#include <algorithm>

#include "vector.hpp"
#include "exceptions.hpp"
#include "vector_functions.hpp"

#include <iostream>


Vector::Vector(pybind11::array_t<float> array): m_array(array) {
    pybind11::buffer_info buffer = array.request();

    if (buffer.format != pybind11::format_descriptor<float>::format()) {
        throw NumpyTypeError("The arrays must be of type np.float32!");
    }
    if (buffer.ndim != 1) {
        throw NumpyShapeError("Only 1 dimensional arrays are supported!");
    }

    m_data = static_cast<float*>(buffer.ptr);
    m_n = buffer.shape[0];
    d_data.create(m_n);
}

Vector::Vector(size_t n, float fill): m_n(n) {
    m_array = pybind11::array_t<float>(m_n);

    pybind11::buffer_info buffer = m_array.request();

    m_data = static_cast<float*>(buffer.ptr);
    std::fill_n(m_data, m_n, fill);

    d_data.create(m_n);
}

void Vector::add(const Vector& vec) {
    vectorInplaceAdd(*this, vec);
}

void Vector::device2Host() {
    d_data.download(m_data);
}

void Vector::host2Device() {
    d_data.upload(m_data, m_n);
}
