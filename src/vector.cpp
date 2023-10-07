#include <algorithm>

#include "vector.hpp"
#include "exceptions.hpp"
#include "vector_functions.cuh"

#include <iostream>

Vector::Vector(pybind11::array_t<float>& array, bool copy = false) {
    pybind11::buffer_info buffer = array.request();

    if (buffer.format != pybind11::format_descriptor<float>::format()) {
        throw NumpyTypeError("The arrays must be of type np.float32!");
    }
    if (buffer.ndim != 1) {
        throw NumpyShapeError("Only 1 dimensional arrays are supported!");
    }

    m_n = buffer.shape[0];
    d_data.create(m_n);

    if (copy) {
         pybind11::buffer_info new_buffer = m_array.request();
         m_data = static_cast<float*>(new_buffer.ptr);
         m_array = pybind11::array_t<float>(buffer);
    } else {
        m_array = array;
        m_data = static_cast<float*>(buffer.ptr);
    }

}

Vector::Vector(size_t n, float fill): m_n(n) {
    m_array = pybind11::array_t<float>(m_n);

    pybind11::buffer_info buffer = m_array.request();

    m_data = static_cast<float*>(buffer.ptr);
    std::fill_n(m_data, m_n, fill);

    d_data.create(m_n);
}

Vector Vector::operator+(const Vector& other) const {
    size_t n1 = getSize();
    size_t n2 = other.getSize();

    if (n1 != n2) {
        throw NumpyLengthError("The two arrays must have the same number of elements!");
    }

    Vector res(n1, 0);

    addCuda(getDeviceData(), other.getDeviceData(), res.getDeviceData(), n1);

    return res;
}

Vector Vector::operator-(const Vector& other) const {
    size_t n1 = getSize();
    size_t n2 = other.getSize();

    if (n1 != n2) {
        throw NumpyLengthError("The two arrays must have the same number of elements!");
    }

    Vector res(n1, 0);

    subCuda(getDeviceData(), other.getDeviceData(), res.getDeviceData(), n1);

    return res;
}

Vector Vector::operator*(float scalar) const {
    Vector res(getSize(), 0);

    scaleCuda(getDeviceData(), scalar, res.getDeviceData(), getSize());

    return res;
}

float Vector::norm(float p) const {
    size_t n = getSize();

    float res = normCuda(getDeviceData(), p, n);

    return res;
}

void Vector::add(const Vector& other) {
    size_t n1 = getSize();
    size_t n2 = other.getSize();

    if (n1 != n2) {
        throw NumpyLengthError("The two arrays must have the same number of elements!");
    }

    addCuda(getDeviceData(), other.getDeviceData(), getDeviceData(), n1);
}

void Vector::sub(const Vector& other) {
    size_t n1 = getSize();
    size_t n2 = other.getSize();

    if (n1 != n2) {
        throw NumpyLengthError("The two arrays must have the same number of elements!");
    }

    subCuda(getDeviceData(), other.getDeviceData(), getDeviceData(), n1);
}

void Vector::scale(float c) {
    scaleCuda(getDeviceData(), c, getDeviceData(), getSize());
}

void Vector::device2Host() {
    d_data.download(m_data);
}

void Vector::host2Device() {
    d_data.upload(m_data, m_n);
}
