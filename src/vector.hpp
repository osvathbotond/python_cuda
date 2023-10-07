#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cudasharedptr.h>


class Vector {
    public:
        Vector(pybind11::array_t<float>& array);
        Vector(pybind11::array_t<float>& array, bool copy);
        Vector(size_t n, float fill);
        Vector operator+(const Vector& other) const;
        Vector operator-(const Vector& other) const;
        Vector operator*(float scalar) const;
        float norm(float p) const;
        // Vector(const Vector &m) = delete;
        // Vector & operator= (const Vector &) = delete;
        void device2Host();
        void host2Device();
        pybind11::array_t<float> getArray() {return m_array;};
        float* getData() {return m_data;};
        float* getDeviceData() {return d_data.data();};
        const float* getDeviceData() const {return d_data.data();};
        size_t getSize() const {return m_n;};
        void add(const Vector& vec);
        void sub(const Vector& vec);
        void scale(float c);
    private:
        float* m_data;
        size_t m_n;
        pybind11::array_t<float> m_array;
        fun::cuda::shared_ptr<float> d_data;
};