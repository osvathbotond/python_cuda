#pragma once

#include <stdexcept>

class NumpyShapeError: public std::exception {
    public:
        NumpyShapeError(const std::string& message): _message(message) {}

        const char* what() const noexcept override {
            return _message.c_str();
        }
    
    private:
        std::string _message;
};

class NumpyTypeError: public std::exception {
    public:
        NumpyTypeError(const std::string& message): _message(message) {}

        const char* what() const noexcept override {
            return _message.c_str();
        }
    
    private:
        std::string _message;
};

class CudaCopyError: public std::exception {
    public:
        CudaCopyError(const std::string& message): _message(message) {}

        const char* what() const noexcept override {
            return _message.c_str();
        }
    
    private:
        std::string _message;
};

class CudaMallocError: public std::exception {
    public:
        CudaMallocError(const std::string& message): _message(message) {}

        const char* what() const noexcept override {
            return _message.c_str();
        }
    
    private:
        std::string _message;
};

class CudaKernelError: public std::exception {
    public:
        CudaKernelError(const std::string& message): _message(message) {}

        const char* what() const noexcept override {
            return _message.c_str();
        }
    
    private:
        std::string _message;
};

class CudaFreeError: public std::exception {
    public:
        CudaFreeError(const std::string& message): _message(message) {}

        const char* what() const noexcept override {
            return _message.c_str();
        }
    
    private:
        std::string _message;
};
