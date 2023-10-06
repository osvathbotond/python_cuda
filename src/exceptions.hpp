#pragma once

#include <stdexcept>


class NumpyShapeError: public std::exception {
    public:
        NumpyShapeError(const std::string& message): m_message(message) {}

        const char* what() const noexcept override {
            return m_message.c_str();
        }
    
    private:
        std::string m_message;
};

class NumpyTypeError: public std::exception {
    public:
        NumpyTypeError(const std::string& message): m_message(message) {}

        const char* what() const noexcept override {
            return m_message.c_str();
        }
    
    private:
        std::string m_message;
};

class NumpyLengthError: public std::exception {
    public:
        NumpyLengthError(const std::string& message): m_message(message) {}

        const char* what() const noexcept override {
            return m_message.c_str();
        }
    
    private:
        std::string m_message;
};

class CudaCopyError: public std::exception {
    public:
        CudaCopyError(const std::string& message): m_message(message) {}

        const char* what() const noexcept override {
            return m_message.c_str();
        }
    
    private:
        std::string m_message;
};

class CudaMallocError: public std::exception {
    public:
        CudaMallocError(const std::string& message): m_message(message) {}

        const char* what() const noexcept override {
            return m_message.c_str();
        }
    
    private:
        std::string m_message;
};

class CudaKernelError: public std::exception {
    public:
        CudaKernelError(const std::string& message): m_message(message) {}

        const char* what() const noexcept override {
            return m_message.c_str();
        }
    
    private:
        std::string m_message;
};

class CudaFreeError: public std::exception {
    public:
        CudaFreeError(const std::string& message): m_message(message) {}

        const char* what() const noexcept override {
            return m_message.c_str();
        }
    
    private:
        std::string m_message;
};
