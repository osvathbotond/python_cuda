# python_cuda
POC for using CUDA from Python via pybind11

# Source files

- `*.cu`: They contain the GPU kernel codes and the C++ interface definitions.
- `vector.hpp`: A header for the Vector class wrapping the numpy array and the CUDA pointer.
- `vector.cpp`: The implementation of the Vector class.
- `vector_functions.cuh`: A header file for the C++ interfaces of the CUDA functions defined in the `.cu` files.
- `vector_functions.hpp`: A header file the Vector-wrapped interfaces of the CUDA functions.
- `vector_functions.cpp`: The implementation of the Vector-wrapped interfaces of the CUDA functions.
- `exceptions.hpp`: A header file for the C++ exceptions.
- `python_cuda.cpp`: It contains the Python interfaces.

# Build process

I am sure there are nicer ways do to compile and link the code, but here is the one used by the `build.sh`:
- It compiles the `.cu` files to `.o` files with `nvcc`
- It compiles the `.cpp` files to `.o` files with `nvcc`
- It links the `.o` files to a `python_cuda.so` file that can be used from Python with `nvcc`

# Docker

I use Docker for several reasons, one of them being easier dependency management (compared to doing it directly in Windows).

To create the docker image, you can use the following command:

```docker build -t python_cuda_dev .```

And to run the container with the current directory mounted to `/workspace`, you can use the following command:

```docker run --rm -ti --gpus all -v .:/workspace python_cuda_dev```

An appropriate NVIDIA GPU is required to use the project, and you need to assign it to the docker container. This is done by the "--gpus all" flag, which makes all of the gpus available for the container. A possible indicator of no available gpu is the following runtime error while trying to use compiled cuda code:

```RuntimeError: CUDA driver version is insufficient for CUDA runtime version```