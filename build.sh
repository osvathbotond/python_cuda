#!/bin/bash
set -e

mkdir -p build

find "src/" -type f -name "*.cu" | while read -r cu_file; do
    o_file="$(basename $cu_file)"
    o_file="build/${o_file%.cu}.o"
    nvcc -O3 -std=c++20 -Xcompiler -fPIC -c $cu_file -o $o_file
done

find "src/" -type f -name "*.cpp" | while read -r cpp_file; do
    o_file="$(basename $cpp_file)"
    o_file="build/${o_file%.cpp}.o"
    nvcc -O3 -std=c++20 -Xcompiler -fPIC -I/workspace/submodules/CudaSharedPtr -c $cpp_file -Xcompiler $(python3 -m pybind11 --includes) -o $o_file

done

nvcc -O3 -std=c++20 -Xcompiler -fPIC -shared build/*.o -o python_cuda$(python3-config --extension-suffix)
