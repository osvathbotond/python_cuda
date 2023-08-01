#!/bin/bash
set -e

mkdir -p build

find "src/" -type f -name "*.cu" | while read -r cu_file; do
    o_file="$(basename $cu_file)"
    o_file="build/${o_file%.cu}.o"
    nvcc -O3 -Xcompiler -fPIC -c $cu_file -o $o_file
done

find "src/" -type f -name "*.cpp" | while read -r cpp_file; do
    o_file="$(basename $cpp_file)"
    o_file="build/${o_file%.cpp}.o"
    nvcc -O3 -Xcompiler -fPIC -I/workspace/submodules/CudaSharedPtr -c $cpp_file -Xcompiler -I/usr/include/python3.10 -I/usr/include/python3.10 -o $o_file

done

nvcc -O3 -Xcompiler -fPIC -lboost_numpy310 -lboost_python310 -shared build/*.o -o python_cuda.so
