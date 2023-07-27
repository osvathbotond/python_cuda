#!/bin/bash

mkdir -p build

find "src/" -type f -name "*.cu" | while read -r cu_file; do
    o_file="$(basename $cu_file)"
    o_file="build/${o_file%.cu}.o"
    nvcc -Xcompiler -fPIC -c $cu_file -o $o_file
done

g++ -fPIC -c src/python_cuda.cpp $(python3-config --cflags --libs) -o build/python_cuda.o

nvcc -lboost_numpy310 -lboost_python310 -Xcompiler -shared build/*.o -o python_cuda.so
