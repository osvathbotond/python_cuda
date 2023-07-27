FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

RUN apt update -y
RUN apt install -y python3
RUN apt install -y python3-pip
RUN apt install -y python3-dev
RUN apt install -y libboost-python-dev
RUN apt install -y libboost-numpy-dev

RUN pip3 install numpy

WORKDIR /workspace