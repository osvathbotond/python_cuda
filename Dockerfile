FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

RUN :\
    && apt update \
    && DEBIAN_FRONTEND=noninteractive apt install -y python3 \
                                                     python3-pip \
                                                     python3-dev \
    && :

RUN python3 -m pip --no-cache-dir --disable-pip-version-check install numpy pybind11

WORKDIR /workspace