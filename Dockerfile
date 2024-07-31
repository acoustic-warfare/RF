#Use an official Python runtime as a parent image
FROM ubuntu:latest as condaContainer

ENV TZ=Europe \
    DEBIAN_FRONTEND=noninteractive \
    DISPLAY=:0.0

SHELL [ "bash", "-c"]

#Install necessary packages
RUN apt-get update && apt-get install -y build-essential
RUN apt-get install -y python3-dev
RUN apt-get install -y \
    lsof \
    #    libgl1-mesa-glx \
    vim \
    nano \
    libzmq3-dev \
    libusb-1.0-0 \
    git \
    cmake \
    clang \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    gstreamer1.0-x \
    gstreamer1.0-alsa \ 
    gstreamer1.0-gl \
    gstreamer1.0-gtk3 \ 
    gstreamer1.0-qt5 \
    gstreamer1.0-pulseaudio \
    pkgconf \
    make

# install ninja for faster build times and to build DynRT-streamer
WORKDIR /
RUN git clone https://github.com/ninja-build/ninja.git
WORKDIR /ninja
RUN mkdir build && \
    cmake -B build -S . && \
    cmake --build build/ -j$(nproc) --target install

# Install kfr
RUN git clone https://github.com/krakenrf/kfr /tmp/kfr && \
    mkdir /tmp/kfr/build && \
    cd /tmp/kfr/build && \
    cmake -G Ninja -DENABLE_CAPI_BUILD=ON -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release .. && \
    cmake --build . -j$(nproc) && \
    cp /tmp/kfr/build/lib/* /usr/local/lib && \
    mkdir /usr/include/kfr && \
    cp /tmp/kfr/include/kfr/capi.h /usr/include/kfr && \
    ldconfig && \
    rm -rf /tmp/kfr


# Make RUN commands use the new environment:
RUN apt-get install -y wget

ENV CONDA_DIR=/opt/miniconda
ENV PATH=$CONDA_DIR/bin:$PATH
ENV CONDA_ALWAYS_YES=true
ENV CONDA_AUTO_UPDATE_CONDA=false

# Install miniconda
RUN mkdir -p ${CONDA_DIR} && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ${CONDA_DIR}/miniconda.sh
RUN  bash $CONDA_DIR/miniconda.sh -u -b -p ${CONDA_DIR}
RUN rm -rf $CONDA_DIR/miniconda.sh

# Use mamba instead for faster environment creation
RUN conda install -c conda-forge mamba

# Copy the requirements file into the container
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml

# RUN conda init
# RUN source ${CONDA_DIR}/etc/profile.d/conda.sh
# RUN conda activate kraken

# Install any dependencies
# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "kraken", "/bin/bash", "-c"]

ARG DYNRT_PATH=/DynRT-aaoesutnhaoeunsth

# Install dynamic rtmp streamer dependency
RUN git clone https://github.com/acoustic-warfare/DynRT-streamer.git $DYNRT_PATH
WORKDIR ${DYNRT_PATH} 
# RUN ./configure.py
RUN meson setup build/ --native-file native-file.ini -Dprefix=$CONDA_DIR/envs/kraken -Dlibdir=lib
RUN cat native-file.ini
RUN ninja -C build/

RUN ninja -C build install

ENV LD_LIBRARY_PATH=/path/to/your/lib:$LD_LIBRARY_PATH

# Copy the start.sh script and make it executable
COPY /src/kraken/start_doa.sh /src/kraken/
RUN chmod +x /src/kraken/start_doa.sh
RUN rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /src/kraken

# Copy the rest of the application code into the container
COPY /src/kraken/ /src/kraken/

ENTRYPOINT ["./start_doa.sh"]
