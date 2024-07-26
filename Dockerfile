#Use an official Python runtime as a parent image
FROM continuumio/miniconda3

#Use mamba instead for faster environment creation
RUN conda install -c conda-forge mamba

#Install necessary packages
RUN apt-get update && apt-get install -y \
    lsof \
    libgl1-mesa-glx \
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
    make && \
    rm -rf /var/lib/apt/lists/*

#Install kfr
RUN git clone https://github.com/krakenrf/kfr /tmp/kfr && \
    mkdir /tmp/kfr/build && \
    cd /tmp/kfr/build && \
    cmake -DENABLE_CAPI_BUILD=ON -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release .. && \
    make && \
    cp /tmp/kfr/build/lib/* /usr/local/lib && \
    mkdir /usr/include/kfr && \
    cp /tmp/kfr/include/kfr/capi.h /usr/include/kfr && \
    ldconfig && \
    rm -rf /tmp/kfr

#Set the working directory in the container
WORKDIR /src/kraken

#Copy the requirements file into the container
COPY environment.yml /src/kraken/

#Install any dependencies
RUN conda env create -f /src/kraken/environment.yml

#Copy the start.sh script and make it executable
COPY /src/kraken/start_doa.sh /src/kraken/
RUN chmod +x /src/kraken/start_doa.sh

#Copy the rest of the application code into the container
COPY /src/kraken/ /src/kraken/

ENTRYPOINT ["./start_doa.sh"]