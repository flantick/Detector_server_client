FROM pytorch/libtorch-cxx11-builder:cuda11.8-9d42578f576838d04fa0166c9d57bf0d23d841aa
WORKDIR /usr/src/app
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -y cmake gcc g++ libopencv-dev build-essential python3-dev autotools-dev libicu-dev libbz2-dev git wget unzip

RUN wget https://download.pytorch.org/libtorch/cu118/libtorch-shared-with-deps-2.0.1%2Bcu118.zip
RUN unzip libtorch-shared-with-deps-2.0.1+cu118.zip

RUN wget https://boostorg.jfrog.io/artifactory/main/release/1.82.0/source/boost_1_82_0.tar.gz
RUN tar xvf boost_1_82_0.tar.gz
WORKDIR /usr/src/app/boost_1_82_0
RUN ./bootstrap.sh --prefix=/usr/
RUN ./b2
RUN ./b2 install

WORKDIR /usr/src/app
RUN git clone https://github.com/flantick/Detector_server_client.git
WORKDIR /usr/src/app/Detector_server_client/detector_server
RUN mkdir build
WORKDIR /usr/src/app/Detector_server_client/detector_server/build
RUN cmake -DCMAKE_PREFIX_PATH=/usr/src/app/libtorch/share/cmake/Torch ..
RUN cmake --build .
CMD ["./build/detector_server"]
#-DCMAKE_PREFIX_PATH=/usr/src/app/libtorch/share/cmake/Torch