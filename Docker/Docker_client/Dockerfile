#flantick/detector_client
FROM ubuntu
WORKDIR /usr/src/app
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
     && apt-get install -y cmake gcc g++ libopencv-dev build-essential python3-dev autotools-dev \
     libicu-dev libbz2-dev git wget \
    && wget https://boostorg.jfrog.io/artifactory/main/release/1.82.0/source/boost_1_82_0.tar.gz \
    && tar xvf boost_1_82_0.tar.gz

WORKDIR /usr/src/app/boost_1_82_0
RUN ./bootstrap.sh --prefix=/usr/ && ./b2 && ./b2 install

WORKDIR /usr/src/app
RUN git clone https://github.com/flantick/Detector_server_client.git

WORKDIR /usr/src/app/Detector_server_client/detector_client
RUN mkdir build
WORKDIR /usr/src/app/Detector_server_client/detector_client/build
RUN cmake .. && cmake --build .

CMD ["./build/detector_client"]