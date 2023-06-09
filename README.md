# Detector_server_client
### Real-Time Object Detection Server
The Real-Time Object Detection Server is a project that provides performing real-time object detection on frames using a pre-trained deep learning model. 
The server is built using C++ and utilizes **Boost.Asio** for networking, **OpenCV** for image processing and **LibTorch** for loading model. It accepts incoming connections, receives frames from clients,
processes them using a loaded object detection model, and sends back frame with the detected objects.

The server utilizes multi-threading to handle multiple client connections concurrently.

<image src="/pictures/example.png" alt="example"><image>

- [Installation](#installation)
  - [installation detector_server](#installation-detector_server)
  - [installation detector_client](#installation-detector_client)
- [Usage](#usage)
  - [Usage detector_server](#Usage-detector_server)
  - [Usage detector_client](#Usage-detector_client)
- [Get torchscript model](#Get-torchscript-model)

## Installation
1. Clone the repository: `git clone https://github.com/flantick/Detector_server_client.git`
2. Install the dependencies:
   - **Boost library**: Download and extract Boost 1.82.0 from [Boost website](https://www.boost.org/users/history/version_1_82_0.html). Set the `BOOST_ROOT` environment variable to the Boost library directory.
   - **OpenCV**: Download and extract OpenCV from the [OpenCV website](https://opencv.org/releases/)). Set the `OPENCV_DIR` environment variable to the Boost library directory.
   - only for server - **LibTorch**: Download and extract LibTorch from the [PyTorch website](https://pytorch.org/get-started/locally/). Set the `CMAKE_PREFIX_PATH` environment variable to the LibTorch directory.

### installation-detector_server
 Build the project using CMake:
   - `cd detector_server`
   - `mkdir build && cd build`
   - `cmake ..`
   - `cmake --build .`

### installation-detector_client
 Build the project using CMake:
   - `cd detector_client`
   - `mkdir build && cd build`
   - `cmake ..`
   - `cmake --build .`

## Usage
  
### Usage-detector_server
1. Run the server: `./detector_server <model_path> <ip> <port> [device]`
   - `model_path`: Path to the model.torchscript file.
   - `ip`: IP address to bind the server.
   - `port`: Port number to listen on.
   - `device`: cpu or gpu (if gpu then model must be for gpu)
2. Connect to the server using a client application.
#### Example usage: 
`./detector_server yolov8n_gpu.torchscript 192.168.0.100 5000 gpu`\
or use [Docker](Docker/Docker_server) 

### Usage-detector_client
1. Run the client application: `./detector_client <server_ip> <server_port> <video_source> [output]`
   - `server_ip`: IP address of the server to connect to.
   - `server_port`: Port number on the server to connect to.
   - `video_source`: Path or index of the video source (e.g., file path or camera index).
   - `output`: Optional output file path to record the received frames.
#### Example usage: 
`./detector_client 192.168.0.100 5000 video.mp4 output.avi`\
or use [Docker](Docker/Docker_client) 

## Get-torchscript-model
```python
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
model.export(format='torchscript', device='0')  # 'cpu' for cpu or '0' for gpu
```
