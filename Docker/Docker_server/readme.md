usage
```commandline
 docker run -p 8888:8888 -v path/to_folder_with_model:/mnt -it --gpus all flantick/detector_server ./detector_server /mnt/yolov8x_gpu.torchscript 0.0.0.0 8888 gpu
```