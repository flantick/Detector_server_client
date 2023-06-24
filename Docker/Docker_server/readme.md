usage
```commandline
 docker run -p 1234:1234 -v path/to_folder_with_model:/mnt -it flantick/detector_server ./detector_server /mnt/yolov8n_gpu.torchscript 127.0.0.1 1234 gpu
```