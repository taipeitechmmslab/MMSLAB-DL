# Lab 12: Object detection (YOLO-V3)

## Usage
- Requirements
    
    - [Matplotlib](https://matplotlib.org/)
    
    - [TensorFlow >= 2.0](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf)
    
    - [Numpy](http://www.numpy.org/)
    
    - [OpenCV](https://opencv.org/)
    
    - [TensorFlow Datasets](https://www.tensorflow.org/datasets/)
    
    - [TensorFlow addons](https://github.com/tensorflow/addons)
    
-  Download the trained model weights of the original YOLO-v3:
    - Method 1: Create folder "model_data" and go to the link: https://pjreddie.com/media/files/yolov3.weights to download trained YOLO-V3 model
    - Method 2: Runing the command
    ```bash
    wget https://pjreddie.com/media/files/yolov3.weights -O model_data/yolov3.weights
    ```

- Converting pre-trained model
    ```bash
    python convert.py
    ```

- Training YOLO-V3 model
    ```bash
    python train.py
    ```

- Training YOLO-V3(multi scale training)
    ```bash
    python train-multi-scale.py
    ```

- Test YOLO-V3
    ```bash
    python test.py
    ```

- TensorBoard
    ```bash
    tensorboard --logdir logs_yolo
    ```

- Image Results


![Results](https://raw.githubusercontent.com/KUASWoodyLIN/TF2-Yolo3/master/output_images/output_results.png)
- References

    - https://github.com/pjreddie/darknet
    - https://github.com/qqwweee/keras-yolo3
    - https://github.com/zzh8829/yolov3-tf2
    - https://github.com/allanzelener/YAD2K
