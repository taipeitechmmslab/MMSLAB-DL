# Lab 12: Object detection (YOLO-V3)

## Usage
  
-  Download the trained model weights of the original YOLO-v3:
    - Method 1: At root folder, create folder "model_data" and go to the link: https://pjreddie.com/media/files/yolov3.weights to download trained YOLO-V3 model.
    - Method 2: Run the command
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

- Prediction Results


![Results](output_images/output_results.png)
- References
    - https://arxiv.org/pdf/1804.02767.pdf
    - https://github.com/pjreddie/darknet
    - https://github.com/qqwweee/keras-yolo3
    - https://github.com/zzh8829/yolov3-tf2
    - https://github.com/allanzelener/YAD2K
    - https://github.com/KUASWoodyLIN/TF2-Yolo3
