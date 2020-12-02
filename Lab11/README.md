# Lab 11: Generative Adversarial Network

## Usage

- Training GAN 
  - train.py

- Image Results
  - Iteration 100
  ![iter 100](image_results/iter_100.JPG)

  - Iteration 1000
  ![iter 1000](image_results/iter_1000.JPG)

  - Iteration 15800
  ![iter 15800](image_results/iter_15800.JPG)

- TensorBoard
  - Open log files ("logs_wgan_gp" is in  C:\Users\hieu>)
  ```bash
  tensorboard --logdir logs_wgan_gp
  ```
  ![TensorBoad](image_results/run.JPG)

  Go to the link http://localhost:6006/ to see the resutl on TensorBoard:
  - result
  ![TensorBoad](image_results/Tensorboard.JPG)

- References: 
  - WGAN-GP paper: https://proceedings.neurips.cc/paper/2017/file/892c3b1c6dccd52936e27cbd0ff683d6-Paper.pdf
  - https://github.com/KUASWoodyLIN/TF2-WGAN

