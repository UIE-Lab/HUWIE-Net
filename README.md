# HUWIE-Net (Hybrid Underwater Image Enhancement Network)

This is an open-source underwater image enhancement method developed using PyTorch.
If you use our code, please consider citing our paper.

![](./im.png)
Top row: real-world underwater images , bottom row: the corresponding enhanced images by HUWIE-Net.

# Dataset
The dataset used in this project is the [UIEB Dataset](https://li-chongyi.github.io/proj_benchmark.html)

# Working Directory
The working directory structure is organized as follows:

    ├── Data
    │   ├── checkpoints
    │   ├── UIEBD
    │   │   ├── gt
    │   │   │   ├── 2_img_.png
    │   │   │   ├── ...
    │   │   ├── raw
    │   │   │   ├── 2_img_.png
    │   │   │   ├── ...
    │   │   └── UIEBD_random_shuffle_3.txt
    ├── Experiments
    │   ├── HUWIE-Net
    │   │   ├── core
    │   │   ├── pre_trained_models
    │   │   │   ├── HUWIE_Net_epoch50.pth
    │   │   │   ├── ...
    │   │   ├── test.py
    │   │   └── train.py

# Requirements
The packages used are listed below. All dependencies are provided in the requirements.txt file.  

- python==3.12.7  
- pytorch==2.3.1  
- tensorboard==2.17.0  
- opencv==4.10.0  
- pillow==11.0.0  
- torchvision==0.18.1  

# Testing and Training

The steps for testing and training are provided below:

- Download the HUWIE-Net repository.  
- Create a working directory to store all required files.  
- Download the dataset and move it to the appropriate directory within the working directory.  
- Install all necessary packages using the requirements.txt file.  
- Place the UIEBD_random_shuffle_3.txt file in the specified directory.  
- Execute the test.py script to evaluate the functionality of the pre-trained HUWIE-Net.  
- Execute the train.py script to train HUWIE-Net.  

Run test.ipynb to test HUWIE-Net in the Colab environment. The cells sequentially perform the following steps:

- Cloning the HUWIE-Net Repository
- Downloading and Extracting the Dataset
- Importing Modules
- Configuration and Setup
- Testing and Evaluation

Run train.ipynb to train HUWIE-Net in the Colab environment.

# Citation
If you use our code, please consider citing our paper.

```
@ARTICLE{10857292,
  author={Demir, Ozan and Aktas, Metin and Eksioglu, Ender M.},
  journal={IEEE Access}, 
  title={Joint Optimization in Underwater Image Enhancement: A Training Framework Integrating Pixel-Level and Physical-Channel Techniques}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Image enhancement;Image color analysis;Optimization;Scattering;Cameras;Computational modeling;Frequency modulation;Feature extraction;Convolution;Absorption;Underwater image enhancement;deep learning;underwater image formation model;dark channel prior;physics-informed deep network;joint optimization},
  doi={10.1109/ACCESS.2025.3536173}}
```

# Contact

If you have any questions, please feel free to contact us at ozandemir22651@gmail.com.

# Usefull Code

[UWCNN](https://li-chongyi.github.io/proj_underwater_image_synthesis.html)  
[UIEC^2-Net](https://github.com/BIGWangYuDong/UWEnhancement)

