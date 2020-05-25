# Session 15 - Monocular Depth Estimation

Depth estimation is a computer vision task designed to estimate depth from a 2D image. Depth information is important for autonomous systems to perceive environments and estimate their own state.  The depth image includes information about the distance of the objects in the image from the viewpoint, which is usually the camera taking the image.

In this project I made a DepthNet Architecture which takes background and a background-foreground image as input and produces their corresponding depth mappings and segmentation masks of the forground.

The project is divided into two segments focusing on the two different outputs of the model. The first one being MaskNet:

 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11drXRdxWF1AFUgtp-0ybKsTYfiHCLsFU?usp=sharing)    


MaskNet


The motive of this architecture is to produce Segmentation masks of the given image.

![mask](/Images/masks.png)  

A image is a matrix for the computer and in this particular image we require only two pixel values, i.e., 0(Black) and 1(White). We know the power of deep learning, it is a machine learning technique that teaches computers to do what comes naturally to humans: learn by example. Deep learning is a key technology behind driverless cars, it is the key to voice control in consumer devices like phones, tablets, TVs, and hands-free speakers. Deep learning is getting lots of attention lately and for good reason. It’s achieving results that were not possible before.

So we now know that predicting two different numbers won't be that difficult for a model to learn.

Taking this into consideration I created a pretty small fully convolutional network for MaskNet which takes background and a background-foreground image as input and outputs segmentation masks of the forground.

The second part of the project focuses on estimating depth maps.  

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BFIp-rdOjE4C-PcV6Jm_A7F4NuQRIhj_?usp=sharing)  
DepthNet  


Architecture
From the work done for developing the segmentation masks, I understood that generating masks shouldn't require deep networks. So came up with the below architecture.  
In the architecture **bg** denotes the background image and **bg_fg** denotes background-foreground image. DepthNet follows a encoder-decoder model, since we want images as an output, we convolve the images to get the features by encoding and then convolve up, namely `UpSample`, decoding the image to reach it's initial dimension.

![depth_architecture](/Images/architecture.png)   
The architecture of Encoder-Decoder is inspired from Resnet. 
![encoder](/Images/encoder.png)   
![decoder](/Images/decoder.png) 


### Parameters and Hyperparameters

- Loss Function: BCE-RMSE Loss (combination of `nn.BCEWithLogitsLoss` and `Root Mean Square Loss`)
- Optimizer: SGD
- Number of epochs: 12
- Comparison Metric: Loss
- Momentum: 0.9
- L2 regularization factor: 1e-8

### Experiments

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1GlKuMTD1tNMbHOesFuByrHrYO2KF2pJG?usp=sharing)  


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Ca-vrgWn92bbCdb5vYanxQwGoT9QXm5i?usp=sharing)  

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16r-esxeYQvNLa7rqUJM6ielukp52QmoE?usp=sharing)

### Dataset Preparation
The complete procedure dataset preparation is explained here ![](https://github.com/rvk007/MODEST-Museum-Dataset). 


### Data Augmentation

Image data is encoded into 3 stacked matrices, each of size height×width. These matrices represent pixel values for an individual RGB color value. Lighting biases are amongst
the most frequently occurring challenges to image recognition problems. Therefore, the efectiveness of color space transformations, also known as `photometric transformations`.
I applied the below transformations.
- HueSaturationValue
- RandomBrightnessContrast

`HueSaturationValue`, it changes the brilliance and intensity of a color and `RandomBrightnessContrast` as the name suggests it randomly changes the brightness and contrast of the image. Since they depict the real world scenario, I chose them as augmentations for the dataset.

### Results
![encoder](/Images/masks1.png)   
![decoder](/Images/depth.png) 

## Project Setup

### On Local System

Install the required packages  
 `$ pip install -r requirements.txt`

### On Google Colab

Upload the files in the root folder and select Python 3 as the runtime type and GPU as the harware accelerator.

