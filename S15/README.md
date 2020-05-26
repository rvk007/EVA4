# Session 15 - Monocular Depth Estimation

Depth estimation is a computer vision task designed to estimate depth from a 2D image. Depth information is important for autonomous systems to perceive environments and estimate their own state.  The depth image includes information about the distance of the objects in the image from the viewpoint, which is usually the camera taking the image.

|                 Segmentation Masks             |                 Depth Map                    | 
| :--------------------------------------------: | :------------------------------------------: |
|   ![seg_mask](images/mask.png)                |         ![depth_map](images/depth.png)      | 

In this project I made a DepthNet Architecture which takes background and a background-foreground image as input and produces their corresponding depth mappings and segmentation masks of the forground.

The project is divided into two segments focusing on the two different outputs of the model. The first one being MaskNet:

 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11drXRdxWF1AFUgtp-0ybKsTYfiHCLsFU?usp=sharing)    


## MaskNet


The motive of this architecture is to produce Segmentation masks of the given image.


<p align="center"><img src="images/mask.png" width="350px"></p>

A image is a matrix for the computer and in this particular image we require only two pixel values, i.e., 0(Black) and 1(White). We know the power of deep learning, it is a machine learning technique that teaches computers to do what comes naturally to humans: learn by example. Deep learning is a key technology behind driverless cars, it is the key to voice control in consumer devices like phones, tablets, TVs, and hands-free speakers. Deep learning is getting lots of attention lately and for good reason. It’s achieving results that were not possible before.

So we now know that predicting two different numbers won't be that difficult for a model to learn.

Taking this into consideration I created a pretty small fully convolutional network for MaskNet which takes background and a background-foreground image as input and outputs segmentation masks of the forground.

<p align="center"><img src="images/masknet.png" width="300px" /></p>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BFIp-rdOjE4C-PcV6Jm_A7F4NuQRIhj_?usp=sharing)  

## DepthNet  


### Architecture
In the architecture **bg** denotes the background image and **bg_fg** denotes background-foreground image. DepthNet follows a encoder-decoder model, since we want images as an output, we convolve the images to get the features by encoding and then convolve up, namely `UpSample`, decoding the image to reach it's initial dimension.

![depthnet](images/depthnet.png)


The model is fully convolutional and includes efficient residual up-sampling blocks — decoder — that track high-dimensional regression problems.   
The first section of the network is proprietary for combining the the inputs together by concatenating them. The second part is a sequence of convolutional and interpolate layers that guide the network in learning its upscaling. In the end a final convolution is applied that yeilds the final predictions. 

<img src="images/encoder.png" width="210px">
<img src="images/decoder.png" width="210px">

### Approach for DepthNet

The encoding for **Segmentation masks** is stopped early than Depth Maps, as I understood that there is no need for masks to undergo a deep network. Also as evident from the architecture the output from decoder of segmentation masks is being fed to decoder of Depth Maps, because masks stores the location of the foreground that can help decoder of Depth Maps to yeild better predictions .

### DepthNet Result
|                 IOU                            |                 Validation Loss              | 
| :--------------------------------------------: | :------------------------------------------: |
|   ![iou](images/best_model_iou.png)           |         ![loss](images/best_model_loss.png)   |

### Predictions

![seg_mask](images/1.png)  

## Parameters and Hyperparameters

- Loss Function: BCE-RMSE Loss (combination of `nn.BCEWithLogitsLoss` and `Root Mean Square Loss`)
- Optimizer: SGD
- Learning Rate: 0.1
- Number of epochs: 12
- Comparison Metric: Loss
- Momentum: 0.9
- L2 regularization factor: 1e-8


## Dataset Preparation
 The dataset consist of 400000 **bg_fg**, 400000 **bg_fg_masks** which are segmentation masks of te foreground and 400000 **bg_fg_depth** which are depth estimation maps and 100 different **bg** and 100 different **fg** on which the whole dataset is created.
 The complete procedure of dataset preparation is explained here ![](https://github.com/rvk007/MODEST-Museum-Dataset).


## Data Augmentation

Image data is encoded into 3 stacked matrices, each of size height×width. These matrices represent pixel values for an individual RGB color value. Lighting biases are amongst
the most frequently occurring challenges to image recognition problems. Therefore, the efectiveness of color space transformations, also known as `photometric transformations`. I applied the below transformations.
- HueSaturationValue
- RandomBrightnessContrast

`HueSaturationValue`, it changes the brilliance and intensity of a color and `RandomBrightnessContrast` as the name suggests it randomly changes the brightness and contrast of the image. Since they depict the real world scenario, I chose them as augmentations for the dataset.

## Loss Functions

### BCEWithLogitsLoss
Binary Cross Entropy with Logits Loss, is used for binary classification, since the segmentation masks and depth also consist of two pixel 0 and 1, the above loss can be used.

### Root Mean Square Loss
RMSE is a quadratic scoring rule that also measures the average magnitude of the error. It’s the square root of the average of squared 
differences between prediction and actual observation.
RMSE does not necessarily increase with the variance of the errors. RMSE increases with the variance of the frequency distribution of error magnitudes.

### SSIM  
Structural SIMilarity is a image assessment algorithm. It analyse the pair of images as perceived by a human for which it takes the three major aspect for comparison:
- Change in luminance, which compares the brightness of the two images.
- Change in contrast, which looks for differences in the range between the brightest and darkest extent of the two images.
- Correlation, which compares the fundamental structure.

It scales the two images to same size and resolution for a pixel-by-pixel comparison. This provides a big advance over MSE(Mean Square Error) and PSNR (Peak Signal to Noise Ratio).

### Dice Loss
`DiceLoss = 1 - Dice Coefficient`  
Dice coefficient, which is essentially a measure of overlap between two samples measure ranges from 0 to 1 where a Dice coefficient of 1 denotes perfect and complete overlap.
Because our target mask is binary, we effectively zero-out any pixels from our prediction which are not "activated" in the target mask. For the remaining pixels, we are essentially penalizing low-confidence predictions; a higher value for this expression, which is in the numerator, leads to a better Dice coefficient.

As these losses are very efficient, I also combined them during prediction of DepthNet. You can see all the applied combinations here.![](\DeepNet\DeepNet\model\losses).

Following is the behaviour of some of the combination of loss functions:

### Results
|                 IOU                            |                 Validation Loss              | RMSE                                       |
| :--------------------------------------------: | :------------------------------------------: |:------------------------------------------:|
|   ![cluster_plot_k3](images/iou.png)           |         ![anchor_bbox_k3](images/loss.png)   |   ![anchor_bbox_k3](images/rmse.png)       |



### Experiments
To look at the all experiments I have performed, go here.![](\DeepNet\Experiments)

## Project Setup

### On Local System

Install the required packages  
 `$ pip install -r requirements.txt`

### On Google Colab

Upload the files in the root folder and select Python 3 as the runtime type and GPU as the harware accelerator.

