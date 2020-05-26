# Experiments 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1GlKuMTD1tNMbHOesFuByrHrYO2KF2pJG?usp=sharing)  


Applied a combination of RMSE, BCEwithLogitsLoss and Dice Loss on the DepthNet model. 
**Result**: Doesn't yeild good results!


### Predictions

|                 Segmentation Masks             |                 Depth Map                    | 
| :--------------------------------------------: | :------------------------------------------: |
|   ![seg_mask](Image/rbd_mask.png)             |     ![depth_map](Image/rbd_depth.png)       | 

### Results
|                 IOU                            |                 Validation Loss              |
| :--------------------------------------------: | :------------------------------------------: |
|   ![iou](Image/rbd_iou.png)                    |         ![loss](Image/rbd_loss.png)          |




[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Ca-vrgWn92bbCdb5vYanxQwGoT9QXm5i?usp=sharing)  


Applied enhanced BCEwithLogitsLoss with Rmse Loss and Learning Rate=0.01.
**Result**: Almost same results as that of DepthNet

### Predictions

|                 Segmentation Masks             |                 Depth Map                    | 
| :--------------------------------------------: | :------------------------------------------: |
|   ![seg_mask](Image/enhance_mask.png)         |   ![depth_map](Image/enhance_depth.png)     | 


### Results
|                 IOU                            |                 Validation Loss              |
| :--------------------------------------------: | :------------------------------------------: |
|   ![iou](Image/enhance_iou.png)                |         ![loss](Image/enhance_loss.png)      |

### Predictions
![output](image/2.png)



[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16r-esxeYQvNLa7rqUJM6ielukp52QmoE?usp=sharing)


Applied Data Augmentation 
- HueSaturationValue
- RandomBrightnessContrast
**Result**: Predicted images are blurry! There is no necessity to use data augmentation transformations as the network is not over fitting.

### Predictions

|                 Segmentation Masks             |                 Depth Map                    | 
| :--------------------------------------------: | :------------------------------------------: |
|   ![seg_mask](Image/albu_mask.png)            |    ![depth_map](Image/albu_depth.png)       | 

### Results
|                 IOU                            |                 Validation Loss              |
| :--------------------------------------------: | :------------------------------------------: |
|   ![iou](Image/albu_iou.png)                   |         ![loss](Image/albu_loss.png)      
   |
### Predictions
![output](image/3.png)



[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_e_lWO2IbfqnTph-Uyq7xt4fY340gLNl?usp=sharing)


Created a architecture similar to U-NET, applied a combination of BCEWithLogitsLoss and SSIM
**Result**: Results were similar to DepthNet, but since there are more than 3x parameters in U-NET, so I preferred DepthNet.

### Predictions

|                 Segmentation Masks             |                 Depth Map                    | 
| :--------------------------------------------: | :------------------------------------------: |
|   ![seg_mask](Image/unet.png)             |    ![depth_map](Image/unet_depth.png)        | 

### Results
|                 IOU                            |                 Validation Loss              |
| :--------------------------------------------: | :------------------------------------------: |
|   ![iou](Image/n_bce_iou.png)                  |         ![loss](Image/n_bce_loss.png)        |
