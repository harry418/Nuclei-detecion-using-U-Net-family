# unet-or-unet-segmentation
This is an implementation of " UNet ,wide-UNet and UNet++ for Medical Image Segmentation " in Python and powered by the Keras deep learning framework (Tensorflow as backend). 

**Table Of Contents**

 U-Net Segmentation
 
 wide-UNet
 
 Nested_UNet or UNet++
 
 Dataset
 
 kaggle notebook
 
 result of comparison

U-Net Segmentation:-
=============
 While converting an image into a vector, we already learned the feature mapping of the image so why not use the same mapping
 to convert it again to image. This is the recipe behind UNet.Use the same feature maps that are used for contraction to 
 expand a vector to a segmented image. This would preserve the structural integrity of the image which would reduce distortion
 enormously. and it is mainly uses in biomedical image segmentation.
 
 total params = 1,941,105
 
 architecure of unet:-
 
![alt text](https://github.com/harry418/unet-or-unet-segmentation/blob/master/images/u-net-architecture.png)

wide-Net Segmentation:-
=============
wide unet also has same structure like unet. but it uses different filters than unet in every convolution or deconvolution.
thus it increases total params. and give a significantly change in accuracy.

total params = 9,282,876

 Nested_UNet or UNet++ Segmentation:-
=============
For the first time, a new architecture, called UNet++ (nested U-Net architecture), is proposed for a more precise
segmentation. We introduced the intermediate layers to U-Nets, which naturally form multiple new up-sampling expanding paths
of different depths, resulting in an ensemble of U-Nets with a partially shared contracting path.

total params = 2,261,889

architecure of Nested-UNet
![alt text](https://github.com/harry418/unet-or-unet-segmentation/blob/master/images/fig_unet++.png)


Dataset
=============
https://www.kaggle.com/c/data-science-bowl-2018

upload stage1_train files here

kaggle notebook
=============
 https://www.kaggle.com/harry418/unet-or-unet-segmentation-keras
 
 before running file make sure to use acceerator as gpu
 
 Result comparison
 =============
 accuracy of unet = 95.99%
 
 accuracy of wide-UNet = 94.87%
 
 accuracy of unet++ = 95.99%
 
 
