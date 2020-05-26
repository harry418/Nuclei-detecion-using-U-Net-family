#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 20:59:48 2020

@author: harit
"""

# import required libraries
from keras.layers import Conv2D 
from keras.layers import MaxPooling2D
from keras.layers import Conv2DTranspose
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import Concatenate
from keras.models import Model
from keras.layers import UpSampling2D


# different convolution blocks
# 2d standard_unit for downsampling in all models
def standard_unit(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    p = MaxPooling2D((2,2))(c)
    return c,p

# upsampling_unit combinataion of deconvolution and skip layer
def upsampling_unit(x, skip,filters, kernel_size=(3, 3), padding="same", strides=1):
    us = Conv2DTranspose(filters ,(2,2),strides=(2, 2), padding='same')(x)
    concat = Concatenate()([us,skip])
    c,_ = standard_unit(concat,filters)
    return c

# bottleneck function for unet
def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c


############################################
"""
standard u-net 
"""

# define model for unet segmentation
def UNet(image_row,image_col):
    f = [16, 32, 64, 128, 256]
    inputs = Input((image_row,image_col, 3))
    
    p0 = inputs
    c1, p1 = standard_unit(p0, f[0]) #128 -> 64
    c2, p2 = standard_unit(p1, f[1]) #64 -> 32
    c3, p3 = standard_unit(p2, f[2]) #32 -> 16
    c4, p4 = standard_unit(p3, f[3]) #16->8
    
    bn = bottleneck(p4, f[4])
    
    u1 = upsampling_unit(bn, c4, f[3]) #8 -> 16
    u2 = upsampling_unit(u1, c3, f[2]) #16 -> 32
    u3 = upsampling_unit(u2, c2, f[1]) #32 -> 64
    u4 = upsampling_unit(u3, c1, f[0]) #64 -> 128
    
    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u4)
    model = Model(inputs, outputs)
    return model


###########################################
"""
wide unet 
"""

# define wide u-net model
def W_UNet(image_row,image_col):
    f = [16,32,64,128,256]
    #f = [35,70,140,280,560]
    img_input = Input((image_row , image_col ,3))
    
    c11,p1 = standard_unit(img_input,f[0])
    c21,p2 = standard_unit(p1,f[1])
    c31,p3 = standard_unit(p2,f[2])
    c41,p4 = standard_unit(p3,f[3])
    c51,_ = standard_unit(p4 ,f[4])
    
    c42 = upsampling_unit(c51,c41,f[3])
    c33 = upsampling_unit(c42,c31,f[2])
    c24 = upsampling_unit(c33,c21,f[1])
    c15 = upsampling_unit(c24,c11,f[0])
    
    outputs = Conv2D(1,(1,1),activation='sigmoid',padding = 'same',strides =1)(c15)
    
    model = Model(img_input,outputs)
    return model


############################################
"""
nested u-net or standard unet++
"""

# define unet++ model
def Nested_Net(image_row,image_col):
    f = [16, 32, 64, 128, 256]
    img_input = Input((image_row,image_col, 3))

    # first downsampling layer
    c11,p1 = standard_unit(img_input,f[0])
    c21,p2 = standard_unit(p1 , f[1])
    c31,p3 = standard_unit(p2,f[2])
    c41,p4 = standard_unit(p3,f[3])
    c51,_ = standard_unit(p4 ,f[4])
    
    u12 = Conv2DTranspose(f[0], (2, 2), strides=(2, 2), padding='same')(c21)
    concat = Concatenate()([u12, c11])
    c12,_ = standard_unit(concat, f[0])

    u22 = Conv2DTranspose(f[1], (2, 2), strides=(2, 2), padding='same')(c31)
    concat = Concatenate()([u22, c21])
    c22,_ = standard_unit(concat, f[1])

    u32 = Conv2DTranspose(f[2], (2, 2), strides=(2, 2), padding='same')(c41)
    concat = Concatenate()([u32, c31])
    c32,_ = standard_unit(concat, f[2])

    u42 = Conv2DTranspose(f[3], (2, 2), strides=(2, 2), padding='same')(c51)
    concat = Concatenate()([u42, c41])
    c42,_ = standard_unit(concat, f[3])
    
    u13 = Conv2DTranspose(f[0], (2, 2), strides=(2, 2), padding='same')(c22)
    concat = Concatenate()([u13, c11,c12])
    c13,_ = standard_unit(concat, f[0])

    u23 = Conv2DTranspose(f[1], (2, 2), strides=(2, 2), padding='same')(c32)
    concat = Concatenate()([u23, c21,c22])
    c23,_ = standard_unit(concat, f[1])

    u33 = Conv2DTranspose(f[2], (2, 2), strides=(2, 2), padding='same')(c42)
    concat = Concatenate()([u33, c31,c32])
    c33,_ = standard_unit(concat, f[2])
    
    u14 = Conv2DTranspose(f[0], (2, 2), strides=(2, 2), padding='same')(c23)
    concat = Concatenate()([u14, c11,c12,c13])
    c14,_ = standard_unit(concat, f[0])

    u24 = Conv2DTranspose(f[1], (2, 2), strides=(2, 2), padding='same')(c33)
    concat = Concatenate()([u24, c21,c22,c23])
    c24,_ = standard_unit(concat, f[1])
    
    u15 = Conv2DTranspose(f[0], (2, 2), strides=(2, 2), padding='same')(c24)
    concat = Concatenate()([u15, c11,c12,c13,c14])
    c15,_ = standard_unit(concat, f[0])
    
    nestnet_output1 = Conv2D(1,(1,1) ,activation= 'sigmoid',padding ='same' ,strides =1)(c12)
    nestnet_output2 = Conv2D(1,(1,1) ,activation= 'sigmoid',padding ='same' ,strides =1)(c13)
    nestnet_output3 = Conv2D(1,(1,1) ,activation= 'sigmoid',padding ='same' ,strides =1)(c14)
    nestnet_output4 = Conv2D(1,(1,1) ,activation= 'sigmoid',padding ='same' ,strides =1)(c15)
    
    model = Model(input = img_input ,output = nestnet_output4)
    return model


