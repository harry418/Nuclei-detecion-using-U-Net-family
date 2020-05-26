#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 21:23:35 2020

@author: harit
"""

# importing required libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import cv2
import tensorflow as tf
import keras

# import generator
from generators import DataGen

# import model file
from models import *

# Hyperparameter
image_size = 128
batch_size = 2
epochs = 5
train_path = "stage1_train/"
train_ids = next(os.walk(train_path))[1]
val_data_size = 10
valid_ids = train_ids[:val_data_size]
train_ids = train_ids[val_data_size:]

# iniatize dataset
gen = DataGen(train_ids,train_path,batch_size = batch_size ,image_size = image_size)
x,y = gen.__getitem__(0)
print(x.shape,y.shape)

# visualize dataset
r = np.random.randint(0,len(x)-1)
fig = plt.figure()
fig.subplots_adjust(hspace =0.4 ,wspace =0.4)
ax1 = fig.add_subplot(1,2,1)
ax1.title.set_text('image')
ax1.imshow(x[r])
ax2 = fig.add_subplot(1,2,2)
ax2.title.set_text('mask')
ax2.imshow(np.reshape(y[r],(image_size , image_size)),cmap = 'gray')

# load data generators
train_gen = DataGen(train_ids,train_path,batch_size=batch_size, image_size=image_size)
valid_gen = DataGen(valid_ids,train_path,batch_size=batch_size, image_size=image_size)

train_steps = len(train_ids)//batch_size
valid_steps = len(valid_ids)//batch_size

"""
standard unet
"""

# compile unet model 
model_unet = UNet(128,128)
model_unet.compile(optimizer='adam' ,loss = 'binary_crossentropy',metrics = ['accuracy'])
model_unet.summary()

# fit model to generator
# fit to unet
model_unet.fit_generator(train_gen,
                  validation_data = valid_gen,
                  steps_per_epoch = train_steps,
                  validation_steps= valid_steps,
                  epochs = epochs)

# predicting output for unet
x,y = valid_gen.__getitem__(1)
result_unet  = model_unet.predict(x)
result_unet = result_unet >0.5


"""
standard wide-unet
"""

# compiling and training wide-unet model
model_w_unet = W_UNet(128,128)
model_w_unet.compile(optimizer='adam' ,loss = 'binary_crossentropy',metrics = ['accuracy'])
model_w_unet.summary()

# fit dataset to wide-unet
model_w_unet.fit_generator(train_gen,
                  validation_data = valid_gen,
                  steps_per_epoch = train_steps,
                  validation_steps= valid_steps,
                  epochs = epochs)

# predicting output for wide-unet
x,y = valid_gen.__getitem__(1)
result_w_unet  = model_w_unet.predict(x)
result_w_unet = result_w_unet >0.5


"""
standard unet++ or nested-net
"""

# compiling nested-unet or standard unet++
model_nested_net = Nested_Net(128,128)
model_nested_net.compile(optimizer='adam' ,loss = 'binary_crossentropy',metrics = ['accuracy'])
model_nested_net.summary()

# fit data to standard nested net
model_nested_net.fit_generator(train_gen,
                  validation_data = valid_gen,
                  steps_per_epoch = train_steps,
                  validation_steps= valid_steps,
                  epochs = epochs)

# predicting output for nested_net
x,y = valid_gen.__getitem__(1)
result_nested_net  = model_nested_net.predict(x)
result_nested_net = result_nested_net >0.5

"""
visualing output for all networks
"""
# visualizing output
fig = plt.figure()
fig.subplots_adjust(hspace =1 ,wspace =1)
ax1 = fig.add_subplot(2,3,1)
ax1.title.set_text('image')
ax1.imshow(x[1])

ax2 = fig.add_subplot(2,3,2)
ax2.title.set_text('mask')
ax2.imshow(np.reshape(y[1]*255,(image_size , image_size)),cmap = 'gray')

ax3 = fig.add_subplot(2, 3, 4)
ax3.title.set_text('U-Net')
ax3.imshow(np.reshape(result_unet[1]*255, (image_size, image_size)), cmap="gray")

ax4 = fig.add_subplot(2, 3,5)
ax4.title.set_text('U-Net')
ax4.imshow(np.reshape(result_w_unet[1]*255, (image_size, image_size)), cmap="gray")

ax5 = fig.add_subplot(2, 3, 6)
ax5.title.set_text('Nested-Net')
ax5.imshow(np.reshape(result_nested_net[1]*255, (image_size, image_size)), cmap="gray")
