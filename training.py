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

#train the model
train_gen = DataGen(train_ids,train_path,batch_size=batch_size, image_size=image_size)
valid_gen = DataGen(valid_ids,train_path,batch_size=batch_size, image_size=image_size)

train_steps = len(train_ids)//batch_size
valid_steps = len(valid_ids)//batch_size

# compile model
model = UNet(128,128)
model.compile(optimizer = 'adam',loss ='binary_crossentropy' ,metrics =['accuracy'])
model.summary()

# fit model to generator
model.fit_generator(train_gen,
                    validation_data = valid_gen,
                    steps_per_epoch = train_steps,
                    validation_steps= valid_steps,
                    epochs = epochs)

# predicting output
x,y = valid_gen.__getitem__(1)
result  = model.predict(x)
result = result >0.5

# visualizing output
r = np.random.randint(0,len(x)-1)
fig = plt.figure()
fig.subplots_adjust(hspace =0.4 ,wspace =0.4)
ax1 = fig.add_subplot(1,3,1)
ax1.title.set_text('image')
ax1.imshow(x[r])
ax2 = fig.add_subplot(1,3,2)
ax2.title.set_text('mask')
ax2.imshow(np.reshape(y[r],(image_size , image_size)),cmap = 'gray')
ax3 = fig.add_subplot(1, 3, 3)
ax3.title.set_text('U-Net')
ax.imshow(np.reshape(result[1]*255, (image_size, image_size)), cmap="gray")