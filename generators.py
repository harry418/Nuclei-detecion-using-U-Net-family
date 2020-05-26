#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 21:20:47 2020

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

#seeding
seed = 2019
np.random.seed = seed
tf.seed = seed


#data generator
class DataGen(keras.utils.Sequence):
    def __init__(self,ids,path,batch_size,image_size):
        self.ids = ids
        self.path = path
        self.batch_size = batch_size
        self.image_size = image_size
        self.on_epoch_end()
        
    def __load__(self,id_name):
        #path
        image_path = os.path.join(self.path,id_name,"images",id_name) + '.png'
        mask_path = os.path.join(self.path,id_name,"masks/")
        all_masks = os.listdir(mask_path)
        
        #reading_images
        try:
            image = cv2.imread(image_path,1)
            image = cv2.resize(image,(self.image_size,self.image_size))
        except Exception as e:
            print(str(e))
        
        mask = np.zeros((self.image_size,self.image_size,1))
        #reading masks
        for name in all_masks:
            try:
                _mask_path = mask_path +name
                _mask_image = cv2.imread(_mask_path,-1)
                _mask_image = cv2.resize(_mask_image,(self.image_size,self.image_size))
                _mask_image = np.expand_dims(_mask_image,axis = -1)
                mask = np.maximum(mask,_mask_image)
            except Exception as e:
                print(str(e))
        
        #normalising
        image = image/255.
        mask = mask/255.
        return image,mask
    
    def __getitem__(self,index):
        if (index+1)*self.batch_size > len(self.ids):
            self.batch_size = len(self.ids) - index*self.batch_size
            
        files_batch = self.ids[index*self.batch_size : (index+1)*self.batch_size]
        
        image = []
        mask = []
        
        for id_name in files_batch:
            _img , _mask = self.__load__(id_name)
            image.append(_img)
            mask.append(_mask)
            
        image = np.array(image)
        mask = np.array(mask)
        
        return image,mask
    
    def on_epoch_end(self):
        pass
    
    def __len__(self):
        return int(np.ceil(len(self.ids)/float(self.batch_size)))
                   