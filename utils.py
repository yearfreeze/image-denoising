# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 15:45:46 2019

@author: freeze
"""
import os
import tensorflow as tf
import numpy as np
from PIL import Image

def data_augmentation(img, mode):
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        img = np.rot90(img)
        return np.flipud(img)
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        img = np.rot90(img, k=2)
        return np.flipud(img)
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        img = np.rot90(img, k=3)
        return np.flipud(img)
    
    
def cal_psnr(im1, im2):
    # assert pixel value range is 0-255 and type is uint8
    mse = ((im1.astype(np.float) - im2.astype(np.float)) ** 2).mean()
    psnr = 10 * np.log10(255 ** 2 / mse)
    return psnr


def tf_psnr(im1, im2):
    # assert pixel value range is 0-1
    mse = tf.losses.mean_squared_error(labels=im2 * 255.0, predictions=im1 * 255.0)
    return 10.0 * (tf.log(255.0 ** 2 / mse) / tf.log(10.0))

    
class data_loader():
    def __init__(self,filepath="D:\\IMG_denoising\\color_noise"):
        self.file_path=filepath
        FL=[]
        for root,_,files in os.walk(filepath,topdown=False):
            for f in files:
                FL.append(f)
        self.file_list=FL
        self.img_num=len(FL)
        
    def load_imgs(self):
        #pixel values range 0-255
        data=[]
        for f in self.file_list:
            im=np.array(Image.open(self.file_path+"\\"+f))
            im=im[:,:,0:3]
            data.append(im)
        
        self.img_data=data
        return data
        
    def batch_data(self):
        #convert to 0-1
        inputs=np.zeros((self.img_num,256,256,3),dtype=np.float32)
        count=0
        for raw in self.img_data:
            exp=data_augmentation(raw,mode=np.random.randint(0,7))
            exp=np.expand_dims(raw,0)
            inputs[count]=exp/256.0
            count=count+1
        
        
        return np.array(inputs)
            
        
        
        
        
        
        
    