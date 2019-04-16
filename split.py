# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 19:33:08 2019

@author: Blayneyyx
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def img_split(file_path,file_name,k=10,size=256):
    fp=os.path.join(file_path,file_name)
    img=np.array(Image.open(fp))
    if img.dtype!=np.uint8:
        raise ValueError( 'error dtype')
        
    shape=img.shape
    if shape[0]<size or shape[1]<size:
        return
    
    L=file_path.split('\\')
    op=''
    for i in range(len(L)-1):
        op=op+L[i]+"\\"
    op=op+"split\\"
    
    #save
    if not os.path.isdir(op):
        os.mkdir(op)
     
    for i in range(k):
        #crop
        if(shape[0]>size and shape[1]>size):
            w_r=np.random.randint(low=0,high=shape[0]-size)
            w_h=np.random.randint(low=0,high=shape[1]-size)
            plt.imsave(op+str(i)+'_'+file_name,img[w_r:w_r+size,w_h:w_h+size])
        



def traverse(root_path):
    step=1
    for root,_,files in os.walk(root_path,topdown=False):
        total=len(files)
        for f in files:
            img_split(root,f)
            print("split img %d / %d"%(step,total))
            step=step+1

traverse("D:\\IMG_denoising\\data")


