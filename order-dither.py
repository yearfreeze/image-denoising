# -*- coding: utf-8 -*-
"""
Created on Sat Mar 2 1:01:14 2019

@author: freeze
#
#有序抖动半调法图像加噪,Bayer模板
#

"""
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

T_mat=np.array([[0,8,2,10],[12,4,14,6],[3,11,1,9],[15,7,13,5]])



def gray_od_noise(img):
    shape=img.shape
    output=np.zeros_like(img)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if((int(img[i,j])>>4)>T_mat[i%4,j%4]):
                output[i,j]=255
            else:
                output[i,j]=img[i,j]
    return output

#添加噪声
def od_noise(file_path,file_name):
    fp=os.path.join(file_path,file_name)
    img=np.array(Image.open(fp))
    if img.dtype!=np.uint8:
        raise ValueError( 'error dtype')
        
    shape=img.shape
    output=np.zeros_like(img)
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[-1]):
                if((int(img[i,j,k])>>4)>T_mat[i%4,j%4]):
                    output[i,j,k]=255
                else:
                    output[i,j,k]=img[i,j,k]
    

    #convert RGB img to gray
    gray=np.array(Image.open(fp).convert('L'))
    gray_n=gray_od_noise(gray)
    
    L=file_path.split('\\')
    op=''
    for i in range(len(L)-1):
        op=op+L[i]+"\\"
    oop=op+"color_noise\\"
    gp=op+"gray\\"
    gnp=op+"gray_noise\\"
    #save
    if not os.path.isdir(oop):
        os.mkdir(oop)
    if not os.path.isdir(gp):
        os.mkdir(gp)
    if not os.path.isdir(gnp):
        os.mkdir(gnp)
    
    #print(output.shape)
    plt.imsave(oop+file_name,output)
    #print(gray.shape)
    plt.imsave(gp+file_name,gray,cmap = plt.get_cmap('gray'))
    #print(gray_n.shape)
    plt.imsave(gnp+file_name,gray_n,cmap = plt.get_cmap('gray'))


def traverse(root_path):
    step=1
    for root,_,files in os.walk(root_path,topdown=False):
        total=len(files)
        for f in files:
            od_noise(root,f)
            print("create img %d / %d"%(step,total))
            step=step+1


traverse("D:\\IMG_denoising\\split")



