# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 22:15:34 2019

@author: freeze
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


h = np.asarray([[1/16, 5/16, 3/16], [7/16, 0, 0], [0, 0, 0]])

def pp_noise(src):
    out = np.zeros(np.shape(src), dtype=np.uint8)
    h_h, h_w = np.shape(h)

    h_center_y = h_w // 2
    h_center_x = h_h // 2

    src_h, src_w, src_c = np.shape(src)

    for cnt_c in range(src_c):
        error = np.zeros(np.shape(src))
        for cnt_y in range(src_h):
            for cnt_x in range(src_w):
                tempppp = 0
                for k_y in range(h_h):
                    for k_x in range(h_w):
                      if cnt_y + k_y - h_center_y < 0 or cnt_y + k_y - h_center_y >= src_h or \
                              cnt_x + k_x - h_center_x < 0 or cnt_x + k_x - h_center_x >= src_w:
                          continue
                      tempppp = tempppp + h[k_y, k_x] * error[cnt_y + k_y - h_center_y, cnt_x + k_x - h_center_x, cnt_c]
                uuuuu = src[cnt_y, cnt_x, cnt_c] + tempppp
                if uuuuu < 128:
                    out[cnt_y, cnt_x, cnt_c] = 0
                else:
                    out[cnt_y, cnt_x, cnt_c] = 255

                error[cnt_y, cnt_x, cnt_c] = uuuuu - out[cnt_y, cnt_x, cnt_c]

    return out

#添加噪声
def od_noise(file_path,file_name):
    fp=os.path.join(file_path,file_name)
    img=np.array(Image.open(fp))
    if img.dtype!=np.uint8:
        raise ValueError( 'error dtype')
        
    output=pp_noise(img)
    


    
    L=file_path.split('\\')
    op=''
    for i in range(len(L)-1):
        op=op+L[i]+"\\"
    oop=op+"gray_noise_another\\"

    if not os.path.isdir(oop):
        os.mkdir(oop)

    plt.imsave(oop+file_name,output)



def traverse(root_path):
    step=1
    for root,_,files in os.walk(root_path,topdown=False):
        total=len(files)
        for f in files:
            od_noise(root,f)
            print("create img %d / %d"%(step,total))
            step=step+1


traverse("D:\\IMG_denoising\\gray")