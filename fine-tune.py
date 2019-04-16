# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 20:27:04 2019

@author: freeze
"""

import os
import numpy as np
import tensorflow as tf
from bn_wrapper import bn_layer
from utils import cal_psnr,tf_psnr,data_loader
import matplotlib.pyplot as plt
from skimage import measure

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def print_img(ax):
    ax=np.clip(ax*255,0,255).astype('uint8')
    plt.imshow(ax)
    return ax

def print_clean_img(ax):
     bx=np.expand_dims(ax,0)
     cx=sess.run(y,{x:bx,it:False})
     cx=cx[0]
     return print_img(cx)

def cal_ssim(ax,cx):
    return measure.compare_ssim(ax, cx, multichannel=True, data_range=255, win_size=11)


def dncnn(inputs, is_training, output_channels=3):
    with tf.variable_scope('block1'):
        output = tf.layers.conv2d(inputs, 64, 3, padding='same', activation=tf.nn.relu)
    for layers in range(2, 16):
        with tf.variable_scope('block%d' % layers):
            output = tf.layers.conv2d(output, 64, 3, padding='same', name='conv%d' % layers, use_bias=False)
            output = tf.nn.relu(bn_layer(output,is_training))
    with tf.variable_scope('block17'):
        output = tf.layers.conv2d(output, output_channels, 3, padding='same')
    return inputs - output


#hypermater 
input_c_dim=3
WIDTH=256
HEIGHT=256
batch_size=42
lr=0.00001
iteration=5000



sess=tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
saver=tf.train.import_meta_graph('trained-step3/model.ckpt-4925.meta')
saver.restore(sess,tf.train.latest_checkpoint('trained-step3/'))

#default graph
graph=tf.get_default_graph()

with tf.device("/gpu:0"):
    x=graph.get_tensor_by_name("noisy_image:0")
    y_=graph.get_tensor_by_name("clean_image:0")
    it=graph.get_tensor_by_name("Placeholder:0")
    LR=graph.get_tensor_by_name("Placeholder_1:0")
    
    y=graph.get_tensor_by_name("sub:0")
    loss=graph.get_tensor_by_name("mul:0")
    eva_psnr=graph.get_tensor_by_name("mul_3:0")
    
    train=graph.get_operation_by_name("Adam_opt")



#IO
dl_noise=data_loader("D:\\IMG_denoising\\color_noise_another")
dl_noise.load_imgs()

dl=data_loader("D:\\IMG_denoising\\split")
dl.load_imgs()

loss_set=[]
psnr_set=[]
step_len=6
max_performance=0
saver=tf.train.Saver(max_to_keep=1)

for i in range(iteration):
    choose=np.random.choice(dl.img_num,dl.img_num,replace=False)
    batch_img=dl.batch_data()
    noisy_img=dl_noise.batch_data()
 
    for j in range(int(dl.img_num/step_len)):
        _,lose,ps=sess.run([train,loss,eva_psnr],{y_:batch_img[choose[j*step_len:(j+1)*step_len]],x:noisy_img[choose[j*step_len:(j+1)*step_len]],it:True,LR:lr})
        loss_set.append(lose)
        psnr_set.append(ps)
        if(ps>max_performance):
            max_performance=ps
            saver.save(sess,'trained-step4/model.ckpt',global_step=i)
    print("iter = %d , loss = %f ,psnr = %f "%(i,lose,ps))
 
    
plt.plot(loss_set,'k-',label='loss')
plt.title('loss per generation')
plt.xlabel('generation')
plt.ylabel('loss')
plt.legend(loc='lower right')
plt.show()

