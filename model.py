# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 16:36:05 2019

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
lr=0.0005
iteration=5000

x=tf.placeholder(tf.float32, [None, WIDTH, HEIGHT, input_c_dim], name = 'noisy_image')
y_=tf.placeholder(tf.float32, [None, WIDTH, HEIGHT, input_c_dim], name = 'clean_image')
it=tf.placeholder(tf.bool)
LR=tf.placeholder(tf.float32)
#define 

with tf.device("/gpu:0"):
    y=dncnn(x,it)
    loss=(1.0/batch_size)*tf.nn.l2_loss(y_-y)
    eva_psnr=tf_psnr(y,y_)
    
    opt=tf.train.AdamOptimizer(learning_rate=LR,beta1=0.5,name='Adam_opt')
    train=opt.minimize(loss)


#sess init
init=tf.global_variables_initializer()
    
sess=tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
sess.run(init)


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
    if i<1000:
        lr=0.0005
    elif i<3000:
        lr=0.00005
    else:
        lr=0.00001
    for j in range(int(dl.img_num/step_len)):
        _,lose,ps=sess.run([train,loss,eva_psnr],{y_:batch_img[choose[j*step_len:(j+1)*step_len]],x:noisy_img[choose[j*step_len:(j+1)*step_len]],it:True,LR:lr})
        loss_set.append(lose)
        psnr_set.append(ps)
        if(ps>max_performance):
            max_performance=ps
            saver.save(sess,'trained-step3/model.ckpt',global_step=i)
    print("iter = %d , loss = %f ,psnr = %f "%(i,lose,ps))
 
    
plt.plot(loss_set,'k-',label='loss')
plt.title('loss per generation')
plt.xlabel('generation')
plt.ylabel('loss')
plt.legend(loc='lower right')
plt.show()

#measure.compare_ssim(ax, cx, multichannel=True, data_range=255, win_size=11)
    

    
    
    
    
    
    
    