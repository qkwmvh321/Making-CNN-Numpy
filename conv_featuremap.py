import numpy as np
import cv2
import pandas as pd

def make_fillter(num,channel,size):
    return np.random.rand(num,channel,size,size)

#paddin계산 함수
def my_pad(x,num):
    
    b,c,w,h = x.shape
    s = num*2
    rw = w+s
    rh = h+s
    pad = np.zeros((b,c,rw,rh))
    for i in range(b):
        pad[i,:,num:rw-num,num:rh-num] = x[i]
    return pad


def create_array(t_h,t_w,k_w,k_h):
    if t_w==1:
        new_array = np.zeros((t_h,k_w*k_h))
    elif t_h==1:
        new_array = np.zeros((t_w,k_w*k_h))
    else:
        new_array = np.zeros((t_w*t_h,k_w*k_h))
    return new_array

#conv1d feature map
def make_feature_map_conv1d(x,kernel,strid,pad=0):
    
    h,w = x.shape[1],x.shape[2] #c*h*w
    k_w =kernel.shape[1]
    k_h =kernel.shape[0]
    
    t_w =(w+pad*2-k_w)//strid+1
    t_h =(h+pad*2-k_h)//strid+1

    new_array = np.zeros((1,t_h*t_w))
    count=0
    
    r_k = kernel.reshape(np.prod(k.shape),1)
    for i in range(0,t_h,strid):
        
        for j in range(0,t_w,strid):
            
            b=x[:,i*strid:k_h+i,j*strid:k_w+j]
            ddd = np.dot(b.flatten(),r_k)
            new_array[:,count] = ddd
            count +=1
            
    
    return new_array.reshape(t_h,t_w) 

#conv2d feature map
def make_feature_map_conv2d(x, kernel, strid, pad=0):
    
    c,h,w = x.shape #c*h*w
    k_w =kernel.shape[2]
    k_h =kernel.shape[1]
    
    t_w =(w+pad*2-k_w)//strid+1
    t_h =(h+pad*2-k_h)//strid+1

    new_array = np.zeros((1,t_h*t_w))
    count=0
    
    r_k = kernel.reshape(k.shape[1]*k.shape[2],3)
    print(new_array.shape)
    
    for i in range(0,t_h,strid):
        
        for j in range(0,t_w,strid):
            
            b=x[:,i*strid:k_h+i,j*strid:k_w+j]
            b2 = b.reshape(3,b.shape[1]*b.shape[2])
            ddd = np.dot(b2,r_k).sum()
            print(ddd)
            new_array[:,count] = ddd

            count +=1
            
    
    return new_array.reshape(t_h,t_w) 


