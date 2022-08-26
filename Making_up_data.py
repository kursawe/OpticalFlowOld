#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 12:45:41 2022

@author: apple
"""
import numpy as np
import matplotlib.pyplot as plt
import skimage
import tifffile

all_data = np.zeros((5,1024,1024))


def my_function_1(x,y):
    return np.exp(-(x-5)**2 - (y-3)**2)#x0=5 y0=3center locations
    #return np.exp(-(x-4)**2 - (y-2)**2)#x0=4 y0=2center locations,Vx=2,Vy=3
x = np.linspace(0,20,1024)#(0,10,100)0-10 100 samples uniform,np.linespace(start,stop,number)
y = np.linspace(0,20,1024)
z = np.zeros((1024,1024))

for x_index, x_value in enumerate(x):
    for y_index,y_value in enumerate(y):
        z[x_index, y_index] = my_function_1(x_value, y_value)
        
all_data[0,:,:] = z

plt.figure()
plt.imshow(z, cmap = 'Greys')
plt.imsave('makeup_data_frame1.tiff',z, cmap = 'Greys',vmin=0,vmax=1)

#different functions= different intensities, same positions to get gamma,change center locations x0 y0 get velocity
def my_function_2(x,y):
    return np.exp(-(x-6)**2 - (y-5)**2)#x0=3 y0=5center locations,Vx=1,Vy=2

for x_index, x_value in enumerate(x):
    for y_index,y_value in enumerate(y):
        z[x_index, y_index] = my_function_2(x_value, y_value)
        
all_data[1,:,:] = z

        
plt.figure()
plt.imshow(z, cmap = 'Greys')
plt.imsave('makeup_data_frame2.tiff',z, cmap = 'Greys',vmin=0,vmax=1)


def my_function_3(x,y):
    return np.exp(-(x-6)**2 - (y-5)**2)+0.5#x0=0 y0=0center locations gamma=0.5

for x_index, x_value in enumerate(x):
    for y_index,y_value in enumerate(y):
        z[x_index, y_index] = my_function_3(x_value, y_value)
        
all_data[2,:,:] = z

plt.figure()
plt.imshow(z, cmap = 'Greys')
#plt.savefig("makeup_data_frame3.tif")
plt.imsave('makeup_data_frame3.tiff',z, cmap = 'Greys',vmin=0,vmax=1)

#skimage.io.imsave('new_made_up_data.tif',all_data)
tifffile.imsave('new_made_up_data.tif',all_data)







