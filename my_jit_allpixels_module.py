#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 18:25:30 2022

@author: apple
"""
import skimage.io
import skimage.filters
import matplotlib.pyplot as plt
import PIL
import numpy as np
from matplotlib import pyplot
import celluloid
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
plt.rcParams["animation.ffmpeg_path"] = '/Users/apple/Desktop/optical flow_Jochen/OpticalFlow/ffmpeg/bin/ffmpeg'
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import jit

@jit
def conduct_optical_flow(box_size = 10, all_images = skimage.io.imread('MB301110_i_4_movie_8 bit.tif')):
    
    mode = 'include_gamma'
    first_image = all_images[0,:,:]
    blurred_images= all_images

    
    
    
    
    I = all_images[0,:,:]
    dIdx = np.zeros_like(I)
    delta_t = 1
    number_of_frames = blurred_images.shape[0]
    Nb = int((1024)/box_size)#Number of boxes
    all_v_x = np.zeros((number_of_frames-1,Nb,Nb))
    all_v_y = np.zeros((number_of_frames-1,Nb,Nb))
    all_gamma = np.zeros((number_of_frames-1,Nb,Nb))
    
    for frame_index in range(1,blurred_images.shape[0]):
        difference_to_previous_frame = blurred_images[frame_index] - blurred_images[frame_index -1]
        current_frame = blurred_images[frame_index]
        previous_frame = blurred_images[frame_index -1]
        dIdx = (current_frame[2:,1:-1] +previous_frame[2:,1:-1] - current_frame[:-2,1:-1]-previous_frame[:-2,1:-1])/(4*delta_t)
        dIdy = (current_frame[1:-1,2:] +previous_frame[1:-1,2:] - current_frame[1:-1,:-2]-previous_frame[1:-1,:-2])/(4*delta_t)
        delta_I_too_big = current_frame-previous_frame
        delta_I = delta_I_too_big[1:1023,1:1023]#0-1024 in total
        b = box_size*box_size 
        v_x = all_v_x[frame_index-1,:,:]
        v_y = all_v_y[frame_index-1,:,:]
        gamma_ = all_gamma[frame_index-1,:,:]
        for box_index_x in range(Nb):
            #from0,...Nb-1
            for box_index_y in range(Nb):
                local_delta_I = delta_I[box_index_x*box_size:((box_index_x+1)*box_size),box_index_y*box_size:(((box_index_y+1)*box_size))]
                local_dIdx = dIdx[box_index_x*box_size:((box_index_x+1)*box_size),box_index_y*box_size:(((box_index_y+1)*box_size))]
                local_dIdy = dIdy[box_index_x*box_size:((box_index_x+1)*box_size),box_index_y*box_size:(((box_index_y+1)*box_size))]
                sum1 = np.sum(local_delta_I*local_dIdx)
                sum2 = np.sum(local_delta_I*local_dIdy)
                #ABC only
                if mode == 'velocity_only':
                    A = np.sum((local_dIdx)**2)
                    B = np.sum(local_dIdx*local_dIdy)
                    C = np.sum((local_dIdy)**2)
                    Vx =(-C*sum1+ B*sum2)/(delta_t*(A*C-B**2))
                    #Assume Delta t=1
                    Vy =(-A*sum2+ B*sum1)/(delta_t*(A*C-B**2))
                    v_x[box_index_x,box_index_y] = Vx
                    v_y[box_index_x,box_index_y] = Vy
    #                 skimage.io.imsave('Vx_velocity_only.tif', v_x[box_index_x,box_index_y])
    #                 skimage.io.imsave('Vy_velocity_only.tif', v_y[box_index_x,box_index_y])
                elif mode == 'include_gamma':
                    A = np.sum((local_dIdx)**2)
                    B = np.sum(local_dIdx*local_dIdy)
                    C = np.sum(local_dIdx)
                    D = np.sum((local_dIdy)**2)
                    E = np.sum(local_dIdy)
                    sum3 = np.sum(local_delta_I)
                    Vx = ((E**2-b*D)*sum1+(b*B-C*E)*sum2+(C*D-B*E)*sum3)/(delta_t*(b*A*D-A*E**2-b*B**2-C**2*D+2*B*C*E))
                    Vy = ((b*B-C*E)*sum1+(C**2-b*A)*sum2+(A*E-B*C)*sum3)/(delta_t*(b*A*D-A*E**2-b*B**2-C**2*D+2*B*C*E))
                    v_x[box_index_x,box_index_y] = Vx
                    v_y[box_index_x,box_index_y] = Vy
                    sum4 = np.sum(local_dIdx*Vx)
                    sum5 = np.sum(local_dIdx*Vy)
                    this_gamma = ((B*E-C*D)*sum1+(B*C-A*E)*sum2+(A*D-B**2)*sum3)/(delta_t*(b*A*D-A*E**2-b*B**2-C**2*D+2*B*C*E))
                    gamma_[box_index_x,box_index_y] = this_gamma
     # skimage.io.imsave('Vx_include_gamma.tif', all_v_x)
     # skimage.io.imsave('Vy_include_gamma.tif', all_v_y)
     # skimage.io.imsave('Gamma_include_gamma.tif', all_gamma)  
    all_pixel_v_x = np.zeros((number_of_frames-1,Nb*box_size,Nb*box_size))
    all_pixel_v_y = np.zeros((number_of_frames-1,Nb*box_size,Nb*box_size))
    all_pixel_gamma = np.zeros((number_of_frames-1,Nb*box_size,Nb*box_size))
    
    all_pixel_contraction = np.zeros((number_of_frames-1,(Nb-2)*box_size,(Nb-2)*box_size))
    all_pixel__flow_contributions = np.zeros((number_of_frames-1,Nb*box_size,Nb*box_size))
    all_pixel__gamma_contributions = np.zeros((number_of_frames-1,Nb*box_size,Nb*box_size))
    all_pixel__flow_gamma = np.zeros((number_of_frames-1,Nb*box_size,Nb*box_size))
    all_pixel__contraction_contributions = np.zeros((number_of_frames-1,(Nb-2)*box_size,(Nb-2)*box_size))
    
    NP = Nb*box_size
    for frame_index in range(1,blurred_images.shape[0]):
        pixel_v_x = all_pixel_v_x[frame_index-1,:,:]
        pixel_v_y = all_pixel_v_y[frame_index-1,:,:]
        pixel_gamma = all_pixel_gamma[frame_index-1,:,:]
        for pixel_index_x in range(NP):
            for pixel_index_y in range(NP):
                pixel_v_x[pixel_index_x,pixel_index_y]= all_v_x[frame_index-1,int(pixel_index_x/box_size),int(pixel_index_y/box_size)]
                pixel_v_y[pixel_index_x,pixel_index_y]= all_v_y[frame_index-1,int(pixel_index_x/box_size),int(pixel_index_y/box_size)]
                pixel_gamma[pixel_index_x,pixel_index_y]= all_gamma[frame_index-1,int(pixel_index_x/box_size),int(pixel_index_y/box_size)]
   #Use all_pixel_v_x->div_v=0. so have to use all_v_x based on box->all_div_v based on box
    all_div_v = np.zeros((number_of_frames-1,Nb-2,Nb-2))
    for frame_index in range(1,blurred_images.shape[0]-1):
        current_all_v_x = all_v_x[frame_index]
        current_all_v_y = all_v_y[frame_index]
        dV_xdx = (current_all_v_x[2:,1:-1] - current_all_v_x[:-2,1:-1])/(2*delta_t)
        dV_ydy = (current_all_v_y[1:-1,2:] - current_all_v_y[1:-1,:-2])/(2*delta_t)
        this_div_v = dV_xdx + dV_ydy
        all_div_v[frame_index,:,:]= this_div_v 

    all_pixel_div_v = np.zeros((number_of_frames-1,(Nb-2)*box_size,(Nb-2)*box_size))#All_v_xframe-1
    for frame_index in range(1,blurred_images.shape[0]-1):
        pixel_div_v = all_pixel_div_v[frame_index-1,:,:]
        for pixel_index_x in range((Nb-2)*box_size):
            for pixel_index_y in range((Nb-2)*box_size):
                pixel_div_v[pixel_index_x,pixel_index_y]= all_div_v[frame_index-1,int(pixel_index_x/box_size),int(pixel_index_y/box_size)]
    # for frame_index in range(0,blurred_images.shape[0]-1):
    #     current_all_pixel_v_x = all_pixel_v_x[frame_index]
    #     current_all_pixel_v_y = all_pixel_v_y[frame_index]
    #     dV_xdx_pixel = (current_all_pixel_v_x[2:,1:-1] - current_all_pixel_v_x[:-2,1:-1])/(2*delta_t)
    #     dV_ydy_pixel = (current_all_pixel_v_y[1:-1,2:] - current_all_pixel_v_y[1:-1,:-2])/(2*delta_t)
    #     this_div_v_pixel = dV_xdx_pixel + dV_ydy_pixel
    #     all_pixel_div_v[frame_index,:,:]= this_div_v_pixel 
    
        
    intensity = blurred_images#84,1024,1024
    total_pixel_intensity = intensity[:,1:NP+1,1:NP+1]#all frames
    all_pixel_intensity = intensity[:-1,1:NP+1,1:NP+1]
    allmtwo_pixel_intensity = all_pixel_intensity[:,1*box_size:-1*box_size,1*box_size:-1*box_size]

    all_pixel_contraction = np.zeros((number_of_frames-1,(Nb-2)*box_size,(Nb-2)*box_size))
    for frame_index in range(1,blurred_images.shape[0]):
        pixel_contraction = all_pixel_contraction[frame_index-1,:,:]
        for pixel_index_x in range((Nb-2)*box_size):
            for pixel_index_y in range((Nb-2)*box_size):             
                    this_pixel_contraction = -(allmtwo_pixel_intensity[frame_index-1,pixel_index_x,pixel_index_y]*all_pixel_div_v[frame_index-1,pixel_index_x,pixel_index_y])
                    pixel_contraction[pixel_index_x,pixel_index_y] = this_pixel_contraction
    
    #Quantify the contributions of the remodeling made to the cytoskeleton dynamics
    all_pixel_deltaI= np.zeros((number_of_frames-1,Nb*box_size,Nb*box_size))
    for frame_index in range(1,blurred_images.shape[0]):
        pixel_deltaI= all_pixel_deltaI[frame_index-1,:,:]
        for pixel_index_x in range(Nb*box_size):
            for pixel_index_y in range(Nb*box_size):  
                this_pixel_deltaI = total_pixel_intensity[frame_index,pixel_index_x,pixel_index_y] - total_pixel_intensity[frame_index-1,pixel_index_x,pixel_index_y]
                pixel_deltaI[pixel_index_x,pixel_index_y] = this_pixel_deltaI 
                
    all_pixel_gamma_contributions = np.zeros((number_of_frames-1,Nb*box_size,Nb*box_size)) 
    for frame_index in range(1,blurred_images.shape[0]):
        pixel_gamma_contributions = all_pixel_gamma_contributions[frame_index-1,:,:]
        for pixel_index_x in range(Nb*box_size):
            for pixel_index_y in range(Nb*box_size):                
                    this_pixel_gamma_contributions = all_pixel_gamma[frame_index-1,pixel_index_x,pixel_index_y]/all_pixel_deltaI[frame_index-1,pixel_index_x,pixel_index_y]
                    pixel_gamma_contributions[pixel_index_x,pixel_index_y] = this_pixel_gamma_contributions


    #Quantify Flow contribution:V delata I = -(VxIx+VyIy)/delta I
    #Degrade Ix, Iy first
    all_pixel_flow_contributions = np.zeros((number_of_frames-1,Nb*box_size,Nb*box_size))#Nb=51
    all_pixel_dIdx = np.zeros((number_of_frames-1,Nb*box_size,Nb*box_size))
    all_pixel_dIdy = np.zeros((number_of_frames-1,Nb*box_size,Nb*box_size))
    for frame_index in range(1,blurred_images.shape[0]):
        current_frame = blurred_images[frame_index]#have to define agian ,othewise use the loop at last frame
        previous_frame = blurred_images[frame_index -1]             
        dIdx = (current_frame[2:,1:-1] +previous_frame[2:,1:-1] - current_frame[:-2,1:-1]-previous_frame[:-2,1:-1])/(4*delta_t)
        dIdy = (current_frame[1:-1,2:] +previous_frame[1:-1,2:] - current_frame[1:-1,:-2]-previous_frame[1:-1,:-2])/(4*delta_t)
        pixel_dIdx = all_pixel_dIdx[frame_index-1,:,:]
        pixel_dIdy = all_pixel_dIdy[frame_index-1,:,:]
        for pixel_index_x in range(Nb*box_size):
            for pixel_index_y in range(Nb*box_size):
                needdIdx = dIdx[1:NP+1,1:NP+1]
                needdIdy = dIdy[1:NP+1,1:NP+1]
                this_pixel_dIdx = needdIdx[pixel_index_x,pixel_index_y]
                pixel_dIdx[pixel_index_x,pixel_index_y] = this_pixel_dIdx
                this_pixel_dIdy = needdIdy[pixel_index_x,pixel_index_y]
                pixel_dIdy[pixel_index_x,pixel_index_y] = this_pixel_dIdy
 
                
    for frame_index in range(1,blurred_images.shape[0]):#blurred_images.shape(84,1024,1024)
        pixel_flow_contributions = all_pixel_flow_contributions[frame_index-1,:,:]
        for pixel_index_x in range(Nb*box_size):
            for pixel_index_y in range(Nb*box_size):                
                this_pixel_flow_contributions = -(all_pixel_v_x[frame_index-1,pixel_index_x,pixel_index_y]*all_pixel_dIdx[frame_index-1,pixel_index_x,pixel_index_y] + all_pixel_v_y[frame_index-1,pixel_index_x,pixel_index_y]*all_pixel_dIdy[frame_index-1,pixel_index_x,pixel_index_y])/all_pixel_deltaI[frame_index-1,pixel_index_x,pixel_index_y]
                pixel_flow_contributions[pixel_index_x,pixel_index_y] = this_pixel_flow_contributions
    #sum check if -V gradient I+gamma -Delta I =0
    all_pixel_sumcheck= np.zeros((number_of_frames-1,Nb*box_size,Nb*box_size))
    for frame_index in range(1,blurred_images.shape[0]):
        pixel_sumcheck = all_pixel_sumcheck[frame_index-1,:,:]
        for pixel_index_x in range(Nb*box_size):
            for pixel_index_y in range(Nb*box_size):                
                    this_pixel_sumcheck = -(all_pixel_v_x[frame_index-1,pixel_index_x,pixel_index_y]*all_pixel_dIdx[frame_index-1,pixel_index_x,pixel_index_y] + all_pixel_v_y[frame_index-1,pixel_index_x,pixel_index_y]*all_pixel_dIdy[frame_index-1,pixel_index_x,pixel_index_y])+all_pixel_gamma[frame_index-1,pixel_index_x,pixel_index_y]-all_pixel_deltaI[frame_index-1,pixel_index_x,pixel_index_y]
                    pixel_sumcheck[pixel_index_x,pixel_index_y] = this_pixel_sumcheck # the values of the sumcheck are almost 0

    #sum check if relative error: (-V gradient I+gamma -Delta I)/Delta I =0
    all_pixel_relative_error = np.zeros((number_of_frames-1,Nb*box_size,Nb*box_size))
    for frame_index in range(1,blurred_images.shape[0]):
        pixel_relative_error = all_pixel_relative_error[frame_index-1,:,:]
        for pixel_index_x in range(Nb*box_size):
            for pixel_index_y in range(Nb*box_size):                
                    this_pixel_relative_error = (-(all_pixel_v_x[frame_index-1,pixel_index_x,pixel_index_y]*all_pixel_dIdx[frame_index-1,pixel_index_x,pixel_index_y] + all_pixel_v_y[frame_index-1,pixel_index_x,pixel_index_y]*all_pixel_dIdy[frame_index-1,pixel_index_x,pixel_index_y])+all_pixel_gamma[frame_index-1,pixel_index_x,pixel_index_y]-all_pixel_deltaI[frame_index-1,pixel_index_x,pixel_index_y])/all_pixel_deltaI[frame_index-1,pixel_index_x,pixel_index_y]
                    pixel_relative_error[pixel_index_x,pixel_index_y] = this_pixel_relative_error 

            
    #Compare the Flows and Remodeling of Actin in the Cells:flow/gamma
    all_pixel_flow_gamma = np.zeros((number_of_frames-1,Nb*box_size,Nb*box_size))
    for frame_index in range(1,blurred_images.shape[0]):
        pixel_flow_gamma = all_pixel_flow_gamma[frame_index-1,:,:]
        for pixel_index_x in range(Nb*box_size):
            for pixel_index_y in range(Nb*box_size):                
                    this_pixel_flow_gamma = abs(-(all_pixel_v_x[frame_index-1,pixel_index_x,pixel_index_y]*all_pixel_dIdx[frame_index-1,pixel_index_x,pixel_index_y] + all_pixel_v_y[frame_index-1,pixel_index_x,pixel_index_y]*all_pixel_dIdy[frame_index-1,pixel_index_x,pixel_index_y]) /all_pixel_gamma[frame_index-1,pixel_index_x,pixel_index_y])
                    pixel_flow_gamma[pixel_index_x,pixel_index_y] = this_pixel_flow_gamma

   #skimage.io.imsave('all_flow_gamma.tif', all_flow_gamma)#np.median(np.abs(all_flow_gamma))=,np.max(np.abs(all_flow_gamma))=,np.min(np.abs(all_flow_gamma))=   
   #Quantify Contraction contribution: |-I div(v)|/Delta I
   #cut off outermost 2 rows of boxes of all_difference_to_previous_frame_box
    mtwobox_pixel_deltaI = all_pixel_deltaI[:,1*box_size:-1*box_size,1*box_size:-1*box_size]
    all_pixel_contraction_contributions = np.zeros((number_of_frames-1,(Nb-2)*box_size,(Nb-2)*box_size))
    for frame_index in range(1,blurred_images.shape[0]):
        pixel_contraction_contributions = all_pixel_contraction_contributions[frame_index-1,:,:]
        for pixel_index_x in range((Nb-2)*box_size):
            for pixel_index_y in range((Nb-2)*box_size):             
                   this_pixel_contraction_contributions = all_pixel_contraction[frame_index-1,pixel_index_x,pixel_index_y]/mtwobox_pixel_deltaI[frame_index-1,pixel_index_x,pixel_index_y]
                   pixel_contraction_contributions[pixel_index_x,pixel_index_y] = this_pixel_contraction_contributions
   
    
   
    return blurred_images, all_pixel_v_x, all_pixel_v_y, all_pixel_gamma, all_pixel_div_v, all_pixel_intensity, all_pixel_contraction, all_pixel_deltaI, all_pixel_sumcheck, all_pixel_relative_error,all_pixel_flow_gamma,all_pixel_contraction_contributions,all_pixel_gamma_contributions,all_pixel_flow_contributions,all_pixel_dIdx, all_pixel_dIdy,all_v_x,all_v_y

def fixjit_optical_flow(smoothing_sigma = 1, box_size = 10, all_images = skimage.io.imread('MB301110_i_4_movie_8 bit.tif')): 
    output = conduct_optical_flow(box_size = 10, all_images = skimage.io.imread('MB301110_i_4_movie_8 bit.tif'))
    dictionary = {}
    dictionary['all_blurred_images'] = output(1)
    dictionary['all_pixel_v_x'] = output(2)
    dictionary['all_pixel_v_y'] = output(3)
    dictionary['all_pixel_gamma'] = output(4)
    dictionary['all_pixel_div_v'] = output(5)
    dictionary['all_pixel_intensity'] = output(6)
    dictionary['all_pixel_contraction'] = output(7)
    dictionary['all_pixel_deltaI'] = output(8)
    dictionary['all_pixel_sumcheck'] = output(9)
    dictionary['all_pixel_relative_error'] = output(0)
    dictionary['all_pixel_flow_gamma'] = output(11)
    dictionary['all_pixel_contraction_contributions'] = output(12)
    dictionary['all_pixel_gamma_contributions'] = output(13)
    dictionary['all_pixel_flow_contributions'] = output(14)
    dictionary['all_pixel_dIdx'] = output(15)
    dictionary['all_pixel_dIdy'] = output(16)
    dictionary['all_v_x'] =output(17)
    dictionary['all_v_y'] =output(18)
    
    return dictionary    

# =============================================================================
#     dictionary = {}
#     dictionary['all_blurred_images'] = blurred_images
#     dictionary['all_pixel_v_x'] = all_pixel_v_x
#     dictionary['all_pixel_v_y'] = all_pixel_v_y
#     dictionary['all_pixel_gamma'] = all_pixel_gamma
#     dictionary['all_pixel_div_v'] = all_pixel_div_v
#     dictionary['all_pixel_intensity'] =  all_pixel_intensity
#     dictionary['all_pixel_contraction'] = all_pixel_contraction
#     dictionary['all_pixel_deltaI'] = all_pixel_deltaI
#     dictionary['all_pixel_sumcheck'] = all_pixel_sumcheck
#     dictionary['all_pixel_relative_error'] = all_pixel_relative_error
#     dictionary['all_pixel_flow_gamma'] = all_pixel_flow_gamma
#     dictionary['all_pixel_contraction_contributions'] = all_pixel_contraction_contributions
#     dictionary['all_pixel_gamma_contributions'] = all_pixel_gamma_contributions
#     dictionary['all_pixel_flow_contributions'] = all_pixel_flow_contributions
#     dictionary['all_pixel_dIdx'] = all_pixel_dIdx
#     dictionary['all_pixel_dIdy'] = all_pixel_dIdy
#     return dictionary
# =============================================================================
 
    
 
def gamma_pixel_fixed_colorbar_movie(all_gamma,filename = " Gamma_pixel_fixed_colorbar.mp4"):    
    plt.figure()
    animation_camera = celluloid.Camera(plt.gcf())#plt.gcf()Get the current figure
    for index in range(all_gamma.shape[0]):
        this_gamma_frame = all_gamma[index,:,:]
        img_gamma = this_gamma_frame 
        plt.imshow(img_gamma, cmap=None, norm=None, aspect=None, interpolation=None, alpha=None, vmin=np.min(all_gamma), vmax=np.max(all_gamma), origin=None, extent=None, filternorm=1, filterrad=4.0, resample=None, url=None)
            #plt.colorbar(ax = plt.gca())
        animation_camera.snap()
    plt.colorbar()
    plt.title("Gamma_include_gamma")
    plt.xlabel("Number of Boxes")
    plt.ylabel("Number of Boxes")
    animation = animation_camera.animate()
    animation.save(filename) 
        
def gamma_pixel_changing_colorbar_movie(all_gamma,blurred_images, filename = "Gamma_pixel_with_changing_colorbar.mp4"):   
    fig = plt.figure()
    ax = fig.add_subplot(111)
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', size='5%', pad='5%')
    tx = ax.set_title('Gamma_pixel Frame 0')
    def animate(i):  
        cax.cla()
        data = all_gamma[i,:,:]
        im = ax.imshow(data)
        fig.colorbar(im,cax = cax)
        tx.set_text('Gamma_pixel Frame {0}'.format(i))   
    ax.set_xlabel("Number of Boxes")
    ax.set_ylabel("Number of Boxes")  
    ani = FuncAnimation(fig, animate, frames=blurred_images.shape[0]-1)
    #ani.save('Animate_Gamma_include_gamma_changing_colorbar.gif')
    ani.save(filename)                
         
def Vx_pixel_changing_colorbar_movie(all_v_x,blurred_images,filename ='Vx_pixel_include_gamma_with_changing_colorbar.mp4'):   
    fig = plt.figure()
    ax = fig.add_subplot(111)
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', size='5%', pad='5%')
    tx = ax.set_title('Vx_include_gamma_pixel Frame 0')
    def animate(i):  
        cax.cla()
        data = all_v_x[i,:,:]
        im = ax.imshow(data)
        fig.colorbar(im,cax = cax)
        tx.set_text('Vx_include_gamma_pixel Frame {0}'.format(i))   
    ax.set_xlabel("Number of Pixels")
    ax.set_ylabel("Number of Pixels")  
    ani = FuncAnimation(fig, animate, frames=blurred_images.shape[0]-1)
    #ani.save('Vx_include_gamma_changing_colorbar.gif')
    ani.save(filename)     
               
def Vy_pixel_changing_colorbar_movie(all_v_y,blurred_images,filename ='Vy_pixel_include_gamma_with_changing_colorbar.mp4'):      
    fig = plt.figure()
    ax = fig.add_subplot(111)
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', size='5%', pad='5%')
    tx = ax.set_title('Vy_include_gamma_pixel Frame 0')
    def animate(i):  
        cax.cla()
        data = all_v_y[i,:,:]
        im = ax.imshow(data)
        fig.colorbar(im,cax = cax)
        tx.set_text('Vy_include_gamma_pixel Frame {0}'.format(i))   
    ax.set_xlabel("Number of Pixels")
    ax.set_ylabel("Number of Pixels")  
    ani = FuncAnimation(fig, animate, frames=blurred_images.shape[0]-1)
    #ani.save('Vy_include_gamma_changing_colorbar.gif')
    ani.save(filename)  
      
def Visualizing_Velocity_box(all_images,blurred_images,all_v_x,all_v_y,box_size,filename = 'Visualizing Velocity.mp4'):   
    fig = plt.figure()
    tx = plt.title('Visualizing Velocity Frame 0')
    def animate(i): 
        plt.cla()
        image_size = blurred_images.shape[1]
        upper_mgrid_limit = int(image_size/box_size)*box_size#to make1024/100=100
        x_pos = np.mgrid[0:upper_mgrid_limit:box_size]
        x_pos += int(box_size/2)
        y_pos = np.mgrid[0:upper_mgrid_limit:box_size]
        y_pos += int(box_size/2)
        x_direct = all_v_x[i,:,:]
        y_direct = all_v_y[i,:,:]
        plt.imshow(all_images[i,:,:])
        plt.quiver(y_pos, x_pos, y_direct, -x_direct, color = 'white')#arrow is in wrong direction because matplt and quiver have different coordanites
        plt.title("Visualizing Velocity") 
        plt.xlabel("Number of Pixels")
        plt.ylabel("Number of Pixels")
        plt.tight_layout()#make sure all lables fit in the frame
    ani = FuncAnimation(fig, animate, frames=blurred_images.shape[0]-1)
    ani.save(filename)    
        


def all_pixel_div_v_fixed_colorbar_movie(all_div_v,filename = "all_pixel_div_v_fixed_colorbar.mp4"): 
    plt.figure()
    animation_camera = celluloid.Camera(plt.gcf())
    for index in range(all_div_v.shape[0]):
        this_div_v_frame = all_div_v[index,:,:]
        img_div_v= this_div_v_frame 
        plt.imshow(img_div_v, cmap=None, norm=None, aspect=None, interpolation=None, alpha=None,  origin=None, extent=None, filternorm=1, filterrad=4.0, resample=None, url=None)
        #plt.colorbar(ax = plt.gca())
        animation_camera.snap()
    plt.colorbar()
    plt.title("The Divergence of Velocity")
    plt.xlabel("Number of Pixels")
    plt.ylabel("Number of Pixels")
    animation = animation_camera.animate()
    animation.save(filename)

def Contractions_Relaxations_Histogram_plot_pixel(all_contraction,filename ='Contractions or Relaxations Histogram_pixel'):
    plt.figure()
    plt.hist(all_contraction.flatten(), bins=100, range=(-1,1), density=False)#most flow contribution around 0
    plt.xlabel('Values of Contractions or Relaxations')
    plt.ylabel('Number of Pixels')
    plt.title('Contractions or Relaxations Histogram') 
    plt.savefig(filename)
        
def Contractions_or_Relaxations_fixed_colorbar_range1_movie_pixel(all_contraction,filename = "Contractions or Relaxations_fixed_colorbar_range(-1,1)_pixel.mp4"): 
    plt.figure()
    animation_camera = celluloid.Camera(plt.gcf())
    for index in range(all_contraction.shape[0]):
        this_contraction_frame = all_contraction[index,:,:]
        img_contraction = this_contraction_frame 
        plt.imshow(img_contraction, cmap=None, norm=None, aspect=None, interpolation=None, alpha=None, vmin=-1, vmax=1, origin=None, extent=None, filternorm=1, filterrad=4.0, resample=None, url=None)
        #plt.colorbar(ax = plt.gca())
        animation_camera.snap()
    plt.colorbar()
    plt.title("Contractions or Relaxations")
    plt.xlabel("Number of Pixels")
    plt.ylabel("Number of Pixels")
    animation = animation_camera.animate()
    animation.save(filename)



def all_gamma_contributions_changing_colorbar_movie_pixel(all_gamma_contributions,blurred_images,filename = "Animate_all_gamma_contributions_changing_colorbar_pixel.mp4"):    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', size='5%', pad='5%')
    tx = ax.set_title('all_gamma_contributions Frame 0')
    def animate(i):  
        cax.cla()
        data = all_gamma_contributions[i,:,:]
        im = ax.imshow(data)
        fig.colorbar(im,cax = cax)
        tx.set_text('all_gamma_contributions Frame {0}'.format(i))   
    ax.set_xlabel("Number of Pixels")
    ax.set_ylabel("Number of Pixels")  
    ani = FuncAnimation(fig, animate, frames=blurred_images.shape[0]-1)
    ani.save(filename)
            
def all_gamma_contributions_fixed_colorbar_range3_movie_pixel(all_gamma_contributions,blurred_images,filename = "All_gamma_contributions_fixed_colorbar_range(-3,3)_pixel.mp4"):    
    plt.figure()
    animation_camera = celluloid.Camera(plt.gcf())
    for index in range(0,blurred_images.shape[0]-1):
        this_gamma_contributions_frame = all_gamma_contributions[index,:,:]
        img_gamma_contributions = this_gamma_contributions_frame 
        plt.imshow(img_gamma_contributions, cmap=None, norm=None, aspect=None, interpolation=None, alpha=None, vmin=-3, vmax=3, origin=None, extent=None, filternorm=1, filterrad=4.0, resample=None, url=None)
        animation_camera.snap()
    plt.colorbar()
    plt.title("Gamma Contributions")
    plt.xlabel("Number of Pixels")
    plt.ylabel("Number of Pixels")
    animation = animation_camera.animate()
    animation.save(filename)
def all_gamma_contributions_fixed_colorbar_range1_movie_pixel(all_gamma_contributions,blurred_images, filename = "Gamma_contributions_fixed_colorbar_range(-1,1)_pixel.mp4"): 
    plt.figure()
    animation_camera = celluloid.Camera(plt.gcf())
    for index in range(0,blurred_images.shape[0]-1):
        this_gamma_contributions_frame = all_gamma_contributions[index,:,:]
        img_gamma_contributions = this_gamma_contributions_frame 
        plt.imshow(img_gamma_contributions, cmap=None, norm=None, aspect=None, interpolation=None, alpha=None, vmin=-1, vmax=1, origin=None, extent=None, filternorm=1, filterrad=4.0, resample=None, url=None)
        animation_camera.snap()
    plt.colorbar()
    plt.title("Gamma Contributions")
    plt.xlabel("Number of Pixels")
    plt.ylabel("Number of Pixels")
    animation = animation_camera.animate()
    animation.save(filename)
def Gamma_contributions_Histogram_plot_pixel(all_gamma_contributions,filename ='Gamma Contribution Histogram_pixel'):
    plt.figure()
    plt.hist(all_gamma_contributions.flatten(), bins=100, range=(-5,5), density=False)
    plt.xlabel('Values of Gamma Contribution')
    plt.ylabel('Number of Pixels')
    plt.title('Gamma Contribution Histogram')
    plt.savefig(filename)

      
def all_flow_contributions_changing_colorbar_movie_pixel(all_flow_contributions,blurred_images, filename = "all_flow_contributions_changing_colorbar_pixel.mp4"):    
    plt.figure()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', size='5%', pad='5%')
    tx = ax.set_title('all_flow_contributions Frame 0')
    def animate(i):  
        cax.cla()
        data = all_flow_contributions[i,:,:]
        im = ax.imshow(data)
        fig.colorbar(im,cax = cax)
        tx.set_text('Flow Contributions Frame {0}'.format(i))   
    ax.set_xlabel("Number of Pixels")
    ax.set_ylabel("Number of Pixels")  
    ani = FuncAnimation(fig, animate, frames=blurred_images.shape[0]-1)
    ani.save(filename)
def Flow_Contribution_Histogram_plot_pixel(all_flow_contributions,filename ='Flow Contribution Histogram_pixel'):
    plt.figure()
    plt.hist(all_flow_contributions.flatten(), bins=100, range=(-5,5), density=False)#most flow contribution around 0
    plt.xlabel('Values of Flow Contribution')
    plt.ylabel('Number of Pixels')
    plt.title('Flow Contribution Histogram')
    plt.savefig(filename)
def all_flow_contributions_fixed_colorbar_range3_movie_pixel(all_flow_contributions,filename = "Flow_contributions_fixed_colorbar range(-3,3)_pixel.mp4"):    
    plt.figure()
    animation_camera = celluloid.Camera(plt.gcf())
    for index in range(all_flow_contributions.shape[0]):
        this_flow_contributions_frame = all_flow_contributions[index,:,:]
        img_flow_contributions = this_flow_contributions_frame 
        plt.imshow(img_flow_contributions, cmap=None, norm=None, aspect=None, interpolation=None, alpha=None, vmin=-3, vmax=3, origin=None, extent=None, filternorm=1, filterrad=4.0, resample=None, url=None)
        #plt.colorbar(ax = plt.gca())
        animation_camera.snap()
    plt.colorbar()
    plt.title("Flow Contributions")
    plt.xlabel("Number of Pixels")
    plt.ylabel("Number of Pixels")
    animation = animation_camera.animate()
    animation.save(filename)
def all_flow_contributions_fixed_colorbar_range1_movie_pixel(all_flow_contributions,filename = "Flow_contributions_fixed_colorbar range(-1,1)_pixel.mp4"):    
    plt.figure()
    animation_camera = celluloid.Camera(plt.gcf())
    for index in range(all_flow_contributions.shape[0]):
        this_flow_contributions_frame = all_flow_contributions[index,:,:]
        img_flow_contributions = this_flow_contributions_frame 
        plt.imshow(img_flow_contributions, cmap=None, norm=None, aspect=None, interpolation=None, alpha=None, vmin=-1, vmax=1, origin=None, extent=None, filternorm=1, filterrad=4.0, resample=None, url=None)
        #plt.colorbar(ax = plt.gca())
        animation_camera.snap()
    plt.colorbar()
    plt.title("Flow Contributions")
    plt.xlabel("Number of Pixels")
    plt.ylabel("Number of Pixels")
    animation = animation_camera.animate()
    animation.save(filename)

def Absolute_Values_of_Flow_over_Gamma_Histogram_plot_pixel(all_flow_gamma,filename ='Absolute Values of Flow over Gamma Histogram_pixel'):
    plt.figure()
    plt.hist(all_flow_gamma.flatten(), bins=100, range=(0,5), density=False)#most flow contribution around 0
    plt.xlabel('Absolute Values of Flow over Gamma')
    plt.ylabel('Number of Pixels')
    plt.title('Absolute Values of Flow over Gamma Histogram')
    plt.savefig(filename)
        
def All_flow_over_gamma_fixed_colorbar_range_0_max_movie_pixel(all_flow_gamma,blurred_images,filename = "All_flow over gamma_fixed_colorbar range(max)_pixel.mp4"):    
    plt.figure()
    animation_camera = celluloid.Camera(plt.gcf())
    for index in range(0,blurred_images.shape[0]-1):
        this_flow_gamma_frame = all_flow_gamma[index,:,:]
        img_flow_gamma = this_flow_gamma_frame
        plt.imshow(img_flow_gamma, cmap=None, norm=None, aspect=None, interpolation=None, alpha=None, vmin=0, vmax=np.max(all_flow_gamma), origin=None, extent=None, filternorm=1, filterrad=4.0, resample=None, url=None)
        animation_camera.snap()
    plt.colorbar()
    plt.title("Flow over Gamma")
    plt.xlabel("Number of Pixels")
    plt.ylabel("Number of Pixels")
    animation = animation_camera.animate()
    animation.save(filename)


def Contraction_or_Relaxations_Contributions_Histogram_plot_pixel(all_contraction_contributions,filename ='Contraction or Relaxations Contributions Histogram_pixel'):
    plt.figure()
    plt.hist(all_contraction_contributions.flatten(), bins=100, range=(-1000,1000), density=False)#most flow contribution around 0
    plt.xlabel('Contraction or Relaxations Contributions')
    plt.ylabel('Number of Pixels')
    plt.title('Contraction or Relaxations Contributions Histogram') 
    plt.savefig(filename)
        
def Contraction_contributions_fixed_colorbar_range_max_movie_pixel(all_contraction_contributions,filename = "Contraction_contributions_fixed_colorbar range(-250,250)_pixel.mp4"):    
    plt.figure()
    animation_camera = celluloid.Camera(plt.gcf())
    for index in range(all_contraction_contributions.shape[0]):
        this_contraction_contributions_frame = all_contraction_contributions[index,:,:]
        img_contraction_contributions = this_contraction_contributions_frame 
        plt.imshow(img_contraction_contributions, cmap=None, norm=None, aspect=None, interpolation=None, alpha=None, vmin=np.min(all_contraction_contributions), vmax=np.max(all_contraction_contributions), origin=None, extent=None, filternorm=1, filterrad=4.0, resample=None, url=None)
        #plt.colorbar(ax = plt.gca())
        animation_camera.snap()
    plt.colorbar()
    plt.title("Contractions or Relaxations Contributions")
    plt.xlabel("Number of Pixels")
    plt.ylabel("Number of Pixels")
    animation = animation_camera.animate()
    animation.save(filename)
def Contraction_contributions_fixed_colorbar_range50_movie_pixel(all_contraction_contributions,filename = "Contraction_contributions_fixed_colorbar range(-50,50)_pixel.mp4"):    
    plt.figure()
    plt.figure()
    animation_camera = celluloid.Camera(plt.gcf())
    for index in range(all_contraction_contributions.shape[0]):
        this_contraction_contributions_frame = all_contraction_contributions[index,:,:]
        img_contraction_contributions = this_contraction_contributions_frame 
        plt.imshow(img_contraction_contributions, cmap=None, norm=None, aspect=None, interpolation=None, alpha=None, vmin=-50, vmax=50, origin=None, extent=None, filternorm=1, filterrad=4.0, resample=None, url=None)
        #plt.colorbar(ax = plt.gca())
        animation_camera.snap()
    plt.colorbar()
    plt.title("Contractions or Relaxations Contributions")
    plt.xlabel("Number of Pixels")
    plt.ylabel("Number of Pixels")
    animation = animation_camera.animate()
    animation.save(filename)
 



def Contraction_contributions_fixed_colorbar_range1_movie_pixel(all_contraction_contributions,filename = "Contraction_contributions_fixed_colorbar range(-1,1)_pixel.mp4"):
    plt.figure()
    animation_camera = celluloid.Camera(plt.gcf())
    for index in range(all_contraction_contributions.shape[0]):
        this_contraction_contributions_frame = all_contraction_contributions[index,:,:]
        img_contraction_contributions = this_contraction_contributions_frame 
        plt.imshow(img_contraction_contributions, cmap=None, norm=None, aspect=None, interpolation=None, alpha=None, vmin=-1, vmax=1, origin=None, extent=None, filternorm=1, filterrad=4.0, resample=None, url=None)
        #plt.colorbar(ax = plt.gca())
        animation_camera.snap()
    plt.colorbar()
    plt.title("Contractions or Relaxations Contributions")
    plt.xlabel("Number of Pixels")
    plt.ylabel("Number of Pixels")
    animation = animation_camera.animate()     
    animation.save(filename)


      
  
  
  
  
  