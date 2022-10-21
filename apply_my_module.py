#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 12:41:03 2022

@author: apple
"""
import my_module
import skimage
import numpy as np
import matplotlib.pyplot as plt


#filename = 'MB301110_i_4_movie_8 bit.tif'
filename = 'MB301110_i_4_movie_8 bit.tif'

my_images = skimage.io.imread(filename)

my_dictionary = my_module.conduct_optical_flow(smoothing_sigma = 1, box_size =100,all_images= my_images)
blurred_images = my_dictionary['all_blurred_images']
all_v_x = my_dictionary['all_v_x']
all_v_y = my_dictionary['all_v_y']
all_gamma = my_dictionary['all_gamma'] 
all_div_v = my_dictionary['all_div_v']
all_intensity_box = my_dictionary['all_intensity_box']
all_contraction = my_dictionary['all_contraction']
all_difference_to_previous_frame_box = my_dictionary['all_difference_to_previous_frame_box']
all_sumcheck_box =  my_dictionary['all_sumcheck_box']
all_relative_error_box =  my_dictionary['all_relative_error_box']
all_flow_gamma =  my_dictionary['all_flow_gamma']
all_contraction_contributions = my_dictionary['all_contraction_contributions']
all_gamma_contributions = my_dictionary['all_gamma_contributions']
all_flow_contributions = my_dictionary['all_flow_contributions']
all_dIdx_box = my_dictionary['all_dIdx_box']
all_dIdy_box = my_dictionary['all_dIdy_box']
print(all_relative_error_box)
median_abs_all_relative_error_box = np.median(np.abs(all_relative_error_box))
print(median_abs_all_relative_error_box)
#Actin
# my_module.Visualizing_Velocity(all_images = my_images, blurred_images = my_dictionary['all_blurred_images'], all_v_x= my_dictionary['all_v_x'], all_v_y= my_dictionary['all_v_y'], box_size= 100,filename = 'Visualizing Velocity_boxsize100.mp4')
# my_module.gamma_fixed_colorbar_movie(all_gamma = my_dictionary['all_gamma'],filename = "actin_gamma_fixed_colorbar_movie_boxsize88.mp4")#must define filename
# my_module.Contractions_Relaxations_Histogram_plot(all_contraction,filename ='Contractions or Relaxations Histogram_boxsize88')
# my_module.Contractions_or_Relaxations_fixed_colorbar_range1_movie(all_contraction,filename = "Contractions or Relaxations_fixed_colorbar_range(-1,1)_boxsize88.mp4")
# my_module.all_gamma_contributions_changing_colorbar_movie(all_gamma_contributions,blurred_images,filename = "Animate_all_gamma_contributions_changing_colorbar_boxsize88.mp4")
# my_module.all_gamma_contributions_fixed_colorbar_range1_movie(all_gamma_contributions,blurred_images, filename = "Gamma_contributions_fixed_colorbar_range(-1,1)_boxsize88.mp4")
# my_module.Gamma_contributions_Histogram_plot(all_gamma_contributions,filename ='Gamma Contribution Histogram_boxsize88')
# my_module.all_flow_contributions_changing_colorbar_movie(all_flow_contributions,blurred_images,filename = "all_flow_contributions_changing_colorbar_boxsize88.mp4")
# my_module.Flow_Contribution_Histogram_plot(all_flow_contributions,filename ='Flow Contribution Histogram_boxsize88')
# my_module.all_flow_contributions_fixed_colorbar_range1_movie(all_flow_contributions,filename = "Flow_contributions_fixed_colorbar range(-1,1)_boxsize88.mp4")
# my_module.Absolute_Values_of_Flow_over_Gamma_Histogram_plot(all_flow_gamma,filename ='Absolute Values of Flow over Gamma Histogram_boxsize88')
# my_module.All_flow_over_gamma_fixed_colorbar_range_0_36_movie(all_flow_gamma,blurred_images,filename = "All_flow over gamma_fixed_colorbar range(0.3.6)_boxsize88.mp4")
# my_module.Contraction_or_Relaxations_Contributions_Histogram_plot(all_contraction_contributions,filename ='Contraction or Relaxations Contributions Histogram_boxsize88')
# my_module.Contraction_contributions_fixed_colorbar_range1_movie(all_contraction_contributions,filename = "Contraction_contributions_fixed_colorbar range(-1,1)_boxsize88.mp4")

#Dumpy
# my_module.Visualizing_Velocity(all_images = my_images, blurred_images = my_dictionary['all_blurred_images'], all_v_x= my_dictionary['all_v_x'], all_v_y= my_dictionary['all_v_y'], box_size= 100,filename = 'Visualizing Velocity_boxsize100.mp4')
# my_module.gamma_fixed_colorbar_movie(all_gamma = my_dictionary['all_gamma'],filename = "Dumpy_gamma_fixed_colorbar_movie_boxsize16.mp4")#must define filename
# my_module.Contractions_Relaxations_Histogram_plot(all_contraction,filename ='Dumpy Contractions or Relaxations Histogram_boxsize16')
# my_module.Contractions_or_Relaxations_fixed_colorbar_range1_movie(all_contraction,filename = "Dumpy Contractions or Relaxations_fixed_colorbar_range(-1,1)_boxsize16.mp4")
# my_module.all_gamma_contributions_changing_colorbar_movie(all_gamma_contributions,blurred_images,filename = "Dumpy_Animate_all_gamma_contributions_changing_colorbar_boxsize16.mp4")
# my_module.all_gamma_contributions_fixed_colorbar_range1_movie(all_gamma_contributions,blurred_images, filename = "Dumpy_Gamma_contributions_fixed_colorbar_range(-1,1)_boxsize16.mp4")
# my_module.Gamma_contributions_Histogram_plot(all_gamma_contributions,filename ='Dumpy Gamma Contribution Histogram_boxsize16')
# my_module.all_flow_contributions_changing_colorbar_movie(all_flow_contributions,blurred_images,filename = "Dumpy_all_flow_contributions_changing_colorbar_boxsize16.mp4")
# my_module.Flow_Contribution_Histogram_plot(all_flow_contributions,filename ='Dumpy Flow Contribution Histogram_boxsize16')
# my_module.all_flow_contributions_fixed_colorbar_range1_movie(all_flow_contributions,filename = "Dumpy_Flow_contributions_fixed_colorbar range(-1,1)_boxsize16.mp4")
# my_module.Absolute_Values_of_Flow_over_Gamma_Histogram_plot(all_flow_gamma,filename ='Dumpy Absolute Values of Flow over Gamma Histogram_boxsize16')
# my_module.All_flow_over_gamma_fixed_colorbar_range_0_36_movie(all_flow_gamma,blurred_images,filename = "Dumpy_All_flow over gamma_fixed_colorbar range(0.3.6)_boxsize16.mp4")
# my_module.Contraction_or_Relaxations_Contributions_Histogram_plot(all_contraction_contributions,filename ='Dumpy Contraction or Relaxations Contributions Histogram_boxsize16')
# my_module.Contraction_contributions_fixed_colorbar_range1_movie(all_contraction_contributions,filename = "Dumpy_Contraction_contributions_fixed_colorbar range(-1,1)_boxsize16.mp4")




#plot line chart
boxsize=[10,15,20,25,30,40,50,55,60,66,70,80,85,87,88,89,90,91,92,95,100,150,200,250,500]
absmedian_allrelative_error = [1.715569573216614e-16,1.581322923591227e-16,1.5316630507487863e-16,1.5133556783046825e-16,1.4936661760989934e-16,1.4756629903874092e-16,1.46047680588553e-16,1.4604758018981124e-16,1.4504150603926482e-16,1.43971979225618e-16,1.4620508537932895e-16,1.446501599808948e-16,1.4352222600339287e-16,1.440906548416389e-16,1.422094689964616e-16,1.4448632862363849e-16,1.4372213610708617e-16,1.443808804464171e-16,1.4502689876033726e-16,1.4774273189717085e-16,1.450049989127822e-16,1.4169498923478925e-16,1.4328156673493794e-16,1.4212662065884832e-16,1.4951458449118565e-16]
plt.plot(boxsize,absmedian_allrelative_error)
plt.title("Abs median all relative error with different boxsize")
plt.xlable("Boxsize")
plt.ylable("Abs median all relative error(box)")
plt.show()


#Actin
#Box
for boxsize in range(100, 201, 50):#range(start, stop, step)#boxsize can not be 1
    my_dictionary = my_module.conduct_optical_flow(smoothing_sigma = 1,box_size=boxsize, all_images= my_images)

    all_gamma = my_dictionary['all_gamma']
    all_contraction = my_dictionary['all_contraction']
    all_sumcheck_box =  my_dictionary['all_sumcheck_box']
    all_relative_error_box =  my_dictionary['all_relative_error_box']
    all_flow_gamma =  my_dictionary['all_flow_gamma']
    all_contraction_contributions = my_dictionary['all_contraction_contributions']
    all_gamma_contributions = my_dictionary['all_gamma_contributions']
    all_flow_contributions = my_dictionary['all_flow_contributions']
    
    my_module.gamma_fixed_colorbar_movie(all_gamma,filename = "actin_gamma_fixed_colorbar_movie" + str(boxsize) + ".mp4")
    my_module.Contractions_Relaxations_Histogram_plot(all_contraction = my_dictionary['all_contraction'],filename = "Actin Contractions or Relaxations Histogram"+ str(boxsize) + ".jpg")
    my_module.Visualizing_Velocity(all_images = my_images, blurred_images = my_dictionary['all_blurred_images'], all_v_x= my_dictionary['all_v_x'], all_v_y= my_dictionary['all_v_y'], box_size= boxsize,filename = 'Visualizing Velocity movie'+ str(boxsize) + ".mp4")
    my_module.Contractions_Relaxations_Histogram_plot(all_contraction,filename ='Contractions or Relaxations Histogram moive'+ str(boxsize) + ".mp4")
    my_module.Contractions_or_Relaxations_fixed_colorbar_range1_movie(all_contraction,filename = "Contractions or Relaxations_fixed_colorbar_range(-1,1)_moive" + str(boxsize) + ".mp4")
    my_module.all_gamma_contributions_changing_colorbar_movie(all_gamma_contributions,blurred_images,filename = "Animate_all_gamma_contributions_changing_colorbar_moive"+ str(boxsize) + ".mp4")
    my_module.all_gamma_contributions_fixed_colorbar_range1_movie(all_gamma_contributions,blurred_images, filename = "Gamma_contributions_fixed_colorbar_range(-1,1)_moive"+ str(boxsize) + ".mp4")
    my_module.Gamma_contributions_Histogram_plot(all_gamma_contributions,filename ='Gamma Contribution Histogram'+ str(boxsize) + ".jpg")
    my_module.all_flow_contributions_changing_colorbar_movie(all_flow_contributions,blurred_images,filename = "all_flow_contributions_changing_colorbar__moive"+ str(boxsize) + ".mp4")
    my_module.Flow_Contribution_Histogram_plot(all_flow_contributions,filename ='Flow Contribution Histogram'+ str(boxsize) + ".jpg")
    my_module.all_flow_contributions_fixed_colorbar_range1_movie(all_flow_contributions,filename = "Flow_contributions_fixed_colorbar range(-1,1)__moive"+ str(boxsize) + ".mp4")
    my_module.Absolute_Values_of_Flow_over_Gamma_Histogram_plot(all_flow_gamma,filename ='Absolute Values of Flow over Gamma Histogram'+ str(boxsize) + ".jpg")
    my_module.All_flow_over_gamma_fixed_colorbar_range_0_36_movie(all_flow_gamma,blurred_images,filename = "All_flow over gamma_fixed_colorbar range(0.3.6)__moive"+ str(boxsize) + ".mp4")
    my_module.Contraction_or_Relaxations_Contributions_Histogram_plot(all_contraction_contributions,filename ='Contraction or Relaxations Contributions Histogram'+ str(boxsize) + ".jpg")
    my_module.Contraction_contributions_fixed_colorbar_range1_movie(all_contraction_contributions,filename = "Contraction_contributions_fixed_colorbar range(-1,1)__moive"+ str(boxsize) + ".mp4")

#Pixel
import my_allpixels_module
import skimage
import numpy as np
import matplotlib.pyplot as plt
filename = 'MB301110_i_4_movie_8 bit.tif'

my_images = skimage.io.imread(filename)

blurred_images = np.zeros_like(my_images, dtype ='double')
for index in range(my_images.shape[0]):
    this_image = my_images[index,:,:]
    this_filtered_image = skimage.filters.gaussian(this_image, sigma =1)
    blurred_images[index,:,:] = this_filtered_image


fewer_images = blurred_images

# for boxsize in range(10, 21, 10):#range(start, stop, step)#boxsize can not be 1
#     my_dictionary = my_allpixels_module.conduct_optical_flow(smoothing_sigma = 1,box_size=boxsize, all_images= my_images)


my_dictionary = my_allpixels_module.conduct_optical_flow(smoothing_sigma = 1, box_size =10,all_images = fewer_images)
my_allpixels_module.gamma_pixel_fixed_colorbar_movie(all_gamma = my_dictionary['all_pixel_gamma'],filename = "actin_gamma_pixel_fixed_colorbar_movieboxsize10.mp4")



for boxsize in range(10, 21, 10):#range(start, stop, step)#boxsize can not be 1
    my_dictionary = my_allpixels_module.conduct_optical_flow(smoothing_sigma = 1,box_size=boxsize, all_images= my_images)

    my_allpixels_module.gamma_pixel_fixed_colorbar_movie(all_gamma = my_dictionary['all_pixel_gamma'],filename = "actin_gamma_pixel_fixed_colorbar_movieboxsize10.mp4")


    blurred_images = my_dictionary['all_blurred_images']
    all_pixel_v_x = my_dictionary['all_pixel_v_x']
    all_pixel_v_y = my_dictionary['all_pixel_v_y']
    all_pixel_gamma = my_dictionary['all_pixel_gamma'] 
    all_pixel_div_v = my_dictionary['all_pixel_div_v']
    all_pixel_intensity = my_dictionary['all_pixel_intensity']
    all_pixel_contraction = my_dictionary['all_pixel_contraction']
    all_pixel_deltaI = my_dictionary['all_pixel_deltaI']
    all_pixel_sumcheck =  my_dictionary['all_pixel_sumcheck']
    all_pixel_relative_error =  my_dictionary['all_pixel_relative_error']
    all_pixel_flow_gamma =  my_dictionary['all_pixel_flow_gamma']
    all_pixel_contraction_contributions = my_dictionary['all_pixel_contraction_contributions']
    all_pixel_gamma_contributions = my_dictionary['all_pixel_gamma_contributions']
    all_pixel_flow_contributions = my_dictionary['all_pixel_flow_contributions']
    all_pixel_dIdx_box = my_dictionary['all_pixel_dIdx_box']
    all_pixel_dIdy_box = my_dictionary['all_pixel_dIdy_box']
    print(all_pixel_relative_error)
    median_abs_all_pixel_relative_error = np.median(np.abs(all_pixel_relative_error))
    print(median_abs_all_pixel_relative_error)
    
    my_allpixels_module.gamma_pixel_fixed_colorbar_movie(all_gamma = my_dictionary['all_pixel_gamma'],filename = "actin_gamma_pixel_fixed_colorbar_movie" + str(boxsize) + ".mp4")
    my_allpixels_module.Contractions_Relaxations_Histogram_plot_pixel(all_contraction = my_dictionary['all_pixel_contraction'],filename = "Actin Contractions or Relaxations Histogram_pixel"+ str(boxsize) + ".jpg")
    my_allpixels_module.Visualizing_Velocity_pixel(all_images = my_images, blurred_images = my_dictionary['all_blurred_images'], all_v_x= my_dictionary['all_pixel_v_x'], all_v_y= my_dictionary['all_pixel_v_y'], box_size= boxsize,filename = 'Visualizing Velocity movie_pixel'+ str(boxsize) + ".mp4")
    my_allpixels_module.Contractions_or_Relaxations_fixed_colorbar_range1_movie_pixel(all_contraction = my_dictionary['all_pixel_contraction'],filename = "Contractions or Relaxations_fixed_colorbar_range(-1,1)_moive_pixel" + str(boxsize) + ".mp4")
    my_allpixels_module.all_gamma_contributions_changing_colorbar_movie_pixel(all_gamma_contributions = my_dictionary['all_pixel_gamma_contributions'],blurred_images = my_dictionary['all_blurred_images'],filename = "Animate_all_gamma_contributions_changing_colorbar_moive_pixel"+ str(boxsize) + ".mp4")   
    my_allpixels_module.all_gamma_contributions_fixed_colorbar_range1_movie_pixel(all_gamma_contributions = my_dictionary['all_pixel_gamma_contributions'],blurred_images = my_dictionary['all_blurred_images'], filename = "Gamma_contributions_fixed_colorbar_range(-1,1)_moive_pixel"+ str(boxsize) + ".mp4")    
    my_allpixels_module.Gamma_contributions_Histogram_plot_pixel(all_gamma_contributions = my_dictionary['all_pixel_gamma_contributions'],filename ='Gamma Contribution Histogram_pixel'+ str(boxsize) + ".jpg")    
    my_allpixels_module.all_flow_contributions_changing_colorbar_movie_pixel(all_flow_contributions = my_dictionary['all_pixel_flow_contributions'],blurred_images = my_dictionary['all_blurred_images'],filename = "all_flow_contributions_changing_colorbar_moive_pixel"+ str(boxsize) + ".mp4")   
    my_allpixels_module.Flow_Contribution_Histogram_plot_pixel(all_flow_contributions = my_dictionary['all_pixel_flow_contributions'],filename ='Flow Contribution Histogram_pixel'+ str(boxsize) + ".jpg")   
    my_allpixels_module.all_flow_contributions_fixed_colorbar_range1_movie_pixel(all_flow_contributions= my_dictionary['all_pixel_flow_contributions'],filename = "Flow_contributions_fixed_colorbar range(-1,1)_moive_pixel"+ str(boxsize) + ".mp4")
    my_allpixels_module.Absolute_Values_of_Flow_over_Gamma_Histogram_plot_pixel(all_flow_gamma = my_dictionary['all_pixel_flow_gamma'],filename ='Absolute Values of Flow over Gamma Histogram_pixel'+ str(boxsize) + ".jpg")
    my_allpixels_module.All_flow_over_gamma_fixed_colorbar_range_0_max_movie_pixel(all_flow_gamma= my_dictionary['all_pixel_flow_gamma'],blurred_images= my_dictionary['all_blurred_images'],filename = "All_flow over gamma_fixed_colorbar range(max)_moive_pixel"+ str(boxsize) + ".mp4")    
    my_allpixels_module.Contraction_or_Relaxations_Contributions_Histogram_plot_pixel(all_contraction_contributions = my_dictionary['all_pixel_contraction_contributions'],filename ='Contraction or Relaxations Contributions Histogram_pixel'+ str(boxsize) + ".jpg")
    my_allpixels_module.Contraction_contributions_fixed_colorbar_range1_movie_pixel(all_contraction_contributions= my_dictionary['all_pixel_contraction_contributions'],filename = "Contraction_contributions_fixed_colorbar range(-1,1)_moive_pixel"+ str(boxsize) + ".mp4")



boxsize=[10,15,20,25,30,40,50,55,60,66,70,80,85,87,88,89,90,91,92,95,100,150,200,250,500]
absmedian_allrelative_error = []
plt.plot(boxsize,absmedian_allrelative_error)
plt.title("Abs median all relative error with different boxsize")
plt.xlable("Boxsize")
plt.ylable("Abs median all relative error(pixel)")
plt.show()



    
