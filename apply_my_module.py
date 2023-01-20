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
plt.xlabel("Boxsize")
plt.ylabel("Abs median all relative error(box)")
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
#notmal run(without @jit)
import my_allpixels_module
import skimage
import numpy as np
import matplotlib.pyplot as plt
filename = 'MB301110_i_4_movie_8 bit.tif'

my_images = skimage.io.imread(filename)
for boxsize in range(10, 21, 10):#range(start, stop, step)#boxsize can not be 1
    my_dictionary = my_allpixels_module.conduct_optical_flow(smoothing_sigma = 1,box_size=boxsize, all_images= my_images)

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
    all_pixel_flow = my_dictionary['all_pixel_flow']
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

#!!USED Single boxsize(dictionary run 17mins)
my_dictionary = my_allpixels_module.conduct_optical_flow(smoothing_sigma = 1, box_size =20,all_images= my_images)
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
all_pixel_dIdx= my_dictionary['all_pixel_dIdx']
all_pixel_dIdy = my_dictionary['all_pixel_dIdy']
all_v_x = my_dictionary['all_v_x']
all_v_y = my_dictionary['all_v_y']
median_abs_all_pixel_relative_error = np.median(np.abs(all_pixel_relative_error))
print(median_abs_all_pixel_relative_error)
median_all_pixel_gamma_firstframe= np.median(all_pixel_gamma[0,:,:])
print(median_all_pixel_gamma_firstframe)#first frame remodeling
median_all_pixel_gamma= np.median(all_pixel_gamma)# median all gamma all frames
print(median_all_pixel_gamma)
print(all_pixel_gamma[0,500,500])

my_allpixels_module.gamma_pixel_fixed_colorbar_movie(all_gamma = all_pixel_gamma,filename = "actin_gamma_pixel_fixed_colorbar_movie_boxsize100.mp4")
my_allpixels_module.Contractions_Relaxations_Histogram_plot_pixel(all_contraction = all_pixel_contraction,filename = "Actin Contractions or Relaxations Histogram_pixel_boxsize100.jpg")
my_allpixels_module.Visualizing_Velocity_pixel(all_images = my_images, blurred_images = blurred_images, all_pixel_v_x= all_pixel_v_x, all_pixel_v_y= all_pixel_v_y, box_size =100,arrow_box_size=20, filename = 'Visualizing Velocity movie_pixel_boxsize100.mp4')
my_allpixels_module.Visualizing_Velocity_box(all_images= my_images,blurred_images= blurred_images,all_v_x =all_v_x,all_v_y=all_v_y,box_size=10,filename = 'Visualizing Velocity movie_box_boxsize10.mp4')
my_allpixels_module.Contractions_or_Relaxations_fixed_colorbar_range1_movie_pixel(all_contraction =all_pixel_contraction,filename = "Contractions or Relaxations_fixed_colorbar_range(-1,1)_moive_pixel_boxsize100.mp4")
#my_allpixels_module.all_gamma_contributions_changing_colorbar_movie_pixel(all_gamma_contributions = all_pixel_gamma_contributions,blurred_images = blurred_images,filename = "Animate_all_gamma_contributions_changing_colorbar_moive_pixel_boxsize10.mp4")   
my_allpixels_module.all_gamma_contributions_fixed_colorbar_range1_movie_pixel(all_gamma_contributions = all_pixel_gamma_contributions,blurred_images = blurred_images, filename = "Gamma_contributions_fixed_colorbar_range(-1,1)_moive_pixel_boxsize100.mp4")    
my_allpixels_module.Gamma_contributions_Histogram_plot_pixel(all_gamma_contributions = all_pixel_gamma_contributions,filename ='Gamma Contribution Histogram_pixel_boxsize100.jpg')    
#my_allpixels_module.all_flow_contributions_changing_colorbar_movie_pixel(all_flow_contributions = all_pixel_flow_contributions,blurred_images = blurred_images,filename = "all_flow_contributions_changing_colorbar_moive_pixel_boxsize10.mp4")   
my_allpixels_module.Flow_Contribution_Histogram_plot_pixel(all_flow_contributions = all_pixel_flow_contributions,filename ="Flow Contribution Histogram_pixel_boxsize15.jpg")   
my_allpixels_module.all_flow_contributions_fixed_colorbar_range1_movie_pixel(all_flow_contributions= all_pixel_flow_contributions,filename = "Flow_contributions_fixed_colorbar range(-1,1)_moive_pixel_boxsize100.mp4")
my_allpixels_module.Absolute_Values_of_Flow_over_Gamma_Histogram_plot_pixel(all_flow_gamma = all_pixel_flow_gamma,filename ="Absolute Values of Flow over Gamma Histogram_pixel_boxsize100.jpg")
my_allpixels_module.All_flow_over_gamma_fixed_colorbar_range_0_max_movie_pixel(all_flow_gamma= all_pixel_flow_gamma,blurred_images= blurred_images,filename = "All_flow over gamma_fixed_colorbar range(0_5)_moive_pixel_boxsize20.mp4")    
my_allpixels_module.Contraction_or_Relaxations_Contributions_Histogram_plot_pixel(all_contraction_contributions = all_pixel_contraction_contributions,filename ="Contraction or Relaxations Contributions Histogram_pixel_boxsize100.jpg")
my_allpixels_module.Contraction_contributions_fixed_colorbar_range1_movie_pixel(all_contraction_contributions= all_pixel_contraction_contributions,filename = "Contraction_contributions_fixed_colorbar range(-1,1)_moive_pixel_boxsize100.mp4")
my_allpixels_module.All_flow_fixed_colorbar_movie_maskpixel(all_pixel_flow=all_pixel_flow,blurred_images=blurred_images,filename = "All_flow_fixed_colorbar_range(max)_maskpixel_boxsize30.mp4")


#median all_pixel_relative_error with different boxsize
plt.figure()
boxsize=[10,15,20,25,30,40,50,60,70,80,90,100,150,200,250,500]
absmedian_allrelative_error = [0.816874055378088,0.8695019747644812,0.8943396651387155,0.9071583488076096,0.9174538766738144,0.9288817334939943,0.9381045760944063,0.9444428974746077,0.9493841796681985,0.9533541465525011,0.9597360093199316,0.9645959394700545,0.9817882636459605,0.9985472241557778,1.0187433864538082,1.0894883857041617]
plt.plot(boxsize,absmedian_allrelative_error)
plt.title("Abs median all relative error with different boxsize")
plt.xlabel("Boxsize")
plt.ylabel("Abs median all relative error(pixel)")
plt.savefig("Abs median all relative error with different boxsize_pixels.jpg")
plt.show()


#first frame gamma with different boxsize
plt.figure()
boxsize=[10,15,20,25,30,40,50,60,70,80,90,100,150,200,250,500]
median_all_pixel_gamma_firstframe = [0.0004640886692643506,0.00045377731864330823,0.0004357495257273809,0.0004489385186648697,0.0004621412995506792,0.00043748957267257364,0.00041044894080283857,0.00043940650512951795,0.0005374083808245047,0.0005534834701024576,0.0005489839480350712,0.0005058654118448723,0.001404793781745974,0.0011342262527625554,0.0014214163590498033,0.0018783756211996831]
plt.plot(boxsize,median_all_pixel_gamma_firstframe)
plt.title("Median remodeling on first frame with different boxsize")
plt.xlabel("Boxsize")
plt.ylabel("Median first frame remodeling(pixel)")
plt.savefig("Median remodeling on first frame with different boxsize_pixels.jpg")
plt.show()

#median all_pixel_gamma on all frames with different boxsize
plt.figure()
boxsize=[10,15,20,25,30,40,50,60,70,80,90,100,150,200,250,500]
median_all_pixel_gamma = [2.3186833070997685e-05,2.2753966921688384e-05,2.0327795241561977e-05,2.4278519772241944e-05,2.3763134626807182e-05,2.8279933753532602e-05,3.2619453302914865e-05,2.904190825397476e-05,4.298611580377751e-05,4.778246647621185e-05,4.709322199662367e-05,4.7699712012983314e-05,7.373333065362917e-05,6.242593259941856e-05,0.0001585193645392378,0.00026549543188595715]
plt.plot(boxsize,median_all_pixel_gamma)
plt.title("Median remodeling on all frames with different boxsize")
plt.xlabel("Boxsize")
plt.ylabel("Median all frames remodeling(pixel)")
plt.savefig("Median remodeling on all frames with different boxsize_pixels.jpg")
plt.show()

#all_pixel_gamma[0,500,500] with different boxsize
plt.figure()
boxsize=[10,15,20,25,30,40,50,60,70,80,90,100,150,200,250,500]
median_specific_pixel_gamma = [-0.013394757597333093,-0.01598090486287545,-0.0044985378558072674,-0.0033835567718252186,-0.008401597657064273,-0.006084176297943904,0.0034526974447265848,-0.0035112515822863077,0.0010041664759156685,-0.0002242826803489487,-0.004057290633899069,0.003964018986277898,-0.0020753671850577884,-0.0022196685390972913,0.01368389296730022,0.005776401406940393]
plt.plot(boxsize,median_specific_pixel_gamma)
plt.title("Median remodeling of[500,500] on first frame with different boxsize")
plt.xlabel("Boxsize")
plt.ylabel("Median remodeling of[500,500] on first frame")
plt.savefig("Median remodeling of[500,500] on first frame with different boxsize_pixels.jpg")
plt.show()
#median all_pixel_relative_error with 10-50boxsize/10-100boxsize
# plt.figure()
# boxsize=[10,15,20,25,30,40,50]
# absmedian_allrelative_error = [0.816874055378088,0.8695019747644812,0.8943396651387155,0.9071583488076096,0.9174538766738144,0.9288817334939943,0.9381045760944063]
# plt.plot(boxsize,absmedian_allrelative_error)
# plt.title("Abs median all relative error with 10-50 different boxsize")
# plt.xlabel("Boxsize")
# plt.ylabel("Abs median all relative error(pixel)")
# plt.savefig("Abs median all relative error with 10-50 different boxsize_pixels.jpg")
# plt.show()

#first frame gamma with 10-100boxsize
# plt.figure()
# boxsize=[10,15,20,25,30,40,50,60,70,80,90,100]
# median_all_pixel_gamma_firstframe = [0.0004640886692643506,0.00045377731864330823,0.0004357495257273809,0.0004489385186648697,0.0004621412995506792,0.00043748957267257364,0.00041044894080283857,0.00043940650512951795,0.0005374083808245047,0.0005534834701024576,0.0005489839480350712,0.0005058654118448723]
# plt.plot(boxsize,median_all_pixel_gamma_firstframe)
# plt.title("Median remodeling on first frame with different 10-100 boxsize")
# plt.xlabel("Boxsize")
# plt.ylabel("Median first frame remodeling(pixel)")
# plt.savefig("Median remodeling on first frame with different 10-100 boxsize_pixels.jpg")
# plt.show()

#allpixels with numba @jit
import my_jit_allpixels_module
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
new_images = blurred_images
#my_dictionary =
my_jit_allpixels_module.conduct_optical_flow(box_size =5,all_images = new_images)
my_jit_allpixels_module.fixjit_optical_flow(smoothing_sigma = 1, box_size = 5, all_images = skimage.io.imread('MB301110_i_4_movie_8 bit.tif'))


my_jit_allpixels_module.gamma_pixel_fixed_colorbar_movie(all_gamma = my_dictionary['all_pixel_gamma'],filename = "actin_gamma_pixel_fixed_colorbar_movieboxsize5.mp4")












#Mask(single boxsize run done)
import my_mask_module
import skimage
import numpy as np
import matplotlib.pyplot as plt
filename = 'Actin_cutcells_data.tif'

my_images = skimage.io.imread(filename)
# =============================================================================
# for boxsize in range(10, 21, 10):#range(start, stop, step)#boxsize can not be 1
#     my_dictionary = my_mask_module.conduct_optical_flow(smoothing_sigma = 1,box_size=boxsize, all_images= my_images)   
# =============================================================================
    
#Single boxsize 
my_dictionary = my_mask_module.conduct_optical_flow(smoothing_sigma = 1, box_size=21, all_images= my_images)
#my_dictionary = my_mask_module.conduct_optical_flow(smoothing_sigma = 1,box_size=10, all_images= my_images[1:6,500:604,500:604])
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
all_pixel_dIdx = my_dictionary['all_pixel_dIdx']
all_pixel_dIdy = my_dictionary['all_pixel_dIdy']
all_v_x = my_dictionary['all_v_x']
all_v_y = my_dictionary['all_v_y'] 
all_pixel_flow =  my_dictionary['all_pixel_flow'] 
all_pixel_abs_gamma_contributions = my_dictionary['all_pixel_abs_gamma_contributions']
all_pixel_abs_flow_contributions =my_dictionary['all_pixel_abs_flow_contributions']
all_pixel_flow_abs_contributions=my_dictionary['all_pixel_flow_abs_contributions']
all_pixel_gamma_abs_contributions=my_dictionary['all_pixel_gamma_abs_contributions']
all_pixel_contraction_abs_contributions=my_dictionary['all_pixel_contraction_abs_contributions']
    

median_abs_all_pixel_relative_error = np.median(np.abs(all_pixel_relative_error))
print(median_abs_all_pixel_relative_error)#boxsize21=9045523476702.229
median_all_pixel_gamma_firstframe= np.median(all_pixel_gamma[0,:,:])
print(median_all_pixel_gamma_firstframe)#first frame remodeling#boxsize21=-1.0904107516023856e+16
median_all_pixel_gamma= np.median(all_pixel_gamma)# median all gamma all frames
print(median_all_pixel_gamma)#boxsize21=-1.1615917558706948e+16
print(all_pixel_gamma[0,500,500])#boxsize21=-1.94706334838492e+16

my_mask_module.gamma_maskpixel_fixed_colorbar_movie(all_gamma = all_pixel_gamma,filename = "actin_gamma_maskpixel_fixed_colorbar_range(-0.05,0.05)_movie_boxsize21.mp4")
#my_mask_module.gamma_maskpixel_changing_colorbar_movie(all_gamma = all_pixel_gamma,blurred_images =blurred_images, filename = "Gamma_maskpixel_with_changing_colorbar_boxsize10.mp4")
my_mask_module.Contractions_Relaxations_Histogram_plot_maskpixel(all_contraction = all_pixel_contraction,filename = "Actin Contractions or Relaxations Histogram_maskpixel_boxsize21.jpg")
my_mask_module.Visualizing_Velocity_maskpixel(all_images = my_images, blurred_images = blurred_images, all_pixel_v_x= all_pixel_v_x, all_pixel_v_y= all_pixel_v_y, box_size =21, arrow_box_size=20,filename = 'Visualizing Velocity movie_maskpixel_arrowboxsize20_boxsize21.mp4')
#my_mask_module.Visualizing_Velocity_box(all_images= my_images,blurred_images= blurred_images,all_v_x =all_v_x,all_v_y=all_v_y,box_size=10,filename = 'Visualizing Velocity movie_maskpixel_box_boxsize10.mp4')
my_mask_module.Contractions_or_Relaxations_fixed_colorbar_range1_movie_maskpixel(all_contraction =all_pixel_contraction,filename = "Contractions or Relaxations_fixed_colorbar_range(-0.1,0.1)_moive_maskpixel_boxsize21.mp4")
#my_mask_module.all_gamma_contributions_changing_colorbar_movie_pixel(all_gamma_contributions = all_pixel_gamma_contributions,blurred_images = blurred_images,filename = "Animate_all_gamma_contributions_changing_colorbar_moive_pixel_boxsize10.mp4")   
my_mask_module.all_gamma_contributions_fixed_colorbar_range1_movie_maskpixel(all_gamma_contributions = all_pixel_gamma_contributions,blurred_images = blurred_images, filename = "Gamma_contributions_fixed_colorbar_range(-1,1)_moive_maskpixel_boxsize21.mp4")    
my_mask_module.Gamma_contributions_Histogram_plot_maskpixel(all_gamma_contributions = all_pixel_gamma_contributions,filename ='Gamma Contribution Histogram_maskpixel_boxsize21.jpg')    
#my_mask_module.all_flow_contributions_changing_colorbar_movie_pixel(all_flow_contributions = all_pixel_flow_contributions,blurred_images = blurred_images,filename = "all_flow_contributions_changing_colorbar_moive_pixel_boxsize10.mp4")   
my_mask_module.Flow_Contribution_Histogram_plot_maskpixel(all_flow_contributions = all_pixel_flow_contributions,filename ="Flow Contribution Histogram_maskpixel_boxsize21.jpg")   
my_mask_module.all_flow_contributions_fixed_colorbar_range1_movie_maskpixel(all_flow_contributions= all_pixel_flow_contributions,filename = "Flow_contributions_fixed_colorbar range(-1,1)_moive_maskpixel_boxsize21.mp4")
my_mask_module.Absolute_Values_of_Flow_over_Gamma_Histogram_plot_maskpixel(all_flow_gamma = all_pixel_flow_gamma,filename ="Absolute Values of Flow over Gamma Histogram_maskpixel_boxsize21.jpg")
my_mask_module.All_flow_over_gamma_fixed_colorbar_range_0_max_movie_maskpixel(all_flow_gamma= all_pixel_flow_gamma,blurred_images= blurred_images,filename = "All_flow over gamma_fixed_colorbar range(0_5)_moive_maskpixel_boxsize21.mp4")    
my_mask_module.Contraction_or_Relaxations_Contributions_Histogram_plot_maskpixel(all_contraction_contributions = all_pixel_contraction_contributions,filename ="Contraction or Relaxations Contributions Histogram_maskpixel_boxsize21.jpg")
my_mask_module.Contraction_contributions_fixed_colorbar_range1_movie_maskpixel(all_contraction_contributions= all_pixel_contraction_contributions,filename = "Contraction_contributions_fixed_colorbar range(-1,1)_moive_maskpixel_boxsize21.mp4")
my_mask_module.All_flow_fixed_colorbar_movie_maskpixel(all_pixel_flow=all_pixel_flow,blurred_images=blurred_images,filename = "All_flow_fixed_colorbar_range(-0.05,0.05)_maskpixel_boxsize21.mp4")
my_mask_module.Intensity_changes_Histogram_plot_maskpixel(all_pixel_deltaI= all_pixel_deltaI,filename ='Actin Intensity Changes Histogram_maskpixel_boxsize21.jpg')
my_mask_module.Intensity_changes_fixed_colorbar_rangemax_movie_maskpixel(all_pixel_deltaI = all_pixel_deltaI,filename = "Actin Intensity Changes_fixed_colorbar range(-0.1,0.1)_maskpixel_boxsize21.mp4")
my_mask_module.Gamma_Histogram_plot_maskpixel(all_gamma= all_pixel_gamma,filename ='Gamma Histogram_maskpixel_boxsize21.jpg')
my_mask_module.Flows_Histogram_plot_maskpixel(all_pixel_flow=all_pixel_flow,filename ='Flows Histogram_maskpixel_boxsize21.jpg')
my_mask_module.Abs_Gamma_contributions_Histogram_plot_maskpixel(all_pixel_abs_gamma_contributions = all_pixel_abs_gamma_contributions,filename ='Abs Gamma Contributions Histogram_maskpixel_boxsize21.jpg')
my_mask_module.Abs_Flows_contributions_Histogram_plot_maskpixel(all_pixel_abs_flow_contributions=all_pixel_abs_flow_contributions,filename ='Abs Flows Contributions Histogram_maskpixel_boxsize21.jpg')
my_mask_module.all_abs_gamma_contributions_fixed_colorbar_range1_movie_maskpixel(all_pixel_abs_gamma_contributions = all_pixel_abs_gamma_contributions,blurred_images=blurred_images, filename = "Abs_Gamma_contributions_fixed_colorbar_range(0,3)_maskpixel_boxsize21.mp4")
my_mask_module.all_abs_flow_contributions_fixed_colorbar_range1_movie_maskpixel(all_pixel_abs_flow_contributions =all_pixel_abs_flow_contributions,blurred_images=blurred_images, filename = "Abs_Flows_contributions_fixed_colorbar_range(0,3)_maskpixel_boxsize21.mp4")
my_mask_module.Contractions_Histogram_plot_maskpixel(all_pixel_contraction=all_pixel_contraction,filename ='Contractions or Relaxations Histogram_maskpixel_boxsize21.jpg')
my_mask_module.Flows_abs_contributions_Histogram_plot_maskpixel(all_pixel_flow_abs_contributions=all_pixel_flow_abs_contributions,filename ='Flows Abs Contributions Histogram_maskpixel_boxsize21.jpg')
my_mask_module.all_flow_abs_contributions_fixed_colorbar_range1_movie_maskpixel(all_pixel_flow_abs_contributions=all_pixel_flow_abs_contributions,blurred_images=blurred_images, filename = "Flows_Abs_contributions_fixed_colorbar_range(-1,1)_maskpixel_boxsize21.mp4")
my_mask_module.Gamma_abs_contributions_Histogram_plot_maskpixel(all_pixel_gamma_abs_contributions=all_pixel_gamma_abs_contributions,filename ='Gamma Abs Contributions Histogram_maskpixel_boxsize21.jpg')
my_mask_module.all_gamma_abs_contributions_fixed_colorbar_range1_movie_maskpixel(all_pixel_gamma_abs_contributions=all_pixel_gamma_abs_contributions,blurred_images=blurred_images, filename = "Gamma_abs_contributions_fixed_colorbar_range(-1,1)_maskpixel_boxsize21.mp4")
my_mask_module.Contraction_abs_contributions_Histogram_plot_maskpixel(all_pixel_contraction_abs_contributions=all_pixel_contraction_abs_contributions,filename ='Contraction Abs Contributions Histogram_maskpixel_boxsize21.jpg')
my_mask_module.all_contraction_abs_contributions_fixed_colorbar_range1_movie_maskpixel(all_pixel_contraction_abs_contributions=all_pixel_contraction_abs_contributions,blurred_images=blurred_images, filename = "Contraction_abs_contributions_fixed_colorbar_range(-1,1)_maskpixel_boxsize21.mp4")




   
#MASK with numba @jit(#change Nb line & delta_I line)
import my_jitmask_module
import skimage
import numpy as np
import matplotlib.pyplot as plt
filename = 'Actin_cutcells_data.tif'

my_images = skimage.io.imread(filename)

blurred_images = np.zeros_like(my_images, dtype ='double')
for index in range(my_images.shape[0]):
    this_image = my_images[index,:,:]
    this_filtered_image = skimage.filters.gaussian(this_image, sigma =1)
    blurred_images[index,:,:] = this_filtered_image
new_images = blurred_images    
    
my_jitmask_module.conduct_optical_flow(box_size=21, all_images= new_images)  
# =============================================================================
# #To test the gamma_contribution(zerodivision error) stuck at which sframe
# for start_frame in range(0,83,3):
#     end_frame = start_frame+3
#     print("testing frame range " + str(start_frame) + " to " + str(end_frame))
#     my_jitmask_module.conduct_optical_flow(box_size=20, all_images= new_images[start_frame:end_frame,:,:])
# =============================================================================

#my_jitmask_module.conduct_optical_flow(box_size=20, all_images= new_images[57:60,:,:])   
my_dictionary = my_jitmask_module.fixjit_optical_flow(smoothing_sigma = 1,box_size = 21, all_images = new_images)
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
all_pixel_dIdx = my_dictionary['all_pixel_dIdx']
all_pixel_dIdy = my_dictionary['all_pixel_dIdy']
all_v_x = my_dictionary['all_v_x']
all_v_y = my_dictionary['all_v_y'] 
all_pixel_flow =  my_dictionary['all_pixel_flow'] 
all_pixel_abs_gamma_contributions = my_dictionary['all_pixel_abs_gamma_contributions']
all_pixel_abs_flow_contributions =my_dictionary['all_pixel_abs_flow_contributions']
all_pixel_flow_abs_contributions=my_dictionary['all_pixel_flow_abs_contributions']
all_pixel_gamma_abs_contributions=my_dictionary['all_pixel_gamma_abs_contributions']
all_pixel_contraction_abs_contributions=my_dictionary['all_pixel_contraction_abs_contributions']

median_abs_all_pixel_relative_error = np.median(np.abs(all_pixel_relative_error))
print(median_abs_all_pixel_relative_error)
median_all_pixel_gamma_firstframe= np.median(all_pixel_gamma[0,:,:])
print(median_all_pixel_gamma_firstframe)#first frame remodeling
median_all_pixel_gamma= np.median(all_pixel_gamma)# median all gamma all frames
print(median_all_pixel_gamma)
print(all_pixel_gamma[0,500,500])
# =============================================================================
# cut cells no blur: 
# print(median_abs_all_pixel_relative_error)    boxsze21=1.0
# print(median_all_pixel_gamma_firstframe)#first frame remodeling   boxsze21=nan
# print(median_all_pixel_gamma)       boxsze21=nan
# print(all_pixel_gamma[0,500,500])   boxsze21=135.9895095667473  
# =============================================================================
    
    

#MASK:median all_pixel_relative_error with different boxsize
# =============================================================================
# original data no cut cells(with blur)
#boxsize=[5,10,20,21,30,40,50,60,]
# absmedian_allrelative_error = [0.589346557918162,0.8038333578065207,0.8842978064541841,0.8833060293782473,0.9081226634747702,0.9183483725603301,0.9255255669359472,0.931472423006644,]
# plt.plot(boxsize,absmedian_allrelative_error)
# =============================================================================
##cue cells with blur
plt.figure()
boxsize=[21,30,40,50,60,]
absmedian_allrelative_error = [0.5882451905755053,]
plt.title("Abs median all relative error with different boxsize")
plt.xlabel("Boxsize")
plt.ylabel("Abs median all relative error(mask pixel)")
plt.savefig("Abs median all relative error with different boxsize_maskpixels.jpg")
plt.show()
#MASK:different boxsize
#MASK:first frame gamma with different boxsize
# =============================================================================
# boxsize=[5,10,20,21,30,40,50,60,]
# median_all_pixel_gamma_firstframe = [0.000486715583868087,0.0005689795426018667,0.0004828370816727977,0.0004401550461997385,0.0004543809004687655,0.0004576347883343074,0.0004558211940845371,0.00043594426635735085,]
# plt.plot(boxsize,median_all_pixel_gamma_firstframe)
# =============================================================================
plt.figure()
boxsize=[21]
median_all_pixel_gamma_firstframe = [0]#0=nan
plt.title("Median remodeling on first frame with different boxsize")
plt.xlabel("Boxsize")
plt.ylabel("Median first frame remodeling(pixel)")
plt.savefig("Median remodeling on first frame with different boxsize_pixels.jpg")
plt.show()


#MASK:median all_pixel_gamma on all frames with different boxsize
# =============================================================================
# boxsize=[5,10,20,21,30,40,50,60,]
# median_all_pixel_gamma = [2.467497060146686e-05,2.622467664901497e-05,2.3243829285662497e-05,2.1730609592684142e-05,2.52800447571008e-05,2.825959746062836e-05,3.117184313550288e-05,3.2103539244226936e-05,]
# plt.plot(boxsize,median_all_pixel_gamma)
# =============================================================================
plt.figure()
boxsize=[21,30,40,50,60,]
median_all_pixel_gamma = [0]#0=nan
plt.title("Median remodeling on all frames with different boxsize")
plt.xlabel("Boxsize")
plt.ylabel("Median all frames remodeling(pixel)")
plt.savefig("Median remodeling on all frames with different boxsize_pixels.jpg")
plt.show()

#MASK:all_pixel_gamma[0,500,500] with different boxsize
# =============================================================================
# boxsize=[5,10,20,21,30,40,50,60,]
# median_specific_pixel_gamma = [-0.002711526016627156,-0.031978412516401614,-0.012323138700650415,-0.011032408108117458,-0.008494073159562075,-0.006468641017650399,-0.003493443212442547,-0.002476063666554035]
# plt.plot(boxsize,median_specific_pixel_gamma)
# =============================================================================
plt.figure()
boxsize=[21]
median_specific_pixel_gamma = [-0.011032408108117458,]
plt.title("Median remodeling of[500,500] on first frame with different boxsize")
plt.xlabel("Boxsize")
plt.ylabel("Median remodeling of[500,500] on first frame")
plt.savefig("Median remodeling of[500,500] on first frame with different boxsize_maskpixels.jpg")
plt.show()

    
my_jitmask_module.Gamma_Histogram_plot_maskpixel(all_gamma= all_pixel_gamma,filename ='Gamma Histogram_maskpixel_boxsize21.jpg')    
my_jitmask_module.gamma_maskpixel_fixed_colorbar_movie(all_gamma = all_pixel_gamma,filename = "actin_gamma_maskpixel_fixed_colorbar_range(-0.05,0.05)_movie_boxsize21.mp4")
#my_mask_module.gamma_maskpixel_changing_colorbar_movie(all_gamma = all_pixel_gamma,blurred_images =blurred_images, filename = "Gamma_maskpixel_with_changing_colorbar_boxsize10.mp4")
my_jitmask_module.Visualizing_Velocity_maskpixel(all_images = my_images, blurred_images = blurred_images, all_pixel_v_x= all_pixel_v_x, all_pixel_v_y= all_pixel_v_y, box_size =21, arrow_box_size=20,filename = 'Visualizing Velocity movie_maskpixel_arrowboxsize20_boxsize21.mp4')
#my_mask_module.Visualizing_Velocity_box(all_images= my_images,blurred_images= blurred_images,all_v_x =all_v_x,all_v_y=all_v_y,box_size=10,filename = 'Visualizing Velocity movie_maskpixel_box_boxsize10.mp4')
my_jitmask_module.Contractions_Relaxations_Histogram_plot_maskpixel(all_contraction = all_pixel_contraction,filename = "Actin Contractions or Relaxations Histogram_maskpixel_boxsize21.jpg")
my_jitmask_module.Contractions_or_Relaxations_fixed_colorbar_range1_movie_maskpixel(all_contraction =all_pixel_contraction,filename = "Contractions or Relaxations_fixed_colorbar_range(-0.15,0.15)_moive_maskpixel_boxsize21.mp4")
#my_allpixels_module.all_gamma_contributions_changing_colorbar_movie_pixel(all_gamma_contributions = all_pixel_gamma_contributions,blurred_images = blurred_images,filename = "Animate_all_gamma_contributions_changing_colorbar_moive_pixel_boxsize10.mp4")   
my_jitmask_module.Gamma_contributions_Histogram_plot_maskpixel(all_gamma_contributions = all_pixel_gamma_contributions,filename ='Gamma Contribution Histogram_maskpixel_boxsize21.jpg')    
my_jitmask_module.all_gamma_contributions_fixed_colorbar_range1_movie_maskpixel(all_gamma_contributions = all_pixel_gamma_contributions,blurred_images = blurred_images, filename = "Gamma_contributions_fixed_colorbar_range(-1,1)_moive_maskpixel_boxsize21.mp4")    
#my_allpixels_module.all_flow_contributions_changing_colorbar_movie_pixel(all_flow_contributions = all_pixel_flow_contributions,blurred_images = blurred_images,filename = "all_flow_contributions_changing_colorbar_moive_pixel_boxsize10.mp4")   
my_jitmask_module.Flow_Contribution_Histogram_plot_maskpixel(all_flow_contributions = all_pixel_flow_contributions,filename ="Flow Contribution Histogram_maskpixel_boxsize21.jpg")   
my_jitmask_module.all_flow_contributions_fixed_colorbar_range1_movie_maskpixel(all_flow_contributions= all_pixel_flow_contributions,filename = "Flow_contributions_fixed_colorbar range(-1,1)_moive_maskpixel_boxsize21.mp4")
my_jitmask_module.Absolute_Values_of_Flow_over_Gamma_Histogram_plot_maskpixel(all_flow_gamma = all_pixel_flow_gamma,filename ="Absolute Values of Flow over Gamma Histogram_maskpixel_boxsize21.jpg")
my_jitmask_module.All_flow_over_gamma_fixed_colorbar_range_0_max_movie_maskpixel(all_flow_gamma= all_pixel_flow_gamma,blurred_images= blurred_images,filename = "All_flow over gamma_fixed_colorbar range(0_5)_moive_maskpixel_boxsize21.mp4")    
my_jitmask_module.Contraction_or_Relaxations_Contributions_Histogram_plot_maskpixel(all_contraction_contributions = all_pixel_contraction_contributions,filename ="Contraction or Relaxations Contributions Histogram_maskpixel_boxsize21.jpg")
my_jitmask_module.Contraction_contributions_fixed_colorbar_range1_movie_maskpixel(all_contraction_contributions= all_pixel_contraction_contributions,filename = "Contraction_contributions_fixed_colorbar range(-1,1)_moive_maskpixel_boxsize21.mp4")
my_jitmask_module.Flows_Histogram_plot_maskpixel(all_pixel_flow=all_pixel_flow,filename ='Flows Histogram_maskpixel_boxsize21.jpg')
my_jitmask_module.All_flow_fixed_colorbar_movie_maskpixel(all_pixel_flow=all_pixel_flow,blurred_images=blurred_images,filename = "All_flow_fixed_colorbar_range(-0.05,0.05)_maskpixel_boxsize21.mp4")
my_jitmask_module.Intensity_changes_Histogram_plot_maskpixel(all_pixel_deltaI= all_pixel_deltaI,filename ='Actin Intensity Changes Histogram_maskpixel_boxsize21.jpg')
my_jitmask_module.Intensity_changes_fixed_colorbar_rangemax_movie_maskpixel(all_pixel_deltaI = all_pixel_deltaI,filename = "Actin Intensity Changes_fixed_colorbar range(-0.05,0.05)_maskpixel_boxsize21.mp4")
my_jitmask_module.Abs_Gamma_contributions_Histogram_plot_maskpixel(all_pixel_abs_gamma_contributions = all_pixel_abs_gamma_contributions,filename ='Abs Gamma Contributions Histogram_maskpixel_boxsize21.jpg')
my_jitmask_module.all_abs_gamma_contributions_fixed_colorbar_range1_movie_maskpixel(all_pixel_abs_gamma_contributions = all_pixel_abs_gamma_contributions,blurred_images=blurred_images, filename = "Abs_Gamma_contributions_fixed_colorbar_range(0,10)_maskpixel_boxsize21.mp4")
my_jitmask_module.Abs_Flows_contributions_Histogram_plot_maskpixel(all_pixel_abs_flow_contributions=all_pixel_abs_flow_contributions,filename ='Abs Flows Contributions Histogram_maskpixel_boxsize21.jpg')
my_jitmask_module.all_abs_flow_contributions_fixed_colorbar_range1_movie_maskpixel(all_pixel_abs_flow_contributions=all_pixel_abs_flow_contributions,blurred_images=blurred_images, filename = "Abs_Flows_contributions_fixed_colorbar_range(0,1)_maskpixel.mp4")
my_jitmask_module.Flows_abs_contributions_Histogram_plot_maskpixel(all_pixel_flow_abs_contributions=all_pixel_flow_abs_contributions,filename ='Flows Abs Contributions Histogram_maskpixel_boxsize21.jpg')
my_jitmask_module.all_flow_abs_contributions_fixed_colorbar_range1_movie_maskpixel(all_pixel_flow_abs_contributions=all_pixel_flow_abs_contributions,blurred_images=blurred_images, filename = "Flows_Abs_contributions_fixed_colorbar_range(-1,1)_maskpixel_boxsize21.mp4")
my_jitmask_module.Gamma_abs_contributions_Histogram_plot_maskpixel(all_pixel_gamma_abs_contributions=all_pixel_gamma_abs_contributions,filename ='Gamma Abs Contributions Histogram_maskpixel_boxsize21.jpg')
my_jitmask_module.all_gamma_abs_contributions_fixed_colorbar_range1_movie_maskpixel(all_pixel_gamma_abs_contributions=all_pixel_gamma_abs_contributions,blurred_images=blurred_images, filename = "Gamma_abs_contributions_fixed_colorbar_range(-1,1)_maskpixel_boxsize21.mp4")
my_jitmask_module.Contraction_abs_contributions_Histogram_plot_maskpixel(all_pixel_contraction_abs_contributions=all_pixel_contraction_abs_contributions,filename ='Contraction Abs Contributions Histogram_maskpixel_boxsize21.jpg')
my_jitmask_module.all_contraction_abs_contributions_fixed_colorbar_range1_movie_maskpixel(all_pixel_contraction_abs_contributions=all_pixel_contraction_abs_contributions,blurred_images=blurred_images, filename = "Contraction_abs_contributions_fixed_colorbar_range(-1,1)_maskpixel_boxsize21.mp4")





#MASK Simulation

import simulation_jitmask_module
import skimage
import numpy as np
import matplotlib.pyplot as plt
my_images = skimage.io.imread('makeup_data123.tif',as_gray=True)#νx = 1,νy = 2,gamma=0.5

blurred_images = np.zeros_like(my_images, dtype ='double')
for index in range(my_images.shape[0]):
    this_image = my_images[index,:,:]
    this_filtered_image = skimage.filters.gaussian(this_image, sigma =1)
    blurred_images[index,:,:] = this_filtered_image
new_images = blurred_images    
    
#my_jitmask_module.conduct_optical_flow(box_size=10, all_images= new_images[1:6,:,:])  
simulation_jitmask_module.conduct_optical_flow(box_size=21, all_images= new_images[:,:,:])  

my_dictionary = simulation_jitmask_module.fixjit_optical_flow(box_size = 21, all_images = new_images)
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
all_pixel_dIdx = my_dictionary['all_pixel_dIdx']
all_pixel_dIdy = my_dictionary['all_pixel_dIdy']
all_v_x = my_dictionary['all_v_x']
all_v_y = my_dictionary['all_v_y']    
all_pixel_flow=my_dictionary['all_pixel_flow']
all_true_local_error_ij=my_dictionary['all_true_local_error_ij']
matrix_Vx_ture=my_dictionary['matrix_Vx_ture']
matrix_Vy_ture=my_dictionary['matrix_Vy_ture']
matrix_gamma_ture=my_dictionary['matrix_gamma_ture']
abs_Vx_error=my_dictionary['abs_Vx_error']
abs_Vy_error=my_dictionary['abs_Vy_error']
abs_gamma_error=my_dictionary['abs_gamma_error']



median_abs_all_pixel_relative_error = np.median(np.abs(all_pixel_relative_error))
print(median_abs_all_pixel_relative_error)#boxsize=10,median error = 0.1994763849337324,boxsize20 is 0.20645668185800237
median_all_pixel_gamma_firstframe= np.median(all_pixel_gamma[0,:,:])
print(median_all_pixel_gamma_firstframe)#first frame remodeling#boxsize=10,20 is 0
median_all_pixel_gamma= np.median(all_pixel_gamma)# median all gamma all frames
print(median_all_pixel_gamma)#boxsize=10,20 is 0
print(all_pixel_gamma[0,500,500])#boxsize=10,20 is 0
np.nonzero(all_pixel_v_x)
np.nonzero(all_pixel_v_y)
np.nonzero(all_pixel_gamma)


simulation_jitmask_module.abs_Vx_error_fixed_colorbar_range1_movie_maskpixel(abs_Vx_error=abs_Vx_error,filename = "Abs_Vx_error_fixed_colorbar range(0,100)_maskpixel.mp4")

#MASK:different sigma









