import skimage.io
import skimage.filters
import matplotlib.pyplot as plt
import PIL
import numpy as np

# define all important algorithm parameters
smoothing_sigma = 9
box_size = 3
#outline for optical flow
#1.Load data
image = skimage.io.imread('MB301110_i_4_movie_8 bit.tif')

#2. For each frame, apply Gaussian blur,choose sigma(manully or by kernel size)
#Using kernel size get sigma and blurred image(picture not moive)
# - either can work as long as we are consistent
# in the following, we use a manually chosen sigma defined above (line 8)

first_image = all_images[0,:,:]

blurred_images = np.zeros_like(image, dtype ='double')

for index in range(image.shape[0]):
    this_image = image[index,:,:]
    this_filtered_image = skimage.filters.gaussian(this_image, sigma =smoothing_sigma)
    blurred_images[index,:,:] = this_filtered_image

#save the file for double-checking
skimage.io.imsave('all_blurred.tif', blurred_images)

for index in range(1,blurred_images.shape[0]):
    #4.Define image gradients \nabla\mu, i.e. ux,uy on each box and each frame
    difference_to_previous_frame = blurred_images[index] - blurred_images[index -1]
    #next, calculate dI/dx dI/dy
    current_frame = blurred_images[index]
    previous_frame = blurred_images[index -1]
    dIdx = (current_frame[1:,:] - current_frame[:-1,:])/delta_x # TODO: adjust this to include previous (or next?) time point
    dIdx = (current_frame[1:,:] +previous_frame[1:,:] - current_frame[:-1,:]-previous_frame[:-1,:])/4delta_x
    dIdy = (current_frame[:,1:] +previous_frame[:,1:] - current_frame[:,:-1]-previous_frame[:,:-1])/4delta_x
    dIdt = ()
    chi^2=
    
    
    
    #3.Produce subregions(boxsize 2*2 or n by n) where there is at least one actin pre subregion
#5.Define error function to correct the advection equation approximately (on each box and each frame)
#6.Define velocity(Vx,Vy and gamma) equations, Each box use Least Squares Minimization to minimize error function 
    #6.1 Compute coefficients A, B, C of Vx,Vy, gamma
    #6.2 find Vx,Vy, gamma (for each box and each frame)
#7. write out data (movies)
