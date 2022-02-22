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
all_images = skimage.io.imread('MB301110_i_4_movie_8 bit.tif')

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
I = all_images[0,:,:]
dIdx = np.zeros_like(I)

for index in range(1,blurred_images.shape[0]):
    #4.Define image gradients \nabla\mu, i.e. ux,uy on each box and each frame
    difference_to_previous_frame = blurred_images[index] - blurred_images[index -1]
    #next, calculate dI/dx dI/dy
    current_frame = blurred_images[index]
    previous_frame = blurred_images[index -1]
    # easier way: 
    # TODO: use the correct equation here (i.e. combining averages from next frame and current frame
    dIdx = (current_frame[2:,:] +previous_frame[2:,:] - current_frame[:-2,:]-previous_frame[:-2,:])/(4*delta_x)
    dIdy = (current_frame[:,2:] +previous_frame[:,2:] - current_frame[:,:-2]-previous_frame[:,:-2])/(4*delta_x)
    dIdt = current_frame-previous_frame
    # at pixel i,j, content of the sum in the error function is
    # dIdt
    box_size =b
    Nb = int((1024-2)/b)
    #define sum
def sum1:
    for k in range(1, Nb)
    sum += difference_to_previous_frame*dIdx
    return sum
def sum2:
    for l in range(1, Nb)
    sum += difference_to_previous_frame*dIdy
    return sum
    Vx =(-C*sum1+ B*sum2)/dIdt(AC-B^2)
    #Is this right? should I follow this way and keep coding? 
    #Do we need to define many sum functions? Since ABC all have sum symbol
    A =
    B =
    C =
    
    
    #3.Produce subregions(boxsize 2*2 or n by n) where there is at least one actin pre subregion
#5.Define error function to correct the advection equation approximately (on each box and each frame)
#6.Define velocity(Vx,Vy and gamma) equations, Each box use Least Squares Minimization to minimize error function 
    #6.1 Compute coefficients A, B, C of Vx,Vy, gamma
    #6.2 find Vx,Vy, gamma (for each box and each frame)
#7. write out data (movies)
