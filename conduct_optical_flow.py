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

for frame_index in range(1,blurred_images.shape[0]):
    #4.Define image gradients \nabla\mu, i.e. ux,uy on each box and each frame
    difference_to_previous_frame = blurred_images[frame_index] - blurred_images[frame_index -1]
    #next, calculate dI/dx dI/dy
    current_frame = blurred_images[frame_index]
    previous_frame = blurred_images[frame_index -1]
    # easier way: 
    # TODO: use the correct equation here (i.e. combining averages from next frame and current frame
    dIdx = (current_frame[2:,:] +previous_frame[2:,:] - current_frame[:-2,:]-previous_frame[:-2,:])/(4*delta_x)
    dIdy = (current_frame[:,2:] +previous_frame[:,2:] - current_frame[:,:-2]-previous_frame[:,:-2])/(4*delta_x)
    dIdt = current_frame-previous_frame
    # at pixel i,j, content of the sum in the error function is
    # dIdt
    box_size =b
    Nb = int((1024-2)/b)
    #initialise v_x as a matrix with one entry for each box
    v_x = np.zeros((Nb,Nb))
    #Return a new array of given shape and type, filled with zeros
    #loop over box-counter in x-direction
    for box_index_x in range(Nb):
        #loop over box index in y direction
        for box_index_y in range(Nb):
            # These next two matrices are 'little' matrices, they are a restriction of the image to each box
            # I hope I got the indexing right!
            local_dIdt = dIdt[box_index_x*Nb:((box_index_x+1)*Nb),box_index_y*Nb:(((box_index_y+1)*Nb))]
            local_dIdx = dIdx[box_index_x*Nb:((box_index_x+1)*Nb),box_index_y*Nb:(((box_index_y+1)*Nb))]
            local_dIdy = dIdy[box_index_x*Nb:((box_index_x+1)*Nb),box_index_y*Nb:(((box_index_y+1)*Nb))]
            sum1 = np.sum(local_dIdt*local_dIdx)
            sum2 = np.sum(local_dIdt*local_dIdy)
            A = sum((local_dIdx)^2)
            B = sum(local_dIdx*local_dIdy)
            C = sum((local_dIdy)^2)
            Vx =(-C*sum1+ B*sum2)/local_dIdt(AC-B^2)
            Vy =(-A*sum2+ B*sum1)/local_dIdt(AC-B^2)
            #Is this right? should I follow this way and keep coding? 
            #A: I think now it is kind of right? Except that A, B and C need to be defined before they are used
            #Do we need to define many sum functions? Since ABC all have sum symbol
            #A: I think we do, but I cannot be sure since I do not know the formula for A,B and C
           
    
    
    #3.Produce subregions(boxsize 2*2 or n by n) where there is at least one actin pre subregion
#5.Define error function to correct the advection equation approximately (on each box and each frame)
#6.Define velocity(Vx,Vy and gamma) equations, Each box use Least Squares Minimization to minimize error function 
    #6.1 Compute coefficients A, B, C of Vx,Vy, gamma
    #6.2 find Vx,Vy, gamma (for each box and each frame)
#7. write out data (movies)
