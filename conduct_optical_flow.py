import cv2

print('this is a test')
print('this is a second test')
print('optical flow')
#outline for optical flow
#1.Load data
#2.Gaussian blur,choose sigma(manully or by kernel size)
#Using kernel size get sigma and blurred image(picture not moive)
import cv2 as cv
import numpy as np
import skimage.io
from matplotlib import pyplot as plt

img = cv.imread('image data3.jpg')
width= img.shape[0]
height=img.shape[1]
depth=img.shape[2]#channels
print(width, height, depth)#960,914,3
blur = cv.GaussianBlur(img,(35,35),0)
#()means size of kernel, both need to be positive and odd numbers
#if sigma X= sigma Y =0, sigma=0.3*(ksize-1)*0.5-1)+0.8
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()

skimage.io.imsave('blurred_imdage data3.jpg', blur)
#choose sigma manully to get blurred movie
import skimage.io
import skimage.filters
import matplotlib.pyplot as plt
import PIL
import numpy as np

image = skimage.io.imread('MB301110_i_4_movie_8 bit.tif')

first_image = image[0,:,:]

blurred_images = np.zeros_like(image, dtype ='double')

print(image.shape[0])
for index in range(image.shape[0]):
    this_image = image[index,:,:]
    this_filtered_image = skimage.filters.gaussian(this_image, sigma =9)
    blurred_images[index,:,:] = this_filtered_image
print(np.max(blurred_images))
skimage.io.imsave('all_blurred.tif', blurred_images)

#3.Produce subregions(boxsize 2*2)
#4.Define image gradients Ix,Iy
#5.Define error function
#6.Define velocity(Vx,Vy)
#7.Each box use Least Squares Minimization to minimize error function to find Vx,Vy
#8.Initialize velocity field: set Vx=0.Vy=0,get the value of each gamma
#9.Compute coefficients of Vx,Vy
#10.Determine velocity(Vx,Vy)
