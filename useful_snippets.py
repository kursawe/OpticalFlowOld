import cv2 as cv
import cv2
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