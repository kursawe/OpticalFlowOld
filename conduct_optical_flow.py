import skimage.io
import skimage.filters
import matplotlib.pyplot as plt
import PIL
import numpy as np
from matplotlib import pyplot
import celluloid
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.animation as animation

# define all important algorithm parameters
smoothing_sigma = 1
box_size = 20
# mode = 'velocity_only'
mode = 'include_gamma'
#outline for optical flow
#1.Load data
all_images = skimage.io.imread('MB301110_i_4_movie_8 bit.tif')

#2. For each frame, apply Gaussian blur,choose sigma(manully or by kernel size)
#Using kernel size get sigma and blurred image(picture not moive)
# - either can work as long as we are consistent
# in the following, we use a manually chosen sigma defined above (line 8)

first_image = all_images[0,:,:]

blurred_images = np.zeros_like(all_images, dtype ='double')

for index in range(all_images.shape[0]):
    this_image = all_images[index,:,:]
    this_filtered_image = skimage.filters.gaussian(this_image, sigma =smoothing_sigma)
    blurred_images[index,:,:] = this_filtered_image

#save the file for double-checking
skimage.io.imsave('all_blurred.tif', blurred_images)
I = all_images[0,:,:]
dIdx = np.zeros_like(I)
delta_t = 1
number_of_frames = blurred_images.shape[0]
Nb = int((1024-2)/box_size)#Number of boxes
all_v_x = np.zeros((number_of_frames-1,Nb,Nb))
all_v_y = np.zeros((number_of_frames-1,Nb,Nb))
all_gamma = np.zeros((number_of_frames-1,Nb,Nb))


for frame_index in range(1,blurred_images.shape[0]):
# for frame_index in range(1,3):
    #4.Define image gradients \nabla\mu, i.e. ux,uy on each box and each frame
    difference_to_previous_frame = blurred_images[frame_index] - blurred_images[frame_index -1]
    #next, calculate dI/dx dI/dy
    current_frame = blurred_images[frame_index]
    previous_frame = blurred_images[frame_index -1]
    # easier way: 
    # TODO: use the correct equation here (i.e. combining averages from next frame and current frame
    dIdx = (current_frame[2:,:] +previous_frame[2:,:] - current_frame[:-2,:]-previous_frame[:-2,:])/(4*delta_t)
    dIdy = (current_frame[:,2:] +previous_frame[:,2:] - current_frame[:,:-2]-previous_frame[:,:-2])/(4*delta_t)
    delta_I = current_frame-previous_frame
    # at pixel i,j, content of the sum in the error function is
    # dIdt
    b = box_size*box_size #The number of pixels in the subregion is b
    #initialise v_x as a matrix with one entry for each box
    v_x = all_v_x[frame_index-1,:,:]
    v_y = all_v_y[frame_index-1,:,:]
    gamma_ = all_gamma[frame_index-1,:,:]
    #set a framework to store each Vx
    #np.zeros:Return a new array of given shape and type, filled with zeros
    #loop over box-counter in x-direction
    for box_index_x in range(Nb):
        #from0,...Nb-1
        #loop over box index in y direction
        for box_index_y in range(Nb):
            # These next two matrices are 'little' matrices, they are a restriction of the image to each box
            # I hope I got the indexing right!
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
                #store each Vx
                v_y[box_index_x,box_index_y] = Vy
#                 skimage.io.imsave('Vx_velocity_only.tif', v_x[box_index_x,box_index_y])
#                 skimage.io.imsave('Vy_velocity_only.tif', v_y[box_index_x,box_index_y])
            elif mode == 'include_gamma':
                #3.Produce subregions(boxsize 2*2 or n by n) where there is at least one actin pre subregion
                #5.Define error function to correct the advection equation approximately (on each box and each frame)
                #6.Define velocity(Vx,Vy and gamma) equations, Each box use Least Squares Minimization to minimize error function 
                #6.1 Compute coefficients A, B, C of Vx,Vy, gamma
                A = np.sum((local_dIdx)**2)
                B = np.sum(local_dIdx*local_dIdy)
                C = np.sum(local_dIdx)
                D = np.sum((local_dIdy)**2)
                E = np.sum(local_dIdy)
                sum3 = np.sum(local_delta_I)
                Vx = ((E**2-b*D)*sum1+(b*B-C*E)*sum2+(C*D-B*E)*sum3)/(delta_t*(b*A*D-A*E**2-b*B**2-C**2*D+2*B*C*E))
                Vy = ((b*B-C*E)*sum1+(C**2-b*A)*sum2+(A*E-B*C)*sum3)/(delta_t*(b*A*D-A*E**2-b*B**2-C**2*D+2*B*C*E))
                v_x[box_index_x,box_index_y] = Vx
                #store each Vx
                v_y[box_index_x,box_index_y] = Vy
                sum4 = np.sum(local_dIdx*Vx)
                sum5 = np.sum(local_dIdx*Vy)
                this_gamma = ((B*E-C*D)*sum1+(B*C-A*E)*sum2+(A*D-B**2)*sum3)/(delta_t*(b*A*D-A*E**2-b*B**2-C**2*D+2*B*C*E))
                gamma_[box_index_x,box_index_y] = this_gamma
    #6.2 find Vx,Vy, gamma (for each box and each frame)
#7. write out data (movies)
skimage.io.imsave('Vx_include_gamma.tif', all_v_x)
skimage.io.imsave('Vy_include_gamma.tif', all_v_y)
skimage.io.imsave('Gamma_include_gamma.tif', all_gamma)

#with fixed colorbar
plt.figure()
animation_camera = celluloid.Camera(plt.gcf())
for index in range(all_gamma.shape[0]):
    this_gamma_frame = all_gamma[index,:,:]
    img_gamma = this_gamma_frame 
    plt.imshow(img_gamma, cmap=None, norm=None, aspect=None, interpolation=None, alpha=None, vmin=None, vmax=None, origin=None, extent=None, filternorm=1, filterrad=4.0, resample=None, url=None)
    #plt.colorbar(ax = plt.gca())
    animation_camera.snap()
plt.colorbar()
animation = animation_camera.animate()
animation.save('Gamma_include_gamma.gif')
animation.save('Gamma_include_gamma.mp4')

plt.figure()
animation_camera = celluloid.Camera(plt.gcf())
for index in range(all_v_x.shape[0]):
    this_v_x_frame = all_v_x[index,:,:]
    img_v_x = this_v_x_frame 
    plt.imshow(img_v_x, cmap=None, norm=None, aspect=None, interpolation=None, alpha=None, vmin=None, vmax=None, origin=None, extent=None, filternorm=1, filterrad=4.0, resample=None, url=None)
    #plt.colorbar(ax = plt.gca())
    animation_camera.snap()
plt.colorbar()
animation = animation_camera.animate()
animation.save('Vx_include_gamma.gif')
animation.save('Vx_include_gamma.mp4')

plt.figure()
animation_camera = celluloid.Camera(plt.gcf())
for index in range(all_v_y.shape[0]):
    this_v_y_frame = all_v_y[index,:,:]
    img_v_y = this_v_y_frame 
    plt.imshow(img_v_y, cmap=None, norm=None, aspect=None, interpolation=None, alpha=None, vmin=None, vmax=None, origin=None, extent=None, filternorm=1, filterrad=4.0, resample=None, url=None)
    #plt.colorbar(ax = plt.gca())
    animation_camera.snap()
plt.colorbar()
animation = animation_camera.animate()
animation.save('Vy_include_gamma.gif')
animation.save('Vy_include_gamma.mp4')


# with changing colorbar
#plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True

# fig = plt.figure()

# ax = fig.add_subplot(111)
# div = make_axes_locatable(ax)
# cax = div.append_axes('right', '5%', '5%')
# im = plt.imshow(img_gamma, cmap=None, norm=None, aspect=None, interpolation=None, alpha=None, vmin=None, vmax=None, origin=None, extent=None, filternorm=1, filterrad=4.0, resample=None, url=None)
# cb = fig.colorbar(im, cax=cax)
# tx = ax.set_title('Frame 0')
# cmap = ["copper", 'RdBu_r', 'Oranges', 'cividis', 'hot', 'plasma']

# def animate(i):
#    cax.cla()
#    fig.colorbar(im, cax=cax)
#    tx.set_text('Frame {0}'.format(i))
# ani = animation.FuncAnimation(fig, animation, frames=83)

# plt.show()

#was tring to implement in this way, also, there are errors, I have fixed some, except this:AttributeError: 'numpy.ndarray' object has no attribute 'axes'
# import matplotlib.pyplot as plt
# from celluloid import Camera
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# from matplotlib.pyplot import axes
# fig = plt.figure()
# img_gamma = all_gamma[0,:,:]
# camera = Camera(img_gamma)
# for i in range(83):
#     plt.plot(all_gamma[0,:,:])
#  #   camera.snap() 
# animation = camera.animate() 
# animation.save('Gamma_include_gamma.mp4')










