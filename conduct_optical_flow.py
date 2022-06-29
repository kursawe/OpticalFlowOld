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
    dIdx = (current_frame[2:,1:-1] +previous_frame[2:,1:-1] - current_frame[:-2,1:-1]-previous_frame[:-2,1:-1])/(4*delta_t)
    dIdy = (current_frame[1:-1,2:] +previous_frame[1:-1,2:] - current_frame[1:-1,:-2]-previous_frame[1:-1,:-2])/(4*delta_t)
    delta_I_too_big = current_frame-previous_frame
    delta_I = delta_I_too_big[1:1023,1:1023]#0-1024 in total
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
animation_camera = celluloid.Camera(plt.gcf())#plt.gcf()Get the current figure
for index in range(all_gamma.shape[0]):
    this_gamma_frame = all_gamma[index,:,:]
    img_gamma = this_gamma_frame 
    plt.imshow(img_gamma, cmap=None, norm=None, aspect=None, interpolation=None, alpha=None, vmin=np.min(all_gamma), vmax=np.max(all_gamma), origin=None, extent=None, filternorm=1, filterrad=4.0, resample=None, url=None)
    #plt.colorbar(ax = plt.gca())
    animation_camera.snap()
plt.colorbar()
#plt.colobar.set_title("Values of gamma")
plt.title("Gamma_include_gamma")
plt.xlabel("Number of Boxes")
plt.ylabel("Number of Boxes")
animation = animation_camera.animate()
animation.save('Gamma_include_gamma.gif')
animation.save('Gamma_include_gamma.mp4')

#with changing colorbar
from matplotlib.animation import FuncAnimation
fig = plt.figure()

ax = fig.add_subplot(111)
div = make_axes_locatable(ax)
cax = div.append_axes('right', size='5%', pad='5%')
# data = np.random.rand(5,5)
# im = ax.imshow(data)
# cb = fig.colorbar(im, cax=cax)
tx = ax.set_title('Gamma_include_gamma Frame 0')
def animate(i):  
    cax.cla()
    data = all_gamma[i,:,:]
    im = ax.imshow(data)
    fig.colorbar(im,cax = cax)
    tx.set_text('Gamma_include_gamma Frame {0}'.format(i))   
ax.set_xlabel("Number of Boxes")
ax.set_ylabel("Number of Boxes")  
ani = FuncAnimation(fig, animate, frames=83)
ani.save('Animate_Gamma_include_gamma_changing_colorbar.gif')
ani.save('gamma_with_changing_colorbar.mp4')
#fig.add_subplot(ROW,COLUMN,POSITION),fig.add_subplot(111)There is only one subplot or graph
#cax.cla() Clear the current axes.
#plt.gca()If the current axes doesn't exist, or isn't a polar one, the appropriate axes will be created and then returned


# plt.figure()
# animation_camera = celluloid.Camera(plt.gcf())
# for index in range(all_v_x.shape[0]):
#     this_v_x_frame = all_v_x[index,:,:]
#     img_v_x = this_v_x_frame 
#     plt.imshow(img_v_x, cmap=None, norm=None, aspect=None, interpolation=None, alpha=None, vmin=np.min(all_v_x), vmax=np.max(all_v_x), origin=None, extent=None, filternorm=1, filterrad=4.0, resample=None, url=None)
#     animation_camera.snap()
# plt.colorbar()
# plt.title("Vx_include_gamma")
# plt.xlabel("Number of Boxes")
# plt.ylabel("Number of Boxes")
# animation = animation_camera.animate()
# animation.save('Vx_include_gamma.gif')
# animation.save('Vx_include_gamma.mp4')

#with changing colorbar
from matplotlib.animation import FuncAnimation
fig = plt.figure()

ax = fig.add_subplot(111)
div = make_axes_locatable(ax)
cax = div.append_axes('right', size='5%', pad='5%')
tx = ax.set_title('Vx_include_gamma Frame 0')
def animate(i):  
    cax.cla()
    data = all_v_x[i,:,:]
    im = ax.imshow(data)
    fig.colorbar(im,cax = cax)
    tx.set_text('Vx_include_gamma Frame {0}'.format(i))   
ax.set_xlabel("Number of Boxes")
ax.set_ylabel("Number of Boxes")  
ani = FuncAnimation(fig, animate, frames=83)
ani.save('Vx_include_gamma_changing_colorbar.gif')
ani.save('Vx_include_gamma_with_changing_colorbar.mp4')



# plt.figure()
# animation_camera = celluloid.Camera(plt.gcf())
# for index in range(all_v_y.shape[0]):
#     this_v_y_frame = all_v_y[index,:,:]
#     img_v_y = this_v_y_frame 
#     plt.imshow(img_v_y, cmap=None, norm=None, aspect=None, interpolation=None, alpha=None, vmin=np.min(all_v_y), vmax=np.max(all_v_y), origin=None, extent=None, filternorm=1, filterrad=4.0, resample=None, url=None)
#     animation_camera.snap()
# plt.colorbar()
# plt.title("Vy_include_gamma")
# plt.xlabel("Number of Boxes")
# plt.ylabel("Number of Boxes")
# animation = animation_camera.animate()
# animation.save('Vy_include_gamma.gif')
# animation.save('Vy_include_gamma.mp4')


#with changing colorbar
from matplotlib.animation import FuncAnimation
fig = plt.figure()

ax = fig.add_subplot(111)
div = make_axes_locatable(ax)
cax = div.append_axes('right', size='5%', pad='5%')
tx = ax.set_title('Vy_include_gamma Frame 0')
def animate(i):  
    cax.cla()
    data = all_v_y[i,:,:]
    im = ax.imshow(data)
    fig.colorbar(im,cax = cax)
    tx.set_text('Vy_include_gamma Frame {0}'.format(i))   
ax.set_xlabel("Number of Boxes")
ax.set_ylabel("Number of Boxes")  
ani = FuncAnimation(fig, animate, frames=83)
ani.save('Vy_include_gamma_changing_colorbar.gif')
ani.save('Vy_include_gamma_with_changing_colorbar.mp4')


# plot and save a figure of feach frame
#for frame_number in range(83):
   # my_fig = plt.figure()
   # data = all_gamma[frame_number,:,:]
   # frame = plt.imshow(data)
   # plt.colorbar()
   # plt.title('frame ' + str(frame_number))
   # my_fig.savefig("frame_" + str(frame_number) +".png")


#Visualizing velocity frame 0
fig = plt.figure()
x_pos = np.mgrid[0:1020:20]
y_pos = np.mgrid[0:1020:20]
x_direct = all_v_x[0,:,:]
y_direct = all_v_y[0,:,:]
plt.imshow(all_images[0,:,:])
plt.title("Visualizing Velocity Frame0")
plt.xlabel("Number of Pixels")
plt.ylabel("Number of Pixels")
plt.quiver(y_pos, x_pos, y_direct, -x_direct, color = 'white')
plt.show()
fig.savefig("visualizing fame0.png")



#Visualizing Velocity moive(this arrow in the center of each boxes)
fig = plt.figure()
tx = plt.title('Visualizing Velocity Frame 0')
def animate(i): 
       plt.cla()
       x_pos = np.mgrid[0:1020:20]
       x_pos += 10
       y_pos = np.mgrid[0:1020:20]
       y_pos += 10
       x_direct = all_v_x[i,:,:]
       y_direct = all_v_y[i,:,:]
       plt.imshow(all_images[i,:,:])
       plt.quiver(y_pos, x_pos, y_direct, -x_direct, color = 'white')#arrow is in wrong direction because matplt and quiver have different coordanites
       plt.title("Visualizing Velocity") 
       plt.xlabel("Number of Pixels")
       plt.ylabel("Number of Pixels")
       plt.tight_layout()#make sure all lables fit in the frame
ani = FuncAnimation(fig, animate, frames=83)
ani.save('Visualizing Velocity.gif')
ani.save('Visualizing Velocity.mp4')

#method 2 works as well(this arrow in the begining of each boxes)
fig = plt.figure()
animation_camera = celluloid.Camera(plt.gcf())
for index in range(83):
       x_pos = np.mgrid[0:1020:20]
       y_pos = np.mgrid[0:1020:20]
       x_direct = all_v_x[index,:,:]
       y_direct = all_v_y[index,:,:]
       plt.imshow(all_images[index,:,:])
       plt.quiver(y_pos, x_pos, y_direct, -x_direct, color = 'white')
       animation_camera.snap()
plt.title("Visualizing Velocity")
plt.xlabel("Number of Pixels")
plt.ylabel("Number of Pixels")        
animation = animation_camera.animate()
animation.save('Visualizing Velocity_arrow in the begining.gif')
#np.meshgrid:create a rectangular grid out of two given one-dimensional arrays representing the Cartdesian indexing or Matrix indexing.
#np.linspace:a tool creating numeric sequences. creates sequences of evenly spaced numbers structured as a NumPy array.
#matplotlib.pyplot.quiver(*args, data=None, **kwargs):Plot a 2D field of arrows.
#quiver([X, Y], U, V, [C], **kw):X, Y define the arrow locations, U, V define the arrow directions, and C optionally sets the color.


#Moive of contraction/extension:-intensity*div(v)
all_div_v = np.zeros((number_of_frames-1,Nb-2,Nb-2))#all_v_x.shape(83,51,51)so 82 fames to calculate div(v)
for frame_index in range(0,83):
    current_all_v_x = all_v_x[frame_index]
    #previous_all_v_x = all_v_x[frame_index -1]
    current_all_v_y = all_v_y[frame_index]
    #previous_all_v_y = all_v_y[frame_index -1]
    #for box_index_x in range(Nb_):
        #for box_index_y in range(Nb_): 
    dV_xdx = (current_all_v_x[2:,1:-1] - current_all_v_x[:-2,1:-1])/(2*delta_t)
    dV_ydy = (current_all_v_y[1:-1,2:] - current_all_v_y[1:-1,:-2])/(2*delta_t)
    this_div_v = dV_xdx + dV_ydy
            #div_v[box_index_x,box_index_y] = this_div_v
    all_div_v[frame_index,:,:]= this_div_v           
skimage.io.imsave('div_v_include_gamma.tif', all_div_v)#all_div_v.shape(83, 49, 49)


#with fixed colorbar
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
plt.xlabel("Number of Boxes")
plt.ylabel("Number of Boxes")
animation = animation_camera.animate()
animation.save('all_div_v_fixed_colorbar.mp4')

#degrade intensity to(83,51,51) cut off last frame
all_intensity_box = np.zeros((number_of_frames-1,Nb,Nb))#keep in consistent with all_div_v.shape(83, 49, 49)  
for frame_index in range(1,blurred_images.shape[0]):
    intensity = blurred_images[frame_index]
    intensity_box = all_intensity_box[frame_index-1,:,:]
    for box_index_x in range(Nb):
        for box_index_y in range(Nb):
                this_intensity_box = np.mean(intensity[box_index_x*20:box_index_x*20+20,box_index_y*20:box_index_y*20+20])
                intensity_box[box_index_x,box_index_y] = this_intensity_box 
skimage.io.imsave('all_intensity_box.tif', all_intensity_box)#all_intensity_box.shape=(83,51,51)

fortynine_intensity_box = all_intensity_box[:,1:-1,1:-1]#or all_intensity_box[:,1:50,1:50]

all_contraction = np.zeros((number_of_frames-1,Nb-2,Nb-2))
for frame_index in range(1,84):
    contraction = all_contraction[frame_index-1,:,:]
    for box_index_x in range(Nb-2):
        for box_index_y in range(Nb-2):             
                this_contraction = -(fortynine_intensity_box[frame_index-1,box_index_x,box_index_y]*all_div_v[frame_index-1,box_index_x,box_index_y])
                contraction[box_index_x,box_index_y] = this_contraction
skimage.io.imsave('all_contraction.tif', all_contraction)#all_contraction.shape(83,49,49)

#histogram
plt.figure()
plt.hist(all_contraction.flatten(), bins=100, range=(-1,1), density=False)#most flow contribution around 0
plt.xlabel('Values of Contractions or Relaxations')
plt.ylabel('Number of Boxes')
plt.title('Contractions or Relaxations Histogram') 
#with fixed colorbar
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
plt.xlabel("Number of Boxes")
plt.ylabel("Number of Boxes")
animation = animation_camera.animate()
animation.save('Contraction_fixed_colorbar.mp4')#np.max(np.abs(all_contraction))=2.003126235420844,np.median(np.abs(all_contraction))=0.059512439332661385,np.min(np.abs(all_contraction))=1.7724330752574374e-07


#Quantify the contributions of the remodeling made to the cytoskeleton dynamics
all_gamma_contributions = np.zeros((number_of_frames-1,Nb,Nb))
all_difference_to_previous_frame_box = np.zeros((number_of_frames-1,51,51))

for frame_index in range(1,blurred_images.shape[0]):
    difference_to_previous_frame = blurred_images[frame_index,1:1023,1:1023] - blurred_images[frame_index -1,1:1023,1:1023]
    difference_to_previous_frame_box = all_difference_to_previous_frame_box[frame_index-1,:,:]
    for box_index_x in range(Nb):
        for box_index_y in range(Nb):
                this_difference_to_previous_frame_box = np.mean(difference_to_previous_frame[box_index_x*20:box_index_x*20+20,box_index_y*20:box_index_y*20+20])#last one isnot included
                difference_to_previous_frame_box[box_index_x,box_index_y] = this_difference_to_previous_frame_box

skimage.io.imsave('all_difference_to_previous_frame_box.tif', all_difference_to_previous_frame_box)    
        

for frame_index in range(1,blurred_images.shape[0]):
    gamma_contributions = all_gamma_contributions[frame_index-1,:,:]
    for box_index_x in range(Nb):
        for box_index_y in range(Nb):                
                this_gamma_contributions = all_gamma[frame_index-1,box_index_x,box_index_y]/all_difference_to_previous_frame_box[frame_index-1,box_index_x,box_index_y]
                #gamma is average of box of each pixel
                gamma_contributions[box_index_x,box_index_y] = this_gamma_contributions

from matplotlib.animation import FuncAnimation
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
ax.set_xlabel("Number of Boxes")
ax.set_ylabel("Number of Boxes")  
ani = FuncAnimation(fig, animate, frames=83)
ani.save('Animate_all_gamma_contributions_include_gamma_changing_colorbar.mp4')
           

#with fixed colorbar
plt.figure()
animation_camera = celluloid.Camera(plt.gcf())
for index in range(0,83):
    this_gamma_contributions_frame = all_gamma_contributions[index,:,:]
    img_gamma_contributions = this_gamma_contributions_frame 
    plt.imshow(img_gamma_contributions, cmap=None, norm=None, aspect=None, interpolation=None, alpha=None, vmin=-3, vmax=3, origin=None, extent=None, filternorm=1, filterrad=4.0, resample=None, url=None)
    animation_camera.snap()
plt.colorbar()
plt.title("Gamma Contributions")
plt.xlabel("Number of Boxes")
plt.ylabel("Number of Boxes")
animation = animation_camera.animate()
animation.save('All_gamma_contributions_fixed_colorbar.mp4')
#histogram
plt.figure()#215883pixes in total(2601each frame)y aixs is number of pixel,x aixs is gamma contribution value
plt.hist(all_gamma_contributions.flatten(), bins=100, range=(-5,5), density=False)
plt.xlabel('Values of Gamma Contribution')
plt.ylabel('Number of Boxes')
plt.title('Gamma Contribution Histogram')
#plt.gca().set_yscale('log')#gca=get correct axis
#np.max(np.abs(all_gamma_contributions))=109053.66072638576，np.median(all_gamma_contributions)=0.993179973065583



#Quantify Flow contribution:V delata I = -(VxIx+VyIy)/delta I
#Degrade Ix, Iy first
all_flow_contributions = np.zeros((number_of_frames-1,Nb,Nb))#Nb=51
all_dIdx_box = np.zeros((number_of_frames-1,Nb,Nb))
all_dIdy_box = np.zeros((number_of_frames-1,Nb,Nb))

for frame_index in range(1,blurred_images.shape[0]):
    current_frame = blurred_images[frame_index]#have to define agian ,othewise use the loop at last frame
    previous_frame = blurred_images[frame_index -1]
    dIdx = (current_frame[2:,1:-1] +previous_frame[2:,1:-1] - current_frame[:-2,1:-1]-previous_frame[:-2,1:-1])/(4*delta_t)
    dIdy = (current_frame[1:-1,2:] +previous_frame[1:-1,2:] - current_frame[1:-1,:-2]-previous_frame[1:-1,:-2])/(4*delta_t)
    dIdx_box = all_dIdx_box[frame_index-1,:,:]
    dIdy_box = all_dIdy_box[frame_index-1,:,:]
    for box_index_x in range(Nb):
        for box_index_y in range(Nb):
                this_dIdx_box = np.mean(dIdx[box_index_x*20:box_index_x*20+20,box_index_y*20:box_index_y*20+20])
                dIdx_box[box_index_x,box_index_y] = this_dIdx_box
                this_dIdy_box = np.mean(dIdy[box_index_x*20:box_index_x*20+20,box_index_y*20:box_index_y*20+20])
                dIdy_box[box_index_x,box_index_y] = this_dIdy_box

skimage.io.imsave('all_dIdx_box.tif', all_dIdx_box)  
skimage.io.imsave('all_dIdy_box.tif', all_dIdy_box)    
    

for frame_index in range(1,blurred_images.shape[0]):#blurred_images.shape(84,1024,1024)
    flow_contributions = all_flow_contributions[frame_index-1,:,:]
    for box_index_x in range(Nb):
        for box_index_y in range(Nb):                
                this_flow_contributions = -(all_v_x[frame_index-1,box_index_x,box_index_y]*all_dIdx_box[frame_index-1,box_index_x,box_index_y] + all_v_y[frame_index-1,box_index_x,box_index_y]*all_dIdy_box[frame_index-1,box_index_x,box_index_y])/all_difference_to_previous_frame_box[frame_index-1,box_index_x,box_index_y]
                flow_contributions[box_index_x,box_index_y] = this_flow_contributions
  
#with changing colorbar                
from matplotlib.animation import FuncAnimation
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
ax.set_xlabel("Number of Boxes")
ax.set_ylabel("Number of Boxes")  
ani = FuncAnimation(fig, animate, frames=83)
ani.save('Animate_all_flow_contributions_changing_colorbar.mp4')
#histogram
plt.figure()
plt.hist(all_flow_contributions.flatten(), bins=100, range=(-5,5), density=False)#most flow contribution around 0
plt.xlabel('Values of Flow Contribution')
plt.ylabel('Number of Boxes')
plt.title('Flow Contribution Histogram')

#with fixed colorbar
plt.figure()
animation_camera = celluloid.Camera(plt.gcf())
for index in range(all_flow_contributions.shape[0]):
    this_flow_contributions_frame = all_flow_contributions[index,:,:]
    img_flow_contributions = this_flow_contributions_frame 
    plt.imshow(img_flow_contributions, cmap=None, norm=None, aspect=None, interpolation=None, alpha=None, vmin=-5, vmax=5, origin=None, extent=None, filternorm=1, filterrad=4.0, resample=None, url=None)
    #plt.colorbar(ax = plt.gca())
    animation_camera.snap()
plt.colorbar()
plt.title("Flow Contributions")
plt.xlabel("Number of Boxes")
plt.ylabel("Number of Boxes")
animation = animation_camera.animate()
animation.save('Flow_contributions_fixed_colorbar.mp4')



#sum check if -V gradient I+gamma -Delta I =0
all_sumcheck_box = np.zeros((number_of_frames-1,51,51))
for frame_index in range(1,blurred_images.shape[0]):#blurred_images.shape(84,1024,1024)
    sumcheck_box = all_sumcheck_box[frame_index-1,:,:]
    for box_index_x in range(Nb):
        for box_index_y in range(Nb):                
                this_sumcheck = -(all_v_x[frame_index-1,box_index_x,box_index_y]*all_dIdx_box[frame_index-1,box_index_x,box_index_y] + all_v_y[frame_index-1,box_index_x,box_index_y]*all_dIdy_box[frame_index-1,box_index_x,box_index_y])+all_gamma[frame_index-1,box_index_x,box_index_y]-all_difference_to_previous_frame_box[frame_index-1,box_index_x,box_index_y]
                sumcheck_box[box_index_x,box_index_y] = this_sumcheck

skimage.io.imsave('all_sumcheck_box.tif', all_sumcheck_box)# the values of the sumcheck are almost 0
#np.max(np.abs(all_sumcheck_box))=1.734723475976807e-16,np.median(np.abs(all_sumcheck_box))=1.0842021724855044e-19
#sum check if relative error: (-V gradient I+gamma -Delta I)/Delta I =0
all_relative_error_box = np.zeros((number_of_frames-1,51,51))
for frame_index in range(1,blurred_images.shape[0]):#blurred_images.shape(84,1024,1024)
    relative_error_box = all_relative_error_box[frame_index-1,:,:]
    for box_index_x in range(Nb):
        for box_index_y in range(Nb):                
                this_relative_error = (-(all_v_x[frame_index-1,box_index_x,box_index_y]*all_dIdx_box[frame_index-1,box_index_x,box_index_y] + all_v_y[frame_index-1,box_index_x,box_index_y]*all_dIdy_box[frame_index-1,box_index_x,box_index_y])+all_gamma[frame_index-1,box_index_x,box_index_y]-all_difference_to_previous_frame_box[frame_index-1,box_index_x,box_index_y])/all_difference_to_previous_frame_box[frame_index-1,box_index_x,box_index_y]
                relative_error_box[box_index_x,box_index_y] = this_relative_error 

skimage.io.imsave('all_relative_error_box.tif', all_relative_error_box)
#np.median(np.abs(all_relative_error_box))=1.5316630507487863e-16，np.max(np.abs(all_relative_error_box))=3.1711151758299057e-12
        

#Compare the Flows and Remodeling of Actin in the Cells:flow/gamma
all_flow_gamma = np.zeros((number_of_frames-1,Nb,Nb))#Nb=51
for frame_index in range(1,blurred_images.shape[0]):#blurred_images.shape(84,1024,1024)
    flow_gamma = all_flow_gamma[frame_index-1,:,:]
    for box_index_x in range(Nb):
        for box_index_y in range(Nb):                
                this_flow_gamma = abs(-(all_v_x[frame_index-1,box_index_x,box_index_y]*all_dIdx_box[frame_index-1,box_index_x,box_index_y] + all_v_y[frame_index-1,box_index_x,box_index_y]*all_dIdy_box[frame_index-1,box_index_x,box_index_y]) /all_gamma[frame_index-1,box_index_x,box_index_y])
                flow_gamma[box_index_x,box_index_y] = this_flow_gamma

skimage.io.imsave('all_flow_gamma.tif', all_flow_gamma)#all_flow_gamma.shape (83, 51, 51)
#np.median(np.abs(all_flow_gamma))=0.09857006975070877,np.max(np.abs(all_flow_gamma))=453170.49709804816,np.min(np.abs(all_flow_gamma))=1.3353133441545005e-07
#histogram
plt.figure()
plt.hist(all_flow_gamma.flatten(), bins=100, range=(0,5), density=False)#most flow contribution around 0
plt.xlabel('Absolute Values of Flow over Gamma')
plt.ylabel('Number of Boxes')
plt.title('Absolute Values of Flow over Gamma Histogram')
#with fixed colorbar
plt.figure()
animation_camera = celluloid.Camera(plt.gcf())
for index in range(0,83):
    this_flow_gamma_frame = all_flow_gamma[index,:,:]
    img_flow_gamma = this_flow_gamma_frame
    plt.imshow(img_flow_gamma, cmap=None, norm=None, aspect=None, interpolation=None, alpha=None, vmin=0, vmax=3.6, origin=None, extent=None, filternorm=1, filterrad=4.0, resample=None, url=None)
    animation_camera.snap()
plt.colorbar()
plt.title("Flow over Gamma")
plt.xlabel("Number of Boxes")
plt.ylabel("Number of Boxes")
animation = animation_camera.animate()
animation.save('Animate_all_flow_gamma_fixed_colorbar.mp4')


#Quantify Contraction contribution: |-I div(v)|/Delta I
#cut off outermost 2 rows of boxes of all_difference_to_previous_frame_box
fortynine_difference_to_previous_frame_box = all_difference_to_previous_frame_box[:,1:-1,1:-1]#fortynine_difference_to_previous_frame_box.shape(83, 49, 49)
all_contraction_contributions = np.zeros((number_of_frames-1,Nb-2,Nb-2))#Nb=51
for frame_index in range(1,84):#blurred_images.shape(84,1024,1024)
    contraction_contributions = all_contraction_contributions[frame_index-1,:,:]
    for box_index_x in range(Nb-2):
        for box_index_y in range(Nb-2):             
                this_contraction_contributions = all_contraction[frame_index-1,box_index_x,box_index_y]/fortynine_difference_to_previous_frame_box[frame_index-1,box_index_x,box_index_y]
                contraction_contributions[box_index_x,box_index_y] = this_contraction_contributions
skimage.io.imsave('all_contraction_contributions.tif', all_contraction_contributions)#np.max(abs(all_contraction_contributions))=105524206.54417692,np.median(abs(all_contraction_contributions))=34.57326644972915

#histogram
plt.figure()
plt.hist(all_contraction_contributions.flatten(), bins=100, range=(-1000,1000), density=False)#most flow contribution around 0
plt.xlabel('Contraction or Relaxations Contributions')
plt.ylabel('Number of Boxes')
plt.title('Contraction or Relaxations Contributions Histogram') 
#with fixed colorbar
plt.figure()
animation_camera = celluloid.Camera(plt.gcf())
for index in range(all_contraction_contributions.shape[0]):
    this_contraction_contributions_frame = all_contraction_contributions[index,:,:]
    img_contraction_contributions = this_contraction_contributions_frame 
    plt.imshow(img_contraction_contributions, cmap=None, norm=None, aspect=None, interpolation=None, alpha=None, vmin=-250, vmax=250, origin=None, extent=None, filternorm=1, filterrad=4.0, resample=None, url=None)
    #plt.colorbar(ax = plt.gca())
    animation_camera.snap()
plt.colorbar()
plt.title("Contractions or Relaxations Contributions")
plt.xlabel("Number of Boxes")
plt.ylabel("Number of Boxes")
animation = animation_camera.animate()
animation.save('Contraction_contributions_fixed_colorbar.mp4')





