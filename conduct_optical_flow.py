import cv2

print('this is a test')
print('this is a second test')
print('optical flow')
#outline for optical flow
#1.Load data
#2.Gaussian blur,choose sigma(manully or by kernel size)
#3.Produce subregions(boxsize 2*2)
#4.Define image gradients Ix,Iy
#5.Define error function
#6.Define velocity(Vx,Vy)
#7.Each box use Least Squares Minimization to minimize error function to find Vx,Vy
#8.Initialize velocity field: set Vx=0.Vy=0,get the value of each gamma
#9.Compute coefficients of Vx,Vy
#10.Determine velocity(Vx,Vy)
