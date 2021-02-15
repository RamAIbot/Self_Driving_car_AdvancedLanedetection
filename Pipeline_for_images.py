# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 09:02:52 2020

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 08:46:30 2020

@author: Admin
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import pickle

dist_pickle = pickle.load(open('./camera_cal/matrix.p','rb'))
dst = dist_pickle["dist"]
mtx = dist_pickle["mtx"]
#Calib test image
#image = cv2.imread('./camera_cal/calibration3.jpg')
fin=[]
out = np.arange(0,250)/250
#print(out.shape)
out1= np.ones(100)
#print(out1.shape)
out2=np.arange(400,350,-1)/400
#print(out2.shape)
out3=np.zeros(400)
#print(out3.shape)

out4=np.arange(800,850,1)/850
#print(out4.shape)
out5=np.ones(100)
#print(out5.shape)
out6 = np.arange(1100,950,-1)/1100
out7=np.zeros(180)

fin = np.concatenate((out, out1, out2,out3,out4,out5,out6,out7))
fin = np.expand_dims(fin,axis=1)

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
   
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    if orient=='x':
        sobel = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=sobel_kernel)
    absolute = np.absolute(sobel)
    scaled = np.uint8(255*absolute/np.max(absolute))
    grad_binary = np.zeros_like(scaled)
    grad_binary[(scaled >= thresh[0])&(scaled <= thresh[1])] = 1
    
    return grad_binary

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    # Apply threshold
    gray_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray_img,cv2.CV_64F,1,0,ksize=sobel_kernel)
    sobely = cv2.Sobel(gray_img,cv2.CV_64F,0,1,ksize=sobel_kernel)
    mag_sobel = np.sqrt((sobelx)**2 + (sobely)**2)
    absolute = np.absolute(mag_sobel)
    scaled = np.uint8(255*absolute/np.max(absolute))
    mag_binary = np.zeros_like(scaled)
    mag_binary[(scaled >= mag_thresh[0])&(scaled <= mag_thresh[1])] = 1
    return mag_binary

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # Apply threshold
    gray_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray_img,cv2.CV_64F,1,0,ksize=sobel_kernel)
    sobely = cv2.Sobel(gray_img,cv2.CV_64F,0,1,ksize=sobel_kernel)
    absx = np.absolute(sobelx)
    absy = np.absolute(sobely)
    direction = np.arctan2(absy,absx)
    dir_binary =  np.zeros_like(gray_img)
    dir_binary[(direction >= thresh[0])&(direction <= thresh[1])] = 1
    return dir_binary

def hls_select(image,thresh=(0,255)):
    hls = cv2.cvtColor(image,cv2.COLOR_BGR2HLS)
    s = hls[:,:,2]
    binary_output = np.zeros_like(s)
    binary_output[(s>thresh[0])&(s<=thresh[1])]=1
    return binary_output





def hist(img):
    #img = img[:,:,0]/255
    img = img/255
    img = np.expand_dims(img,axis=-1)
    bottom_half = img[img.shape[0]//2:,:]
    histogram = np.sum(bottom_half,axis=0)
#    out = np.arange(600)
#    out1 = np.arange(600,-1,-1)
#    out3=np.zeros(79)
#    out2=np.concatenate((out, out1, out3))
#    fin = np.expand_dims(out2,axis=1)
    histogram = np.multiply(histogram,fin)
    #print(img.shape)
    out_img = np.dstack((img,img,img))
    #print(out_img.shape)
    #print(histogram.shape)
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:])+midpoint
    
    nwindows = 9
    margin = 100
    minpix =50
    
    window_height = np.int(img.shape[0]//nwindows)
    nonzero = img.nonzero()
    #**Beware y and then x**
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    left_lane_ids=[]
    right_lane_ids=[]
    
    for window in range(nwindows):
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - (window)*window_height
        
        win_xleft_low = leftx_current - margin
        win_xleft_high =leftx_current + margin
        
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0),2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0),2)
        
        good_left_inds = ((nonzeroy >= win_y_low )& (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) &(nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds =  ((nonzeroy >= win_y_low )& (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) &(nonzerox < win_xright_high)).nonzero()[0]
        
       
        left_lane_ids.append(good_left_inds)
        right_lane_ids.append(good_right_inds)
        
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        
    try:
        left_lane_ids = np.concatenate(left_lane_ids)
        right_lane_ids = np.concatenate(right_lane_ids)
    except ValueError:
        pass
    
    leftx = nonzerox[left_lane_ids]
    lefty = nonzeroy[left_lane_ids]
    rightx = nonzerox[right_lane_ids]
    righty = nonzeroy[right_lane_ids]

        
    
    return histogram,leftx,lefty,rightx,righty,out_img


#image = cv2.imread('./test_images/straight_lines1.jpg')
#image = cv2.imread('./test_images/straight_lines2.jpg')
#image = cv2.imread('./test_images/test1.jpg')
#image = cv2.imread('./test_images/test2.jpg')
#image = cv2.imread('./test_images/test3.jpg')
#image = cv2.imread('./test_images/test4.jpg')
#image = cv2.imread('./test_images/test5.jpg')
#image = cv2.imread('./test_images/test6.jpg')
#image = cv2.imread('./test_images/testing.jpg')
image = cv2.imread('D:/Self Driving Car Engineer/Course 4/SampleImages/1040.jpg')
    
ksize = 3 
img_undist = cv2.undistort(image,mtx,dst,None,mtx)
final_img = np.copy(img_undist)
gradx = abs_sobel_thresh(img_undist, orient='x', sobel_kernel=ksize, thresh=(52, 238))
grady = abs_sobel_thresh(img_undist, orient='y', sobel_kernel=ksize, thresh=(59, 249))
mag_binary = mag_thresh(img_undist, sobel_kernel=ksize, mag_thresh=(68, 255))
dir_binary = dir_threshold(img_undist, sobel_kernel=ksize, thresh=(0.02, 1.57))
#s_binary = hls_select(img_undist,thresh=(212,255))
s_binary = hls_select(img_undist,thresh=(151,255))
#was 254 perviously but 255 works better for test 1


combined = np.zeros_like(dir_binary)
combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) |(s_binary == 1)] = 1
#top left,bottom left,bottom right,top right Initally the yoffset was 10
src = np.float32([[585-20, 460+10],[203-20, 720],[1127+30, 720],[695+30, 460+10]])

points = np.int32(np.copy(src))
cv2.polylines(img_undist,[points] ,True,(0,0,255),5)
#** Key here is keep the destination top boundary as closer as possible for effective transform**
dst = np.array([[320-20, 0],[320-20, 720],[960+30, 720],[960+30, 0]],dtype='float32')
img_size=(combined.shape[1],combined.shape[0])
M = cv2.getPerspectiveTransform(src,dst)
Minv = cv2.getPerspectiveTransform(dst,src)
warped = cv2.warpPerspective(combined,M,img_size,flags=cv2.INTER_LINEAR)

histogram_img,leftx,lefty,rightx,righty,out_img = hist(warped)

left_fit = np.polyfit(lefty,leftx,2)
right_fit = np.polyfit(righty,rightx,2)

ploty = np.linspace(0,warped.shape[0]-1,warped.shape[0])
try:
    leftfitx = left_fit[0]*ploty**2 + left_fit[1]*ploty+left_fit[2]
    rightfitx = right_fit[0]*ploty**2+right_fit[1]*ploty+right_fit[2]
except TypeError:
    print('The function failed to fit a line!')
    
final_out_img = np.copy(out_img).astype(np.uint8)
    
out_img[lefty,leftx] = [255,0,0]
out_img[righty,rightx] = [0,0,255]
leftpoints_draw = (np.asarray([leftfitx,ploty]).T).astype(np.int32)
rightpoints_draw = (np.asarray([rightfitx,ploty]).T).astype(np.int32)

cv2.polylines(out_img,[leftpoints_draw],False,(0,255,255),3)
cv2.polylines(out_img,[rightpoints_draw],False,(0,255,255),3)

#**Measuring Curvature radius**
y_eval = np.max(ploty)
ym_per_pixel = 30/720 #meters per pixel in y dimension
xm_per_pixel = 3.7/700 #meters per pixel in x dimension
left_curved = ((1 + (2*left_fit[0]*y_eval*ym_per_pixel + left_fit[1])**2)**1.5)/(np.absolute(2*left_fit[0]))
right_curved = ((1 + (2*right_fit[0]*y_eval*ym_per_pixel + right_fit[1])**2)**1.5)/(np.absolute(2*right_fit[0]))

print('left_curved: '+str(left_curved))
print('right_curved: '+str(right_curved))

#**Drwaing on image the lane**
pts_left = np.array([np.transpose(np.vstack([leftfitx, ploty]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([rightfitx, ploty])))])
#flipud is just reversing the order of the points which are from top to bottom to make them bottom to top so that we can have an anticlockwise ordering of the corners.
pts = np.hstack((pts_left, pts_right))
#print(pts.shape)


cv2.fillPoly(final_out_img,np.int_([pts]),(0,255,0))
#cv2.imwrite('./test_images/test.jpg',combined*255)
newwarp = cv2.warpPerspective(final_out_img, Minv, (image.shape[1], image.shape[0])) 
result = cv2.addWeighted(final_img, 1, newwarp, 0.3, 0)
cv2.imshow('undistorted',img_undist)
plt.plot(histogram_img)
plt.show()
cv2.imshow('combined',combined*255)
cv2.imshow('warped',warped*255)
cv2.imshow('out_img',out_img)
cv2.imshow('result',result)
#cv2.imwrite('./output_images/test5.jpg',result)

cv2.waitKey(0)
cv2.destroyAllWindows()








