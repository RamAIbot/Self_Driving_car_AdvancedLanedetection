# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 16:55:33 2020

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

def hls_select_light(image,thresh=(0,255)):
    hls = cv2.cvtColor(image,cv2.COLOR_BGR2HLS)
    h = hls[:,:,1]
    binary_output = np.zeros_like(h)
    binary_output[(h>thresh[0])&(h<=thresh[1])]=1
    return binary_output

def yellow(image,thresh=(0,255)):
    y_aloneG = image[:,:,1]
    y_aloneR = image[:,:,2]
    binary_output = np.zeros_like(y_aloneG)
    binary_output[(y_aloneG>thresh[0])&(y_aloneG<=thresh[1])&(y_aloneR>thresh[0])&(y_aloneR<=thresh[1])]=1
    return binary_output


image = cv2.imread('./test_images/straight_lines1.jpg')
#image = cv2.imread('./test_images/straight_lines2.jpg')
#image = cv2.imread('./test_images/test1.jpg')
#image = cv2.imread('./test_images/test2.jpg')
#image = cv2.imread('./test_images/test3.jpg')
#image = cv2.imread('./test_images/test4.jpg')
#image = cv2.imread('./test_images/test5.jpg')
#image = cv2.imread('./test_images/test6.jpg')
    
ksize = 3 
img_undist = cv2.undistort(image,mtx,dst,None,mtx)
#***Values for straight_lines1.jpg**
#gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(105, 185))
#grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(105, 189))
#mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(82, 215))
#dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.77, 1.30))
#s_binary = hls_select(image,thresh=(210,229))


gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(86, 249))
grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(86, 247))
mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(109, 238))
dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(1.01, 1.56))
s_binary = hls_select(image,thresh=(135,254))
l_binary = hls_select_light(image,thresh=(237,255))

combined = np.zeros_like(dir_binary)
combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) |(s_binary == 1) |(l_binary==1)] = 1

yellow_image = yellow(image,thresh=(240,255))
cv2.imwrite('./test_images/test.jpg',combined*255)
cv2.imshow('image',image)
cv2.imshow('yellow',yellow_image*255)
#cv2.imshow('undistorted',img_undist)
cv2.imshow('combined',combined*255)
cv2.waitKey(0)
cv2.destroyAllWindows()


