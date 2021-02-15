# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 12:37:53 2020

@author: Admin
"""

import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob
import pickle

def nothing(x):
    pass

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

def yuv_select_lumin(image,thresh=(0,255)):
    yuv_img = cv2.cvtColor(image,cv2.COLOR_BGR2YUV)
    lumin = yuv_img[:,:,0]
    binary_output = np.zeros_like(lumin)
    binary_output[(lumin>thresh[0])&(lumin<=thresh[1])]=1
    return binary_output

# Choose a Sobel kernel size
ksize = 3 # Choose a larger odd number to smooth gradient measurements
#image = cv2.imread('./test_images/straight_lines1.jpg')
#image = cv2.imread('./test_images/test5.jpg')
#image = cv2.imread('./test_images/testing.jpg')

image = cv2.imread('D:/Self Driving Car Engineer/Course 4/SampleImages/1035.jpg')


# Apply each of the thresholding functions
#gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
#grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
#mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(20, 200))
#dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))
#
#combined = np.zeros_like(dir_binary)
#combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
#Remove comments to see each image
#cv2.imshow('image',image)
#cv2.imshow('gradx',gradx*255)
#cv2.imshow('grady',grady*255)
#cv2.imshow('Mag_bin',mag_binary*255)
#cv2.imshow('Direction',dir_binary*255)
#cv2.imshow('Combined',combined*255)

#cv2.waitKey(0)
#cv2.destroyAllWindows()

cv2.namedWindow("Thresholding")
cv2.createTrackbar('Sobelx_min','Thresholding',1,255,nothing)
cv2.createTrackbar('Sobelx_max','Thresholding',100,255,nothing)
cv2.createTrackbar('Sobely_min','Thresholding',1,255,nothing)
cv2.createTrackbar('Sobely_max','Thresholding',100,255,nothing)

cv2.createTrackbar('mag_min','Thresholding',0,255,nothing)
cv2.createTrackbar('mag_max','Thresholding',100,255,nothing)
cv2.createTrackbar('dir_min','Thresholding',0,157,nothing)
cv2.createTrackbar('dir_max','Thresholding',0,157,nothing)

cv2.createTrackbar('s_min','Thresholding',0,255,nothing)
cv2.createTrackbar('s_max','Thresholding',100,255,nothing)

cv2.createTrackbar('l_min','Thresholding',0,255,nothing)
cv2.createTrackbar('l_max','Thresholding',0,255,nothing)

cv2.createTrackbar('y_min','Thresholding',0,255,nothing)
cv2.createTrackbar('y_max','Thresholding',0,255,nothing)

cv2.createTrackbar('lumin_min','Thresholding',0,255,nothing)
cv2.createTrackbar('lumin_max','Thresholding',0,255,nothing)

while(1):
    print('while')
    sx_min = cv2.getTrackbarPos('Sobelx_min','Thresholding')
    sx_max = cv2.getTrackbarPos('Sobelx_max','Thresholding')
    sy_min = cv2.getTrackbarPos('Sobely_min','Thresholding')
    sy_max = cv2.getTrackbarPos('Sobelx_min','Thresholding')
    
    mag_min = cv2.getTrackbarPos('mag_min','Thresholding')
    mag_max = cv2.getTrackbarPos('mag_max','Thresholding')
    dir_min = cv2.getTrackbarPos('dir_min','Thresholding')
    dir_max = cv2.getTrackbarPos('dir_max','Thresholding')
    
    smin = cv2.getTrackbarPos('s_min','Thresholding')
    smax = cv2.getTrackbarPos('s_max','Thresholding')
    
    lmin = cv2.getTrackbarPos('l_min','Thresholding')
    lmax = cv2.getTrackbarPos('l_max','Thresholding')
    
    ymin = cv2.getTrackbarPos('y_min','Thresholding')
    ymax = cv2.getTrackbarPos('y_max','Thresholding')
    
    lumin_min = cv2.getTrackbarPos('lumin_min','Thresholding')
    lumin_max = cv2.getTrackbarPos('lumin_max','Thresholding')
    
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(sx_min, sx_max))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(sy_min, sy_max))
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(mag_min, mag_max))
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(dir_min/100, dir_max/100))
    
    s_binary = hls_select(image,thresh=(smin,smax))
    l_binary = hls_select_light(image,thresh=(lmin,lmax))
    
    y_binary=yellow(image,thresh=(ymin,ymax))
    #h_binary = hls_select_hue(image,thresh=(hmin,hmax))
    luminescence = yuv_select_lumin(image,thresh=(lumin_min,lumin_max)) 
    
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) |(s_binary == 1) &(luminescence==1)] = 1
    cv2.imshow('Combined',combined*255)
    cv2.waitKey(1)
cv2.destroyAllWindows()
    

