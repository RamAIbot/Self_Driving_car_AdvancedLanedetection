# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 09:53:19 2020

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
image = cv2.imread('./camera_cal/calibration20.jpg')

nx = 9
ny = 6
img_size=(image.shape[1],image.shape[0])
undist = cv2.undistort(image,mtx,dst,None,mtx)
gray_img = cv2.cvtColor(undist,cv2.COLOR_BGR2GRAY)
ret,corners = cv2.findChessboardCorners(gray_img,(nx,ny),None)
offset = 100
if ret:
    cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
    src=np.float32([corners[0],corners[nx-1],corners[-1],corners[-nx]])
    dst = np.float32([[offset, offset], [img_size[0]-offset, offset], 
                                     [img_size[0]-offset, img_size[1]-offset], 
                                     [offset, img_size[1]-offset]])
    M = cv2.getPerspectiveTransform(src,dst)
    warped = cv2.warpPerspective(undist,M,img_size,flags=cv2.INTER_LINEAR)
    
cv2.imshow('image',image)
#cv2.imshow('undistorted',img_undist)
cv2.imshow('warped',warped)
cv2.waitKey(0)
cv2.destroyAllWindows()
