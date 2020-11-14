# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 16:58:02 2020

@author: Admin
"""
#Camera Calibration
import cv2
import numpy as np
import matplotlib.image as mpimg
import glob
import pickle

#Camera Calibration
objp = np.zeros((9*6,3),np.float32)
#print(objp)
print(objp.shape)
objp[:,:2]=np.mgrid[0:9,0:6].T.reshape(-1,2)
#print(objp)
#print(objp.shape)

#objp[:,:2] = np.meshgrid(0:9,0:6)
#print(np.mgrid[0:9,0:6].T)
#print(np.mgrid[0:9,0:6].T.reshape(-1,2))

objectp=[]
imagep=[]
images = glob.glob('./camera_cal/calibration*.jpg')
for idx,fname in enumerate(images):
    img = cv2.imread(fname)
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray_img,(9,6),None)
    print(corners)
       
    if ret:
        objectp.append(objp)
        imagep.append(corners)
        cv2.drawChessboardCorners(img,(9,6),corners,ret)
        #cv2.imshow('images',img)
        #cv2.waitKey(0)
        
#cv2.destroyAllWindows();
#Calibration and testing
test_img = cv2.imread('./camera_cal/calibration1.jpg')
img_size = (test_img.shape[1],test_img.shape[0])
ret,mtx,dist,rvecs,tvecs=cv2.calibrateCamera(objectp,imagep,img_size,None,None)
dist_pickle={}
dist_pickle["mtx"] = mtx
dist_pickle["dist"]=dist

pickle.dump(dist_pickle,open('./camera_cal/matrix.p','wb'))

dst = cv2.undistort(test_img,mtx,dist,None,mtx)
cv2.imshow('test_img',test_img)
cv2.imshow('outputimg',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
    



