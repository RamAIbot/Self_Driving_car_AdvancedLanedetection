# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 10:14:50 2020

@author: Admin
"""

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

def nothing(x):
    pass


def hist(img):
    img = img[:,:,0]/255
    img = np.expand_dims(img,axis=-1)
    bottom_half = img[img.shape[0]//2:,:]
    histogram = np.sum(bottom_half,axis=0)
    print(img.shape)
    out_img = np.dstack((img,img,img))
    print(out_img.shape)
    print(histogram.shape)
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

dist_pickle = pickle.load(open('./camera_cal/matrix.p','rb'))
dst = dist_pickle["dist"]
mtx = dist_pickle["mtx"]
#Calib test image
image = cv2.imread('./test_images/test.jpg')
#Use binary image to generate polygon for lanes don't use color
#image = cv2.imread('./test_images/straight_lines1.jpg')


img_size=(image.shape[1],image.shape[0])
undist = cv2.undistort(image,mtx,dst,None,mtx)
gray_img = cv2.cvtColor(undist,cv2.COLOR_BGR2GRAY)

offset = 300
leftx = 0
lefty = (img_size[1]//2)+70

rightx = img_size[0]
righty = (img_size[1]//2)+70

bottomrightx = img_size[0]
bottomrighty = img_size[1]

bottomleftx = 0
bottomlefty = img_size[1]

##src=np.float32([[leftx+600,lefty],[rightx-600,righty],[bottomrightx,bottomrighty],[bottomleftx,bottomlefty]])
src = np.float32([[585, 460],[203, 720],[1127, 720],[695, 460]])
points = np.int32(np.copy(src))
##points = points.reshape((-1, 1, 2))
#print(points.shape)
image = cv2.polylines(image,[points] ,True,(0,0,255),5)
#
##dst = np.float32([[bottomleftx,bottomlefty],[leftx+500,lefty],[bottomrightx,bottomrighty],[rightx-500,righty]])
##distance between x coordinate of bottom-right and bottom-left
#widthA = np.sqrt((bottomrightx-bottomleftx)**2+(bottomrighty-bottomlefty)**2)
##Distance between x coordinate of top-right and top-left
#widthB = np.sqrt((rightx - leftx)**2+(righty - lefty)**2)
#maxWidth = max(int(widthA),int(widthB))
#
##distance between y value top-right and bottom-right
#heightA = np.sqrt((rightx - bottomrightx)**2+(righty - bottomrighty)**2)
##distance between y value top-left and bottom-left
#heightB = np.sqrt((leftx - bottomleftx)**2+(lefty - bottomlefty)**2)
#maxHeight = max(int(heightA),int(heightB))


#cv2.namedWindow("Perspective")
#
#cv2.createTrackbar('tl_x','Perspective',0,img_size[0],nothing)
#cv2.createTrackbar('tl_y','Perspective',0,img_size[1],nothing)
#cv2.createTrackbar('tr_x','Perspective',0,img_size[0],nothing)
#cv2.createTrackbar('tr_y','Perspective',0,img_size[1],nothing)
#
#cv2.createTrackbar('br_x','Perspective',0,img_size[0],nothing)
#cv2.createTrackbar('br_y','Perspective',0,img_size[1],nothing)
#cv2.createTrackbar('bl_x','Perspective',0,img_size[0],nothing)
#cv2.createTrackbar('bl_y','Perspective',0,img_size[1],nothing)

#while(1):
#    print('while')
#    tl_x = cv2.getTrackbarPos('tl_x','Perspective')
#    tl_y = cv2.getTrackbarPos('tl_y','Perspective')
#    tr_x = cv2.getTrackbarPos('tr_x','Perspective')
#    tr_y = cv2.getTrackbarPos('tr_y','Perspective')
#    
#    br_x = cv2.getTrackbarPos('br_x','Perspective')
#    br_y = cv2.getTrackbarPos('br_y','Perspective')
#    bl_x = cv2.getTrackbarPos('bl_x','Perspective')
#    bl_y = cv2.getTrackbarPos('bl_y','Perspective')
#    
#    dst = np.array([[tl_x,tl_y],[tr_x,tr_y],[br_x,br_y],[bl_x,bl_y]],dtype='float32')
#dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0,maxHeight-1]],dtype='float32')
dst = np.array([[320, 0],[320, 720],[960, 720],[960, 0]],dtype='float32')
pointsdst=np.int32(np.copy(dst))

M = cv2.getPerspectiveTransform(src,dst)
warped = cv2.warpPerspective(undist,M,img_size,flags=cv2.INTER_LINEAR)
warped1 = np.copy(warped)
#warped = cv2.warpPerspective(undist,M,(maxWidth, maxHeight),flags=cv2.INTER_LINEAR)
warped1 = cv2.polylines(warped1,[pointsdst] ,True,(0,0,255),5)
cv2.imshow('image',image)
#cv2.imshow('undist',undist)
#cv2.imshow('undistorted',img_undist)
cv2.imshow('warped',warped)
#print(warped.shape)

histogram_img,leftx,lefty,rightx,righty,out_img = hist(warped)

left_fit = np.polyfit(lefty,leftx,2)
right_fit = np.polyfit(righty,rightx,2)

ploty = np.linspace(0,warped.shape[0]-1,warped.shape[0])
try:
    leftfitx = left_fit[0]*ploty**2 + left_fit[1]*ploty+left_fit[2]
    rightfitx = right_fit[0]*ploty**2+right_fit[1]*ploty+right_fit[2]
except TypeError:
    print('The function failed to fit a line!')
    
out_img[lefty,leftx] = [255,0,0]
out_img[righty,rightx] = [0,0,255]
leftpoints_draw = (np.asarray([leftfitx,ploty]).T).astype(np.int32)
rightpoints_draw = (np.asarray([rightfitx,ploty]).T).astype(np.int32)

cv2.polylines(out_img,[leftpoints_draw],False,(0,255,255),3)
cv2.polylines(out_img,[rightpoints_draw],False,(0,255,255),3)

    
#plt.plot(histogram_img)
#plt.plot(leftfitx,ploty,color='yellow')
#plt.plot(rightfitx,ploty,color='yellow')
cv2.imshow('out_img',out_img)
#plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()



