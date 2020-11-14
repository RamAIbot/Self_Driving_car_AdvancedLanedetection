## Advanced Lane Finding

<img src ='./output_images/straight_lines1.jpg' alt='image'/>

  The project is an advanced version of the previous lane detection algorithm which uses canny edges and hough transformt to detect lanes. The problem with the previous algotihm is that it fails to detect lanes of different curvature and lighting. This project makes is broken down into many steps to solve the lane detection problem. The general idea behind the project is that we first perform the lens correction, then we find out the lanes using sobel opration and color spaces thresholding. Now after getting a binary warped image we then make a perspective transform which is the bird's eye view of the road. Now we fit a second order polynomial to get the lane boundaries and mark the region inside the lane which is safe for the car to travel. We also find out the radius of curvature.
  

 
<h3> Camera Calibration </h3>

<p> We  use around 20 images to correct the distortion in our camera lenses. This distortion usually results in extension of lanes or shifitng of lines from desired position. So we need to calibrate camera to correct it before further processing. 
  We find the corners of the chessboard image. A corner is the intersection of 4 sqaures. The image has 9X6 corners in total. We generate the total corners and compare it with the detected corners to find out the camera matrix and distortion coefficient. </p>

```
//Move to the project directory.
Run project.py
The output matrices are stored in ./camera_cal/matrix.p

```
<h2> Distorted image </h2>

<img src="./" alt="distorted image"/>

<h2> Undistorted image after calibration </h2>

<img src="./" alt="undistorted image"/>

<p> We can see that there is a bulging effect at the bottom of the undistorted image which is being corrected by calibration of camera lenses </p>

<h3> Color and Gradient Thresholding </h3>

```
//Goto the project directory
Run color_gradient_threshold.py for tuning the threshold for vaious techniques.
Run hlsandgradfinalthresholding.py for complete version of the pipeline for thresholding and use the threshold values from previous tuning

```
<p> We apply various thresholding to image to detect only lanes. The Sobel operator which finds the corners is first performed along x axis, then along y axis. We then take the magnitude and direction of both x and y of the sobel operator. The threshold values are chosen based on experiment using the color_gradient_threshold.py. Now we apply color thresholding to the image to get finer distinction. The image is converted to HLS format and a threshold is selected for saturation. The final image is obtained by below combination of the various thresholding</p>

```
combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) |(s_binary == 1)] = 1

```

<img src="./" alt="thresholding_image"/>
<img src="./" alt="tuner_image"/>




