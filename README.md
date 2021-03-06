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
<h4> Distorted image </h4>

<img src="or3.JPG" alt="distorted image"/>

<h4> Undistorted image after calibration </h4>

<img src="un3.JPG" alt="undistorted image"/>

<p> We can see that there is a bulging effect at the bottom of the undistorted image which is being corrected by calibration of camera lenses </p>

<h3> Color and Gradient Thresholding </h3>

```
//Goto the project directory
Run color_gradient_threshold.py for tuning the threshold for vaious techniques.
Run hlsandgradfinalthresholding.py for complete version of the pipeline for thresholding and use the threshold values from previous tuning

```
<p> We apply various thresholding to image to detect only lanes. The Sobel operator which finds the corners is first performed along x axis, then along y axis. We then take the magnitude and direction of both x and y of the sobel operator. The threshold values are chosen based on experiment using the color_gradient_threshold.py. Now we apply color thresholding to the image to get finer distinction. The image is converted to HLS format and a threshold is selected for saturation. The final image is obtained by below combination of the various thresholding.To add more accuracy in shaded areas, the image is converted in YUV format and the luminiscence value is tuned.</p>

```
combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) |(s_binary == 1)&(luminiscence==1)] = 1

```

<img src="./thresh1.JPG" alt="thresholding_image"/>
<img src="./thresh.JPG" alt="tuner_image"/>

<h3> Perspective Transform </h3>

<p> To detect the curvature of the a line we cannot directly determine with the image from the camera. These images may appear curved even when they are lot for points in long distance from camera. So we use a perspective transform to solve this. We get a bird's eye view of the image after the transform. We test the transformtion using the same chess board image. But this time the object is away from the center of the image. The perspective warps this and brings the image to the center. We also use Bilinear interpolation to to fill up the areas while transforming </p>

```
Goto project directory
Run prespectvetransform.py
```
<h4> Image where object is facing different angle and not camera </h4>
<img src="ptun.JPG" alt="image at corner"/>

<h4> Perspective transform output </h4>
<img src="ptfinal.JPG" alt="perspective transformed"/>

<h3> Final Pipeline </h3>

```
Go to project directory
For detecting lanes in images
Run Pipeline_for_images.py
For detecting lanes in videos
Run Pipeline_for_videos.py

The output is stored in ./output_images

```

<p> We combine all the techniques and detect lanes in the videos . We read the image and set the kernel size as 3 for all the gradient thresholding. The image is undistorted using the camera matrix and distortion coefficients in ./camera_cal/matrix.p pickle file. The sobel along x & y is performed. Then magnitude and gradient is computed. Finally the image is converted to HLS and Saturation values are adjusted. All these are combined to get the final image. </p>

<h5> Perspective transform and Polynomial Fitting </h5>

<p> The trapezoidal region of the warped image is chosen as the source. The sources covers the left and right lanes with slight margin to fit for different images. Destination points are chosen to transform the source points to a rectangular region in the transformed image</p>

<p>Now we fit a second polynomial for the lanes. The histogram of the image is taken at the bottom half of the image. The magnitude of the histogram at the lanes is very high and so we choose that as the point of start. </p>

<h6> Histogram image </h6>

<img src="hist.JPG" alt="histogram"/>

<p> Sometimes in shady or bright areas we may have additional peaks due to thresholding which might affect the lane detection. So to suppress the additional unwanted peaks other than lanes we set weights to the histogram areas,since we know the location of lanes. The weights will amplify the peaks at the location of lanes and will suppress other unwanted noises. We use the below weight filter for this purpose.</p>

<h6> Histogram Weights </h6>

<img src="hisoweights.JPG" alt="histogramweights"/>

<p> We use sliding window to find out the lanes along y direction since x direction lanes don't change much we fit a polynomial for y direction. We move the sliding window in y direction and find out the maximum in the left half and right half of the histogram for each window. A selected margin is used to make a box around the lane. If the box has more lane pixels than minimum pixel we then recenter the box along x for the next window. After that select the corrseponding left lane and right lanes x,y coordinates. Now we use the polyfit function to detect both lanes in the image.(y=a*x^2+b*x+c) </p>

<h6> Source points in the image for perspective transform </h6>

<img src="trapezoidal.JPG" alt="trapezoidal"/>

<h6> Perspective transform of road </h6>

<img src="perspective.JPG" alt="perspective"/>

<h6> polynomial fitting </h6>

<img src="finall.JPG" alt="finall"/>

<h6>Erosion and dilation </h6>

<p> Due to thresolding operations,sometimes we might encounter high level of discontinuity in bright areas. This might cause the polyfit to fit noise regions and disrupt the lane detection. So to avoid this we use the dilation and erosion approach. We use a vertical kernel since our lanes are vertical of size 320 X 1 and perform dilation first followed by erosion for one interation of the perspective transformed image before feeding to polyfit as this connects discontinuities in lane regions. </p>

<img src="erosiondilation.JPG" alt="erosiondilation"/>

<h6> Sanity checks </h6>

<p> Not always the lane detected will be accurate and so we need to perform validation step each time we fit the polynomial. As we scan the region around the detected lane to find out the new lanes there might be lesser chances of error,but thresholding noises may cause error in detection. So we validate by comparing the polynomial fitting coefficients which are quadratic here. Whenever the ratio of corresponding polynomial coefficients becomes greater than threshold(which is determined by tuning) we use the polynomial ceofficients of the previous frame and when the error keeps on occuring we perform lane detection through the entire sliding window approach and not limiting to the areas of previous detected lane.</p>

<h5> Measuring the radius of curvature </h5>

<img src="rad.JPG" alt="rad"/>
<p>
The radius of curvature measure here is based on pixel value. To convert to real word numbers we need to do some conversion. The lane is about 30 meters long and 3.7 meters wide. Or, if you prefer to derive a conversion from pixel space to world space in your own images, compare your images with U.S. regulations that require a minimum lane width of 12 feet or 3.7 meters, and the dashed lane lines are 10 feet or 3 meters long each.</p>

<p>
  Low numbers imply strong curves, high numbers imply straight road. In theory, in case of a perfectly straight road the radius of curvature is infinite. In the practice, a number higher than 2-3000 means a pretty straight road, and the variance in the numbers is relatively high in case of straight lines. So having ~3500 and ~7100 for the two lane lines can be completely OK, it means that the road is more or less straight.
</p>

<p> Let's say that our camera image has 720 relevant pixels in the y-dimension, and we'll say roughly 700 relevant pixels in the x-dimension between 2 lanes from histogram. Therefore, to convert from pixels to real-world meter measurements, we can use: </p>

```
Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

```

<p> The key point here is that polynomial coefficients are in pixels value and not in metres to compute the curvature. So we need to convert them to metres based on the below relation.</p>

<img src="polyfitcoeffmetresconv.JPG" alt="polyfitcoeffmetresconv"/>

<h6> Locating the car's center </h6>

<UL>
  <LI>Assume that camera is placed at the center of the car - so image width divided on 2 is an actual car position.</LI>
  <LI>Actual lane center you can define if you summarize x-coordinates near the car for the left line and the x-coordinates for the right line and divide it by 2 (actual center of the lane based on determined left and right lines).</LI>
  <LI>After that, if you subtract value from item 1 from value from item 2 above you will receive position towards the center.</LI>
</UL>

<p> With lane width about 3.7m and average car width about 2m, offset can't be greater than 0.8-1m. </p>

<h5> Adjustment for video </h5> 

<p> Finally for a video we need to perform all the calculation for each frame. Since finding the lanes takes much time, we can make a margin around the previoulsy detected lanes and serach for lanes pixelx from there alone. We can also introduce a count variable and perform the lane detection from complete image after certain steps of the video. This helps to solve an miscalculation and wrong lane detection which might occur while searching in the margin of previous lane alone, if the previous lane is wrongly detected, the error may accumulate. So we recalculate the lanes after certain time frames again. </p>

<h6> Final Output Image </h6>

```
The video output is uploaded in drive with the link below

https://drive.google.com/file/d/1x4S4riasVGZzkHgzVnQdo20JWUjxCC4k/view?usp=sharing

```

<img src="finallout.JPG" alt="final image"/>

<h5> Problems and further improvements: </h5>

<p> The pipeline works very well for roads where there is no additional lighting. It detects clearly in areas when there is not much sunlight falling on the road. But during afternoon the detection fails becuase of the direct sunlight on the road, the lanes are lighted up more causing the yellow and white lanes less visible by the dominating sunlight. This is solved at the cost of processing time. The threshold values are tuned to minimize the effect,but the entire image region must be searched for each frame and we cannot approximate the pixels near the lane for finding the lanes in the next frame. A more spohisticated algorithm similar to semantic segmentation can be used to reduce the processing time and improve accuracy. </p>

