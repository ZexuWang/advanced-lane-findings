
# Advanced Lane Finding Project
---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.







## Import packages


```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
import glob
```

## Camera calibrition
---
In this section, the chessboard images provided will be un-distorted and the calibration matrix will be calculated.


```python
imgs = glob.glob('camera_cal/calibration*.jpg')

# arrays to store object points and image points from all the images
objpoints = [] 
imgpoints = []

# obj points preparation
objp = np.zeros((6*9,3),np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) # x,y coordinate 
for fname in imgs:
    img = mpimg.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,corners = cv2.findChessboardCorners(gray , (9,6), None)
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)
        #cv2.drawChessboardCorners(img, (9,6), corners,ret)
        #f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))
        #ax1.imshow(cv2.cvtColor(mpimg.imread(fname), cv2.COLOR_BGR2RGB))
        #ax1.set_title('Original Image', fontsize=18)
        #ax2.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        #ax2.set_title('With Corners', fontsize=18)

```

Here we will define a calculation of undistort function:


```python
def cal_undistort(img,read,plot):
    if read == True:
        img = mpimg.imread(img)
    img_size = (img.shape[1],img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    if plot == True:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24,9))
        f.tight_layout()
        ax1.imshow(img)
        ax1.set_title('Original Image',fontsize = 50)
        ax2.imshow(undist)
        ax2.set_title('Undistorted Image',fontsize = 50)
    return undist   
```

By using the objpoints and imgpoints matrix from the chessboard, we can calculate the distortion matrix. And we take the chessboard figure 'calibration2' as an example to undistort.


```python
img = mpimg.imread('camera_cal/calibration2.jpg')
calibration2 = cal_undistort(img,False,True)

```


![png](output_8_0.png)


# Pipeline construction

In this pipline, we will follow the course instruction's order to process the images:

* Undistort the original images
* Find the binary selection of the images by combined color and gradient threshold to the undistorted images
* Apply perspective transformation to the binary images
* Measure the curvature and fit a polynomial for the lane lines
* Plot out the found lane lines on the image/vedio


##   Undistort the test images

Now we use the algorithms we developed previously to undistort the test images 


```python
images = glob.glob('test_images/test*.jpg')
for image in images:
    undist = cal_undistort(image,True,True)
```


![png](output_10_0.png)



![png](output_10_1.png)



![png](output_10_2.png)



![png](output_10_3.png)



![png](output_10_4.png)



![png](output_10_5.png)


## Find the binary selections of the images

Here we combine the color and gradient threshold to get the best binary thresholded image.


```python
def combine_binay(img,read,display):
    if read == True:
        img = mpimg.imread(img)        
    
    
    img = np.copy(img)
#     # Convert to HLS color space and separate the V channel
#     hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
#     h_channel = hls[:,:,0]
#     l_channel = hls[:,:,1]
#     s_channel = hls[:,:,2]
#     sobel_kernel = 15
#     # direction selection
#     ang_thresh = (0.5,1)
#     sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0,ksize=sobel_kernel) # Take the derivative in x
#     abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
#     scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
#     sobely = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1,ksize=sobel_kernel) # Take the derivative in y
#     abs_sobely = np.absolute(sobely) # Absolute y derivative to accentuate lines away from horizontal
#     scaled_sobely = np.uint8(255*abs_sobely/np.max(abs_sobely))
#     direction = np.arctan2(abs_sobely, abs_sobelx)
#     direction_bi = np.zeros_like(direction)
#     direction_bi[(direction >= ang_thresh[0]) & (direction <= ang_thresh[1])] = 1

#     # Threshold x gradient
#     sx_thresh=(80, 100)
#     sxbinary = np.zeros_like(h_channel)
#     sxbinary[(scaled_sobelx >= sx_thresh[0]) & (scaled_sobelx <= sx_thresh[1])] = 1
    
#     # Threshold color channel
#     s_thresh=(220, 255)
#     s_binary = np.zeros_like(s_channel)
#     s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    


    
    # Convert to HLS color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h_channel = hsv[:,:,0]
    s_channel = hsv[:,:,1]
    v_channel = hsv[:,:,2]
    
    img = np.copy(img)   
    #Define parameters
    white_low = (0, 0, 233)
    white_high =(90, 63, 255)
    yellow_low = (15, 127, 213)
    yellow_high = (30, 255, 255)
    
    
    
    white_binary = cv2.inRange(hsv,white_low,white_high)
    yellow_binary = cv2.inRange(hsv,yellow_low,yellow_high)
#     white_binary[(h_channel >= white_low[0]) & (h_channel<= white_high[0])
#                    & (s_channel>= white_low[1]) & (s_channel<= white_high[1])
#                    & (v_channel>= white_low[2]) & (v_channel<= white_high[2])] = 1
#     yellow_binary = np.zeros_like(h_channel)
#     yellow_binary[(h_channel >= yellow_low[0]) & (h_channel<= yellow_high[0])
#                    & (s_channel>= yellow_low[1]) & (s_channel<= yellow_high[1])
#                    & (v_channel>= yellow_low[2]) & (v_channel<= yellow_high[2])] = 1
#     combine_bi = np.zeros_like(h_channel)
    combine_bi = cv2.bitwise_or(white_binary,yellow_binary)
#     combine_bi[(s_binary ==1) | (sxbinary ==1)] =1
    if display == True:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24,9))
        f.tight_layout()
        ax1.imshow(cal_undistort(img,False,False))
        ax1.set_title('Undistorted Image',fontsize = 50)
        ax2.imshow(combine_bi, cmap='gray')
        ax2.set_title('Selected Binary', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    return combine_bi
```


```python
image = mpimg.imread('test_images/test1.jpg')
binary = combine_binay(image,False,True)
```


![png](output_13_0.png)



```python
images = glob.glob('test_images/test*.jpg')
for image in images:
    binary = combine_binay(image,True,True)
```


![png](output_14_0.png)



![png](output_14_1.png)



![png](output_14_2.png)



![png](output_14_3.png)



![png](output_14_4.png)



![png](output_14_5.png)


## Perspective transformation



```python
def corners_unwarp(img,read,display,inv):
    if read == True:
        img = mpimg.imread(img)
    src = np.float32(
        [[500,480],
         [800,480],
         [1200,700],
         [100,700]])
    dst = np.float32(
        [[0,0],
         [1200,0],
         [1200,700],
         [0,700]])
    img_size = (img.shape[1],img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    warped_bi = combine_binay(warped,False,False)
    if display == True:
        f, (ax1,ax2) = plt.subplots(1,2,figsize = (20,10))
        ax1.set_title('Original')
        ax1.imshow(img)
        ax2.set_title('Wraped')
        ax2.imshow(warped,cmap='gray')
    if inv == True:
        return M_inv,warped_bi, warped
    else:
        return warped_bi, warped
```


```python
images = glob.glob('test_images/test*.jpg')
for image in images:
    binary,warped = corners_unwarp(image,True,True,False)
```


![png](output_17_0.png)



![png](output_17_1.png)



![png](output_17_2.png)



![png](output_17_3.png)



![png](output_17_4.png)



![png](output_17_5.png)


## Lane finding using peaks in a histogram

### Finding the lane line with fitted curves


```python
def fit_polynomial(img):
    M_inv,binary_warped,warped = corners_unwarp(img,False,False,True)
    binary_all = combine_binay(img,False,False)
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Check or not the windows
    w_check = False
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin   # Update this
        win_xleft_high = leftx_current + margin  # Update this
        win_xright_low = rightx_current - margin  # Update this
        win_xright_high = rightx_current + margin  # Update this
        
        if w_check == True:
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),
            (win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),
            (win_xright_high,win_y_high),(0,255,0), 2) 
        
        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    dots = False

    left_fit = np.polyfit(lefty,leftx,2)
    right_fit = np.polyfit(righty,rightx,2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        y_eval = np.max(ploty)
        y_eval = y_eval*ym_per_pix
        left_fit_real = np.polyfit(lefty*ym_per_pix,leftx*xm_per_pix,2)
        right_fit_real = np.polyfit(righty*ym_per_pix,rightx*xm_per_pix,2)
        left_curverad = ((1 + (2*left_fit_real[0]*y_eval + left_fit_real[1])**2)**1.5) / np.absolute(2*left_fit_real[0])
        right_curverad = ((1 + (2*right_fit_real[0]*y_eval + right_fit_real[1])**2)**1.5) / np.absolute(2*right_fit_real[0])
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty
        
        
    # Find the position of the vehicle
    start_ind = binary_warped.shape[0]-1
    left_start = left_fitx[start_ind]
    right_start = right_fitx[start_ind]
    center_pos = (left_start+right_start)/2
    offset = 640 - center_pos
        
    # Create an image to draw the lines on    
    warp_zero = np.zeros_like(binary_all).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.flipud(np.transpose(np.vstack([left_fitx, ploty])))])
    pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.polylines(color_warp, np.int_([pts]), isClosed=False, color=(0,0,255), thickness = 40)
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, M_inv, (binary_all.shape[1], binary_all.shape[0]))
    
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.5, 0)
    
    f, (ax1, ax2) = plt.subplots(1,2, figsize=(9, 6))
    f.tight_layout()
    ax1.imshow(warped)
    ax1.set_xlim(0, 1280)
    ax1.set_ylim(0, 720)
    ax1.plot(left_fitx, ploty, color='green', linewidth=3)
    ax1.plot(right_fitx, ploty, color='green', linewidth=3)
    ax1.set_title('Top view of the Lane line', fontsize=16)
    ax1.invert_yaxis() # to visualize as we do the images
    ax2.imshow(result)
    ax2.set_title('Normal View of the lane line', fontsize=16)
    ax2.text(100,100, 'Radius of left lane curvature is : % .1f m' %left_curverad
            , color='white', fontsize=10)
    ax2.text(100,150, 'Radius of right lane curvature is : % .1f m' %right_curverad
            , color='white', fontsize=10)
    if offset > 0:
        ax2.text(100,200, 'The vehicle is : % .1f m left to the center' %np.absolute(offset)
            , color='white', fontsize=10)
    elif offset < 0:
        ax2.text(100,200, 'The vehicle is : % .1f m right to the center' %np.absolute(offset)
            , color='white', fontsize=10)
    else:
        ax2.text(100,200, 'The vehicle is at the center'
            , color='white', fontsize=10)
```


```python
images = glob.glob('test_images/test*.jpg')
for image in images:
    img = mpimg.imread(image)
    fit_polynomial(img)
```


![png](output_21_0.png)



![png](output_21_1.png)



![png](output_21_2.png)



![png](output_21_3.png)



![png](output_21_4.png)



![png](output_21_5.png)


# Vedio Pipline


```python
from moviepy.editor import VideoFileClip
from IPython.display import HTML
```


```python
def vedio(img):
    M_inv,binary_warped,warped = corners_unwarp(img,False,False,True)
    binary_all = combine_binay(img,False,False)
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Check or not the windows
    w_check = False
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin   # Update this
        win_xleft_high = leftx_current + margin  # Update this
        win_xright_low = rightx_current - margin  # Update this
        win_xright_high = rightx_current + margin  # Update this
        
        if w_check == True:
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),
            (win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),
            (win_xright_high,win_y_high),(0,255,0), 2) 
        
        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    dots = False
    left_fit = np.polyfit(lefty,leftx,2)
    right_fit = np.polyfit(righty,rightx,2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        y_eval = np.max(ploty)
        y_eval = y_eval*ym_per_pix
        left_fit_real = np.polyfit(lefty*ym_per_pix,leftx*xm_per_pix,2)
        right_fit_real = np.polyfit(righty*ym_per_pix,rightx*xm_per_pix,2)
        left_curverad = ((1 + (2*left_fit_real[0]*y_eval + left_fit_real[1])**2)**1.5) / np.absolute(2*left_fit_real[0])
        right_curverad = ((1 + (2*right_fit_real[0]*y_eval + right_fit_real[1])**2)**1.5) / np.absolute(2*right_fit_real[0])
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty
        
        
    # Find the position of the vehicle
    start_ind = binary_warped.shape[0]-1
    left_start = left_fitx[start_ind]
    right_start = right_fitx[start_ind]
    center_pos = (left_start+right_start)/2
    offset = 640 - center_pos
        
    # Create an image to draw the lines on    
    warp_zero = np.zeros_like(binary_all).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.flipud(np.transpose(np.vstack([left_fitx, ploty])))])
    pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.polylines(color_warp, np.int_([pts]), isClosed=False, color=(0,0,255), thickness = 40)
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, M_inv, (binary_all.shape[1], binary_all.shape[0]))
    
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.5, 0)
    return result
```


```python
white_output = 'challenge_out.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(vedio).subclip(0,5)
%time white_clip.write_videofile(white_output,audio = False)
```

    [MoviePy] >>>> Building video challenge_out.mp4
    [MoviePy] Writing video challenge_out.mp4


    
      0%|          | 0/126 [00:00<?, ?it/s][A
      2%|▏         | 2/126 [00:00<00:13,  9.24it/s][A
      2%|▏         | 3/126 [00:00<00:13,  9.45it/s][A
      4%|▍         | 5/126 [00:00<00:12,  9.67it/s][A
      5%|▍         | 6/126 [00:00<00:12,  9.54it/s][A
      6%|▌         | 7/126 [00:00<00:12,  9.35it/s][A
      6%|▋         | 8/126 [00:00<00:12,  9.32it/s][A
      7%|▋         | 9/126 [00:00<00:12,  9.30it/s][A
      8%|▊         | 10/126 [00:01<00:12,  9.40it/s][A
      9%|▊         | 11/126 [00:01<00:12,  9.25it/s][A
     10%|▉         | 12/126 [00:01<00:12,  9.25it/s][A
     10%|█         | 13/126 [00:01<00:12,  9.36it/s][A
     11%|█         | 14/126 [00:01<00:12,  9.23it/s][A
     12%|█▏        | 15/126 [00:01<00:12,  9.24it/s][A
     13%|█▎        | 16/126 [00:01<00:11,  9.24it/s][A
     13%|█▎        | 17/126 [00:01<00:11,  9.15it/s][A
     14%|█▍        | 18/126 [00:01<00:11,  9.18it/s][A
     15%|█▌        | 19/126 [00:02<00:11,  9.30it/s][A
     16%|█▌        | 20/126 [00:02<00:11,  9.49it/s][A
     17%|█▋        | 21/126 [00:02<00:11,  9.43it/s][A
     17%|█▋        | 22/126 [00:02<00:11,  9.38it/s][A
     18%|█▊        | 23/126 [00:02<00:11,  9.24it/s][A
     19%|█▉        | 24/126 [00:02<00:11,  9.25it/s][A
     20%|█▉        | 25/126 [00:02<00:10,  9.25it/s][A
     21%|██        | 26/126 [00:02<00:10,  9.36it/s][A
     21%|██▏       | 27/126 [00:02<00:10,  9.53it/s][A
     22%|██▏       | 28/126 [00:02<00:10,  9.25it/s][A
     23%|██▎       | 29/126 [00:03<00:10,  9.35it/s][A
     24%|██▍       | 30/126 [00:03<00:10,  9.33it/s][A
     25%|██▌       | 32/126 [00:03<00:09,  9.57it/s][A
     27%|██▋       | 34/126 [00:03<00:09,  9.81it/s][A
     29%|██▊       | 36/126 [00:03<00:09,  9.92it/s][A
     29%|██▉       | 37/126 [00:03<00:09,  9.61it/s][A
     31%|███       | 39/126 [00:04<00:08,  9.78it/s][A
     33%|███▎      | 41/126 [00:04<00:08,  9.90it/s][A
     34%|███▍      | 43/126 [00:04<00:11,  7.04it/s][A
     35%|███▍      | 44/126 [00:04<00:12,  6.62it/s][A
     36%|███▌      | 45/126 [00:05<00:13,  6.11it/s][A
     37%|███▋      | 46/126 [00:05<00:13,  5.87it/s][A
     37%|███▋      | 47/126 [00:05<00:13,  5.86it/s][A
     38%|███▊      | 48/126 [00:05<00:13,  5.83it/s][A
     39%|███▉      | 49/126 [00:05<00:13,  5.73it/s][A
     40%|███▉      | 50/126 [00:06<00:13,  5.81it/s][A
     40%|████      | 51/126 [00:06<00:13,  5.68it/s][A
     41%|████▏     | 52/126 [00:06<00:13,  5.66it/s][A
     42%|████▏     | 53/126 [00:06<00:13,  5.58it/s][A
     43%|████▎     | 54/126 [00:06<00:13,  5.40it/s][A
     44%|████▎     | 55/126 [00:06<00:12,  5.51it/s][A
     44%|████▍     | 56/126 [00:07<00:12,  5.39it/s][A
     45%|████▌     | 57/126 [00:07<00:13,  5.21it/s][A
     46%|████▌     | 58/126 [00:07<00:12,  5.40it/s][A
     47%|████▋     | 59/126 [00:07<00:12,  5.39it/s][A
     48%|████▊     | 60/126 [00:07<00:12,  5.29it/s][A
     48%|████▊     | 61/126 [00:08<00:12,  5.32it/s][A
     49%|████▉     | 62/126 [00:08<00:11,  5.39it/s][A
     50%|█████     | 63/126 [00:08<00:11,  5.45it/s][A
     51%|█████     | 64/126 [00:08<00:11,  5.44it/s][A
     52%|█████▏    | 65/126 [00:08<00:11,  5.36it/s][A
     52%|█████▏    | 66/126 [00:08<00:10,  5.53it/s][A
     53%|█████▎    | 67/126 [00:09<00:10,  5.39it/s][A
     54%|█████▍    | 68/126 [00:09<00:10,  5.40it/s][A
     55%|█████▍    | 69/126 [00:09<00:10,  5.39it/s][A
     56%|█████▌    | 70/126 [00:09<00:10,  5.46it/s][A
     56%|█████▋    | 71/126 [00:09<00:10,  5.39it/s][A
     57%|█████▋    | 72/126 [00:10<00:10,  5.40it/s][A
     58%|█████▊    | 73/126 [00:10<00:10,  5.15it/s][A
     59%|█████▊    | 74/126 [00:10<00:09,  5.27it/s][A
     60%|█████▉    | 75/126 [00:10<00:09,  5.28it/s][A
     60%|██████    | 76/126 [00:10<00:09,  5.19it/s][A
     61%|██████    | 77/126 [00:11<00:09,  5.33it/s][A
     62%|██████▏   | 78/126 [00:11<00:08,  5.37it/s][A
     63%|██████▎   | 79/126 [00:11<00:08,  5.37it/s][A
     63%|██████▎   | 80/126 [00:11<00:08,  5.39it/s][A
     64%|██████▍   | 81/126 [00:11<00:08,  5.46it/s][A
     65%|██████▌   | 82/126 [00:11<00:08,  5.49it/s][A
     66%|██████▌   | 83/126 [00:12<00:07,  5.50it/s][A
     67%|██████▋   | 84/126 [00:12<00:07,  5.43it/s][A
     67%|██████▋   | 85/126 [00:12<00:07,  5.55it/s][A
     68%|██████▊   | 86/126 [00:12<00:07,  5.42it/s][A
     69%|██████▉   | 87/126 [00:12<00:07,  5.34it/s][A
     70%|██████▉   | 88/126 [00:13<00:07,  5.16it/s][A
     71%|███████   | 89/126 [00:13<00:06,  5.42it/s][A
     71%|███████▏  | 90/126 [00:13<00:06,  5.35it/s][A
     72%|███████▏  | 91/126 [00:13<00:06,  5.30it/s][A
     73%|███████▎  | 92/126 [00:13<00:06,  5.49it/s][A
     74%|███████▍  | 93/126 [00:14<00:06,  5.44it/s][A
     75%|███████▍  | 94/126 [00:14<00:05,  5.52it/s][A
     75%|███████▌  | 95/126 [00:14<00:05,  5.61it/s][A
     76%|███████▌  | 96/126 [00:14<00:05,  5.59it/s][A
     77%|███████▋  | 97/126 [00:14<00:05,  5.43it/s][A
     78%|███████▊  | 98/126 [00:14<00:05,  5.43it/s][A
     79%|███████▊  | 99/126 [00:15<00:04,  5.43it/s][A
     79%|███████▉  | 100/126 [00:15<00:04,  5.43it/s][A
     80%|████████  | 101/126 [00:15<00:04,  5.40it/s][A
     81%|████████  | 102/126 [00:15<00:04,  5.44it/s][A
     82%|████████▏ | 103/126 [00:15<00:04,  5.28it/s][A
     83%|████████▎ | 104/126 [00:16<00:04,  5.33it/s][A
     83%|████████▎ | 105/126 [00:16<00:03,  5.39it/s][A
     84%|████████▍ | 106/126 [00:16<00:03,  5.44it/s][A
     85%|████████▍ | 107/126 [00:16<00:03,  5.33it/s][A
     86%|████████▌ | 108/126 [00:16<00:03,  5.53it/s][A
     87%|████████▋ | 109/126 [00:16<00:03,  5.40it/s][A
     87%|████████▋ | 110/126 [00:17<00:03,  5.31it/s][A
     88%|████████▊ | 111/126 [00:17<00:02,  5.44it/s][A
     89%|████████▉ | 112/126 [00:17<00:02,  5.23it/s][A
     90%|████████▉ | 113/126 [00:17<00:02,  5.30it/s][A
     90%|█████████ | 114/126 [00:17<00:02,  5.17it/s][A
     91%|█████████▏| 115/126 [00:18<00:02,  5.38it/s][A
     92%|█████████▏| 116/126 [00:18<00:01,  5.32it/s][A
     93%|█████████▎| 117/126 [00:18<00:01,  5.51it/s][A
     94%|█████████▎| 118/126 [00:18<00:01,  5.54it/s][A
     94%|█████████▍| 119/126 [00:18<00:01,  5.44it/s][A
     95%|█████████▌| 120/126 [00:19<00:01,  5.29it/s][A
     96%|█████████▌| 121/126 [00:19<00:00,  5.28it/s][A
     97%|█████████▋| 122/126 [00:19<00:00,  5.19it/s][A
     98%|█████████▊| 123/126 [00:19<00:00,  5.38it/s][A
     98%|█████████▊| 124/126 [00:19<00:00,  5.39it/s][A
     99%|█████████▉| 125/126 [00:19<00:00,  5.42it/s][A
    [A

    [MoviePy] Done.
    [MoviePy] >>>> Video ready: challenge_out.mp4 
    
    CPU times: user 10.2 s, sys: 233 ms, total: 10.4 s
    Wall time: 23 s



```python
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))
```





<video width="960" height="540" controls>
  <source src="challenge_out.mp4">
</video>




# Discussions

The lane detection pipline I developed works fine in the image finding. While in the video stream, it seems like the threshold value set for the binary selection is too strict and when the car comes to some place with limited light or the lane lines are shadowed by some obstacles (tree for example), the pipline will not work properly. To match the deadline as close as possible, I skip the part of tracking, sanity check, look-ahead filter, reset and smoothing. Since I'm a bit unfamiliar with oop, this remaining work will be continued with some more looks into the object oriented programming. 


```python
# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None 
```
