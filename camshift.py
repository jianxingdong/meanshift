#!/usr/bin/python
import numpy as np
import cv2
from draw_rect import Rectangle

#define the window
cv2.namedWindow('image',cv2.CV_WINDOW_AUTOSIZE)

#define the VideoCapture for our video
cap = cv2.VideoCapture("../video_rec/video.avi")

# the user sets up the desired ROI
ret,frame = cap.read()

rectangle = Rectangle(frame)
cv2.setMouseCallback('image',rectangle.draw_rectangle)

while(1):
  cv2.imshow('image',rectangle.img_show)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

r,h,c,w = (rectangle.p1[1], rectangle.height, rectangle.p1[0], rectangle.width)  
track_window = (c,r,w,h)


# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]
hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

while(1):
    ret ,frame = cap.read()

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        # apply meanshift to get the new location
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)

        # Draw it on image
        pts = cv2.cv.BoxPoints(ret)
        pts = np.int0(pts)
        cv2.polylines(frame,[pts],True, 255,2)
        cv2.imshow('frame',frame)

        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break

    else:
        break

cv2.destroyAllWindows()
cap.release()
