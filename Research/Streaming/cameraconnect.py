'''
Haar Cascade Face detection with OpenCV  
    Based on tutorial by pythonprogramming.net
    Visit original post: https://pythonprogramming.net/haar-cascade-face-eye-detection-python-opencv-tutorial/  
Adapted by Marcelo Rovai - MJRoBot.org @ 7Feb2018 
'''

import numpy as np
import cv2

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
#faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')

#cam = cv2.VideoCapture('/Imaging Devices/PS3Eye Camera')
cam = cv2.VideoCapture(0)
#cam = cv2.VideoCapture('rtsp://admin:Brabb2000@172.16.160.201:554/cam/realmonitor?stream=1')
#cam = cv2.VideoCapture('rtsp://172.16.160.202/user=admin&password=Brabb2000&channel=1&stream=0.sdp')
#cam.set(3,640) # set Width
#cam.set(4,480) # set Height

# Check if the webcam is opened correctly
if not cam.isOpened():
    raise IOError("Cannot open the camera")

cv2.namedWindow("Camera Connect Interface")

#img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame or connect to the camera")
        break
    cv2.imshow("Camera Connect Interface", frame)

    k = cv2.waitKey(1)
    if k%256 == 27: # press 'ESC' to quit
        print('ESC pressed and Video closed')
        break
    elif k%256 == 32: # press Space to take picture
        img_name = "Research/Streaming/opencv_frame-snapshot_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter +=1
    
cam.release()
cv2.destroyAllWindows()