import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import os
from cvzone.SelfiSegmentationModule import SelfiSegmentation

seg = SelfiSegmentation()

cam = cv2.VideoCapture(0)

bg_image = np.zeros([480,640,3],dtype=np.uint8)
bg_image.fill(255)

while True:
    ret, frame = cam.read()
    height , width, channel = frame.shape
    
    if frame is None:
        break
    
    imgout = seg.removeBG(frame,(255,255,255))
    
    try:
        cv2.imshow('remove bg',imgout)
        cv2.imshow('original',frame)
        
    except Exception as e:
        print("wait", e)



    keyboard = cv2.waitKey(30)

    if keyboard == ord('q') or keyboard == 27:
        print(height,width,channel)
        break
    


cam.release()
cv2.destroyAllWindows()