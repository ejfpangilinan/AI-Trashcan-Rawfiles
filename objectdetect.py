import numpy as np
import cv2
from matplotlib import pyplot as plt
import math

cam = cv2.VideoCapture(0)

bg_image = np.zeros([480,640,3],dtype=np.uint8)
bg_image.fill(255)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

# fgbg = cv2.createBackgroundSubtractorKNN(detectShadows=False)
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
# fgbg = cv2.createBackgroundSubtractorMOG2()
# fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()

while True:
    ret, frame = cam.read()
    
    if frame is None:
        break
    
    fgmask = fgbg.apply(frame)
    fgmask = cv2.medianBlur(fgmask,5)
    # # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)


    try:
        edges = cv2.Canny(fgmask,100,200)

        gray = np.float32(edges)
        corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
        corners = np.int0(corners)

        minx = math.inf
        miny = math.inf
        maxx = 0
        maxy = 0
        for corner in corners:
            x,y = corner.ravel()
            maxx = max(x,maxx)
            maxy = max(y,maxx)
            minx = min(x,minx)
            miny = min(y,miny)

        object_start = (minx,miny)
        object_end = (maxx,maxy)
        
        cv2.rectangle(frame,object_start,object_end , (255, 0, 0), 2)

        cv2.imshow('Canny', edges)
        cv2.imshow('Frame', frame)
        cv2.imshow('FG MASK Frame', fgmask)

    except Exception as e:
        print("wait", e)

    keyboard = cv2.waitKey(30)
    
    if keyboard == ord('q') or keyboard == 27:
        break
    
cam.release()
cv2.destroyAllWindows()