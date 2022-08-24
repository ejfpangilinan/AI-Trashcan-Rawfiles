from tkinter import filedialog
import numpy as np
from keras.preprocessing import image
from keras.models import load_model

from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import random


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import glob
import math


import cv2
from cvzone.SelfiSegmentationModule import SelfiSegmentation

seg = SelfiSegmentation()


#Prepare Data
VALIDATION_DIR = './VALIDATE'


model = load_model('./rps.h5')


cam = cv2.VideoCapture(0)


# result, bg_image = cam.read()

# bg_image = np.zeros([480,640,3],dtype=np.uint8)
# bg_image.fill(255)

previous = time.time()
delta = 0


while True:
    
    current = time.time()
    delta += current - previous
    previous = current
    
    ret, frame = cam.read()

    if frame is None:
        break

    grayscaleImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, threshedImage = cv2.threshold(grayscaleImage, 75, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    ## Find the contours with biggest area
    contours, _ = cv2.findContours(threshedImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    maxContour = max(contours, key=cv2.contourArea)
    ## Create mask
    mask = np.zeros(frame.shape[:2], np.uint8)
    cv2.drawContours(mask, [maxContour], -1, 255, -1)
    fgmask = np.zeros(frame.shape[:3], np.uint8)
    height = frame.shape[:2][1]
    fgmask[:, 0:height] = (255, 255, 255)

    locations = np.where(mask != 0)
    fgmask[locations[0], locations[1]] = frame[locations[0], locations[1]]

    # fgmask = seg.removeBG(frame,(255,255,255))

    try:
        edges = cv2.Canny(fgmask,150,300)

        gray = np.float32(edges)
        corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
        corners = np.int0(corners)

        x_pos = []
        y_pos = []

        for corner in corners:
            x,y = corner.ravel()
            x_pos.append(x)
            y_pos.append(y)
        
        minx = min(x_pos)
        miny = min(y_pos)
        maxx = max(x_pos)
        maxy = max(y_pos)

        text = "minx, miny"
        coordinates = (minx,miny)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (255,0,0)
        thickness = 1
        edges = cv2.putText(edges, text, coordinates, font, fontScale, color, thickness, cv2.LINE_AA)

        text = "minx, maxy"
        coordinates = (minx,maxy)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (255,0,0)
        thickness = 1
        edges = cv2.putText(edges, text, coordinates, font, fontScale, color, thickness, cv2.LINE_AA)

        text = "maxx, miny"
        coordinates = (maxx,miny)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (255,0,0)
        thickness = 1
        edges = cv2.putText(edges, text, coordinates, font, fontScale, color, thickness, cv2.LINE_AA)

        text = "maxx, maxy"
        coordinates = (maxx,maxy)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (255,0,0)
        thickness = 1
        edges = cv2.putText(edges, text, coordinates, font, fontScale, color, thickness, cv2.LINE_AA)


        object_start = (minx,miny)
        object_end = (maxx,maxy)

        cv2.rectangle(frame,object_start,object_end , (255, 0, 0), 2)
        cv2.rectangle(edges,object_start,object_end , (255, 255, 255), 2)

        cv2.imshow('Canny', edges)
        cv2.imshow('BG Remove', fgmask)
        cv2.imshow('Frame', frame)
        
        no_image = False
        
        keyboard = cv2.waitKey(30)

        if keyboard == ord('q') or keyboard == 27:
            cropped = fgmask[miny:maxy, minx:maxx]
            cv2.imshow('output.png',cropped)
            cv2.waitKey(0)
            break

    except Exception as e:
        no_image = True
        print("wait", e)

    # if delta>=10 and not no_image:
    #     try:
    #         cropped = fgmask[miny:maxy-miny, minx:maxx-minx]
    #         cv2.imshow('output.png',cropped)
    #         cv2.waitKey(0)
    #         break
    #     except Exception as e:
    #         print('redo', e)


cam.release()
cv2.destroyAllWindows()



# #CATEGORIZED
img = cv2.resize(cropped,(200,200))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict(images, batch_size=10)

if classes[0][0]<0.5:
    plt.title("Biodegradable")
    print(classes,": Biodegradable")
else:
    plt.title("Non-Biodegradable")
    print(classes,": Non-Biodegradable")

# img = mpimg.imread('./output.png')
RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

imgplot = plt.imshow(RGB_img)

plt.show(block=False)
plt.pause(5)
plt.close('all')
