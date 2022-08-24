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



#Prepare Data
VALIDATION_DIR = './VALIDATE'


model = load_model('./rps.h5')


cam = cv2.VideoCapture(0)

# result, pic = cam.read()

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
fgbg = cv2.createBackgroundSubtractorKNN(detectShadows=False)

previous = time.time()
delta = 0


while True:
    
    current = time.time()
    delta += current - previous
    previous = current
    
    ret, frame = cam.read()

    if frame is None:
        break
    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

    try:
        edges = cv2.Canny(fgmask,100,200)

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

        
        object_start = (minx,miny)
        object_end = (maxx,maxy)

        cv2.rectangle(frame,object_start,object_end , (255, 0, 0), 2)

        cv2.imshow('Canny', edges)
        cv2.imshow('Frame', frame)
        cv2.imshow('FG MASK Frame', fgmask)
        
        no_image = False
        
        keyboard = cv2.waitKey(30)

        if keyboard == 'q' or keyboard == 27:
            break

    except Exception as e:
        no_image = True
        print("wait", e)

    if delta>=10 and not no_image:
        try:
            cropped = frame[miny:maxy-miny, minx:maxx-minx]
            cv2.imshow('output.png',cropped)
            cv2.waitKey(0)
            break
        except Exception as e:
            print('redo', e)


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
