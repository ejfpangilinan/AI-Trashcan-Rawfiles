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

def prep_img(frame):
    
    if frame is None:
        return
    
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
    fgbg = cv2.createBackgroundSubtractorKNN(detectShadows=False)
    
    fgmask = fgbg.apply(frame)
    
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    cnt = 0
    cropped = None 
    
    print(type(fgmask))
    
    edges = cv2.Canny(fgmask,100,200)
    gray = np.float32(edges)
    corners = cv2.goodFeaturesToTrack(gray,100,0.01,10)

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
    
    cropped = frame[miny:maxy-miny, minx:maxx-minx]
    
    return cropped


#Prepare Data
VALIDATION_DIR = './TEST'


model = load_model('./rps.h5')
mylist = [f for f in glob.glob(VALIDATION_DIR+"/B/*.jpg")]
mylist1 = [f for f in glob.glob(VALIDATION_DIR+"/N/*.jpg")]

mylist = mylist+mylist1

random.shuffle(mylist)
start = 1
while start:
    print("----------------------------")
    print("MENU")
    print("----------------------------")
    print("[1] Take Photo")
    print("[2] Open Video Classify")
    print("[3] Random Classify")
    print("[0] Exit")
    print("----------------------------")
    choice = int(input("Enter Choice: "))

    if choice==1:
        cam = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        
        result, pic = cam.read()

        if result:
            cv2.normalize(pic, pic, 0, 255, cv2.NORM_MINMAX)
            
            cv2.imwrite("./output.png", pic)

            
            img = image.load_img('output.png', target_size=(200, 200))
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

            img = mpimg.imread('./output.png')
            imgplot = plt.imshow(img)

            
            plt.show(block=False)
            plt.pause(5)
            plt.close('all')
            
            
            
            
            # window
            cv2.waitKey(0)
            cam.release()
            cv2.destroyAllWindows()
          
        # If captured image is corrupted, moving to else part
        else:
            print("No image detected. Please! try again")
            
    elif choice == 2:
        
        previous = time.time()
        delta = 0
        cam = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        text = "Initial"
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        while(True):
            
            current = time.time()
            delta += current - previous
            previous = current
            
            if delta > 2 :
                result, pic = cam.read()
                
                cv2.normalize(pic, pic, 0, 255, cv2.NORM_MINMAX)
                
                cv2.imwrite("./output.png", pic)
                
                img = image.load_img('output.png', target_size=(200, 200))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)

                images = np.vstack([x])
                classes = model.predict(images, batch_size=10)

                if classes[0][0]<0.5:
                    text = "Biodegradable"
                    # plt.title("Biodegradable")
                    print(classes,": Biodegradable")
                else:
                    text = "Non-Biodegradable"
                    # plt.title("Non-Biodegradable")
                    print(classes,": Non-Biodegradable")
                    
                
                delta = 0
            
            ret, frame = cam.read()
            cv2.putText(frame,
                text,
                (250, 400),
                font, 1,
                (0, 200, 200),
                2,
                cv2.LINE_4)
  
                # Display the resulting frame
            cv2.imshow('frame', frame)
            
            
                # the 'q' button is set as the
                # quitting button you may use any
                # desired button of your choice
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # After the loop release the cap object
        cam.release()
        # Destroy all the windows
        cv2.destroyAllWindows()
                
                
    elif choice == 3:
        
        for i in range(10):
            fn = random.randrange(0,len(mylist1))
            path = mylist[fn]



            img = image.load_img(path, target_size=(200, 200))
            
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)

            images = np.vstack([x])
            classes = model.predict(images, batch_size=10)

            if classes[0][0]<0.5:
                if mylist[fn][11]=='B':
                    plt.title("Biodegradable - Correct")
                else:
                    plt.title("Biodegradable - Wrong")
                print(mylist[fn],":",classes,": Biodegradable")
            else:
                if mylist[fn][11]=='N':
                    plt.title(" Non-Biodegradable - Correct")
                else:
                    plt.title(" Non-Biodegradable - Wrong")
                print(mylist[fn],":",classes,": Non-Biodegradable")

            img = mpimg.imread(path)
            imgplot = plt.imshow(img)

            
            plt.show(block=False)
            plt.pause(2)
            plt.close('all')
            
    elif choice == 0:
        print("BYE")
        start=0
        try:
            cam.release()
            cv2.destroyAllWindows()
        except:
            pass
        break
    else:
        print('Invalid input!')





# for i in range(20):
#     fn = random.randrange(0,len(mylist1))
#     path = mylist[fn]



#     img = image.load_img(path, target_size=(200, 200))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)

#     images = np.vstack([x])
#     classes = model.predict(images, batch_size=10)

#     if classes[0][0]<0.5:
#         plt.title("Biodegradable")
#         print(mylist[fn],":",classes,": Biodegradable")
#     else:
#         plt.title("Non-Biodegradable")
#         print(mylist[fn],":",classes,": Non-Biodegradable")

#     img = mpimg.imread(path)
#     imgplot = plt.imshow(img)

    
#     plt.show(block=False)
#     plt.pause(2)
#     plt.close('all')