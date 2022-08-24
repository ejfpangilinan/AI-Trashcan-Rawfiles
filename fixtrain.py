import numpy as np
import cv2
import os
import glob
import random

from PIL import Image

TRAINING_DIR = './TRAINING'

ransam = random.sample(range(540),200)
print(ransam)

ctt =0
ctv = 0

for enum , im in enumerate(glob.glob("./TRAINING/B/*")):
    if enum in ransam and ctt<140:
        im = Image.open(im)
        newsize = (200, 200)
        im = im.resize(newsize)
        rgb_im = im.convert('RGB')
        rgb_im.save("./VALIDATE/B/B-"+str(ctt).zfill(4)+".jpg")
        ctt+=1
    elif enum in ransam and ctt>=140 and ctt<200:
        im = Image.open(im)
        newsize = (200, 200)
        im = im.resize(newsize)
        rgb_im = im.convert('RGB')
        rgb_im.save("./TRAINSET/B/B-"+str(ctv).zfill(4)+".jpg")
        rgb_im.save("./VALIDATE/B/B-"+str(ctt).zfill(4)+".jpg")
        ctt+=1
        ctv+=1
    else:
        im = Image.open(im)
        newsize = (200, 200)
        im = im.resize(newsize)
        rgb_im = im.convert('RGB')
        rgb_im.save("./TRAINSET/B/B-"+str(ctv).zfill(4)+".jpg")
        ctv+=1

ctt =0
ctv = 0
for enum , im in enumerate(glob.glob("./TRAINING/N/*")):
    if enum in ransam and ctt<140:
        im = Image.open(im)
        newsize = (200, 200)
        im = im.resize(newsize)
        rgb_im = im.convert('RGB')
        rgb_im.save("./VALIDATE/N/N-"+str(ctt).zfill(4)+".jpg")
        ctt+=1
    elif enum in ransam and ctt>=140 and ctt<200:
        im = Image.open(im)
        newsize = (200, 200)
        im = im.resize(newsize)
        rgb_im = im.convert('RGB')
        rgb_im.save("./TRAINSET/N/N-"+str(ctv).zfill(4)+".jpg")
        rgb_im.save("./VALIDATE/N/N-"+str(ctt).zfill(4)+".jpg")
        ctt+=1
        ctv+=1
    else:
        im = Image.open(im)
        newsize = (200, 200)
        im = im.resize(newsize)
        rgb_im = im.convert('RGB')
        rgb_im.save("./TRAINSET/N/N-"+str(ctv).zfill(4)+".jpg")
        ctv+=1


    