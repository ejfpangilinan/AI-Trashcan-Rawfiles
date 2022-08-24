
import numpy as np
import cv2
import os
import glob




NEW_DIR = './TRAIN'
TRAINING_DIR = './TRAIN.1'



for enum , im in enumerate(glob.glob("./TRAINSET/B/*.jpg")):
    # if enum>=10:
    #     break
    img = cv2.imread(im)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # # thresh = cv2.adaptiveThreshold(blurred, 255,
    # # 	cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    # # cv2.imshow("Mean Adaptive Thresholding", thresh)
    # # cv2.waitKey(0)

    # thresh = cv2.adaptiveThreshold(blurred, 255,
    # cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 501, 3)

    # fgbg = cv2.createBackgroundSubtractorMOG2().apply(thresh)

    # for i, row in enumerate(fgbg):
    #     # get the pixel values by iterating
    #     for j, pixel in enumerate(fgbg):
    #         if(fgbg[i][j] == 255):
    #                 # update the pixel value to black
    #             img[i][j] = [255,255,255]
    
    gree_img  = np.full(img.shape, (0,255,0), np.uint8)
    # fused_img  = cv2.add(img,red_img)
    fused_img  = cv2.addWeighted(img, 0.9, gree_img, 0.1, 0)
    
    grayscaleImage = cv2.cvtColor(fused_img, cv2.COLOR_BGR2GRAY)
    _, threshedImage = cv2.threshold(grayscaleImage, 75, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    ## Find the contours with biggest area
    contours, _ = cv2.findContours(threshedImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    maxContour = max(contours, key=cv2.contourArea)
    ## Create mask
    mask = np.zeros(fused_img.shape[:2], np.uint8)
    cv2.drawContours(mask, [maxContour], -1, 255, -1)
    fgmask = np.zeros(fused_img.shape[:3], np.uint8)
    height = fused_img.shape[:2][1]
    fgmask[:, 0:height] = (255, 255, 255)

    locations = np.where(mask != 0)
    fgmask[locations[0], locations[1]] = fused_img[locations[0], locations[1]]
    
    
    

    print('./TRAIN/B/TRAIN_NEW_BIODEG_ORI_%d.jpg'%(enum))
    cv2.imwrite(r'./TRAIN/B/TRAIN_NEW_BIODEG_ORI_%d.jpg'%enum, fgmask)
    
for enum , im in enumerate(glob.glob("./TRAINSET/N/*.jpg")):
    # if enum>=10:
    #     break
    img = cv2.imread(im)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # # thresh = cv2.adaptiveThreshold(blurred, 255,
    # # 	cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    # # cv2.imshow("Mean Adaptive Thresholding", thresh)
    # # cv2.waitKey(0)

    # thresh = cv2.adaptiveThreshold(blurred, 255,
    # cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 501, 3)

    # fgbg = cv2.createBackgroundSubtractorMOG2().apply(thresh)

    # for i, row in enumerate(fgbg):
    #     # get the pixel values by iterating
    #     for j, pixel in enumerate(fgbg):
    #         if(fgbg[i][j] == 255):
    #                 # update the pixel value to black
    #             img[i][j] = [255,255,255]
    
    gree_img  = np.full(img.shape, (0,255,0), np.uint8)
    # fused_img  = cv2.add(img,red_img)
    fused_img  = cv2.addWeighted(img, 0.9, gree_img, 0.1, 0)
    
    grayscaleImage = cv2.cvtColor(fused_img, cv2.COLOR_BGR2GRAY)
    _, threshedImage = cv2.threshold(grayscaleImage, 75, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    ## Find the contours with biggest area
    contours, _ = cv2.findContours(threshedImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    maxContour = max(contours, key=cv2.contourArea)
    ## Create mask
    mask = np.zeros(fused_img.shape[:2], np.uint8)
    cv2.drawContours(mask, [maxContour], -1, 255, -1)
    fgmask = np.zeros(fused_img.shape[:3], np.uint8)
    height = fused_img.shape[:2][1]
    fgmask[:, 0:height] = (255, 255, 255)

    locations = np.where(mask != 0)
    fgmask[locations[0], locations[1]] = fused_img[locations[0], locations[1]]
    
    print('./TRAIN/N/TRAIN_NEW_NONBIO_%d.jpg'%(enum))
    cv2.imwrite(r'./TRAIN/N/TRAIN_NEW_NONBIO_%d.jpg'%enum, fgmask)

for enum , im in enumerate(glob.glob("./VALIDATE/B/*.jpg")):
    # if enum>=10:
    #     break
    img = cv2.imread(im)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # # thresh = cv2.adaptiveThreshold(blurred, 255,
    # # 	cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    # # cv2.imshow("Mean Adaptive Thresholding", thresh)
    # # cv2.waitKey(0)

    # thresh = cv2.adaptiveThreshold(blurred, 255,
    # cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 501, 3)

    # fgbg = cv2.createBackgroundSubtractorMOG2().apply(thresh)

    # for i, row in enumerate(fgbg):
    #     # get the pixel values by iterating
    #     for j, pixel in enumerate(fgbg):
    #         if(fgbg[i][j] == 255):
    #                 # update the pixel value to black
    #             img[i][j] = [255,255,255]
    
    gree_img  = np.full(img.shape, (0,255,0), np.uint8)
    # fused_img  = cv2.add(img,red_img)
    fused_img  = cv2.addWeighted(img, 0.9, gree_img, 0.1, 0)
    
    grayscaleImage = cv2.cvtColor(fused_img, cv2.COLOR_BGR2GRAY)
    _, threshedImage = cv2.threshold(grayscaleImage, 75, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    ## Find the contours with biggest area
    contours, _ = cv2.findContours(threshedImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    maxContour = max(contours, key=cv2.contourArea)
    ## Create mask
    mask = np.zeros(fused_img.shape[:2], np.uint8)
    cv2.drawContours(mask, [maxContour], -1, 255, -1)
    fgmask = np.zeros(fused_img.shape[:3], np.uint8)
    height = fused_img.shape[:2][1]
    fgmask[:, 0:height] = (255, 255, 255)

    locations = np.where(mask != 0)
    fgmask[locations[0], locations[1]] = fused_img[locations[0], locations[1]]
    
    
    

    print('./VALIDATE_NEW/B/VALIDATE_NEW_BIODEG_ORI_%d.jpg'%(enum))
    cv2.imwrite(r'./VALIDATE_NEW/B/VALIDATE_NEW_BIODEG_ORI_%d.jpg'%enum, fgmask)
    
for enum , im in enumerate(glob.glob("./VALIDATE/N/*.jpg")):
    # if enum>=10:
    #     break
    img = cv2.imread(im)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # # thresh = cv2.adaptiveThreshold(blurred, 255,
    # # 	cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    # # cv2.imshow("Mean Adaptive Thresholding", thresh)
    # # cv2.waitKey(0)

    # thresh = cv2.adaptiveThreshold(blurred, 255,
    # cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 501, 3)

    # fgbg = cv2.createBackgroundSubtractorMOG2().apply(thresh)

    # for i, row in enumerate(fgbg):
    #     # get the pixel values by iterating
    #     for j, pixel in enumerate(fgbg):
    #         if(fgbg[i][j] == 255):
    #                 # update the pixel value to black
    #             img[i][j] = [255,255,255]
    
    gree_img  = np.full(img.shape, (0,255,0), np.uint8)
    # fused_img  = cv2.add(img,red_img)
    fused_img  = cv2.addWeighted(img, 0.9, gree_img, 0.1, 0)
    
    grayscaleImage = cv2.cvtColor(fused_img, cv2.COLOR_BGR2GRAY)
    _, threshedImage = cv2.threshold(grayscaleImage, 75, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    ## Find the contours with biggest area
    contours, _ = cv2.findContours(threshedImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    maxContour = max(contours, key=cv2.contourArea)
    ## Create mask
    mask = np.zeros(fused_img.shape[:2], np.uint8)
    cv2.drawContours(mask, [maxContour], -1, 255, -1)
    fgmask = np.zeros(fused_img.shape[:3], np.uint8)
    height = fused_img.shape[:2][1]
    fgmask[:, 0:height] = (255, 255, 255)

    locations = np.where(mask != 0)
    fgmask[locations[0], locations[1]] = fused_img[locations[0], locations[1]]
    
    print('./VALIDATE_NEW/N/VALIDATE_NEW_NONBIO_%d.jpg'%(enum))
    cv2.imwrite(r'./VALIDATE_NEW/N/VALIDATE_NEW_NONBIO_%d.jpg'%enum, fgmask)
    
    
    
    
    


# for im in os.listdir(TRAINING_DIR+'/B'):
# 	img = cv2.imread(str(im))
 
    
 
# 	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 	blurred = cv2.GaussianBlur(gray, (7, 7), 0)

# 	# thresh = cv2.adaptiveThreshold(blurred, 255,
# 	# 	cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
# 	# cv2.imshow("Mean Adaptive Thresholding", thresh)
# 	# cv2.waitKey(0)

# 	thresh = cv2.adaptiveThreshold(blurred, 255,
# 	cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 303, 3)

# 	fgbg = cv2.createBackgroundSubtractorMOG2().apply(thresh)

# 	for i, row in enumerate(fgbg):
# 		# get the pixel values by iterating
# 		for j, pixel in enumerate(fgbg):
# 			if(fgbg[i][j] == 255):
# 					# update the pixel value to black
# 				img[i][j] = [255,255,255]
    
# 	cv2.imwrite(NEW_DIR+'/B/'+im, img)