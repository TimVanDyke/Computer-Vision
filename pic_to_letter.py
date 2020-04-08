#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import sys
import numpy as np
from PIL import Image
from get_letter import letters_from_imgArr
import glob

IMG_SIZE = 28
NETWORK = "83%_0-2_0-4_0-4x2000"

def img_to_array(img_path):
    # creates the right format for the image
    def resize_image():
        # load
        image = cv2.imread('./pictures/temp.png', cv2.IMREAD_GRAYSCALE)
        # resize to 28x28
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        # rotate and flip to match the training dataset
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        image = cv2.flip(image, 0)
        arr = []
        arr.append(image)
        arr = np.array(arr)
        arr = arr.astype('float32') / 256
        return arr

    def binarize_image(img_path):
        img = Image.open(img_path)
        thresh = 110
        #transfer the image to 1 and 0
        def fn(x): return 0 if x > thresh else 255
        r = img.convert('L').point(fn, mode='1')
        # we could find a better way then making a temp picture
        r.save('./pictures/temp.png')

    binarize_image(img_path)
    return resize_image()


#read all the PNG files in the specified folder
pictures_in_folder = glob.glob("{}/*.png".format(sys.argv[1]))

#create a list of pictures in the folder
dataArr = []
for i in range(len(pictures_in_folder)):
    data = img_to_array(pictures_in_folder[i])
    dataArr.append(data)


#get letter from the array of pictures
dataArr = np.array(dataArr)
datattr = dataArr.astype('float32') / 256
letters = letters_from_imgArr('balanced', NETWORK , dataArr)

# print the results
for i in range(len(letters)):
    print(letters[i])
