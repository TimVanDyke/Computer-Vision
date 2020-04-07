#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from get_letter import letters_from_imgArr


IMG_SIZE = 28


def img_to_array(img_path):
    # creates the right format for the image
    def resize_image():
        # load
        image = cv2.imread('./pictures/temp.png', cv2.IMREAD_GRAYSCALE)
        # resize to 28x28
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        # invert white and black
        image = cv2.bitwise_not(image)
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
        thresh = 100
        def fn(x): return 255 if x > thresh else 0
        r = img.convert('L').point(fn, mode='1')
        # we could find a better way then making a temp picture
        r.save('./pictures/temp.png')

    binarize_image(img_path)
    return resize_image()

#this array can be used to store multiple images
dataArr = []

#this ia a single image
data = img_to_array(sys.argv[1])
dataArr.append(data)

#get letter from the array of pictures
dataArr = np.array(dataArr)
datattr = dataArr.astype('float32') / 256
letters_from_imgArr('balanced',"83%_0-12_0-5_0-5x2000", dataArr)
