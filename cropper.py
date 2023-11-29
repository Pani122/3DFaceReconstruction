import cv2 as cv
import numpy as np
import scipy
import math
import os
import sys

import matplotlib
import pylab as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import face_alignment

import os
import subprocess

### initial 
#
FA = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, enable_cuda=False, flip_input=False)
#
input_directory = 'dataset/'
input_files = [f for f in os.listdir(input_directory) if f.endswith('.jpg')]
### get landmarks from test image
for input_file in input_files:

    image_file = os.path.join(input_directory, input_file)
    image = cv.imread(image_file)

    # print(image.shape)

    image_height, image_width, image_depth = image.shape
    preds = FA.get_landmarks(image)
    if(preds == None):
        continue
    ### landmarks  vis
    canvas = image.copy()
    minX=1000
    maxX=0
    minY=1000
    maxY=0
    for var in  preds[0]:
        if minX > var[0]:
            minX = var[0]
        if maxX < var[0]:
            maxX = var[0]
        if minY > var[1]:
            minY = var[1]
        if maxY < var[1]:
            maxY = var[1]

    ### crop face image
    scale=90/math.sqrt((minX-maxX)*(minY-maxY))
    width=maxX-minX
    height=maxY-minY
    cenX=width/2
    cenY=height/2

    x= int( (minX+cenX)*scale )
    y= int( (minY+cenY)*scale )
    #print x,y,scale

    resized_image = cv.resize(image, (0,0), fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
    rh,rw,rc =  resized_image.shape

    #
    crop_width = 192
    crop_height = 192
    left = 0
    top = 0
    right = 0
    bottom = 0
    cx = x
    cy = y
    if x < crop_width/2:
        left = crop_width/2 - x
        cx = x + left
    if y < crop_height/2:
        top = crop_height/2 - y
        cy = y + top
    if rw - x < crop_width/2:
        right =  crop_width/2 + x - rw
    if rh - y < crop_height/2:
        bottom = crop_height/2 + y - rh
    #
    crop_image = cv.copyMakeBorder(resized_image,int(top), int(bottom), int(left), int(right),cv.BORDER_REFLECT)
    crop_image = crop_image[int(cy-crop_height/2):int(cy+crop_height/2), int(cx-crop_width/2):int(cx+crop_width/2), :]
    width, height = 192 / 100, 192 / 100
    fig, ax = plt.subplots(figsize=(width, height))
    ax.axis('off')
    ax.imshow(crop_image)
    plt.savefig(f'inputs/{input_file}')
    plt.close()
   

