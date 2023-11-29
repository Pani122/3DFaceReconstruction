import cv2 as cv
import numpy as np
import scipy
import math
import os
import sys
import torchfile
import matplotlib
import pylab as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import face_alignment
import vrn_unguided
from functools import partial
VRN = vrn_unguided.vrn_unguided
VRN.load_state_dict(torch.load('vrn_unguided.pth'))
### initial 
VRN.eval()
enable_cuda = True
#
FA = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, enable_cuda=True, flip_input=False)
#
VRN = vrn_unguided.vrn_unguided
VRN.load_state_dict(torch.load('models/vrn_unguided.pth'))
if enable_cuda:
    VRN.cuda()

### get landmarks from test image
image_file = 'samples/sample_input.jpg'
image = cv.imread(image_file)
try:
    image_height, image_width, image_depth = image.shape
except:
    print('cannot load image:', image_file)
#
preds = FA.get_landmarks(image)

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
    
plt.imshow(canvas[:,:,[2,1,0]])
plt.show()
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


### vrn output
inp = torch.from_numpy(crop_image.transpose((2, 0, 1))).float().unsqueeze_(0)
if enable_cuda:
    inp = inp.cuda()
with torch.no_grad():
    out = VRN(Variable(inp))[-1].data.cpu()
# out = VRN(Variable(inp))
print(out)


### save to obj file
import mcubes
# from sklearn.neighbors import NearestNeighbors

im =  crop_image[:,:,[2,1,0]] #RGB
vol = out.numpy()
vol = vol.reshape((200,192,192))*255
vol = vol.astype(float)
import cv2
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D

# Load your volume data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
verts, faces, normals, values = measure.marching_cubes(vol, 10)
mesh = Poly3DCollection(verts[faces])
ax.add_collection3d(mesh)
ax.set_xlim(0, 200) 
ax.set_ylim(0, 200)  
ax.set_zlim(0, 200)  

plt.tight_layout()
plt.show()
