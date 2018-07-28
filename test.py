import os
import sys
import numpy as np
from scipy import stats
import json
import tensorflow as tf
import cv2 as cv
sys.path.append('/Users/ryanho/Documents/python/HAND/processing/')
sys.path.append('/Users/ryanho/Documents/python/HAND/models/')
from data_reader import DataReader

"""

data_size (2062, 64, 64)

"""


# img= np.zeros((512, 512, 3), np.uint8)
# img[:,:]= (0,0,0)
#
# img = cv.rectangle(img,(400,0),(512,100),(255,255,255),-1)
#
# img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
# img = cv.imread('/Users/ryanho/Documents/python/HAND/data/test_img.jpg')
raw = np.load('/Users/ryanho/Documents/python/HAND/data/X.npy')
raw = raw[0]

print(raw.shape)

# cv.imshow('img', raw)
# cv.waitKey(7000)

