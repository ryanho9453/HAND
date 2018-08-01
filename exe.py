import os
import sys
import numpy as np
from scipy import stats
import json
import tensorflow as tf
import cv2 as cv
from processing.data_reader import DataReader

"""

data_size (2062, 64, 64)

"""
with open('/Users/ryanho/Documents/python/HAND/config.json', 'r') as f:
    config = json.load(f)
reader_config = config['processing']['reader']

data_reader = DataReader(reader_config)
while data_reader.finish_all_data is False:
    img, label = data_reader.next_batch()
    cv.imshow('img', img[0])
    cv.waitKey()



# cv.imshow('img', img)
# cv.waitKey(3000)

