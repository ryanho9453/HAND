import cv2 as cv
import numpy as np
import random
import os
import re

"""
data_size (2062, 64, 64)

"""


class DataReader:
    def __init__(self, config, mode=None, process=None):
        self.config = config
        self.index = 0

        self.exclude_folder = ['.DS_Store']

        self.finish_all_data = False

        self.number_n_batch = 0

        X, self.Y = self.__load_data(mode=mode)

        self.data_size = len(X)

        self.X = self.__preprocess(X, process=process)


    def next_batch(self):
        start_idx = self.index
        end_idx = self.index + self.config['batch_size']
        imgs = self.X[start_idx: end_idx]
        labels = self.Y[start_idx: end_idx]

        # complement
        if len(imgs) < self.config['batch_size']:
            complement = self.config['batch_size'] - len(imgs)
            imgs += self.X[:complement]
            labels += self.Y[:complement]
            random.shuffle(imgs)
            random.shuffle(labels)
            self.index = self.data_size

        self.index = end_idx

        if self.index == self.data_size:
            self.finish_all_data = True

        self.number_n_batch += 1

        return imgs, labels

    def __load_data(self, mode=None):
        """

        img array value btw 0~1 , so multiply 255

        """

        if mode == 'train':
            imgs = np.load(self.config['data_path'] + 'X.npy')[:self.config['train_size']]
            labels = np.load(self.config['data_path'] + 'Y.npy')[:self.config['train_size']]

        elif mode == 'test':
            imgs = np.load(self.config['data_path'] + 'X.npy')[self.config['train_size']:]
            labels = np.load(self.config['data_path'] + 'Y.npy')[self.config['train_size']:]

        else:
            imgs = np.load(self.config['data_path'] + 'X.npy')
            labels = np.load(self.config['data_path'] + 'Y.npy')

        imgs = imgs * 255
        imgs = imgs.astype('uint8')

        return imgs, labels

    def __preprocess(self, imgs, process=None):
        """

        if process = contour , threshold first, then find contour

        """

        if process == 'threshold':
            X = self.__threshold_process(imgs)
            return X

        elif process == 'contour':
            thresholds = self.__threshold_process(imgs)
            X = self.__find_contour(thresholds)
            return X

    def __threshold_process(self, imgs):
        thresholds = []
        for img in imgs:
            ret, thresh = cv.threshold(img, 127, 255, cv.THRESH_OTSU)
            thresholds.append(thresh)

        return np.array(thresholds)

    def __find_contour(self, thresholds):
        img_size = thresholds[0].shape

        contour_pool = []
        for thresh in thresholds:
            _, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
            canvas = np.zeros(img_size, dtype='uint8')
            for cnt in contours:
                cv.drawContours(canvas, [cnt], 0, (0, 255, 0), 1)
            contour_pool.append(canvas)

        return np.array(contour_pool)

