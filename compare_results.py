import os
from glob import glob
from tqdm import tqdm
import cv2
import csv
import json
from sklearn.metrics import accuracy_score#, precission_score, recall
import numpy as np
import matplotlib.pyplot as plt


def get_path(path):
    if not os.path.exists(path):
	os.system('mkdir -p %s' % path)
    return path


def union3d(arr1, arr2):
    total = np.sum(arr1 == 0) + np.sum(arr2 == 0)  # cardinal(arr1)+cardinal(arr2)
    inter = interspect3d(arr1, arr2)
    # print 'union',total/3.0-inter
    return total / 3.0 - inter


def interspect3d(arr1, arr2):
    cnt = 0
    for i1, i2 in zip(range(arr1.shape[0]), range(arr2.shape[0])):
        for j1, j2 in zip(range(arr1.shape[1]), range(arr2.shape[1])):
            for k1, k2 in zip(range(arr1.shape[2]), range(arr2.shape[2])):
                if arr1[i1][j1][k1] == 0 and arr2[i2][j2][k2] == 0 and arr1[i1][j1][k1] == arr2[i2][j2][k2]:
                    cnt += 1
    # print 'inter',cnt/3.0
    return cnt / 3.0


def region_fitting_error(gen, gt, thresh=None):
    '''
    region fitting error (RFE), extrated from
    "Meaningful Object Segmentation From SAR Images via a Multiscale Nonlocal Active Contour Model"
    :param gen:  a generated image
    :param gt:  a ground truth image
    :return:
    '''
    gen=cv2.imread(gen)
    gt=cv2.imread(gt)
    if thresh is not None:
        _, gt = cv2.threshold(gt, thresh, 255, cv2.THRESH_BINARY)
        _, gen = cv2.threshold(gen, thresh, 255, cv2.THRESH_BINARY)
    inter = interspect3d(gen, gt)
    union = union3d(gen, gt)
    RFE = np.abs(union - inter) / np.sum(gt == 0)
    return RFE

def calculate_accuracy(inp, outp, targ, thresh=None):
    '''
    calculate segmentation accuracy
    :param inp: path of input img
    :param outp: path of output img
    :param targ: path of target img
    :param thres: threshold of creating binary img, integer
    :return: accuracy
    '''
    input_arr = cv2.imread(inp)
    target_arr = cv2.imread(targ)
    output_arr = cv2.imread(outp)

    if thresh is not None:
        _, input = cv2.threshold(input_arr, thresh, 255, cv2.THRESH_BINARY)
        _, target = cv2.threshold(target_arr, thresh, 255, cv2.THRESH_BINARY)
        _, output = cv2.threshold(output_arr, thresh, 255, cv2.THRESH_BINARY)
    else:
        input = input_arr
        output = output_arr
        target = target_arr
    acc = accuracy_score(target.flatten(), output.flatten())
    return acc




