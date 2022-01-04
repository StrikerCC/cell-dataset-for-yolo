# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 11/25/21 5:53 PM
"""
import copy

import numpy as np
import math
import cv2

import utils


def get_annotation(img_path, label_path):
    annotations = []
    img = cv2.imread(img_path)
    if img is not None:
        m, n = img.shape[:2]
        labels = np.loadtxt(label_path).reshape(-1, 5)
        for label in labels:
            cls, center_x_ratio, center_y_ratio, w_ratio, h_ratio = label
            x_min, y_min, x_max, y_max = xywhratio_2_xyxy(m, n, center_x_ratio, center_y_ratio, w_ratio, h_ratio)
            print(x_min, y_min, x_max, y_max)
            annotations.append([cls, x_min, y_min, x_max, y_max])
    return img, annotations


def xywhratio_2_xyxy(m, n, center_x_ratio, center_y_ratio, w_ratio, h_ratio):
    x_min, x_max = int(n * (center_x_ratio - w_ratio / 2)), int(n * (center_x_ratio + w_ratio / 2))
    y_min, y_max = int(m * (center_y_ratio - h_ratio / 2)), int(m * (center_y_ratio + h_ratio / 2))
    return x_min, y_min, x_max, y_max


def xyxy_2_xywhratio(m, n, x_min, y_min, x_max, y_max):
    center_x_ratio = float((x_min + x_max) / 2 / n)
    center_y_ratio = float((y_min + y_max) / 2 / m)
    w_ratio = float((x_max - x_min) / n)
    h_ratio = float((y_max - y_min) / m)
    return center_x_ratio, center_y_ratio, w_ratio, h_ratio


def xyxy_expand(m, n, x_min, y_min, x_max, y_max, ratio):
    center_x_ratio, center_y_ratio, w_ratio, h_ratio = xyxy_2_xywhratio(m, n, x_min, y_min, x_max, y_max)
    w_ratio *= ratio
    h_ratio *= ratio
    x_min, y_min, x_max, y_max = xywhratio_2_xyxy(m, n, center_x_ratio, center_y_ratio, w_ratio, h_ratio)
    return max(0, x_min), max(0, y_min), min(n-1, x_max), min(m-1, y_max)


def xyxy_xywh_test():
    m, n = 200, 100
    x_min, y_min, x_max, y_max = 0, 0, 100, 100
    x, y, w, h = xyxy_2_xywhratio(m, n, x_min, y_min, x_max, y_max)
    x_min_, y_min_, x_max_, y_max_ = xywhratio_2_xyxy(m, n, x, y, w, h)
    print(x, y, w, h)
    print(x_min_, y_min_, x_max_, y_max_)


def main():
    """"""
    xyxy_xywh_test()
    # img_label_test_patcher()
    # datamatrix_test_patcher()


if __name__ == '__main__':
    main()


