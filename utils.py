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


class ImgPatch:
    def __init__(self, img_size=(3000, 4096), patch_size=(1500, 2048), overlapping_size=(300, 300)):
        self.img_size = img_size
        self.d_x, self.d_y = int(patch_size[1]), int(patch_size[0])
        self.o_x, self.o_y = int(overlapping_size[1]), int(overlapping_size[0])
        self.cell_width, self.cell_height = self.d_x + self.o_x, self.d_y + self.o_y
        self.top_left_corner_coord = self.compute_top_left_corners()
        # self.small_bbox_side_ignore = 1
        self.min_ratio_to_keep_bbox = 0.3

    def compute_top_left_corners(self):
        len_y, len_x = self.img_size
        top_left_corner_coord = []
        num_cell_y, num_cell_x = math.ceil(len_y / self.d_y), math.ceil(len_x / self.d_x)
        for i in range(num_cell_y):
            top_left_corner_coord_i = []
            for j in range(num_cell_x):

                # extend half of overlap to all direction
                index_y = i * self.d_y - int(self.o_y / 2)
                if i == 0:  # extend overlap down
                    index_y = i * self.d_y
                elif i == num_cell_y - 1:  # extend overlap up
                    index_y = i * self.d_y - self.o_y

                index_x = j * self.d_x - int(self.o_x / 2)
                if j == 0:
                    index_x = j * self.d_x
                elif j == num_cell_x - 1:
                    index_x = j * self.d_x - self.o_x

                top_left_corner_coord_i.append((index_y, index_x))
            top_left_corner_coord.append(top_left_corner_coord_i)
        return top_left_corner_coord

    def img_2_cell(self, img):
        assert img.shape[:2] == self.img_size, 'expect ' + str(self.img_size) + ' got ' + str(img.shape[:2])
        cells = [[None for j in range(len(self.top_left_corner_coord[0]))] for i in
                 range(len(self.top_left_corner_coord))]
        for i, y_x_row in enumerate(self.top_left_corner_coord):
            for j, (y, x) in enumerate(y_x_row):
                cells[i][j] = img[y:y + self.d_y + self.o_y, x:x + self.d_x + self.o_x, :]
        return cells

    def labels_from_img_coord_2_cell_coord(self, labels):
        bboxs_in_cell = [[[] for j in range(len(self.top_left_corner_coord[0]))] for i in
                         range(len(self.top_left_corner_coord))]
        for label in labels:
            cls, bbox = label[0], label[1:]
            # check each cell
            for i, y_x_row in enumerate(self.top_left_corner_coord):
                for j, (y, x) in enumerate(y_x_row):
                    x_min, y_min, x_max, y_max = bbox
                    w, h = x_max - x_min, y_max - y_min

                    x_min_cell, y_min_cell, x_max_cell, y_max_cell = max(x, x_min), \
                                                                     max(y, y_min), \
                                                                     min(x + self.d_x + self.o_x - 1, x_max), \
                                                                     min(y + self.d_y + self.o_y - 1, y_max)
                    w_cell, h_cell = x_max_cell - x_min_cell, y_max_cell - y_min_cell
                    # if enough part of bbox in this cell
                    if w_cell / w > self.min_ratio_to_keep_bbox and h_cell / h > self.min_ratio_to_keep_bbox:
                        bboxs_in_cell[i][j].append([cls, x_min_cell - x, y_min_cell - y, x_max_cell - x, y_max_cell - y])
        return bboxs_in_cell

    def img_and_xyxy_label_from_img_2_cell(self, img, label):
        """"""
        img_cell = self.img_2_cell(img)
        bbox_in_cell = self.labels_from_img_coord_2_cell_coord(label)
        return img_cell, bbox_in_cell

    def img_and_xywh_label_from_img_2_cell(self, img, labels):
        """"""
        imgs_cell = self.img_2_cell(img)
        if not isinstance(labels[0], list): labels = [labels]
        '''format annotation to xyxy'''
        for i, label in enumerate(labels):
            xyxy = utils.xywhratio_2_xyxy(*img.shape[:2], *label[1:])
            labels[i] = [label[0], *xyxy]

        labels_in_cell = self.labels_from_img_coord_2_cell_coord(labels)

        '''format back to xywh'''
        for i_patch, (img_col, label_row) in enumerate(zip(imgs_cell, labels_in_cell)):
            for j_patch, (img_cell, labels_in_one_cell) in enumerate(zip(img_col, label_row)):
                for i_label, label in enumerate(labels_in_one_cell):
                    xywh = utils.xyxy_2_xywhratio(*img_cell.shape[:2], *label[1:])
                    labels_in_cell[i_patch][j_patch][i_label] = [label[0], *xywh]
        return imgs_cell, labels_in_cell

    def example_cells(self):
        img = np.arange(0, self.img_size[0] * self.img_size[1])
        img = img.reshape((self.img_size[0], self.img_size[1], -1))
        img = np.concatenate([img, img, img], axis=-1)
        img_cells = self.img_2_cell(img)
        return img_cells


def sudo_img_test_patcher():
    """sudo testing"""
    img = np.arange(0, 100)
    # img = img.reshape((10, 10,))
    img = img.reshape((10, 10))

    # img = np.expand_dims(img, axis=-1)
    # img = np.concatenate([img, img, img], axis=-1)

    # print(img[:2, :2].shape)
    # print(img[:2, :2])
    patcher = ImgPatch(img.shape[:2], patch_width=2, patch_height=2, width_overlapping=2, height_overlapping=2)
    imgs = patcher.img_2_cell(img)
    label = patcher.labels_from_img_coord_2_cell_coord([[1, 0, 0, 3, 2],
                                                        [2, 0, 0, 3, 2]])

    print(patcher.top_left_corner_coord)
    for i, (img_row, label_row) in enumerate(zip(imgs, label)):
        for j, (img, label) in enumerate(zip(img_row, label_row)):
            print(i, j)
            print(img)
            print(label)


def datamatrix_test_patcher():
    """"""
    # m, n = 3000, 4096
    pathcer = ImgPatch()
    cells = pathcer.top_left_corner_coord
    img_cells = pathcer.example_cells()
    for cell_row in cells:
        print(cell_row)

    for img_row in img_cells:
        for img in img_row:
            print(img.shape, end='')
        print()


def img_label_test_patcher():
    '''anno testing'''

    img = cv2.imread('./DataMatrixCoCoFormat/images/Image_20211125133414987.bmp')
    label = np.loadtxt('./DataMatrixCoCoFormat/labels/Image_20211125133414987.txt').tolist()

    '''change label format'''
    m, n = img.shape[:2]
    for i in range(len(label)):
        cls, center_x_ratio, center_y_ratio, w_ratio, h_ratio = label[i]
        x_min, y_min, x_max, y_max = xywhratio_2_xyxy(m, n, center_x_ratio, center_y_ratio, w_ratio, h_ratio)
        label[i] = [cls, x_min, y_min, x_max, y_max]
    '''org'''
    # img_copy = copy.deepcopy(img)
    # for b in label:
    #     cls, x_min, y_min, x_max, y_max = b
    #     cv2.rectangle(img_copy, (x_min, y_min), (x_max, y_max), (0, 0, 255), 5)
    # cv2.namedWindow('cell ' + str(i) + ' ' + str(j), cv2.WINDOW_NORMAL)
    # cv2.imshow('cell ' + str(i) + ' ' + str(j), img_copy)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    patcher = ImgPatch(img.shape[:2])
    imgs = patcher.img_2_cell(img)
    label = patcher.labels_from_img_coord_2_cell_coord(label)

    print(patcher.top_left_corner_coord)
    print(label)
    for i, (img_row, label_row) in enumerate(zip(imgs, label)):
        for j, (img, label) in enumerate(zip(img_row, label_row)):
            print(i, j)
            print(label)
            for b in label:
                cls, x_min, y_min, x_max, y_max = b
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 5)
            cv2.imshow('cell ' + str(i) + ' ' + str(j), img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def xyxy_xywh_test():
    m, n = 200, 100
    x_min, y_min, x_max, y_max = 0, 0, 100, 100
    x, y, w, h = xyxy_2_xywhratio(m, n, x_min, y_min, x_max, y_max)
    x_min_, y_min_, x_max_, y_max_ = xywhratio_2_xyxy(m, n, x, y, w, h)
    print(x, y, w, h)
    print(x_min_, y_min_, x_max_, y_max_)


def main():
    """"""
    # xyxy_xywh_test()
    # img_label_test_patcher()
    datamatrix_test_patcher()



if __name__ == '__main__':
    main()


