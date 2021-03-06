# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 12/7/21 7:25 PM
"""
import cv2
import os
import shutil
from main import get_img_dir_paths, get_label_dir_paths, vis_debug
import utils


def main():
    root_dir = './no/'
    dataset_dir_path = root_dir + '/DataMatrixPatch/'
    img_paths = get_img_dir_paths(root_dir + '/img')
    label_paths = get_label_dir_paths(root_dir + '/label')

    print(list(img_paths.keys())[0], img_paths[list(img_paths.keys())[0]])
    print(list(label_paths.keys())[0], label_paths[list(label_paths.keys())[0]])

    '''check labeling'''
    for view in label_paths.keys():
        label_path = label_paths[view]
        for img_path in img_paths[view]:
            if img_path['name'][:-3] in label_path:
                img, labels = utils.get_annotation(img_path['path'], label_path)

    '''make patches'''
    '''clear old dataset dir and make new one'''
    if os.path.isdir(dataset_dir_path):
        shutil.rmtree(dataset_dir_path)
    os.mkdir(dataset_dir_path)
    # os.mkdir(images_dir_path)
    # os.mkdir(labels_dir_path)
    for view in label_paths.keys():
        label_path = label_paths[view]
        for img_path in img_paths[view]:
            if img_path['name'][:-3] in label_path:
                img, labels = utils.get_annotation(img_path['path'], label_path)
                m, n = img.shape[:2]
                for i_label, label in enumerate(labels):
                    cls, x_min, y_min, x_max, y_max = label
                    x_min, y_min, x_max, y_max = utils.xyxy_expand(m, n, x_min, y_min, x_max, y_max, 1.2)
                    dm_patch = img[y_min:y_max, x_min:x_max]
                    print(dataset_dir_path+img_path['name'][:-4]+'_'+str(i_label)+'.jpg')
                    cv2.imwrite(dataset_dir_path+img_path['name'][:-4]+'_'+str(i_label)+'.jpg', dm_patch)


if __name__ == '__main__':
    main()
