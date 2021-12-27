# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 11/25/21 3:33 PM
"""
import os
import shutil
import cv2
import numpy as np
import utils


def get_img_dir_paths(root_dir_path):
    img_paths = {}
    view_dir_paths = os.listdir(root_dir_path)
    for view_dir in view_dir_paths:
        img_paths_ = os.listdir(root_dir_path + '/' + view_dir)
        for img_path in img_paths_:
            if view_dir not in img_paths.keys():
                img_paths[view_dir] = []
            img_paths[view_dir].append({'name': img_path,
                                        'path': root_dir_path + '/' + view_dir + '/' + img_path})
    return img_paths


def get_label_dir_paths(root_dir_path):
    label_paths = {}
    view_dir_paths = os.listdir(root_dir_path)
    for view_dir in view_dir_paths:
        label_paths_ = os.listdir(root_dir_path + '/' + view_dir)
        # make sure at least one label exist
        for label_path in label_paths_:
            if '.txt' in label_path and not label_path == 'classes.txt':
                label_paths[view_dir] = root_dir_path + '/' + view_dir + '/' + label_path
    return label_paths


def make_yolov5_dataset(dataset_dir_path, img_paths, label_paths, patch):
    """"""
    images_dir_path = dataset_dir_path + '/images/'
    labels_dir_path = dataset_dir_path + '/labels/'

    patcher = utils.ImgPatch()
    print(patcher.top_left_corner_coord)
    '''clear old dataset dir and make new one'''
    if os.path.isdir(dataset_dir_path):
        shutil.rmtree(dataset_dir_path)
    os.mkdir(dataset_dir_path)
    os.mkdir(images_dir_path)
    os.mkdir(labels_dir_path)

    '''pick common view for img and label'''
    for i_view, view in enumerate(label_paths.keys()):
        if view in img_paths.keys():
            '''label for same view should be the same'''
            label_path = label_paths[view]
            label = np.loadtxt(label_path).tolist()
            img_patch, label_patch = None, None

            for i_img, img_name_and_path in enumerate(img_paths[view]):
                img_name, img_path = img_name_and_path['name'], img_name_and_path['path']
                if patch:

                    '''build annotation for patch cell'''
                    if i_img == 0:
                        '''make patch img and label'''
                        img = cv2.imread(img_path)
                        img_patch, label_patch = patcher.img_and_xywh_label_from_img_2_cell(img, label)

                    '''read img'''
                    img = cv2.imread(img_path)
                    img_patch, _ = patcher.img_and_xywh_label_from_img_2_cell(img, label)
                    # saving
                    for i_patch, (img_col, label_row) in enumerate(zip(img_patch, label_patch)):
                        for j_patch, (img_cell, label_cell) in enumerate(zip(img_col, label_row)):
                            img_cell_path = images_dir_path + '/' + img_name[:-4] + '_' + str(i_patch) + '_' + str(j_patch) + img_name[-4:]
                            label_cell_path = labels_dir_path + '/' + img_name[:-4] + '_' + str(i_patch) + '_' + str(j_patch) + '.txt'
                            cv2.imwrite(img_cell_path, img_cell)
                            np.savetxt(label_cell_path, label_cell)
                else:
                    '''copy img into images folder'''
                    shutil.copyfile(img_path, images_dir_path + '/' + img_name)
                    '''copy txt into labels folder'''
                    shutil.copyfile(label_path, labels_dir_path + '/' + img_name[:-3] + 'txt')
                # break

        print('View', view, 'done, ', len(img_paths[view]), 'img copied')
    return


def vis_debug(img_path, label_path):
    img, labels = utils.get_annotation(img_path, label_path)
    m, n = img.shape[:2]
    for label in labels:
        cls, x_min, y_min, x_max, y_max = label
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 5)
        print(x_min, y_min, x_max, y_max)

    print(img_path)
    print(label_path)
    cv2.namedWindow('labelling', cv2.WINDOW_NORMAL)
    cv2.imshow('labelling', img)
    cv2.waitKey(0)


def vis_debug_dataset(img_dir_path, label_dir_path, prob=0.5):
    img_names, label_names = os.listdir(img_dir_path), set(os.listdir(label_dir_path))
    for i, img_name in enumerate(img_names):
        label_name = img_name[:-3] + 'txt'
        if label_name not in label_names:
            print('Cannot find label' + label_name + ' for ', img_name)
            continue

        if np.random.random() < prob:
            print(i, '/', len(img_names))
            vis_debug(img_dir_path + '/' + img_name, label_dir_path + '/' + label_name)


def main():
    dataset_root_path = './det/'
    dataset_output_dir_path = dataset_root_path + '/DataMatrixCoCoFormat/'
    img_paths = get_img_dir_paths(dataset_root_path + '/img')
    label_paths = get_label_dir_paths(dataset_root_path + '/label')

    print(list(img_paths.keys())[0], img_paths[list(img_paths.keys())[0]])
    print(list(label_paths.keys())[0], label_paths[list(label_paths.keys())[0]])

    '''check labeling'''
    for view in label_paths.keys():
        label_path = label_paths[view]
        for img_path in img_paths[view]:
            if img_path['name'][:-3] in label_path:
                vis_debug(img_path['path'], label_path)
        # break

    make_yolov5_dataset(dataset_output_dir_path, img_paths, label_paths, patch=True)

    vis_debug_dataset(dataset_output_dir_path + '/images/',
                      dataset_output_dir_path + '/labels/', prob=0.005)

    return


if __name__ == '__main__':
    main()
