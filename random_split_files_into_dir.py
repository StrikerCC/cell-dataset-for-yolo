import os
import random
import shutil


def random_mv(src_dir, tgt_dirs, prob=0.05):
    assert os.path.isdir(src_dir)
    if not isinstance(tgt_dirs, list):
        tgt_dirs = [tgt_dirs]

    for tgt_dir in tgt_dirs:
        if not os.path.isdir(tgt_dir):
            os.mkdir(tgt_dir)

        '''mv for each target dir'''
        for file_name in os.listdir(src_dir):
            if random.random() < prob:
                shutil.move(src_dir + '/' + file_name, tgt_dir + '/' + file_name)

        '''update prob'''
        prob = (1-prob)*prob
    return


def merge(src_dirs, tgt_dir):
    if not os.path.isdir(tgt_dir):
        os.mkdir(tgt_dir)

    if not isinstance(src_dirs, list):
        src_dirs = [src_dirs]

    for src_dir in src_dirs:
        assert os.path.isdir(src_dir)
        '''mv for each target dir'''
        for file_name in os.listdir(src_dir):
            shutil.move(src_dir + '/' + file_name, tgt_dir + '/' + file_name)
    return


def mv_accordingly(src_dir, index_dir, tgt_dir):
    assert os.path.isdir(src_dir)
    assert os.path.isdir(index_dir)
    if not os.path.isdir(tgt_dir):
        os.mkdir(tgt_dir)

    index_name = set()
    for file_name in os.listdir(index_dir):
        index = file_name[:-4]
        index_name.add(index)

    '''nv file in src and index'''
    for file_name in os.listdir(src_dir):
        if file_name[:-4] in index_name:
            shutil.move(src_dir + '/' + file_name, tgt_dir + '/' + file_name)
    return


def main():
    root_dir = './DataMatrixCoCoFormat/'
    img_train_dir = root_dir + '/images/train/'
    img_val_dir = root_dir + '/images/val/'
    label_train_dir = root_dir + '/labels/train/'
    label_val_dir = root_dir + '/labels/val/'

    # root_dir = './test/'
    # src_dir = root_dir + '/src/'
    # tgt_dir = root_dir + '/tgt/'

    '''fake files'''
    if len(os.listdir(img_train_dir)) == 0:
        for i in range(1000):
            f = open(img_train_dir + '/' + str(i) + '.txt', 'w')
            # f.write('# ' + str(i))
            f.close()
    random_mv(img_train_dir, img_val_dir)

    print('src', len(os.listdir(img_train_dir)))
    print('tgt', len(os.listdir(img_val_dir)))

    # merge([img_src_dir, img_tgt_dir], img_src_dir)

    mv_accordingly(label_train_dir, img_val_dir, label_val_dir)

    return


if __name__ == '__main__':
    main()

