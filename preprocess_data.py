#!/usr/bin/python

""" 
+ Apply standard preprocessing for training data - 
    subtract image mean
    square crop
    resize
+ move data to trainingdata directory
+ generate train and test text files for caffe
+ generate hdf5 files
"""
import os
import cv2
import json
import glob
import random
import init_paths
import numpy as np

ALL_IMAGES = 'origdata/AllImages'
VAL_DIR = 'trainingdata/val'
TRAIN_DIR = 'trainingdata/train'

handcam_root = 'origdata/HandCam' # test images
imagenet_root = 'origdata/ImageNet' # training
deepgrasping_root = 'origdata/DeepGrasping' #training

labelmap = {
        'power' :       0,
        'Power' :       0,
        '3 jaw chuck' : 1,
        'tool' :        2,
        'pinch' :       3,
        'key':          4
}

train_data_dirs = [imagenet_root, deepgrasping_root]
test_data_dirs = [handcam_root]

def get_data_filenames(data_dirs, full_paths=False):
    """ gets all trainingdata full paths in one list """
    full_filenames = []
    for data_dir in data_dirs:
        files = os.listdir(os.path.join(data_dir, 'Images'))
        if full_paths:
            full_filenames.extend(map(lambda f: os.path.join(data_dir, f), files))
        else:
            full_filenames.extend(files)
    return full_filenames

def get_annotations(data_dirs):
    """ loads and merges annotation dicts for data_dir"""
    anno_filename = glob.glob('{}/*.json'.format(data_dirs[0]))[0]
    with open(anno_filename, 'r') as anno_file:
        annotations = json.load(anno_file)
    for data_dir in data_dirs[1:]:
        anno_filename = glob.glob('{}/*.json'.format(data_dir))[0]
        with open(anno_filename, 'r') as anno_file:
            annotations.update(json.load(anno_file))
    return annotations

def train_val_split(data_filenames, train_percent=.8):
    """ self explanatory """
    num_train = int(train_percent * len(data_filenames))
    shuffled = random.sample(data_filenames, len(data_filenames))
    # return train, val
    return shuffled[:num_train], shuffled[num_train:]

def write_textfiles(train, val, anno):
    """ writes out train and val txt files for caffe """
    """ used only for side effect """
    # fn creates text file line in form '<filename> <int mapping of grip type>'
    # for each train/val file and writes each line in corresponding
    # train.txt/val.txt for training
    # to_line = lambda fname:'{} {}'.format(fname,labelmap[anno[fname]['grip']])
    to_line = lambda fname:'{} {}'.format(fname,labelmap[anno[fname]['grip']])
    train_str = '\n'.join(map(to_line, train))
    val_str = '\n'.join(map(to_line, val))

    with open('train.txt', 'w') as trainfile:
        trainfile.write(train_str)
    with open('val.txt', 'w') as valfile:
        valfile.write(val_str)

def move_and_transform(train, val, size, resize=True, center_crop=False):
    """ moves train/val files to trainingdata dir after applying optional
        transformations of resizing and centercropping """
    load_image = lambda fname: cv2.imread(os.path.join(ALL_IMAGES, fname))
    resize_image = lambda img: cv2.resize(img, (size, size))
    move_image = lambda (img, fname): cv2.imwrite(fname, img)

    train_imgs = map(load_image, train)
    val_imgs = map(load_image, val)

    if center_crop:
        pass # TODO
    if resize:
        train_imgs = map(resize_image, train_imgs)
        val_imgs = map(resize_image, val_imgs)
    
    train_fnames = map(lambda fname: os.path.join(TRAIN_DIR, fname), train)
    val_fnames = map(lambda fname: os.path.join(VAL_DIR, fname), val)

    map(move_image, zip(train_imgs, train_fnames))
    map(move_image, zip(val_imgs, val_fnames))

if __name__ == '__main__':
    data = get_data_filenames(train_data_dirs)
    annotations = get_annotations(train_data_dirs)
    train, val  = train_val_split(data)
    write_textfiles(train, val, annotations)
    move_and_transform(train, val, 256)
