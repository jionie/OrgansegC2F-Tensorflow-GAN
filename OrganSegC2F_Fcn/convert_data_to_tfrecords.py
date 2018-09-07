# -*- coding = utf-8 -*-

from __future__ import absolute_import,division,print_function
import os
import numpy as np
import tensorflow as tf
import time
from scipy.misc import imread,imresize
from os import  walk
from os.path import join
import sys
import math
import random
from PIL import Image



data_path = "/home/jionie/OrganSegC2F"
current_fold = 0
organ_number = 1
low_range = 100
high_range = 240
slice_threshold = 0.98
slice_thickness = 3
organ_ID = 1
GPU_ID = 0
mode = 'training'
batch_size = 1



####################################################################################################
# returning the volume filename as in the testing stage
def volume_filename_testing(result_directory, t, i):
    return os.path.join(result_directory, str(t) + '_' + str(i + 1) + '.npz')


####################################################################################################
# returning the volume filename as in the fusion stage
def volume_filename_fusion(result_directory, code, i):
    return os.path.join(result_directory, code + '_' + str(i + 1) + '.npz')


####################################################################################################
# returning the volume filename as in the coarse-to-fine testing stage
def volume_filename_coarse2fine(result_directory, r, i):
    return os.path.join(result_directory, 'R' + str(r) + '_' + str(i + 1) + '.npz')


####################################################################################################
# defining the common variables used throughout the entire flowchart
image_path = os.path.join(data_path, 'images')
image_path_ = {}
for plane in ['X', 'Y', 'Z']:
    image_path_[plane] = os.path.join(data_path, 'images_' + plane)
    if not os.path.exists(image_path_[plane]):
        os.makedirs(image_path_[plane])
label_path = os.path.join(data_path, 'labels')
label_path_ = {}
for plane in ['X', 'Y', 'Z']:
    label_path_[plane] = os.path.join(data_path, 'labels_' + plane)
    if not os.path.exists(label_path_[plane]):
        os.makedirs(label_path_[plane])
list_path = os.path.join(data_path, 'lists')
if not os.path.exists(list_path):
    os.makedirs(list_path)
list_training = {}
for plane in ['X', 'Y', 'Z']:
    list_training[plane] = os.path.join(list_path, 'training_' + plane + '.txt')
list_testing = {}
for plane in ['X', 'Y', 'Z']:
    list_testing[plane] = os.path.join(list_path, 'testing_' + plane + '.txt')


plane = 'Y'

####################################################################################################
# returning the binary label map by the organ ID (especially useful under overlapping cases)
#   label: the label matrix
#   organ_ID: the organ ID
def is_organ(label, organ_ID):
    return label == organ_ID


####################################################################################################
# determining if a sample belongs to the training set by the fold number
#   total_samples: the total number of samples
#   i: sample ID, an integer in [0, total_samples - 1]
#   folds: the total number of folds
#   current_fold: the current fold ID, an integer in [0, folds - 1]
def in_training_set(total_samples, i, folds, current_fold):
    fold_remainder = folds - total_samples % folds
    fold_size = (total_samples - total_samples % folds) / folds
    start_index = fold_size * current_fold + max(0, current_fold - fold_remainder)
    end_index = fold_size * (current_fold + 1) + max(0, current_fold + 1 - fold_remainder)
    return not (i >= start_index and i < end_index)


####################################################################################################
# returning the filename of the training set according to the current fold ID
def training_set_filename(current_fold):
    return os.path.join(list_path, 'training_' + 'FD' + str(current_fold) + '.txt')


####################################################################################################
# returning the filename of the testing set according to the current fold ID
def testing_set_filename(current_fold):
    return os.path.join(list_path, 'testing_' + 'FD' + str(current_fold) + '.txt')


class Data():

    def training_setup(self):
        self.random = False
        image_list = open(training_set_filename(current_fold), 'r').read().splitlines() #for example traing_FD0.txt 20 /DATA3_DB7/data/zhcao/segmentation/DATA/images/0021.npy /DATA3_DB7/data/zhcao/segmentation/DATA/labels/0021.npy
        self.training_image_set = np.zeros((len(image_list)), dtype = np.int)
        for i in range(len(image_list)):
            s = image_list[i].split(' ')
            self.training_image_set[i] = int(s[0])
        slice_list = open(list_training[plane], 'r').read().splitlines() # for example traing_X.txt 0 0 /DATA3_DB7/data/zhcao/segmentation/DATA/images_X/0001/0000.npy /DATA3_DB7/data/zhcao/segmentation/DATA/labels_X/0001/0000.npy 100.0 0 0 0 0 0
        self.slices = len(slice_list) #total slices
        self.image_ID = np.zeros((self.slices), dtype = np.int)   # record image index
        self.slice_ID = np.zeros((self.slices), dtype = np.int)   # record slice index of image index
        self.image_filename = ['' for l in range(self.slices)]    # record image_filename for every slice
        self.label_filename = ['' for l in range(self.slices)]    # record label_filename for every slice
        self.average = np.zeros((self.slices))
        self.pixels = np.zeros((self.slices), dtype = np.int)
        for l in range(self.slices):
            s = slice_list[l].split(' ')
            self.image_ID[l] = s[0]                               # for example in training_X.txt first col 0 of line 1 indicate image 0
            self.slice_ID[l] = s[1]                               # for example in training_X.txt second col 0 of line 1 indicate slice 0 of image 0
            self.image_filename[l] = s[2]                         # for example in training_X.txt /DATA3_DB7/data/zhcao/segmentation/DATA/images_X/0001/0000.npy
            self.label_filename[l] = s[3]                         # for example in training_X.txt /DATA3_DB7/data/zhcao/segmentation/DATA/labels_X/0001/0000.npy
            self.average[l] = float(s[4])
            self.pixels[l] = int(s[organ_ID * 5])
            
        if slice_threshold <= 1:
            pixels_index = sorted(range(self.slices), key = lambda l: self.pixels[l])   #  sorted range(self.slices) based on self.pixels[l], l belongs to range(self.slices), lambda l returns self.pixels[l]
            last_index = int(math.floor((self.pixels > 0).sum() * slice_threshold))     # floor the integer
            min_pixels = self.pixels[pixels_index[-last_index]]                         # the last last_index pixel (sorted)

        else:
            min_pixels = slice_threshold
        self.active_index = [l for l, p in enumerate(self.pixels) if p >= min_pixels]   # return index, pixels which pixels>=min_pixel in unsorted pixels to get slices which has organs
        self.index_ = -1
        self.next_slice_index()

    def testing_setup(self):
        self.random = False
        image_list = open(testing_set_filename(current_fold), 'r').read().splitlines() #for example traing_FD0.txt 20 /DATA3_DB7/data/zhcao/segmentation/DATA/images/0021.npy /DATA3_DB7/data/zhcao/segmentation/DATA/labels/0021.npy
        self.testing_image_set = np.zeros((len(image_list)), dtype = np.int)
        for i in range(len(image_list)):
            s = image_list[i].split(' ')
            self.testing_image_set[i] = int(s[0])
        slice_list = open(list_testing[plane], 'r').read().splitlines() # for example traing_X.txt 0 0 /DATA3_DB7/data/zhcao/segmentation/DATA/images_X/0001/0000.npy /DATA3_DB7/data/zhcao/segmentation/DATA/labels_X/0001/0000.npy 100.0 0 0 0 0 0
        self.slices = len(slice_list) #total slices
        self.image_ID = np.zeros((self.slices), dtype = np.int)   # record image index
        self.slice_ID = np.zeros((self.slices), dtype = np.int)   # record slice index of image index
        self.image_filename = ['' for l in range(self.slices)]    # record image_filename for every slice
        self.label_filename = ['' for l in range(self.slices)]    # record label_filename for every slice
        self.average = np.zeros((self.slices))
        self.pixels = np.zeros((self.slices), dtype = np.int)
        for l in range(self.slices):
            s = slice_list[l].split(' ')
            self.image_ID[l] = s[0]                               # for example in training_X.txt first col 0 of line 1 indicate image 0
            self.slice_ID[l] = s[1]                               # for example in training_X.txt second col 0 of line 1 indicate slice 0 of image 0
            self.image_filename[l] = s[2]                         # for example in training_X.txt /DATA3_DB7/data/zhcao/segmentation/DATA/images_X/0001/0000.npy
            self.label_filename[l] = s[3]                         # for example in training_X.txt /DATA3_DB7/data/zhcao/segmentation/DATA/labels_X/0001/0000.npy
            self.average[l] = float(s[4])
            self.pixels[l] = int(s[organ_ID * 5])
            
        if slice_threshold <= 1:
            pixels_index = sorted(range(self.slices), key = lambda l: self.pixels[l])   #  sorted range(self.slices) based on self.pixels[l], l belongs to range(self.slices), lambda l returns self.pixels[l]
            last_index = int(math.floor((self.pixels > 0).sum() * slice_threshold))     # floor the integer
            min_pixels = self.pixels[pixels_index[-last_index]]                         # the last last_index pixel (sorted)

        else:
            min_pixels = slice_threshold
        self.active_index = [l for l, p in enumerate(self.pixels) if p >= min_pixels]   # return index, pixels which pixels>=min_pixel in unsorted pixels to get slices which has organs
        self.index_ = -1
        self.next_slice_index()

    def next_slice_index(self):
        while True:
            if self.random:
                self.index_ = random.randint(0, len(self.active_index) - 1)
            else:
                self.index_ += 1
                if self.index_ == len(self.active_index):
                    self.index_ = -1
            self.index1 = self.active_index[self.index_]
            if self.image_ID[self.index1] in self.training_image_set:
                break
        self.index0 = self.index1 - 1
        if self.index1 == 0 or self.slice_ID[self.index0] != self.slice_ID[self.index1] - 1:
            self.index0 = self.index1
        self.index2 = self.index1 + 1
        if self.index1 == self.slices - 1 or \
            self.slice_ID[self.index2] != self.slice_ID[self.index1] + 1:
            self.index2 = self.index1



    def load_data(self):

        if slice_thickness == 1:

            label1 = np.load(self.label_filename[self.index1])
            width = label1.shape[0]
            height = label1.shape[1]
            label = label1.reshape(1, width, height, 1)
                
            image1 = np.load(self.image_filename[self.index1])
            image = np.repeat(image1.reshape(width, height, 1), 3, axis = 2)
            image = image.reshape(1, width, height, 3)
            

        elif slice_thickness == 3:
                
            label_load = np.load(self.label_filename[self.index1])
            label1 = label_load.copy()
            
            width = label1.shape[0]
            height = label1.shape[1]

            label_load = np.load(self.label_filename[self.index0])
            label0 = label_load.copy()
            
            #label0 = np.load(self.label_filename[self.index0])

            label_load = np.load(self.label_filename[self.index2])
            label2 = label_load.copy()
            
            #label2 = np.load(self.label_filename[self.index2])


            image_load = np.load(self.image_filename[self.index0])
            image0 = image_load.copy()
            
            #image0 = np.load(self.image_filename[self.index0])

            image_load = np.load(self.image_filename[self.index1])
            image1 = image_load.copy()
            
            #image1 = np.load(self.image_filename[self.index1])

            image_load = np.load(self.image_filename[self.index2])
            image2 = image_load.copy()
            
            #image2 = np.load(self.image_filename[self.index2])

            del label_load
            del image_load
            
            image1_rgb = np.repeat(image1.reshape(width, height, 1), 3, axis = 2)
            image0_rgb = np.repeat(image0.reshape(width, height, 1), 3, axis = 2)
            image2_rgb = np.repeat(image2.reshape(width, height, 1), 3, axis = 2)


            image = np.concatenate((image0_rgb.reshape(1, width, height, 3), \
                image1_rgb.reshape(1, width, height, 3), image2_rgb.reshape(1, width, height, 3)), axis=0)

            label = np.concatenate((label0.reshape(1, width, height, 1), \
                label1.reshape(1, width, height, 1), label2.reshape(1, width, height, 1)), axis=0)

        image = image.astype(np.float32)
        image[image < low_range] = low_range
        image[image > high_range] = high_range
        image = (image - low_range) / (high_range - low_range)
        label = is_organ(label, organ_ID).astype(np.int32)
        print(self.label_filename[self.index1])

        print(self.image_filename[self.index1])

        return image, label


def get_next_batch(data, batch_size=32):

    batches = []
    image_mini_batch = []
    label_mini_batch = []
    index_mini_batch = 0

    while index_mini_batch < batch_size:

        data.next_slice_index()

        image, label = data.load_data()
        image_mini_batch.append(image)
        label_mini_batch.append(label)

        index_mini_batch += 1

    batches.append((np.array(image_mini_batch), np.array(label_mini_batch)))

    return batches

def read_images(data, batch_size):

    batch = get_next_batch(data, batch_size)
    image_width = batch[0][0][0].shape[1]
    image_height = batch[0][0][0].shape[2]
    image_channel = batch[0][0][0].shape[3]
    label_width = batch[0][1][0].shape[1]
    label_height = batch[0][1][0].shape[2]
    label_channel = batch[0][1][0].shape[3]
    image = np.reshape(batch[0][0], [batch_size, slice_thickness, image_width, image_height, image_channel])
    label = np.reshape(batch[0][1], [batch_size, slice_thickness, label_width, label_height, label_channel])

    return image, label

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert(data, name):


    tfrecord_dir = '/media/jionie/Disk1/TRAINING_DATA/'
    filename = name + '.tfrecords'
    print('Writting',filename)

    
    writer = tf.python_io.TFRecordWriter(os.path.join(tfrecord_dir, filename))

    while (data.index_ != -1):
        
        print('reading image begin')

        start_time = time.time()
        image, label = read_images(data, batch_size)
        duration = time.time() - start_time

        print("reading image end , cost %d sec" %duration)

        #convert to tfrecords
        print('convert to tfrecord begin')
        start_time = time.time()
        IMG_BATCH_SIZE = image.shape[0]
        IMG_SLICE_THICKNESS = image.shape[1]
        IMG_WIDTH = image.shape[2]
        IMG_HEIGHT = image.shape[3]
        IMG_CHANNELS = image.shape[4]

        LABEL_BATCH_SIZE = label.shape[0]
        LABEL_SLICE_THICKNESS = label.shape[1]
        LABEL_WIDTH = label.shape[2]
        LABEL_HEIGHT = label.shape[3]
        LABEL_CHANNELS = label.shape[4]

        img_raw = image.tobytes()
        label_raw = label.tobytes()
            
        example = tf.train.Example(features=tf.train.Features(feature={
            'image': _bytes_feature(img_raw),
            'image_batch_size': _int64_feature(IMG_BATCH_SIZE),
            'image_slice_thickness': _int64_feature(IMG_SLICE_THICKNESS),
            'image_width': _int64_feature(IMG_WIDTH),
            'image_height': _int64_feature(IMG_HEIGHT),
            'image_cahnnels': _int64_feature(IMG_CHANNELS),
            'label': _bytes_feature(label_raw),
            'label_batch_size': _int64_feature(LABEL_BATCH_SIZE),
            'label_slice_thickness': _int64_feature(LABEL_SLICE_THICKNESS),
            'label_width': _int64_feature(LABEL_WIDTH),
            'label_height': _int64_feature(LABEL_HEIGHT),
            'label_channels': _int64_feature(LABEL_CHANNELS)
            }))
        duration = time.time() - start_time
        print('convert to tfrecord end , cost %d sec' %duration)
       
        writer.write(example.SerializeToString())

    writer.close()
    print('Writting End')

def main(argv):

    data = Data()


    if mode == 'training':
        data.training_setup()

    if mode == 'testing':
        data.testing_setup()

    convert(data, 'training_' + plane)

if __name__ == '__main__':
    tf.app.run()