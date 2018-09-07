#!/usr/bin/env Python
# coding=utf-8
import numpy as np
import os
import sys
import shutil
import time
import logging
import urllib
import math
import random
from PIL import Image
import tensorflow as tf
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral, create_pairwise_gaussian, unary_from_softmax

sys.path.append("/media/jionie/Disk1/OrganSegC2F/models-master/slim/")

# Add path to the cloned library
sys.path.append("/media/jionie/Disk1/OrganSegC2F/tf-image-segmentation-master")

from tf_image_segmentation.models.fcn_8s_new import FCN_8s_new
from tf_image_segmentation.models.fcn_8s_dis import FCN_8s_dis
from loss import *
import matplotlib.pyplot as plt


train_plane = "Y"
fcn_8s_checkpoint_path = '/media/jionie/Disk1/checkpoints/fcn_8s_checkpoint/model_fcn8s_final.ckpt'  
data_path = "/media/jionie/Disk1"
current_fold = 3
organ_number = 1
low_range = 100
high_range = 240
slice_threshold = 1
slice_thickness = 3
training_margin = 20
testing_margin = 20
random_prob = 0.5
sample_batch = 1
organ_ID = 1
wd = 5e-4
initial_learning_rate = 1e-4
gamma = 0.5
checkpoint_path = '/media/jionie/Disk1/checkpoints/fine/fcn_vgg/'
summary_save_dir1 = '/media/jionie/Disk1/train_summary/fine/fcn_vgg/fine-' + train_plane + str(current_fold)
summary_save_dir2 = '/media/jionie/Disk1/train_summary/fine_eval/fcn_vgg/fine-' + train_plane + str(current_fold)
batch_size = 1
is_save = True
number_of_classes = 21
max_iterations = round(60000/batch_size)
result_path = os.path.join(data_path, '/results')
is_retrain = False



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
coarse_list_training = {}
fine_list_training = {}
coarse_list_retraining = {}
fine_list_retraining = {}
for plane in ['X', 'Y', 'Z']:
    coarse_list_training[plane] = os.path.join(list_path, 'coarse_' + plane + '.txt')
    fine_list_training[plane] = os.path.join(list_path, 'fine_' + plane + '.txt')



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

    def __init__(self, mode):
        self.mode = mode
    
    def setup(self):
        if self.mode == 'train':
            with open(training_set_filename(current_fold), 'r') as f1:
                image_list = f1.read().splitlines() #for example traing_FD0.txt 20 /DATA3_DB7/data/zhcao/segmentation/DATA/images/0021.npy /DATA3_DB7/data/zhcao/segmentation/DATA/labels/0021.npy
        if self.mode == 'test':
            with open(testing_set_filename(current_fold), 'r') as f1:
                image_list = f1.read().splitlines() 
        self.training_image_set = np.zeros((len(image_list)), dtype = np.int)
        for i in range(len(image_list)):
            s = image_list[i].split(' ')
            self.training_image_set[i] = int(s[0])

        if is_retrain :
            with open(coarse_list_retraining[train_plane], 'r') as f2:
                slice_list = f2.read().splitlines()    
        else:
            with open(coarse_list_training[train_plane], 'r') as f2:
                slice_list = f2.read().splitlines() # for example traing_X.txt 0 0 /DATA3_DB7/data/zhcao/segmentation/DATA/images_X/0001/0000.npy /DATA3_DB7/data/zhcao/segmentation/DATA/labels_X/0001/0000.npy 100.0 0 0 0 0 0
        
        self.slices = len(slice_list)
        self.image_ID = np.zeros((self.slices), dtype = np.int)
        self.slice_ID = np.zeros((self.slices), dtype = np.int)
        self.image_filename = ['' for l in range(self.slices)]
        self.label_filename = ['' for l in range(self.slices)]
        self.average = np.zeros((self.slices))
        self.pixels = np.zeros((self.slices), dtype = np.int)
        self.minA = np.zeros((self.slices), dtype = np.int)
        self.maxA = np.zeros((self.slices), dtype = np.int)
        self.minB = np.zeros((self.slices), dtype = np.int)
        self.maxB = np.zeros((self.slices), dtype = np.int)

        for l in range(self.slices):
            s = slice_list[l].split(' ')
            self.image_ID[l] = s[0]
            self.slice_ID[l] = s[1]
            self.image_filename[l] = s[2]
            self.label_filename[l] = s[3]
            self.average[l] = float(s[4])
            self.pixels[l] = int(s[organ_ID * 5])
            self.minA[l] = int(s[organ_ID * 5 + 1])
            self.maxA[l] = int(s[organ_ID * 5 + 2])
            self.minB[l] = int(s[organ_ID * 5 + 3])
            self.maxB[l] = int(s[organ_ID * 5 + 4])

        if slice_threshold <= 1:
            pixels_index = sorted(range(self.slices), key = lambda l: self.pixels[l])
            last_index = int(math.floor((self.pixels > 0).sum() * slice_threshold))
            min_pixels = self.pixels[pixels_index[-last_index]]
        else:
            min_pixels = slice_threshold

        self.active_index = [l for l, p in enumerate(self.pixels) if p >= min_pixels]   # return index, pixels which pixels>=min_pixel in unsorted pixels to get slices which has organs
        self.active_index = [p for l, p in enumerate(self.active_index) if self.image_ID[p] in self.training_image_set ]
        self.index_ = -1
        self.next_slice_index()


    def shuffle_data(self):

        random.shuffle(self.active_index)

    def next_slice_index(self):
        while True:
            self.index_ += 1
            if self.index_ == len(self.active_index):
                self.index_ = 0
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
        margin_mean = training_margin
        margin_random = margin_mean
        if random.randint(0, 100 - 1) >= 100 * random_prob:
            self.left = margin_mean
            self.right = margin_mean
            self.top = margin_mean
            self.bottom = margin_mean
        else:
            a = np.zeros(sample_batch * 4, dtype = np.uint16)
            for i in range(sample_batch * 4):
                a[i] = random.randint(margin_mean - margin_random, margin_mean + margin_random)
            self.left = int(a[0: sample_batch].sum() / sample_batch)
            self.right = int(a[sample_batch: sample_batch * 2].sum() / sample_batch)
            self.top = int(a[sample_batch * 2: sample_batch * 3].sum() / sample_batch)
            self.bottom = int(a[sample_batch * 3: sample_batch * 4].sum() / sample_batch)



    def load_data(self):

        if slice_thickness == 1:

            label1 = np.load(self.label_filename[self.index1])
            width = label1.shape[0]
            height = label1.shape[1]
            label = np.repeat(label1.reshape(width, height, 1), 3, axis = 2)
                
            image1 = np.load(self.image_filename[self.index1]).astype(np.float32)
            image = np.repeat(image1.reshape(width, height, 1), 3, axis = 2)

            minA = self.minA[self.index1]
            maxA = self.maxA[self.index1]
            minB = self.minB[self.index1]
            maxB = self.maxB[self.index1]

        elif slice_thickness == 3:

            label1 = np.load(self.label_filename[self.index1])
            
            width = label1.shape[0]
            height = label1.shape[1]

            label0 = np.load(self.label_filename[self.index0])
            label2 = np.load(self.label_filename[self.index2])

            image1 = np.load(self.image_filename[self.index1]).astype(np.float32)
            image0 = np.load(self.image_filename[self.index0]).astype(np.float32)
            image2 = np.load(self.image_filename[self.index2]).astype(np.float32)
            

            image = np.concatenate((image0.reshape(width, height, 1), \
                image1.reshape(width, height, 1), image2.reshape(width, height, 1)), axis=2)

            label = np.concatenate((label0.reshape(width, height, 1), \
                label1.reshape(width, height, 1), label2.reshape(width, height, 1)), axis=2)


            minA = min(self.minA[self.index0], self.minA[self.index1], self.minA[self.index2])
            maxA = max(self.maxA[self.index0], self.maxA[self.index1], self.maxA[self.index2])
            minB = min(self.minB[self.index0], self.minB[self.index1], self.minB[self.index2])
            maxB = max(self.maxB[self.index0], self.maxB[self.index1], self.maxB[self.index2])


        image = image[max(minA - self.left, 0): min(maxA + self.right + 1, width), \
            max(minB - self.top, 0): min(maxB + self.bottom + 1, height), :].astype(np.float32)
        image[image < low_range] = low_range
        image[image > high_range] = high_range
        image = (image - low_range) / (high_range - low_range)
        image = (image*255).astype(np.int32)
        label = label[max(minA - self.left, 0): min(maxA + self.right + 1, width), \
            max(minB - self.top, 0): min(maxB + self.bottom + 1, height), :].astype(np.int32)
        label = is_organ(label, organ_ID).astype(np.int32)
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



def resized_parameter(input_parameter, multiple):

    input_parameter = float(input_parameter)
    output_parameter = math.ceil(input_parameter / multiple) * multiple

    return output_parameter


####################################################################################################
# computing the DSC together with other values based on the label and prediction volumes
def DSC_computation(label, pred):
    pred_sum = pred.sum()
    label_sum = label.sum()
    inter_sum = np.multiply(pred, label)
    inter_sum = inter_sum.sum()
    return 2 * float(inter_sum) / (pred_sum + label_sum), inter_sum, pred_sum, label_sum


####################################################################################################

def post_processing(F, S, threshold, organ_ID):
    if F.sum() == 0:
        return F
    if F.sum() >= np.product(F.shape) / 2:
        return F
    height  = F.shape[0]
    width = F.shape[1]
    depth = F.shape[2]
    ll = np.array(np.nonzero(S))
    marked = np.zeros_like(F, dtype = np.bool)
    queue = np.zeros((F.sum(), 3), dtype = np.int)
    volume = np.zeros(F.sum(), dtype = np.int)
    head = 0
    tail = 0
    bestHead = 0
    bestTail = 0
    bestHead2 = 0
    bestTail2 = 0
    for l in range(ll.shape[1]):
        if not marked[ll[0, l], ll[1, l], ll[2, l]]:
            temp = head
            marked[ll[0, l], ll[1, l], ll[2, l]] = True
            queue[tail, :] = [ll[0, l], ll[1, l], ll[2, l]]
            tail = tail + 1
            while (head < tail):
                t1 = queue[head, 0]
                t2 = queue[head, 1]
                t3 = queue[head, 2]
                if t1 > 0 and F[t1 - 1, t2, t3] and not marked[t1 - 1, t2, t3]:
                    marked[t1 - 1, t2, t3] = True
                    queue[tail, :] = [t1 - 1, t2, t3]
                    tail = tail + 1
                if t1 < height - 1 and F[t1 + 1, t2, t3] and not marked[t1 + 1, t2, t3]:
                    marked[t1 + 1, t2, t3] = True
                    queue[tail, :] = [t1 + 1, t2, t3]
                    tail = tail + 1
                if t2 > 0 and F[t1, t2 - 1, t3] and not marked[t1, t2 - 1, t3]:
                    marked[t1, t2 - 1, t3] = True
                    queue[tail, :] = [t1, t2 - 1, t3]
                    tail = tail + 1
                if t2 < width - 1 and F[t1, t2 + 1, t3] and not marked[t1, t2 + 1, t3]:
                    marked[t1, t2 + 1, t3] = True
                    queue[tail, :] = [t1, t2 + 1, t3]
                    tail = tail + 1
                if t3 > 0 and F[t1, t2, t3 - 1] and not marked[t1, t2, t3 - 1]:
                    marked[t1, t2, t3 - 1] = True
                    queue[tail, :] = [t1, t2, t3 - 1]
                    tail = tail + 1
                if t3 < depth - 1 and F[t1, t2, t3 + 1] and not marked[t1, t2, t3 + 1]:
                    marked[t1, t2, t3 + 1] = True
                    queue[tail, :] = [t1, t2, t3 + 1]
                    tail = tail + 1
                head = head + 1
            if tail - temp > bestTail - bestHead:
                bestHead2 = bestHead
                bestTail2 = bestTail
                bestHead = temp
                bestTail = tail
            elif tail - temp > bestTail2 - bestHead2:
                bestHead2 = temp
                bestTail2 = tail
            volume[temp: tail] = tail - temp
    volume = volume[0: tail]
    target_voxel = np.where(volume >= (bestTail - bestHead) * threshold)
    F0 = np.zeros_like(F, dtype = np.bool)
    F0[tuple(map(tuple, np.transpose(queue[target_voxel, :])))] = True
    return F0

####################################################################################################
# dense CRF
def dense_crf(probs, img=None, n_iters=10, n_classes=2,
              sxy_gaussian=(1,1), compat_gaussian=4,
              kernel_gaussian=dcrf.DIAG_KERNEL,
              normalisation_gaussian=dcrf.NORMALIZE_SYMMETRIC,
              sxy_bilateral=(10, 10), compat_bilateral=5,
              srgb_bilateral=(5, 5, 5),
              kernel_bilateral=dcrf.DIAG_KERNEL,
              normalisation_bilateral=dcrf.NORMALIZE_SYMMETRIC):
    """DenseCRF over unnormalised predictions.
       More details on the arguments at https://github.com/lucasb-eyer/pydensecrf.
    
    Args:
      probs: class probabilities per pixel.
      img: if given, the pairwise bilateral potential on raw RGB values will be computed.
      n_iters: number of iterations of MAP inference.
      sxy_gaussian: standard deviations for the location component of the colour-independent term.
      compat_gaussian: label compatibilities for the colour-independent term (can be a number, a 1D array, or a 2D array).
      kernel_gaussian: kernel precision matrix for the colour-independent term (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
      normalisation_gaussian: normalisation for the colour-independent term (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).
      sxy_bilateral: standard deviations for the location component of the colour-dependent term.
      compat_bilateral: label compatibilities for the colour-dependent term (can be a number, a 1D array, or a 2D array).
      srgb_bilateral: standard deviations for the colour component of the colour-dependent term.
      kernel_bilateral: kernel precision matrix for the colour-dependent term (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
      normalisation_bilateral: normalisation for the colour-dependent term (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).
      
    Returns:
      Refined predictions after MAP inference.
    """
    _, h, w, _ = probs.shape
    
    
    probs = probs[0].transpose(2, 0, 1).copy(order='C') # Need a contiguous array.
    
    d = dcrf.DenseCRF2D(w, h, n_classes) # Define DenseCRF model.
    U = unary_from_softmax(probs) # Unary potential.
    U = U.reshape((n_classes, -1)) # Needs to be flat.
    d.setUnaryEnergy(U)
    energy = create_pairwise_gaussian(sxy_gaussian, [w, h])
    d.addPairwiseEnergy(energy, compat=compat_gaussian)

    if img is not None:
        assert(img.shape[1:3] == (h, w)), "The image height and width must coincide with dimensions of the logits."
        energy = create_pairwise_bilateral(sdims=sxy_bilateral, schan=srgb_bilateral[0], img=img, chdim=-1)
        d.addPairwiseEnergy(energy, compat=compat_bilateral)

    Q = d.inference(n_iters)
    preds = np.array(Q, dtype=np.float32).reshape((n_classes, h, w)).transpose(1, 2, 0)
    return np.expand_dims(preds, 0)
