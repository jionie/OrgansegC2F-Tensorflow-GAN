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
from utils import *


sys.path.append("/media/jionie/Disk1/OrganSegC2F/models-master/slim/")

# Add path to the cloned library
sys.path.append("/media/jionie/Disk1/OrganSegC2F/tf-image-segmentation-master")

from tf_image_segmentation.models.fcn_8s import FCN_8s
from tf_image_segmentation.models.fcn_8s_new import FCN_8s_new
from loss import *
import matplotlib.pyplot as plt


data_path = "/media/jionie/Disk1"
organ_number = 1
low_range = 100
high_range = 240
slice_thickness = 3
threshold = 1
training_margin = 20
testing_margin = 20
organ_ID = 1
max_rounds = 10
number_of_classes=21
batch_size=1

def FCN_model(images, labels, training, reuse=False, finetune=False):

    upsampled_logits_batch_1, fcn_8s_variables_mapping = FCN_8s_new(image_batch_tensor=images,
                                            number_of_classes=number_of_classes,
                                            new_number_of_classes=3,
                                            is_training=training,
                                            is_reuse=reuse,
                                            is_finetune=finetune)

    upsampled_logits_batch_1 = tf.cast(upsampled_logits_batch_1, tf.float32)
    pred = tf.multiply(tf.nn.sigmoid(upsampled_logits_batch_1), tf.constant(255.0, dtype=tf.float32))
    pred0 = tf.where(tf.greater(pred, tf.multiply(tf.ones_like(pred, tf.float32), tf.constant(128.0, dtype=tf.float32))), \
    tf.ones_like(pred, tf.float32), tf.zeros_like(pred, tf.float32))

    up_loss, down_loss, up_DSC, down_DSC = DSC_loss_new(upsampled_logits_batch_1, pred0, labels)

    total_up_loss = up_loss
    total_down_loss = down_loss
    total_up_DSC = up_DSC
    total_down_DSC = down_DSC

               
    total_DSC = tf.div(2*total_up_DSC, total_down_DSC)
    total_loss = tf.subtract(tf.constant(1.0, dtype=tf.float32),tf.div(total_up_loss, total_down_loss))

    return total_loss, total_DSC, fcn_8s_variables_mapping, pred0, upsampled_logits_batch_1





if __name__ == '__main__':


    images = tf.placeholder(dtype=tf.uint8, shape=[1, None, None, 3],
                                name='training_image')

    labels = tf.placeholder(dtype=tf.int32, shape=[1, None, None, 3],
                                name='training_label')

    _, total_DSC, _, pred0, logits= FCN_model(images, labels, False, False, False)

    for current_fold in range(0, 2):
        volume_list = open(testing_set_filename(current_fold), 'r').read().splitlines()

        while volume_list[len(volume_list) - 1] == '':
            volume_list.pop()


        coarse_result_directory = '/media/jionie/Disk1/results/coarse/fcn_vgg/test/fusion:X_Y_Z_' + str(current_fold) + '/'



        DSC = np.zeros((max_rounds + 1, len(volume_list)))
        DSC_90 = np.zeros((len(volume_list)))
        DSC_95 = np.zeros((len(volume_list)))
        DSC_98 = np.zeros((len(volume_list)))
        DSC_99 = np.zeros((len(volume_list)))
        coarse2fine_result_directory = '/media/jionie/Disk1/results/coarse2fine/fcn_vgg/test/fusion:X_Y_Z_' + str(current_fold) + '/'

        if not os.path.exists(coarse2fine_result_directory):
            os.makedirs(coarse2fine_result_directory)

        result_file = coarse2fine_result_directory + 'results.txt'

        output = open(result_file, 'w')
        output.close()



        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        
        
        with tf.Session() as sess:
        
            for i in range(len(volume_list)):
                start_time = time.time()
                print('Testing ' + str(i + 1) + ' out of ' + str(len(volume_list)) + ' testcases.')
                output = open(result_file, 'a+')
                output.write('  Testcase ' + str(i + 1) + ':\n')
                output.close()
                s = volume_list[i].split(' ')
                label = np.load(s[2])
                label = is_organ(label, organ_ID).astype(np.uint8)
                
                image = np.load(s[1])

                print('  Data loading is finished: ' + str(time.time() - start_time) + ' second(s) elapsed.')
                terminated = False
                for r in range(max_rounds + 1):

                    volume_file = coarse2fine_result_directory + 'R' + str(r) + '_' +  str(i+1) + '.npz'

                    if terminated or not os.path.isfile(volume_file):
                        terminated = True
                        pred = np.zeros_like(label, dtype = np.int8)
                        if r == 0:
                            coarse_volume_file = coarse_result_directory + 'F2P_' + str(i+1) + '.npz'
                            volume_data = np.load(coarse_volume_file)
                            pred = volume_data['volume']
                            print('    Fusion is finished: ' + \
                                str(time.time() - start_time) + ' second(s) elapsed.')
                        else:
                            if pred_prev.sum() == 0:
                                continue
                            pred_ = np.zeros_like(label, dtype = np.float32)
                            for plane in ['X', 'Y', 'Z']:
                                
                                if plane == 'X':
                                    model_file = '/media/jionie/Disk1/checkpoints/fine/fcn_vgg/X/X_' + str(current_fold) + '/' + 'fine-X-' + str(current_fold)
                                
                                elif plane == 'Y':
                                    model_file = '/media/jionie/Disk1/checkpoints/fine/fcn_vgg/Y/Y_' + str(current_fold) + '/' + 'fine-Y-' + str(current_fold)
                                
                                else:
                                    model_file = '/media/jionie/Disk1/checkpoints/fine/fcn_vgg/Z/Z_' + str(current_fold) + '/' + 'fine-Z-' + str(current_fold)
                                
                                sess.run(init)

                                print('Tesing model of ' + model_file)

                                saver.restore(sess, model_file)

                                pred__ = np.zeros_like(image, dtype = np.float32)
                                
                                minR = 0
                                if plane == 'X':
                                    maxR = label.shape[0]
                                elif plane == 'Y':
                                    maxR = label.shape[1]
                                elif plane == 'Z':
                                    maxR = label.shape[2]


                                for j in range(minR, maxR):

                                    print('Processing ' + str(j) + ' slice of ' + str(maxR) + ' slices.')


                                    if slice_thickness == 1:
                                        sID = [j, j, j]

                                        if plane == 'X':

                                            image_ = image[j, :, :]
                                            label_ = pred_prev[j, :, :]
                                            width = image_.shape[0]
                                            height = image_.shape[1]

                                            test_image = np.repeat(image_.reshape(1, width, height, 1), 3, axis = 3)
                                            test_label = np.repeat(label_.reshape(1, width, height, 1), 3, axis = 3)
                                                    

                                        elif plane == 'Y':

                                            image_ = image[:, j, :]
                                            label_ = pred_prev[:, j, :]
                                            width = image_.shape[0]
                                            height = image_.shape[1]
                                            
                                            test_image = np.repeat(image_.reshape(1, width, height, 1), 3, axis = 3)
                                            test_label = np.repeat(label_.reshape(1, width, height, 1), 3, axis = 3)

                                        elif plane == 'Z':

                                            image_ = image[:, :, j]
                                            label_ = pred_prev[:, :, j]
                                            width = image_.shape[0]
                                            height = image_.shape[1]

                                            test_image = np.repeat(image_.reshape(1, width, height, 1), 3, axis = 3)
                                            test_label = np.repeat(label_.reshape(1, width, height, 1), 3, axis = 3)

                                    elif slice_thickness == 3:

                                        sID = [max(minR, j - 1), j, min(maxR - 1, j + 1)]

                                        if plane == 'X':

                                            image_0 = image[max(minR, j - 1), :, :]
                                            image_1 = image[j, :, :]
                                            image_2 = image[min(maxR - 1, j + 1), :, :]
                                            label_0 = pred_prev[max(minR, j - 1), :, :]
                                            label_1 = pred_prev[j, :, :]
                                            label_2 = pred_prev[min(maxR - 1, j + 1), :, :]
                                            width = image_0.shape[0]
                                            height = image_0.shape[1]

                                            test_image = np.concatenate((image_0.reshape(1, width, height, 1), \
                                            image_1.reshape(1, width, height, 1), image_2.reshape(1, width, height, 1)), axis=3)
                                            test_label = np.concatenate((label_0.reshape(1, width, height, 1), \
                                            label_1.reshape(1, width, height, 1), label_2.reshape(1, width, height, 1)), axis=3)

                                        elif plane == 'Y':

                                            image_0 = image[:, max(minR, j - 1), :]
                                            image_1 = image[:, j, :]
                                            image_2 = image[:, min(maxR - 1, j + 1), :]
                                            label_0 = pred_prev[:, max(minR, j - 1), :]
                                            label_1 = pred_prev[:, j, :]
                                            label_2 = pred_prev[:, min(maxR - 1, j + 1), :]
                                            width = image_0.shape[0]
                                            height = image_0.shape[1]

                                            test_image = np.concatenate((image_0.reshape(1, width, height, 1), \
                                            image_1.reshape(1, width, height, 1), image_2.reshape(1, width, height, 1)), axis=3)
                                            test_label = np.concatenate((label_0.reshape(1, width, height, 1), \
                                            label_1.reshape(1, width, height, 1), label_2.reshape(1, width, height, 1)), axis=3)

                                        elif plane == 'Z':

                                            image_0 = image[:, :, max(minR, j - 1)]
                                            image_1 = image[:, :, j]
                                            image_2 = image[:, :, min(maxR - 1, j + 1)]
                                            label_0 = pred_prev[:, :, max(minR, j - 1)]
                                            label_1 = pred_prev[:, :, j]
                                            label_2 = pred_prev[:, :, min(maxR - 1, j + 1)]
                                            width = image_0.shape[0]
                                            height = image_0.shape[1]

                                            test_image = np.concatenate((image_0.reshape(1, width, height, 1), \
                                            image_1.reshape(1, width, height, 1), image_2.reshape(1, width, height, 1)), axis=3)
                                            test_label = np.concatenate((label_0.reshape(1, width, height, 1), \
                                            label_1.reshape(1, width, height, 1), label_2.reshape(1, width, height, 1)), axis=3)
                                    
                                    if (test_label.sum()) == 0:
                                                continue

                                    width = test_label.shape[1]
                                    height = test_label.shape[2]
                                    arr = np.nonzero(test_label)
                                    minA = min(arr[1])
                                    maxA = max(arr[1])
                                    minB = min(arr[2])
                                    maxB = max(arr[2])
                                    print(minA, maxA, minB, maxB)
                                    
                                    test_image = test_image[:, max(minA - testing_margin, 0): \
                                        min(maxA + testing_margin + 1, width), \
                                        max(minB - testing_margin, 0): min(maxB + testing_margin + 1, height), :]

                                    test_label = test_label[:, max(minA - testing_margin, 0): \
                                        min(maxA + testing_margin + 1, width), \
                                        max(minB - testing_margin, 0): min(maxB + testing_margin + 1, height), :]
                                
                                    test_image[test_image > high_range] = high_range
                                    test_image[test_image < low_range] = low_range
                                    test_image = (test_image - low_range) / (high_range - low_range)
                                    test_image = (test_image*255).astype(np.uint8)

                                    
                                    image_width = test_image.shape[1]
                                    image_height = test_image.shape[2]
                                    image_channel = test_image.shape[3]
                                    label_width = test_label.shape[1]
                                    label_height = test_label.shape[2]
                                    label_channel = test_label.shape[3]
                                    


                                    resized_image_width = int(resized_parameter(image_width, 32))
                                    resized_image_height = int(resized_parameter(image_height, 32))
                                    resized_label_width = int(resized_parameter(label_width, 32))
                                    resized_label_height = int(resized_parameter(label_height, 32))


                                    add_zeros_image_width = np.zeros((batch_size, resized_image_width-image_width, image_height, image_channel)).astype(np.float32)
                                    test_image = np.concatenate((test_image, add_zeros_image_width), axis=1)
                                    
                                    add_zeros_image_height = np.zeros((batch_size, resized_image_width, resized_image_height-image_height, image_channel)).astype(np.float32)
                                    test_image = np.concatenate((test_image, add_zeros_image_height), axis=2)
                                    
                                    add_zeros_label_width = np.zeros((batch_size, resized_label_width-label_width, label_height, label_channel)).astype(np.int32)
                                    test_label = np.concatenate((test_label, add_zeros_label_width), axis=1)
                                    
                                    add_zeros_label_height = np.zeros((batch_size, resized_label_width, resized_label_height-label_height, label_channel)).astype(np.int32)
                                    test_label = np.concatenate((test_label, add_zeros_label_height), axis=2)

                                    print(test_image.shape, test_label.shape)

                                    feed_dict = {images: test_image,
                                            labels: test_label
                                            }



                                    DSC_0, pred_1, logits_0 = sess.run([total_DSC, pred0, logits], feed_dict=feed_dict)
                                    pred_final = pred_1
                                    #pred_final = np.zeros_like(logits_0, dtype = np.float32)

                                    #for k in range(logits_0.shape[3]):

                                        #logits_sparse_0 = np.ones_like(logits_0[:, :, : ,k], dtype=np.float32) - np.copy(logits_0[:, :, : ,k])
                                        #logits_sparse_1 = np.copy(logits_0[:, :, : ,k])

                                        #batch_ = logits_0.shape[0]
                                        #width_ = logits_0.shape[1]
                                        #height_ = logits_0.shape[2]
                                    

                                        #images_sparse = np.copy(test_image[:, :, :, k]).reshape([batch_, width_, height_, 1])
                                        #logits_sparse = np.concatenate([logits_sparse_0.reshape([batch_, width_, height_, 1]), \
                                        #logits_sparse_1.reshape([batch_, width_, height_, 1])], axis=3).astype(np.float32)
                                        #logits_sparse = dense_crf(logits_sparse, images_sparse)
                                        #pred_final[:, :, :, k] =  np.argmax(logits_sparse, axis=3).astype(np.float32)
                                        

                                    if slice_thickness == 1:
                                        
                                        pred_final = np.reshape(pred_final, [resized_label_width, resized_label_height, image_channel])

                                        if plane == 'X':

                                            pred__[j, max(minA - testing_margin, 0): \
                                                min(maxA + testing_margin + 1, width), \
                                                max(minB - testing_margin, 0): \
                                                min(maxB + testing_margin + 1, height)] = pred_final[0:image_width, 0:image_height, 0]

                                        elif plane == 'Y':

                                            pred__[max(minA - testing_margin, 0): \
                                                min(maxA + testing_margin + 1, width), j, \
                                                max(minB - testing_margin, 0): \
                                                min(maxB + testing_margin + 1, height)] = pred_final[0:image_width, 0:image_height, 0]

                                        elif plane == 'Z':

                                            pred__[max(minA - testing_margin, 0): \
                                                min(maxA + testing_margin + 1, width), \
                                                max(minB - testing_margin, 0): \
                                                min(maxB + testing_margin + 1, height), j] = pred_final[0:image_width, 0:image_height, 0]


                                    elif slice_thickness == 3:
                                        
                                        pred_final = np.reshape(pred_final, [resized_label_width, resized_label_height, image_channel])

                                        if plane == 'X':

                                            pred__[max(minR, j - 1): min(maxR, j + 2), \
                                                max(minA - testing_margin, 0): \
                                                min(maxA + testing_margin + 1, width), \
                                                max(minB - testing_margin, 0): \
                                                min(maxB + testing_margin + 1, height)] += \
                                                pred_final[0:image_width, 0:image_height, (1 if (j==0) else 0):(2 if (j==maxR-1) else 3)].transpose(2, 0, 1)

                                        elif plane == 'Y':

                                            pred__[max(minA - testing_margin, 0): \
                                                min(maxA + testing_margin + 1, width), \
                                                max(minR, j - 1): min(maxR, j + 2), \
                                                max(minB - testing_margin, 0): \
                                                min(maxB + testing_margin + 1, height)] += \
                                                pred_final[0:image_width, 0:image_height, (1 if (j==0) else 0):(2 if (j==maxR-1) else 3)].transpose(0, 2, 1)

                                        elif plane == 'Z':

                                            pred__[max(minA - testing_margin, 0): \
                                                min(maxA + testing_margin + 1, width), \
                                                max(minB - testing_margin, 0): \
                                                min(maxB + testing_margin + 1, height), \
                                                max(minR, j - 1): min(maxR, j + 2)] += \
                                                pred_final[0:image_width, 0:image_height, (1 if (j==0) else 0):(2 if (j==maxR-1) else 3)]


                                if slice_thickness == 3:

                                    if plane == 'X':

                                        pred__[minR, :, :] /= 2
                                        pred__[minR + 1: maxR - 1, :, :] /= 3
                                        pred__[maxR - 1, :, :] /= 2

                                    elif plane == 'Y':

                                        pred__[:, minR, :] /= 2
                                        pred__[:, minR + 1: maxR - 1, :] /= 3
                                        pred__[:, maxR - 1, :] /= 2

                                    elif plane == 'Z':
                                                
                                        pred__[:, :, minR] /= 2
                                        pred__[:, :, minR + 1: maxR - 1] /= 3
                                        pred__[:, :, maxR - 1] /= 2

                                print(' Testing is finished: ' + str(time.time() - start_time) + ' second(s) elapsed.')
                                pred_ = pred_ + pred__
                                
                            pred[pred_ > 3/2] = 1
                            #pred = post_processing(pred, pred, 0.5, organ_ID)
                            print('    Testing is finished: ' + \
                                str(time.time() - start_time) + ' second(s) elapsed.')
                        np.savez_compressed(volume_file, volume = pred)
                    else:
                        volume_data = np.load(volume_file)
                        pred = volume_data['volume']
                        print('    Testing result is loaded: ' + \
                            str(time.time() - start_time) + ' second(s) elapsed.')



                        
                    DSC[r, i], inter_sum, pred_sum, label_sum = DSC_computation(label, pred)
                    print('      DSC = 2 * ' + str(inter_sum) + ' / (' + str(pred_sum) + ' + ' + \
                        str(label_sum) + ') = ' + str(DSC[r, i]) + ' .')
                    output = open(result_file, 'a+')
                    output.write('    Round ' + str(r) + ', ' + 'DSC = 2 * ' + str(inter_sum) + ' / (' + \
                        str(pred_sum) + ' + ' + str(label_sum) + ') = ' + str(DSC[r, i]) + ' .\n')
                    output.close()
                    if pred_sum == 0 and label_sum == 0:
                        DSC[r, i] = 0
                    if r > 0:
                        inter_DSC, inter_sum, pred_sum, label_sum = DSC_computation(pred_prev, pred)
                        if pred_sum == 0 and label_sum == 0:
                            inter_DSC = 1
                        print('        Inter-iteration DSC = 2 * ' + str(inter_sum) + ' / (' + \
                            str(pred_sum) + ' + ' + str(label_sum) + ') = ' + str(inter_DSC) + ' .')
                        output = open(result_file, 'a+')
                        output.write('      Inter-iteration DSC = 2 * ' + str(inter_sum) + ' / (' + \
                            str(pred_sum) + ' + ' + str(label_sum) + ') = ' + str(inter_DSC) + ' .\n')
                        output.close()
                        if DSC_90[i] == 0 and (r == max_rounds or inter_DSC >= 0.90):
                            DSC_90[i] = DSC[r, i]
                        if DSC_95[i] == 0 and (r == max_rounds or inter_DSC >= 0.95):
                            DSC_95[i] = DSC[r, i]
                        if DSC_98[i] == 0 and (r == max_rounds or inter_DSC >= 0.98):
                            DSC_98[i] = DSC[r, i]
                        if DSC_99[i] == 0 and (r == max_rounds or inter_DSC >= 0.99):
                            DSC_99[i] = DSC[r, i]
                    if r <= max_rounds:
                        pred_prev = np.copy(pred)
            for r in range(max_rounds + 1):
                print('Round ' + str(r) + ', ' + 'Average DSC = ' + str(np.mean(DSC[r, :])) + ' .')
                output = open(result_file, 'a+')
                output.write('Round ' + str(r) + ', ' + 'Average DSC = ' + str(np.mean(DSC[r, :])) + ' .\n')
                output.close()
            print('DSC threshold = 0.90, ' + 'Average DSC = ' + str(np.mean(DSC_90)) + ' .')
            print('DSC threshold = 0.95, ' + 'Average DSC = ' + str(np.mean(DSC_95)) + ' .')
            print('DSC threshold = 0.98, ' + 'Average DSC = ' + str(np.mean(DSC_98)) + ' .')
            print('DSC threshold = 0.99, ' + 'Average DSC = ' + str(np.mean(DSC_99)) + ' .')
            output = open(result_file, 'a+')
            output.write('DSC threshold = 0.90, ' + 'Average DSC = ' + str(np.mean(DSC_90)) + ' .\n')
            output.write('DSC threshold = 0.95, ' + 'Average DSC = ' + str(np.mean(DSC_95)) + ' .\n')
            output.write('DSC threshold = 0.98, ' + 'Average DSC = ' + str(np.mean(DSC_98)) + ' .\n')
            output.write('DSC threshold = 0.99, ' + 'Average DSC = ' + str(np.mean(DSC_99)) + ' .\n')
            output.close()
            print('The coarse-to-fine testing process is finished.')
