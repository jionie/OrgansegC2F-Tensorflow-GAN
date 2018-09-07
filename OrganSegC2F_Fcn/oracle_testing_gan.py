#!/usr/bin/env Python
# coding=utf-8
from input_data_fine import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

slim = tf.contrib.slim

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


    test_result_epoch = '0'
    volume_list = open(testing_set_filename(current_fold), 'r').read().splitlines()

    while volume_list[len(volume_list) - 1] == '':
        volume_list.pop()

    DSC = np.zeros(len(volume_list))

    result_directory = '/media/jionie/Disk1/results/oracle/fcn_vgg/test/oracle-' + train_plane + str(testing_margin) + '_gan/' + \
    'fine-' + train_plane + '_epoch_' + test_result_epoch + '/' + '6000' + '/'


    if not os.path.exists(result_directory):
        os.makedirs(result_directory)

    result_file = result_directory  + 'result.txt'

    output = open(result_file, 'w')
    output.close()
    output = open(result_file, 'a+')
    output.write('Evaluating model ' + ':\n')
    output.close()


    images = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 3],
                                name='training_image')

    labels = tf.placeholder(dtype=tf.int32, shape=[1, None, None, 3],
                                name='training_label')

    _, total_DSC, _, pred0, logits= FCN_model(images, labels, False, False, False)


    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    restore_variables = [v for v in tf.trainable_variables()]
    model_file = '/media/jionie/Disk1/checkpoints/fine/fcn_vgg/Y_gan/Y_0_0/6000/fine-Y-6000'
    init_fn = slim.assign_from_checkpoint_fn(model_path=model_file, var_list=restore_variables, ignore_missing_vars=True)

    with tf.Session() as sess:

        sess.run(init)

        init_fn(sess)

        for i in range(len(volume_list)):
            
            start_time = time.time()
            s = volume_list[i].split(' ')
            volume_file =  result_directory + str(i + 1) + '.npz'
            label = np.load(s[2])
            label = is_organ(label, organ_ID).astype(np.int32)
            image = np.load(s[1]).astype(np.float32)
            print(s[1])


            pred = np.zeros_like(image, dtype = np.float32)
            
            if label.sum() > 0:
            
                minR = 0

                if train_plane == 'X':
                    maxR = label.shape[0]
                elif train_plane == 'Y':
                    maxR = label.shape[1]
                elif train_plane == 'Z':
                    maxR = label.shape[2]


                for j in range(minR, maxR):
                    
                    print('Processing ' + str(j) + ' slice of ' + str(maxR) + ' slices.')


                    if slice_thickness == 1:
                        sID = [j, j, j]

                        if train_plane == 'X':

                            image_ = image[j, :, :]
                            label_ = label[j, :, :]
                            width = image_.shape[0]
                            height = image_.shape[1]

                            test_image = np.repeat(image_.reshape(1, width, height, 1), 3, axis = 3)
                            test_label = np.repeat(label_.reshape(1, width, height, 1), 3, axis = 3)
                                    

                        elif train_plane == 'Y':

                            image_ = image[:, j, :]
                            label_ = label[:, j, :]
                            width = image_.shape[0]
                            height = image_.shape[1]
                            
                            test_image = np.repeat(image_.reshape(1, width, height, 1), 3, axis = 3)
                            test_label = np.repeat(label_.reshape(1, width, height, 1), 3, axis = 3)

                        elif train_plane == 'Z':

                            image_ = image[:, :, j]
                            label_ = label[:, :, j]
                            width = image_.shape[0]
                            height = image_.shape[1]

                            test_image = np.repeat(image_.reshape(1, width, height, 1), 3, axis = 3)
                            test_label = np.repeat(label_.reshape(1, width, height, 1), 3, axis = 3)

                    elif slice_thickness == 3:

                        sID = [max(minR, j - 1), j, min(maxR - 1, j + 1)]

                        if train_plane == 'X':

                            image_0 = image[max(minR, j - 1), :, :]
                            image_1 = image[j, :, :]
                            image_2 = image[min(maxR - 1, j + 1), :, :]
                            label_0 = label[max(minR, j - 1), :, :]
                            label_1 = label[j, :, :]
                            label_2 = label[min(maxR - 1, j + 1), :, :]
                            width = image_0.shape[0]
                            height = image_0.shape[1]

                            test_image = np.concatenate((image_0.reshape(1, width, height, 1), \
                            image_1.reshape(1, width, height, 1), image_2.reshape(1, width, height, 1)), axis=3)
                            test_label = np.concatenate((label_0.reshape(1, width, height, 1), \
                            label_1.reshape(1, width, height, 1), label_2.reshape(1, width, height, 1)), axis=3)

                        elif train_plane == 'Y':

                            image_0 = image[:, max(minR, j - 1), :]
                            image_1 = image[:, j, :]
                            image_2 = image[:, min(maxR - 1, j + 1), :]
                            label_0 = label[:, max(minR, j - 1), :]
                            label_1 = label[:, j, :]
                            label_2 = label[:, min(maxR - 1, j + 1), :]
                            width = image_0.shape[0]
                            height = image_0.shape[1]

                            test_image = np.concatenate((image_0.reshape(1, width, height, 1), \
                            image_1.reshape(1, width, height, 1), image_2.reshape(1, width, height, 1)), axis=3)
                            test_label = np.concatenate((label_0.reshape(1, width, height, 1), \
                            label_1.reshape(1, width, height, 1), label_2.reshape(1, width, height, 1)), axis=3)

                        elif train_plane == 'Z':

                            image_0 = image[:, :, max(minR, j - 1)]
                            image_1 = image[:, :, j]
                            image_2 = image[:, :, min(maxR - 1, j + 1)]
                            label_0 = label[:, :, max(minR, j - 1)]
                            label_1 = label[:, :, j]
                            label_2 = label[:, :, min(maxR - 1, j + 1)]
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
                    test_image = test_image*255

                    
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



                    DSC_0, pred_final, logits_final = sess.run([total_DSC, pred0, logits], feed_dict=feed_dict)


                    

                    if slice_thickness == 1:
                        
                        logits_final = np.reshape(logits_final, [resized_label_width, resized_label_height, image_channel])

                        if train_plane == 'X':

                            pred[j, max(minA - testing_margin, 0): \
                                min(maxA + testing_margin + 1, width), \
                                max(minB - testing_margin, 0): \
                                min(maxB + testing_margin + 1, height)] = logits_final[0:image_width, 0:image_height, 0]

                        elif train_plane == 'Y':

                            pred[max(minA - testing_margin, 0): \
                                min(maxA + testing_margin + 1, width), j, \
                                max(minB - testing_margin, 0): \
                                min(maxB + testing_margin + 1, height)] = logits_final[0:image_width, 0:image_height, 0]

                        elif train_plane == 'Z':

                            pred[max(minA - testing_margin, 0): \
                                min(maxA + testing_margin + 1, width), \
                                max(minB - testing_margin, 0): \
                                min(maxB + testing_margin + 1, height), j] = logits_final[0:image_width, 0:image_height, 0]


                    elif slice_thickness == 3:
                        
                        logits_final = np.reshape(logits_final, [resized_label_width, resized_label_height, image_channel])

                        if train_plane == 'X':

                            pred[max(minR, j - 1): min(maxR, j + 2), \
                                max(minA - testing_margin, 0): \
                                min(maxA + testing_margin + 1, width), \
                                max(minB - testing_margin, 0): \
                                min(maxB + testing_margin + 1, height)] += \
                                logits_final[0:image_width, 0:image_height, (1 if (j==0) else 0):(2 if (j==maxR-1) else 3)].transpose(2, 0, 1)

                        elif train_plane == 'Y':

                            pred[max(minA - testing_margin, 0): \
                                min(maxA + testing_margin + 1, width), \
                                max(minR, j - 1): min(maxR, j + 2), \
                                max(minB - testing_margin, 0): \
                                min(maxB + testing_margin + 1, height)] += \
                                logits_final[0:image_width, 0:image_height, (1 if (j==0) else 0):(2 if (j==maxR-1) else 3)].transpose(0, 2, 1)

                        elif train_plane == 'Z':

                            pred[max(minA - testing_margin, 0): \
                                min(maxA + testing_margin + 1, width), \
                                max(minB - testing_margin, 0): \
                                min(maxB + testing_margin + 1, height), \
                                max(minR, j - 1): min(maxR, j + 2)] += \
                                logits_final[0:image_width, 0:image_height, (1 if (j==0) else 0):(2 if (j==maxR-1) else 3)]


                if slice_thickness == 3:

                    if train_plane == 'X':

                        pred[minR, :, :] /= 2
                        pred[minR + 1: maxR - 1, :, :] /= 3
                        pred[maxR - 1, :, :] /= 2

                    elif train_plane == 'Y':

                        pred[:, minR, :] /= 2
                        pred[:, minR + 1: maxR - 1, :] /= 3
                        pred[:, maxR - 1, :] /= 2

                    elif train_plane == 'Z':
                                
                        pred[:, :, minR] /= 2
                        pred[:, :, minR + 1: maxR - 1] /= 3
                        pred[:, :, maxR - 1] /= 2

            print(' Testing is finished: ' + str(time.time() - start_time) + ' second(s) elapsed.')

            pred = np.int32(np.around(pred * 255))
            print(' Data saving is finished: ' + \
                str(time.time() - start_time) + ' second(s) elapsed.')

                
            pred_temp = np.zeros_like(pred, dtype = np.int32)
            pred_temp[np.where(pred >= 128)] = 1
            np.savez_compressed(volume_file, volume = pred_temp)

                
            DSC[i], inter_sum, pred_sum, label_sum = DSC_computation(label, pred_temp)
           

            print(' DSC = 2 * ' + str(inter_sum) + ' / (' + str(pred_sum) + \
                ' + ' + str(label_sum) + ') = ' + str(DSC[i]) + ' .')
                
            output = open(result_file, 'a+')
            output.write('  Testcase ' + str(i + 1) + ': DSC = 2 * ' + str(inter_sum) + ' / (' + \
                str(pred_sum) + ' + ' + str(label_sum) + ') = ' + str(DSC[i]) + ' .\n')
            output.close()
                
            if pred_sum == 0 and label_sum == 0:
                
                DSC[i] = 0

        print(' DSC computation is finished: ' + \
            str(time.time() - start_time) + ' second(s) elapsed.')
            
        print('average DSC = ' + str(np.mean(DSC[:])) + ' .')
        output = open(result_file, 'a+')
        output.write('average DSC = ' + str(np.mean(DSC[:])) + ' .\n')
        output.close()








