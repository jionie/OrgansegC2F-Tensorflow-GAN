#!/usr/bin/env Python
# coding=utf-8
from input_data_coarse import *

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
    total_loss = tf.subtract(tf.constant(1.0, dtype=tf.float32),tf.div(2*total_up_loss, total_down_loss))

    return total_loss, total_DSC, fcn_8s_variables_mapping, pred0



if __name__ == '__main__':

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    data = Data('train')
    data.setup()

    batch = get_next_batch(data, batch_size)
    image_channel = batch[0][0][0].shape[2]
    label_channel = batch[0][1][0].shape[2]

    eval_data = Data('test')
    eval_data.setup()
    batch_eval = get_next_batch(eval_data, batch_size)
    image_channel_eval = batch_eval[0][0][0].shape[2]
    label_channel_eval = batch_eval[0][1][0].shape[2]


    images = tf.placeholder(dtype=tf.int32, shape=[batch_size, None, None, image_channel],
                                name='train_images')

    labels = tf.placeholder(dtype=tf.int32, shape=[batch_size, None, None, label_channel],
                                name='train_labels')

    eval_images = tf.placeholder(dtype=tf.int32, shape=[batch_size, None, None, image_channel_eval],
                                name='eval_images')

    eval_labels = tf.placeholder(dtype=tf.int32, shape=[batch_size, None, None, label_channel_eval],
                                name='eval_labels')

    #config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.5

    loss_, DSC, fcn_8s_variables_mapping, pred1= FCN_model(images, labels, True, False, True)

    train_var_list = [v for v in tf.trainable_variables()
                        if 'beta' not in v.name and 'gamma' not in v.name]

    with tf.variable_scope("total_loss"):
        loss = loss_ + wd * tf.add_n([tf.nn.l2_loss(v) for v in train_var_list])


    eval_loss, eval_DSC, fcn_8s_variables_mapping, _= FCN_model(eval_images, eval_labels, False, True, False)

    
    global_step = tf.train.get_or_create_global_step()

    
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-5)
    # Batch norm requires update ops to be added as a dependency to the train_op
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    
    with tf.control_dependencies(update_ops):
        train_op = slim.learning.create_train_op(loss,
                                                optimizer,
                                                global_step=global_step,
                                                variables_to_train=train_var_list)

    tf.summary.scalar(name='train_loss', tensor=loss)
    tf.summary.scalar(name='train_accuracy', tensor=DSC)
    tf.summary.scalar(name='eval_loss', tensor=eval_loss)
    tf.summary.scalar(name='eval_accuracy', tensor=eval_DSC)

    train_summary_op = tf.summary.merge([tf.get_collection(tf.GraphKeys.SUMMARIES,'train_loss'), tf.get_collection(tf.GraphKeys.SUMMARIES,'train_accuracy')])
    eval_summary_op = tf.summary.merge([tf.get_collection(tf.GraphKeys.SUMMARIES,'eval_loss'), tf.get_collection(tf.GraphKeys.SUMMARIES,'eval_accuracy')])
   
    init = tf.global_variables_initializer()
    
    restore_variables = [v for v in tf.trainable_variables()]
    model_path='/DATA/data/zhcao/segmentation/DATA/checkpoints/coarse/fcn_vgg/Y/Y_0_2_v2/coarse-Y-2'

    init_fn = slim.assign_from_checkpoint_fn(model_path=model_path,
                                             var_list=restore_variables, ignore_missing_vars=True)


    saver = tf.train.Saver(max_to_keep=10)



    with tf.Session() as sess:

        sess.run(init)

        train_writer = tf.summary.FileWriter(summary_save_dir1 + '_continue_v2', sess.graph)
        eval_writer = tf.summary.FileWriter(summary_save_dir2 + '_continue_v2', sess.graph)

        init_fn(sess)

        print(len(data.active_index))
        num_epoch = round(40000/len(data.active_index)) + 1
        print(num_epoch)

        for epochs in range(3, int(num_epoch)):
            data.shuffle_data()
            eval_data.shuffle_data()
            for iterations in range(len(data.active_index)):
                    
                image_width = batch[0][0][0].shape[0]
                image_height = batch[0][0][0].shape[1]
                image_channel = batch[0][0][0].shape[2]
                label_width = batch[0][1][0].shape[0]
                label_height = batch[0][1][0].shape[1]
                label_channel = batch[0][1][0].shape[2]

                train_image = np.reshape(batch[0][0], [batch_size, image_width, image_height, image_channel]).astype(np.float32)
                train_label = np.reshape(batch[0][1], [batch_size, label_width, label_height, label_channel]).astype(np.int32)

                resized_image_width = int(resized_parameter(image_width, 32))
                resized_image_height = int(resized_parameter(image_height, 32))
                resized_label_width = int(resized_parameter(label_width, 32))
                resized_label_height = int(resized_parameter(label_height, 32))
                
                print(image_width, image_height, resized_image_width, resized_image_height)

                print(label_width, label_height, resized_label_width, resized_label_height)

                add_zeros_image_width = np.zeros((batch_size, resized_image_width-image_width, image_height, image_channel)).astype(np.float32)
                train_image = np.concatenate((train_image, add_zeros_image_width), axis=1)
                
                add_zeros_image_height = np.zeros((batch_size, resized_image_width, resized_image_height-image_height, image_channel)).astype(np.float32)
                train_image = np.concatenate((train_image, add_zeros_image_height), axis=2)
                
                add_zeros_label_width = np.zeros((batch_size, resized_label_width-label_width, label_height, label_channel)).astype(np.int32)
                train_label = np.concatenate((train_label, add_zeros_label_width), axis=1)
                
                add_zeros_label_height = np.zeros((batch_size, resized_label_width, resized_label_height-label_height, label_channel)).astype(np.int32)
                train_label = np.concatenate((train_label, add_zeros_label_height), axis=2)
                
                feed_dict_train = {images: train_image,
                            labels: train_label,
                            eval_images: train_image,
                            eval_labels: train_label
                            }

                train_loss, train_DSC, train_summary, _ = sess.run([loss, DSC, train_summary_op, train_op], feed_dict=feed_dict_train)

                print("iterations:        ", iterations)
                print("training_DSC:      ", train_DSC)
                print("training_loss:     ", train_loss)
                
                train_writer.add_summary(train_summary, global_step=epochs*len(data.active_index) + iterations)
                batch = get_next_batch(data, batch_size)

                if iterations % int((len(data.active_index))/(len(eval_data.active_index))) == 0:

                    image_width_eval = batch_eval[0][0][0].shape[0]
                    image_height_eval = batch_eval[0][0][0].shape[1]
                    image_channel_eval = batch_eval[0][0][0].shape[2]
                    label_width_eval = batch_eval[0][1][0].shape[0]
                    label_height_eval = batch_eval[0][1][0].shape[1]
                    label_channel_eval = batch_eval[0][1][0].shape[2]

                    eval_image = np.reshape(batch_eval[0][0], [batch_size, image_width_eval, image_height_eval, image_channel_eval]).astype(np.float32)
                    eval_label = np.reshape(batch_eval[0][1], [batch_size, label_width_eval, label_height_eval, label_channel_eval]).astype(np.int32)

                    resized_image_width_eval = int(resized_parameter(image_width_eval, 32))
                    resized_image_height_eval = int(resized_parameter(image_height_eval, 32))
                    resized_label_width_eval = int(resized_parameter(label_width_eval, 32))
                    resized_label_height_eval = int(resized_parameter(label_height_eval, 32))


                    print(image_width_eval, image_height_eval, resized_image_width_eval, resized_image_height_eval)

                    print(label_width_eval, label_height_eval, resized_label_width_eval, resized_label_height_eval)

                    add_zeros_image_width_eval = np.zeros((batch_size, resized_image_width_eval-image_width_eval, image_height_eval, image_channel_eval)).astype(np.float32)
                    eval_image = np.concatenate((eval_image, add_zeros_image_width_eval), axis=1)
                    
                    add_zeros_image_height_eval = np.zeros((batch_size, resized_image_width_eval, resized_image_height_eval-image_height_eval, image_channel_eval)).astype(np.float32)
                    eval_image = np.concatenate((eval_image, add_zeros_image_height_eval), axis=2)
                    
                    add_zeros_label_width_eval = np.zeros((batch_size, resized_label_width_eval-label_width_eval, label_height_eval, label_channel_eval)).astype(np.int32)
                    eval_label = np.concatenate((eval_label, add_zeros_label_width_eval), axis=1)
                    
                    add_zeros_label_height_eval = np.zeros((batch_size, resized_label_width_eval, resized_label_height_eval-label_height_eval, label_channel_eval)).astype(np.int32)
                    eval_label = np.concatenate((eval_label, add_zeros_label_height_eval), axis=2)

                    feed_dict_eval = {images: eval_image,
                            labels: eval_label,
                            eval_images: eval_image,
                            eval_labels: eval_label
                            }

                    
                    eval_loss0, eval_DSC0, eval_summary = sess.run([eval_loss, eval_DSC, eval_summary_op], feed_dict=feed_dict_eval)
                    eval_writer.add_summary(eval_summary, global_step= epochs*len(data.active_index) + iterations)
                    print("iterations:    ", iterations)
                    print("eval_DSC:      ", eval_DSC0)
                    print("eval_loss:     ", eval_loss0)

                    batch_eval = get_next_batch(eval_data, batch_size)


                if (is_save and iterations % 500 == 0):

                    if not os.path.exists(checkpoint_path + train_plane + '/' + train_plane + '_' \
                                          + str(current_fold) + '_' + str(epochs)  + '_v2/'  \
                                          + str(epochs*len(data.active_index) + iterations) + '/'): 

                        os.makedirs(checkpoint_path + train_plane + '/' + train_plane + '_' \
                                          + str(current_fold) + '_' + str(epochs)  + '_v2/'  \
                                          + str(epochs*len(data.active_index) + iterations) + '/')

                    saver.save(sess, checkpoint_path + train_plane + '/' + train_plane + '_' \
                                      + str(current_fold) + '_' + str(epochs)  + '_v2/' \
                                      + str(epochs*len(data.active_index) + iterations) + '/'  +'coarse-' \
                                      + train_plane, global_step=epochs*len(data.active_index) + iterations)







