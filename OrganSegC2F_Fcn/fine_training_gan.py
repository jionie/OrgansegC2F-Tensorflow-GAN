#!/usr/bin/env Python
# coding=utf-8
from input_data_fine import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

slim = tf.contrib.slim

wd = 5e-4


def discriminator(images, training, reuse=False):
    
    logits, fcn_8s_dis_variables_mapping = FCN_8s_dis(image_batch_tensor=images,
                              number_of_classes=3,
                              is_training=training,
                              is_reuse=reuse)
    

    return tf.nn.sigmoid(logits), logits, fcn_8s_dis_variables_mapping

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

    return total_loss, total_DSC, fcn_8s_variables_mapping, pred0, tf.nn.sigmoid(upsampled_logits_batch_1)


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
    #config.gpu_options.allow_growth = True

    loss_, DSC, _, pred0, upsampled_logits = FCN_model(images, labels, True, False, True)
    eval_loss, eval_DSC, _, _, _= FCN_model(eval_images, eval_labels, False, True, False)

    
    def sigmoid_cross_entropy_with_logits(x, y):
      try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
      except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)
    
    true_input = labels
    fake_input = upsampled_logits

    D, D_logits, fcn_8s_dis_variables_mapping = discriminator(true_input, training=True, reuse=False)
    D_fake, D_fake_logits, _ = discriminator(fake_input, training=True, reuse=True)

    D_true_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(D_logits, tf.ones_like(D)))*0.1
    D_fake_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(D_fake_logits, tf.zeros_like(D_fake)))*0.1


    
    train_var_list_gen = [v for v in tf.trainable_variables()
                        if 'FCN_slice' in v.name and 'beta' not in v.name and 'gamma' not in v.name]

    train_var_list_dis = [v for v in tf.trainable_variables()
                        if 'FCN_dis' in v.name and 'beta' not in v.name and 'gamma' not in v.name]

    with tf.variable_scope("generator_loss"):
        G_loss = loss_ + wd * tf.add_n([tf.nn.l2_loss(v) for v in train_var_list_gen]) +\
        tf.reduce_mean(sigmoid_cross_entropy_with_logits(D_fake_logits, tf.ones_like(D_fake)))*0.1

    with tf.variable_scope("discriminator_loss"):
        D_loss = D_true_loss + D_fake_loss


    global_step = tf.train.get_or_create_global_step()
    
    optimizer_gen = tf.train.AdamOptimizer(learning_rate=1e-4)
    optimizer_dis = tf.train.AdamOptimizer(learning_rate=1e-4)
    # Batch norm requires update ops to be added as a dependency to the train_op
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    
    with tf.variable_scope("gen_adam_vars"):

        with tf.control_dependencies(update_ops):

            gen_optimizer = slim.learning.create_train_op(G_loss,
                                                          optimizer_gen,
                                                          global_step=global_step,
                                                          variables_to_train=train_var_list_gen)

    with tf.variable_scope("dis_adam_vars"):

        with tf.control_dependencies(update_ops):

            dis_optimizer = slim.learning.create_train_op(D_true_loss,
                                                          optimizer_dis,
                                                          global_step=global_step,
                                                          variables_to_train=train_var_list_dis)

    
    tf.add_to_collection("gen_optimizer", gen_optimizer)
    tf.add_to_collection("dis_optimizer_true", dis_optimizer)


    
    
    tf.summary.scalar(name='gen_loss', tensor=G_loss)
    tf.summary.scalar(name='dis_loss', tensor=D_loss)
    tf.summary.scalar(name='train_accuracy', tensor=DSC)
    tf.summary.scalar(name='eval_loss', tensor=eval_loss)
    tf.summary.scalar(name='eval_accuracy', tensor=eval_DSC)



    train_summary_op = tf.summary.merge([tf.get_collection(tf.GraphKeys.SUMMARIES,'gen_loss'), \
                                         tf.get_collection(tf.GraphKeys.SUMMARIES,'dis_loss'), \
                                         tf.get_collection(tf.GraphKeys.SUMMARIES,'train_accuracy')])

    eval_summary_op = tf.summary.merge([tf.get_collection(tf.GraphKeys.SUMMARIES,'eval_loss'), \
                                         tf.get_collection(tf.GraphKeys.SUMMARIES,'eval_accuracy')])


    init = tf.global_variables_initializer()

    restore_variables = [v for v in tf.trainable_variables()]

    
   
    model_file = checkpoint_path + train_plane + '/' + train_plane + '_' + str(current_fold) + '/' + 'fine-' + train_plane + '-' + str(current_fold)
    vgg_16_path = '/media/jionie/Disk1/checkpoints/vgg_16/vgg_16.ckpt'
    init_fn = slim.assign_from_checkpoint_fn(model_path=model_file, var_list=restore_variables, ignore_missing_vars=True)
    saver = tf.train.Saver(max_to_keep=10)

    with tf.Session() as sess:

        sess.run(init)
        train_writer = tf.summary.FileWriter(summary_save_dir1 + '-gan', sess.graph)
        eval_writer = tf.summary.FileWriter(summary_save_dir2 + '-gan', sess.graph)
        num_epoch = round(30000/len(data.active_index)) + 1  

        init_fn(sess)   

        mean_DSC_train = 0
        mean_DSC_eval = 0
        count = 0

        for epochs in range(num_epoch+1):
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

                resized_image_width = resized_parameter(image_width, 32)
                resized_image_height = resized_parameter(image_height, 32)
                resized_label_width = resized_parameter(label_width, 32)
                resized_label_height = resized_parameter(label_height, 32)

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

                
                _ = sess.run([dis_optimizer], feed_dict=feed_dict_train)
                _ = sess.run([gen_optimizer], feed_dict=feed_dict_train)
                _ = sess.run([gen_optimizer], feed_dict=feed_dict_train)
                _ = sess.run([gen_optimizer], feed_dict=feed_dict_train)
                _ = sess.run([gen_optimizer], feed_dict=feed_dict_train)
                train_loss, train_DSC, train_summary, _ = sess.run([G_loss, DSC, train_summary_op, gen_optimizer], feed_dict=feed_dict_train)
                mean_DSC_train = (mean_DSC_train*count + train_DSC)/(count+1)


                print("iterations:        ", iterations)
                print("training_DSC:      ", mean_DSC_train)
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

                    resized_image_width_eval = resized_parameter(image_width_eval, 32)
                    resized_image_height_eval = resized_parameter(image_height_eval, 32)
                    resized_label_width_eval = resized_parameter(label_width_eval, 32)
                    resized_label_height_eval = resized_parameter(label_height_eval, 32)

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
                    mean_DSC_eval = (mean_DSC_eval*count + eval_DSC0)/(count+1)

                    print("iterations:    ", iterations)
                    print("eval_DSC:      ", mean_DSC_eval)
                    print("eval_loss:     ", eval_loss0)

                    batch_eval = get_next_batch(eval_data, batch_size)

            if is_save :

                if not os.path.exists(checkpoint_path + train_plane + '/' + train_plane + '_' + str(current_fold) + '_' + str(epochs) + '/' ):     
                    os.makedirs(checkpoint_path + train_plane + '/' + train_plane + '_' + str(current_fold) + '_' + str(epochs) + '/' )

                saver.save(sess, checkpoint_path + train_plane + '/' + train_plane + '_' + str(current_fold) + '_' + str(epochs) + '/'  +'fine-' \
                + train_plane, global_step=epochs)




