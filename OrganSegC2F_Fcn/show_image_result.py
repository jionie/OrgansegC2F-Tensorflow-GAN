#!/usr/bin/env Python
# coding=utf-8
from input_data_fine import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

slim = tf.contrib.slim

def FCN_model(images, labels, slice_thickness):


    batch_size = tf.shape(images)[0]
    image_width = tf.shape(images)[2]
    image_height = tf.shape(images)[3]
    image_channel = tf.shape(images)[4]
    label_width = tf.shape(labels)[2]
    label_height = tf.shape(labels)[3]
    label_channel = tf.shape(labels)[4]

    image_slice = tf.reshape(images[:, 0, :, :, :], [batch_size, image_width, image_height, image_channel])
    label_slice = tf.reshape(labels[:, 0, :, :, :], [batch_size, label_width, label_height, label_channel])

    upsampled_logits_batch_1, fcn_8s_variables_mapping = FCN_8s(image_batch_tensor=image_slice,
                                            number_of_classes=number_of_classes,
                                            new_number_of_classes=3,
                                            is_training=True,
                                            is_reuse=False)

    upsampled_logits_batch_1 = tf.cast(upsampled_logits_batch_1, tf.float32)
    pred = tf.multiply(tf.nn.sigmoid(upsampled_logits_batch_1), tf.constant(255.0, dtype=tf.float32))
    pred0 = tf.where(tf.greater(pred, tf.multiply(tf.ones_like(pred, tf.float32), tf.constant(128.0, dtype=tf.float32))), \
    tf.ones_like(pred, tf.float32), tf.zeros_like(pred, tf.float32))

    up_loss, down_loss, up_DSC, down_DSC = DSC_loss(upsampled_logits_batch_1, pred0, label_slice)

    total_up_loss = up_loss
    total_down_loss = down_loss
    total_up_DSC = up_DSC
    total_down_DSC = down_DSC

               
    total_DSC = tf.div(2*total_up_DSC, total_down_DSC)
    total_loss = tf.subtract(tf.constant(1.0, dtype=tf.float32),tf.div(total_up_loss, total_down_loss))

    return total_loss, total_DSC, fcn_8s_variables_mapping, pred0

def DSC_computation(label, pred):
    pred_sum = pred.sum()
    label_sum = label.sum()
    inter_sum = np.multiply(pred, label)
    inter_sum = inter_sum.sum()
    return 2 * float(inter_sum) / (pred_sum + label_sum), inter_sum, pred_sum, label_sum


if __name__ == '__main__':

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    #npy_file_list = ["/media/jionie/Disk1/images/0031.npy", "/media/jionie/Disk1/labels/0031.npy"]
    
    #npy_file_list = ["/media/jionie/Disk1/results/coarse_gan/fcn_vgg/test/coarse_X_1/11.npz", "/media/jionie/Disk1/labels/0031.npy"]

    #npy_file_list = ["/media/jionie/Disk1/results/oracle_gan/fcn_vgg/test/oracle_X_1/11.npz", "/media/jionie/Disk1/labels/0031.npy"]

    npy_file_list = ["/media/jionie/Disk1/results/coarse2fine/fcn_vgg/test/fusion:X_Y_Z_1/R1_11.npz", "/media/jionie/Disk1/labels/0031.npy"]
    
    print(npy_file_list)
    volume_data = np.load(npy_file_list[0])
    image = volume_data['volume']
    #image[image < low_range] = low_range
    #image[image > high_range] = high_range
    label = np.load(npy_file_list[1])
    fig1 = plt.figure()
    
    image_x = image[282, :, :]
    label_x = label[282, :, :]
    image_y = image[:, 232, :]
    label_y = label[:, 232, :]
    image_z = image[:, :, 137]
    label_z = label[:, :, 137]

    #ax1 = fig1.add_subplot(1,2,1)
    #ax1.imshow(image_x, cmap = 'gray')

    #ax2 = fig1.add_subplot(1,2,2)
    #ax2.imshow(label_x, cmap = 'gray')

    #ax3 = fig1.add_subplot(1,2,1)
    #ax3.imshow(image_y, cmap = 'gray')

    #ax4 = fig1.add_subplot(1,2,2)
    #ax4.imshow(label_y, cmap = 'gray')

    

    ax5 = fig1.add_subplot(1,2,1)
    ax5.imshow(image_z, cmap = 'gray')

    ax6 = fig1.add_subplot(1,2,2)
    ax6.imshow(label_z, cmap = 'gray')
    
    plt.show()

    DSC = DSC_computation(image_z, label_z)
    print(DSC)


    
    
    










