#!/usr/bin/env Python
# coding=utf-8
from input_data_pretrain_coarse import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

slim = tf.contrib.slim


if __name__ == '__main__':

    coarse_retraining_Z = "/media/jionie/Disk1/lists/coarse_retraining_Z_tmp.txt"
    coarse_training_Z = "/media/jionie/Disk1/lists/coarse_training_Z.txt"
    coarse_retraining_Z_new = "/media/jionie/Disk1/lists/coarse_retraining_Z.txt"


    list1 = open(coarse_training_Z, 'r').read().splitlines()
    list2 = open(coarse_retraining_Z, 'r').read().splitlines()
    list3 = open(coarse_retraining_Z_new, 'a+')

    i=0
    j=0

    while(i<(len(list2)-1)):

        list2_slice_Z = list2[i].split(' ')

        while(j<(len(list1)-1)):

            list1_slice_Z = list1[j].split(' ')

            if (list2_slice_Z[0] == list1_slice_Z[2]):

                content = list1_slice_Z[0] + ' ' + list1_slice_Z[1] + ' ' + list1_slice_Z[2] + ' ' + list1_slice_Z[3] + ' ' + \
                list1_slice_Z[4] + ' ' +  list1_slice_Z[5] + ' ' + list1_slice_Z[6] + ' ' + list1_slice_Z[7] + ' ' + list1_slice_Z[8] + ' ' + list1_slice_Z[9] + '\n'

                list3.write(content)

                break

            else:

                j+=1

        i+=1

    
    list3.close()





