#!/usr/bin/env Python
# coding=utf-8
from input_data_coarse import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

slim = tf.contrib.slim


if __name__ == '__main__':


    for current_fold in range(0, 4):

        volume_list = open(testing_set_filename(current_fold), 'r').read().splitlines()
        while volume_list[len(volume_list) - 1] == '':
            volume_list.pop()

        result_name = 'fusion:' + 'X_Y_Z_' + str(current_fold) + '/'
        result_directory = '/media/jionie/Disk1/results/coarse/fcn_vgg/test/' + result_name

        if not os.path.exists(result_directory):
            os.makedirs(result_directory)

        result_file = result_directory + 'results.txt'
        
        output = open(result_file, 'w')
        output.close()
        output = open(result_file, 'a+')
        output.write('Fusing results ' + ':\n')
        output.close()

        DSC_X = np.zeros((len(volume_list)))
        DSC_Y = np.zeros((len(volume_list)))
        DSC_Z = np.zeros((len(volume_list)))
        DSC_F1 = np.zeros((len(volume_list)))
        DSC_F2 = np.zeros((len(volume_list)))
        DSC_F3 = np.zeros((len(volume_list)))
        DSC_F1P = np.zeros((len(volume_list)))
        DSC_F2P = np.zeros((len(volume_list)))
        DSC_F3P = np.zeros((len(volume_list)))

        for i in range(len(volume_list)):

            start_time = time.time()
            print('Testing ' + str(i + 1) + ' out of ' + str(len(volume_list)) + ' testcases.')
            
            output = open(result_file, 'a+')
            output.write('  Testcase ' + str(i + 1) + ':\n')
            output.close()
            
            s = volume_list[i].split(' ')
            label = np.load(s[2])
            label = is_organ(label, organ_ID).astype(np.int32)

            for plane in ['X', 'Y', 'Z']:

                if plane == 'X':
                    volume_file = '/media/jionie/Disk1/results/coarse_gan/fcn_vgg/test/coarse_X_' + str(current_fold) + '/' +\
                    str(i + 1) + '.npz'
                
                elif plane == 'Y':
                    volume_file = '/media/jionie/Disk1/results/coarse_gan/fcn_vgg/test/coarse_Y_' + str(current_fold) + '/' +\
                    str(i + 1) + '.npz'
                
                else:
                    volume_file = '/media/jionie/Disk1/results/coarse_gan/fcn_vgg/test/coarse_Z_' + str(current_fold) + '/' +\
                    str(i + 1) + '.npz'

                volume_data = np.load(volume_file)
                pred = volume_data['volume']

                DSC_, inter_sum, pred_sum, label_sum = DSC_computation(label, pred)
                print('    DSC_' + plane + ' = 2 * ' + str(inter_sum) + ' / (' + \
                str(pred_sum) + ' + ' + str(label_sum) + ') = ' + str(DSC_) + ' .')

                output = open(result_file, 'a+')
                output.write('    DSC_' + plane + ' = 2 * ' + str(inter_sum) + ' / (' + \
                    str(pred_sum) + ' + ' + str(label_sum) + ') = ' + str(DSC_) + ' .\n')
                output.close()

                if pred_sum == 0 and label_sum == 0:

                    DSC_ = 0

                if plane == 'X':

                    pred_X = np.copy(pred).astype(np.bool)
                    DSC_X[i] = DSC_

                elif plane == 'Y':

                    pred_Y = np.copy(pred).astype(np.bool)
                    DSC_Y[i] = DSC_

                elif plane == 'Z':

                    pred_Z = np.copy(pred).astype(np.bool)
                    DSC_Z[i] = DSC_


            volume_file_F1 = result_directory + 'F1_' + str(i+1) + '.npz'

            if os.path.isfile(volume_file_F1):

                volume_data = np.load(volume_file_F1)
                pred_F1 = volume_data['volume']

            else:

                pred_F1 = np.logical_or(np.logical_or(pred_X, pred_Y), pred_Z)
                np.savez_compressed(volume_file_F1, volume = pred_F1)
            

            DSC_F1[i], inter_sum, pred_sum, label_sum = DSC_computation(label, pred_F1)
            print('    DSC_F1 = 2 * ' + str(inter_sum) + ' / (' + str(pred_sum) + ' + ' \
                + str(label_sum) + ') = ' + str(DSC_F1[i]) + ' .')

            output = open(result_file, 'a+')
            output.write('    DSC_F1 = 2 * ' + str(inter_sum) + ' / (' + \
                str(pred_sum) + ' + ' + str(label_sum) + ') = ' + str(DSC_F1[i]) + ' .\n')
            output.close()

            if pred_sum == 0 and label_sum == 0:

                DSC_F1[i] = 0



            volume_file_F2 = result_directory + 'F2_' + str(i+1) + '.npz'
            
            if os.path.isfile(volume_file_F2):
                
                volume_data = np.load(volume_file_F2)
                pred_F2 = volume_data['volume']

            else:
                
                pred_F2 = np.logical_or(np.logical_or(np.logical_and(pred_X, pred_Y), \
                    np.logical_and(pred_X, pred_Z)), np.logical_and(pred_Y, pred_Z))
                np.savez_compressed(volume_file_F2, volume = pred_F2)


            DSC_F2[i], inter_sum, pred_sum, label_sum = DSC_computation(label, pred_F2)
            print('    DSC_F2 = 2 * ' + str(inter_sum) + ' / (' + str(pred_sum) + ' + ' + \
                str(label_sum) + ') = ' + str(DSC_F2[i]) + ' .')

            output = open(result_file, 'a+')
            output.write('    DSC_F2 = 2 * ' + str(inter_sum) + ' / (' + \
                str(pred_sum) + ' + ' + str(label_sum) + ') = ' + str(DSC_F2[i]) + ' .\n')
            output.close()

            if pred_sum == 0 and label_sum == 0:

                DSC_F2[i] = 0



            volume_file_F3 = result_directory + 'F3_' + str(i+1) + '.npz'

            if os.path.isfile(volume_file_F3):

                volume_data = np.load(volume_file_F3)
                pred_F3 = volume_data['volume']

            else:

                pred_F3 = np.logical_and(np.logical_and(pred_X, pred_Y), pred_Z)
                np.savez_compressed(volume_file_F3, volume = pred_F3)

            DSC_F3[i], inter_sum, pred_sum, label_sum = DSC_computation(label, pred_F3)
            print('    DSC_F3 = 2 * ' + str(inter_sum) + ' / (' + str(pred_sum) + ' + ' + \
                str(label_sum) + ') = ' + str(DSC_F3[i]) + ' .')
            
            output = open(result_file, 'a+')
            output.write('    DSC_F3 = 2 * ' + str(inter_sum) + ' / (' + \
                str(pred_sum) + ' + ' + str(label_sum) + ') = ' + str(DSC_F3[i]) + ' .\n')
            output.close()


            if pred_sum == 0 and label_sum == 0:

                DSC_F3[i] = 0

            S = pred_F3

            if (S.sum() == 0):

                S = pred_F2

            if (S.sum() == 0):

                S = pred_F1



            volume_file_F1P = result_directory + 'F1P_' + str(i+1) + '.npz'

            if os.path.isfile(volume_file_F1P):

                volume_data = np.load(volume_file_F1P)
                pred_F1P = volume_data['volume']

            else:

                pred_F1P = post_processing(pred_F1, S, 0.5, organ_ID)
                np.savez_compressed(volume_file_F1P, volume = pred_F1P)


            DSC_F1P[i], inter_sum, pred_sum, label_sum = DSC_computation(label, pred_F1P)
            print('    DSC_F1P = 2 * ' + str(inter_sum) + ' / (' + str(pred_sum) + ' + ' + \
                str(label_sum) + ') = ' + str(DSC_F1P[i]) + ' .')


            output = open(result_file, 'a+')
            output.write('    DSC_F1P = 2 * ' + str(inter_sum) + ' / (' + \
                str(pred_sum) + ' + ' + str(label_sum) + ') = ' + str(DSC_F1P[i]) + ' .\n')
            output.close()


            if pred_sum == 0 and label_sum == 0:

                DSC_F1P[i] = 0



            volume_file_F2P = result_directory + 'F2P_' + str(i+1) + '.npz'

            if os.path.isfile(volume_file_F2P):

                volume_data = np.load(volume_file_F2P)
                pred_F2P = volume_data['volume']

            else:
                pred_F2P = post_processing(pred_F2, S, 0.5, organ_ID)
                np.savez_compressed(volume_file_F2P, volume = pred_F2P)


            DSC_F2P[i], inter_sum, pred_sum, label_sum = DSC_computation(label, pred_F2P)
            print('    DSC_F2P = 2 * ' + str(inter_sum) + ' / (' + str(pred_sum) + ' + ' + \
                str(label_sum) + ') = ' + str(DSC_F2P[i]) + ' .')


            output = open(result_file, 'a+')
            output.write('    DSC_F2P = 2 * ' + str(inter_sum) + ' / (' + \
                str(pred_sum) + ' + ' + str(label_sum) + ') = ' + str(DSC_F2P[i]) + ' .\n')
            output.close()


            if pred_sum == 0 and label_sum == 0:

                DSC_F2P[i] = 0



            volume_file_F3P = result_directory + 'F3P_' + str(i+1) + '.npz'

            if os.path.isfile(volume_file_F3P):

                volume_data = np.load(volume_file_F3P)
                pred_F3P = volume_data['volume']

            else:

                pred_F3P = post_processing(pred_F3, S, 0.5, organ_ID)
                np.savez_compressed(volume_file_F3P, volume = pred_F3P)


            DSC_F3P[i], inter_sum, pred_sum, label_sum = DSC_computation(label, pred_F3P)
            print('    DSC_F3P = 2 * ' + str(inter_sum) + ' / (' + str(pred_sum) + ' + ' + \
                str(label_sum) + ') = ' + str(DSC_F3P[i]) + ' .')


            output = open(result_file, 'a+')
            output.write('    DSC_F3P = 2 * ' + str(inter_sum) + ' / (' + \
                str(pred_sum) + ' + ' + str(label_sum) + ') = ' + str(DSC_F3P[i]) + ' .\n')
            output.close()


            if pred_sum == 0 and label_sum == 0:
                DSC_F3P[i] = 0



        output = open(result_file, 'a+')
        print('Average DSC_X = ' + str(np.mean(DSC_X)) + ' .')
        output.write('Average DSC_X = ' + str(np.mean(DSC_X)) + ' .\n')
        print('Average DSC_Y = ' + str(np.mean(DSC_Y)) + ' .')
        output.write('Average DSC_Y = ' + str(np.mean(DSC_Y)) + ' .\n')
        print('Average DSC_Z = ' + str(np.mean(DSC_Z)) + ' .')
        output.write('Average DSC_Z = ' + str(np.mean(DSC_Z)) + ' .\n')
        print('Average DSC_F1 = ' + str(np.mean(DSC_F1)) + ' .')
        output.write('Average DSC_F1 = ' + str(np.mean(DSC_F1)) + ' .\n')
        print('Average DSC_F2 = ' + str(np.mean(DSC_F2)) + ' .')
        output.write('Average DSC_F2 = ' + str(np.mean(DSC_F2)) + ' .\n')
        print('Average DSC_F3 = ' + str(np.mean(DSC_F3)) + ' .')
        output.write('Average DSC_F3 = ' + str(np.mean(DSC_F3)) + ' .\n')
        print('Average DSC_F1P = ' + str(np.mean(DSC_F1P)) + ' .')
        output.write('Average DSC_F1P = ' + str(np.mean(DSC_F1P)) + ' .\n')
        print('Average DSC_F2P = ' + str(np.mean(DSC_F2P)) + ' .')
        output.write('Average DSC_F2P = ' + str(np.mean(DSC_F2P)) + ' .\n')
        print('Average DSC_F3P = ' + str(np.mean(DSC_F3P)) + ' .')
        output.write('Average DSC_F3P = ' + str(np.mean(DSC_F3P)) + ' .\n')
        output.close()
        print('The fusion process is finished.')