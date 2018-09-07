import string
import os
import sys



if __name__ == '__main__':

    result_file = '/media/jionie/Disk1/results/analyze/analyze_coarse_gan.txt' 
    
    output = open(result_file, 'w')
    output.close()
            
    total_samples = 82

    mean_coarse_X = 0
    mean_coarse_Y = 0
    mean_coarse_Z = 0
    mean_coarse_F1P = 0
    mean_coarse_F2P = 0
    deviation_coarse_X = 0
    deviation_coarse_Y = 0
    deviation_coarse_Z = 0
    deviation_coarse_F1P = 0
    deviation_coarse_F2P = 0
    max_coarse_F1P = 0
    min_coarse_F1P = 1
    max_coarse_F2P = 0
    min_coarse_F2P = 1

    
    for fold in range(0, 4):

        file_coarse = '/media/jionie/Disk1/results/coarse_gan/fcn_vgg/test/fusion:X_Y_Z_' + str(fold) + '/results.txt'
         
        with open(file_coarse, 'r') as f0:
            
            content = f0.read().splitlines()
            
            for i in range(len(content)):
                
                s = content[i].split('=')
                if (s[0].replace(' ', '')=='DSC_X'):
                    mean_coarse_X += float((s[2].split(' '))[1])
                if (s[0].replace(' ', '')=='DSC_Y'):
                    mean_coarse_Y += float((s[2].split(' '))[1])
                if (s[0].replace(' ', '')=='DSC_Z'):
                    mean_coarse_Z += float((s[2].split(' '))[1])
                if (s[0].replace(' ', '')=='DSC_F1P'):
                    mean_coarse_F1P += float((s[2].split(' '))[1])
                    if (min_coarse_F1P > float((s[2].split(' '))[1])):
                        min_coarse_F1P = float((s[2].split(' '))[1])
                    if (max_coarse_F1P < float((s[2].split(' '))[1])):
                        max_coarse_F1P = float((s[2].split(' '))[1]) 
                if (s[0].replace(' ', '')=='DSC_F2P'):
                    mean_coarse_F2P += float((s[2].split(' '))[1])
                    if (min_coarse_F2P > float((s[2].split(' '))[1])):
                        min_coarse_F2P = float((s[2].split(' '))[1])
                    if (max_coarse_F2P < float((s[2].split(' '))[1])):
                        max_coarse_F2P = float((s[2].split(' '))[1]) 

    mean_coarse_X /= total_samples
    mean_coarse_Y /= total_samples
    mean_coarse_Z /= total_samples
    mean_coarse_F1P /= total_samples
    mean_coarse_F2P /= total_samples


    output = open(result_file, 'a+')
    output.write('mean_coarse_X: ' + str(mean_coarse_X) + '\n')
    output.write('mean_coarse_Y: ' + str(mean_coarse_Y) + '\n')
    output.write('mean_coarse_Z: ' + str(mean_coarse_Z) + '\n')
    output.write('mean_coarse_F1P: ' + str(mean_coarse_F1P) + '\n')
    output.write('min_coarse_F1P: ' + str(min_coarse_F1P) + '\n')
    output.write('max_coarse_F1P: ' + str(max_coarse_F1P) + '\n')
    output.write('mean_coarse_F2P: ' + str(mean_coarse_F2P) + '\n')
    output.write('min_coarse_F2P: ' + str(min_coarse_F2P) + '\n')
    output.write('max_coarse_F2P: ' + str(max_coarse_F2P) + '\n')

    
    output.close()


    for fold in range(0, 4):

        file_coarse = '/media/jionie/Disk1/results/coarse_gan/fcn_vgg/test/fusion:X_Y_Z_' + str(fold) + '/results.txt'
         
        with open(file_coarse, 'r') as f0:
            
            content = f0.read().splitlines()
            
            for i in range(len(content)):
                
                s = content[i].split('=')
                if (s[0].replace(' ', '')=='DSC_X'):
                    deviation_coarse_X += (float((s[2].split(' '))[1]) - mean_coarse_X)**2
                if (s[0].replace(' ', '')=='DSC_Y'):
                    deviation_coarse_Y += (float((s[2].split(' '))[1]) - mean_coarse_Y)**2
                if (s[0].replace(' ', '')=='DSC_Z'):
                    deviation_coarse_Z += (float((s[2].split(' '))[1]) - mean_coarse_Z)**2
                if (s[0].replace(' ', '')=='DSC_F1P'):
                    deviation_coarse_F1P += (float((s[2].split(' '))[1]) - mean_coarse_F1P)**2
                if (s[0].replace(' ', '')=='DSC_F2P'):
                    deviation_coarse_F2P += (float((s[2].split(' '))[1]) - mean_coarse_F2P)**2



    deviation_coarse_X = (deviation_coarse_X/total_samples)**0.5
    deviation_coarse_Y = (deviation_coarse_Y/total_samples)**0.5
    deviation_coarse_Z = (deviation_coarse_Z/total_samples)**0.5
    deviation_coarse_F1P = (deviation_coarse_F1P/total_samples)**0.5
    deviation_coarse_F2P = (deviation_coarse_F2P/total_samples)**0.5

   
    output = open(result_file, 'a+')
    output.write('deviation_coarse_X: ' + str(deviation_coarse_X) + '\n')
    output.write('deviation_coarse_Y: ' + str(deviation_coarse_Y) + '\n')
    output.write('deviation_coarse_Z: ' + str(deviation_coarse_Z) + '\n')
    output.write('deviation_coarse_F1P: ' + str(deviation_coarse_F1P) + '\n')
    output.write('deviation_coarse_F2P: ' + str(deviation_coarse_F2P) + '\n')

    output.close()

    
                    
                
           
