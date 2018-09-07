import string
import os
import sys



if __name__ == '__main__':
    
    folder = '0'
    #result_file = '/media/jionie/Disk1/results/analyze/analyze_oracle_' + folder + '.txt' 
    result_file = '/media/jionie/Disk1/results/analyze/analyze_oracle.txt' 
    
    output = open(result_file, 'w')
    output.close()
            
    total_samples = 82


    mean_fine_X = 0
    mean_fine_Y = 0
    mean_fine_Z = 0
    mean_fine_F1P = 0
    mean_fine_F2P = 0
    max_fine_F1P = 0
    min_fine_F1P = 1
    max_fine_F2P = 0
    min_fine_F2P = 1
    deviation_fine_X = 0
    deviation_fine_Y = 0
    deviation_fine_Z = 0
    deviation_fine_F1P = 0
    deviation_fine_F2P = 0

    
    for fold in range(0, 4):

        file_fine = '/media/jionie/Disk1/results/oracle/fcn_vgg/test/fusion:X_Y_Z_' + str(fold) + '/results.txt'
         
        with open(file_fine, 'r') as f1:
            
            content = f1.read().splitlines()
            
            for i in range(len(content)):
                
                s = content[i].split('=')
                if (s[0].replace(' ', '')=='DSC_X'):
                    mean_fine_X += float((s[2].split(' '))[1])
                if (s[0].replace(' ', '')=='DSC_Y'):
                    mean_fine_Y += float((s[2].split(' '))[1])
                if (s[0].replace(' ', '')=='DSC_Z'):
                    mean_fine_Z += float((s[2].split(' '))[1])
                if (s[0].replace(' ', '')=='DSC_F1P'):
                    mean_fine_F1P += float((s[2].split(' '))[1])
                    if (min_fine_F1P > float((s[2].split(' '))[1])):
                        min_fine_F1P = float((s[2].split(' '))[1])
                    if (max_fine_F1P < float((s[2].split(' '))[1])):
                        max_fine_F1P = float((s[2].split(' '))[1]) 
                if (s[0].replace(' ', '')=='DSC_F2P'):
                    mean_fine_F2P += float((s[2].split(' '))[1])
                    if (min_fine_F2P > float((s[2].split(' '))[1])):
                        min_fine_F2P = float((s[2].split(' '))[1])
                    if (max_fine_F2P < float((s[2].split(' '))[1])):
                        max_fine_F2P = float((s[2].split(' '))[1]) 

    mean_fine_X /= total_samples
    mean_fine_Y /= total_samples
    mean_fine_Z /= total_samples
    mean_fine_F1P /= total_samples
    mean_fine_F2P /= total_samples


    output = open(result_file, 'a+')

    output.write('mean_fine_X: ' + str(mean_fine_X) + '\n')
    output.write('mean_fine_Y: ' + str(mean_fine_Y) + '\n')
    output.write('mean_fine_Z: ' + str(mean_fine_Z) + '\n')
    output.write('mean_fine_F1P: ' + str(mean_fine_F1P) + '\n')
    output.write('min_fine_F1P: ' + str(min_fine_F1P) + '\n')
    output.write('max_fine_F1P: ' + str(max_fine_F1P) + '\n')
    output.write('mean_fine_F2P: ' + str(mean_fine_F2P) + '\n')
    output.write('min_fine_F2P: ' + str(min_fine_F2P) + '\n')
    output.write('max_fine_F2P: ' + str(max_fine_F2P) + '\n')
    
    output.close()


    for fold in range(0, 4):

        file_fine = '/media/jionie/Disk1/results/oracle/fcn_vgg/test/fusion:X_Y_Z_' + str(fold) + '/results.txt'
         
        with open(file_fine, 'r') as f1:
            
            content = f1.read().splitlines()
            
            for i in range(len(content)):
                
                s = content[i].split('=')
                if (s[0].replace(' ', '')=='DSC_X'):
                    deviation_fine_X += (float((s[2].split(' '))[1])- mean_fine_X)**2
                if (s[0].replace(' ', '')=='DSC_Y'):
                    deviation_fine_Y += (float((s[2].split(' '))[1])- mean_fine_Y)**2
                if (s[0].replace(' ', '')=='DSC_Z'):
                    deviation_fine_Z += (float((s[2].split(' '))[1])- mean_fine_Z)**2
                if (s[0].replace(' ', '')=='DSC_F1P'):
                    deviation_fine_F1P += (float((s[2].split(' '))[1])- mean_fine_F1P)**2
                if (s[0].replace(' ', '')=='DSC_F2P'):
                    deviation_fine_F2P += (float((s[2].split(' '))[1])- mean_fine_F2P)**2




    deviation_fine_X = (deviation_fine_X/total_samples)**0.5
    deviation_fine_Y = (deviation_fine_Y/total_samples)**0.5
    deviation_fine_Z = (deviation_fine_Z/total_samples)**0.5
    deviation_fine_F1P = (deviation_fine_F1P/total_samples)**0.5
    deviation_fine_F2P = (deviation_fine_F2P/total_samples)**0.5

    output = open(result_file, 'a+')

    output.write('deviation_fine_X: ' + str(deviation_fine_X) + '\n')
    output.write('deviation_fine_Y: ' + str(deviation_fine_Y) + '\n')
    output.write('deviation_fine_Z: ' + str(deviation_fine_Z) + '\n')
    output.write('deviation_fine_F1P: ' + str(deviation_fine_F1P) + '\n')
    output.write('deviation_fine_F2P: ' + str(deviation_fine_F2P) + '\n')

    output.close()

    
                    
                
           
