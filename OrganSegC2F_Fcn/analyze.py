import string
import os
import sys



if __name__ == '__main__':

    result_file = '/media/jionie/Disk1/results/analyze/analyze.txt' 
    
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

    mean_fine_X = 0
    mean_fine_Y = 0
    mean_fine_Z = 0
    mean_fine_F1P = 0
    mean_fine_F2P = 0
    deviation_fine_X = 0
    deviation_fine_Y = 0
    deviation_fine_Z = 0
    deviation_fine_F1P = 0
    deviation_fine_F2P = 0

    mean_coarse2fine_1 = 0
    mean_coarse2fine_2 = 0
    mean_coarse2fine_3 = 0
    mean_coarse2fine_5 = 0
    mean_coarse2fine_10 = 0
    deviation_coarse2fine_1 = 0
    deviation_coarse2fine_2 = 0
    deviation_coarse2fine_3 = 0
    deviation_coarse2fine_5 = 0
    deviation_coarse2fine_10 = 0

    max_DSC_1 = 0
    min_DSC_1 = 1
    max_DSC_2 = 0
    min_DSC_2 = 1
    max_DSC_3 = 0
    min_DSC_3 = 1
    max_DSC_5 = 0
    min_DSC_5 = 1
    max_DSC_10 = 0
    min_DSC_10 = 1

    
    for fold in range(0, 4):

        file_coarse = '/media/jionie/Disk1/results/coarse/fcn_vgg/test/fusion:X_Y_Z_' + str(fold) + '/results.txt'
         
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
                if (s[0].replace(' ', '')=='DSC_F2P'):
                    mean_coarse_F2P += float((s[2].split(' '))[1])

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
                if (s[0].replace(' ', '')=='DSC_F2P'):
                    mean_fine_F2P += float((s[2].split(' '))[1])

        
        file_coarse2fine = '/media/jionie/Disk1/results/coarse2fine/fcn_vgg/test/fusion:X_Y_Z_' + str(fold) + '/results.txt'
         
        with open(file_coarse2fine, 'r') as f2:
            
            content = f2.read().splitlines()
            
            for i in range(len(content)):
                
                s = content[i].split(',')
                
                if (s[0].replace(' ', '')=='Round1'):

                    if((s[1].split(' '))[1]=='DSC'):
                        mean_coarse2fine_1 += float((s[1].split(' '))[11])
                        if(float((s[1].split(' '))[11])>max_DSC_1):
                            max_DSC_1 = float((s[1].split(' '))[11])
                        if(float((s[1].split(' '))[11])<min_DSC_1):
                            min_DSC_1 = float((s[1].split(' '))[11])

                if (s[0].replace(' ', '')=='Round2'):
                   
                    if((s[1].split(' '))[1]=='DSC'):
                        mean_coarse2fine_2 += float((s[1].split(' '))[11])
                        if(float((s[1].split(' '))[11])>max_DSC_2):
                            max_DSC_2 = float((s[1].split(' '))[11])
                        if(float((s[1].split(' '))[11])<min_DSC_2):
                            min_DSC_2 = float((s[1].split(' '))[11])

                if (s[0].replace(' ', '')=='Round3'):
                    
                    if((s[1].split(' '))[1]=='DSC'):
                        mean_coarse2fine_3 += float((s[1].split(' '))[11])
                        if(float((s[1].split(' '))[11])>max_DSC_3):
                            max_DSC_3 = float((s[1].split(' '))[11])
                        if(float((s[1].split(' '))[11])<min_DSC_3):
                            min_DSC_3 = float((s[1].split(' '))[11])

                if (s[0].replace(' ', '')=='Round5'):
                   
                    if((s[1].split(' '))[1]=='DSC'):
                        mean_coarse2fine_5 += float((s[1].split(' '))[11])
                        if(float((s[1].split(' '))[11])>max_DSC_5):
                            max_DSC_5 = float((s[1].split(' '))[11])
                        if(float((s[1].split(' '))[11])<min_DSC_5):
                            min_DSC_5 = float((s[1].split(' '))[11])

                if (s[0].replace(' ', '')=='Round10'):
                    
                    if((s[1].split(' '))[1]=='DSC'):
                        mean_coarse2fine_10 += float((s[1].split(' '))[11])
                        if(float((s[1].split(' '))[11])>max_DSC_10):
                            max_DSC_10 = float((s[1].split(' '))[11])
                        if(float((s[1].split(' '))[11])<min_DSC_10):
                            min_DSC_10 = float((s[1].split(' '))[11])

    mean_coarse_X /= 82
    mean_coarse_Y /= 82
    mean_coarse_Z /= 82
    mean_coarse_F1P /= 82
    mean_coarse_F2P /= 82

    mean_fine_X /= 82
    mean_fine_Y /= 82
    mean_fine_Z /= 82
    mean_fine_F1P /= 82
    mean_fine_F2P /= 82

    mean_coarse2fine_1 /= 82
    mean_coarse2fine_2 /= 82
    mean_coarse2fine_3 /= 82
    mean_coarse2fine_5 /= 82
    mean_coarse2fine_10 /= 82

    output = open(result_file, 'a+')
    output.write('mean_coarse_X: ' + str(mean_coarse_X) + '\n')
    output.write('mean_coarse_Y: ' + str(mean_coarse_Y) + '\n')
    output.write('mean_coarse_Z: ' + str(mean_coarse_Z) + '\n')
    output.write('mean_coarse_F1P: ' + str(mean_coarse_F1P) + '\n')
    output.write('mean_coarse_F2P: ' + str(mean_coarse_F2P) + '\n')

    output.write('mean_fine_X: ' + str(mean_fine_X) + '\n')
    output.write('mean_fine_Y: ' + str(mean_fine_Y) + '\n')
    output.write('mean_fine_Z: ' + str(mean_fine_Z) + '\n')
    output.write('mean_fine_F1P: ' + str(mean_fine_F1P) + '\n')
    output.write('mean_fine_F2P: ' + str(mean_fine_F2P) + '\n')

    output.write('mean_coarse2fine_1: ' + str(mean_coarse2fine_1) + '\n')
    output.write('mean_coarse2fine_2: ' + str(mean_coarse2fine_2) + '\n')
    output.write('mean_coarse2fine_3: ' + str(mean_coarse2fine_3) + '\n')
    output.write('mean_coarse2fine_5: ' + str(mean_coarse2fine_5) + '\n')
    output.write('mean_coarse2fine_10: ' + str(mean_coarse2fine_10) + '\n')
    output.write('max_DSC_1: ' + str(max_DSC_1) + '\n')
    output.write('max_DSC_2: ' + str(max_DSC_2) + '\n')
    output.write('max_DSC_3: ' + str(max_DSC_3) + '\n')
    output.write('max_DSC_5: ' + str(max_DSC_5) + '\n')
    output.write('max_DSC_10: ' + str(max_DSC_10) + '\n')
    output.write('min_DSC_1: ' + str(min_DSC_1) + '\n')
    output.write('min_DSC_2: ' + str(min_DSC_2) + '\n')
    output.write('min_DSC_3: ' + str(min_DSC_3) + '\n')
    output.write('min_DSC_5: ' + str(min_DSC_5) + '\n')
    output.write('min_DSC_10: ' + str(min_DSC_10) + '\n')

    
    output.close()


    for fold in range(0, 4):

        file_coarse = '/media/jionie/Disk1/results/coarse/fcn_vgg/test/fusion:X_Y_Z_' + str(fold) + '/results.txt'
         
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

        
        file_coarse2fine = '/media/jionie/Disk1/results/coarse2fine/fcn_vgg/test/fusion:X_Y_Z_' + str(fold) + '/results.txt'
         
        with open(file_coarse2fine, 'r') as f2:
            
            content = f2.read().splitlines()
            
            for i in range(len(content)):
                
                s = content[i].split(',')
                
                if (s[0].replace(' ', '')=='Round1'):

                    if((s[1].split(' '))[1]=='DSC'):
                        deviation_coarse2fine_1 += (float((s[1].split(' '))[11]) - mean_coarse2fine_1)**2

                if (s[0].replace(' ', '')=='Round2'):
                   
                    if((s[1].split(' '))[1]=='DSC'):
                        deviation_coarse2fine_2 += (float((s[1].split(' '))[11]) - mean_coarse2fine_2)**2

                if (s[0].replace(' ', '')=='Round3'):
                    
                    if((s[1].split(' '))[1]=='DSC'):
                        deviation_coarse2fine_3 += (float((s[1].split(' '))[11]) - mean_coarse2fine_3)**2

                if (s[0].replace(' ', '')=='Round5'):
                   
                    if((s[1].split(' '))[1]=='DSC'):
                        deviation_coarse2fine_5 += (float((s[1].split(' '))[11]) - mean_coarse2fine_5)**2

                if (s[0].replace(' ', '')=='Round10'):
                    
                    if((s[1].split(' '))[1]=='DSC'):
                        deviation_coarse2fine_10 += (float((s[1].split(' '))[11]) - mean_coarse2fine_10)**2


    deviation_coarse_X = (deviation_coarse_X/82)**0.5
    deviation_coarse_Y = (deviation_coarse_Y/82)**0.5
    deviation_coarse_Z = (deviation_coarse_Z/82)**0.5
    deviation_coarse_F1P = (deviation_coarse_F1P/82)**0.5
    deviation_coarse_F2P = (deviation_coarse_F2P/82)**0.5

    deviation_fine_X = (deviation_fine_X/82)**0.5
    deviation_fine_Y = (deviation_fine_Y/82)**0.5
    deviation_fine_Z = (deviation_fine_Z/82)**0.5
    deviation_fine_F1P = (deviation_fine_F1P/82)**0.5
    deviation_fine_F2P = (deviation_fine_F2P/82)**0.5

    deviation_coarse2fine_1 = (deviation_coarse2fine_1/82)**0.5
    deviation_coarse2fine_2 = (deviation_coarse2fine_2/82)**0.5
    deviation_coarse2fine_3 = (deviation_coarse2fine_3/82)**0.5
    deviation_coarse2fine_5 = (deviation_coarse2fine_5/82)**0.5
    deviation_coarse2fine_10 = (deviation_coarse2fine_10/82)**0.5

    output = open(result_file, 'a+')
    output.write('deviation_coarse_X: ' + str(deviation_coarse_X) + '\n')
    output.write('deviation_coarse_Y: ' + str(deviation_coarse_Y) + '\n')
    output.write('deviation_coarse_Z: ' + str(deviation_coarse_Z) + '\n')
    output.write('deviation_coarse_F1P: ' + str(deviation_coarse_F1P) + '\n')
    output.write('deviation_coarse_F2P: ' + str(deviation_coarse_F2P) + '\n')

    output.write('deviation_fine_X: ' + str(deviation_fine_X) + '\n')
    output.write('deviation_fine_Y: ' + str(deviation_fine_Y) + '\n')
    output.write('deviation_fine_Z: ' + str(deviation_fine_Z) + '\n')
    output.write('deviation_fine_F1P: ' + str(deviation_fine_F1P) + '\n')
    output.write('deviation_fine_F2P: ' + str(deviation_fine_F2P) + '\n')

    output.write('deviation_coarse2fine_1: ' + str(deviation_coarse2fine_1) + '\n')
    output.write('deviation_coarse2fine_2: ' + str(deviation_coarse2fine_2) + '\n')
    output.write('deviation_coarse2fine_3: ' + str(deviation_coarse2fine_3) + '\n')
    output.write('deviation_coarse2fine_5: ' + str(deviation_coarse2fine_5) + '\n')
    output.write('deviation_coarse2fine_10: ' + str(deviation_coarse2fine_10) + '\n')
    
    output.close()

    
                    
                
           
