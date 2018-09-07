import string
import os
import sys



if __name__ == '__main__':

    result_file = '/media/jionie/Disk1/results/analyze/analyze_coarse2fine_gan.txt' 
    
    output = open(result_file, 'w')
    output.close()
            
    total_samples = 82

    mean_coarse2fine_1 = 0
    mean_coarse2fine_2 = 0
    mean_coarse2fine_3 = 0
    mean_coarse2fine_4 = 0
    mean_coarse2fine_5 = 0
    mean_coarse2fine_6 = 0
    mean_coarse2fine_7 = 0
    mean_coarse2fine_8 = 0
    mean_coarse2fine_9 = 0
    mean_coarse2fine_10 = 0

    deviation_coarse2fine_1 = 0
    deviation_coarse2fine_2 = 0
    deviation_coarse2fine_3 = 0
    deviation_coarse2fine_4 = 0
    deviation_coarse2fine_5 = 0
    deviation_coarse2fine_6 = 0
    deviation_coarse2fine_7 = 0
    deviation_coarse2fine_8 = 0
    deviation_coarse2fine_9 = 0
    deviation_coarse2fine_10 = 0

    max_DSC_1 = 0
    min_DSC_1 = 1
    max_DSC_2 = 0
    min_DSC_2 = 1
    max_DSC_3 = 0
    min_DSC_3 = 1
    max_DSC_4 = 0
    min_DSC_4 = 1
    max_DSC_5 = 0
    min_DSC_5 = 1
    max_DSC_6 = 0
    min_DSC_6 = 1
    max_DSC_7 = 0
    min_DSC_7 = 1
    max_DSC_8 = 0
    min_DSC_8 = 1
    max_DSC_9 = 0
    min_DSC_9 = 1
    max_DSC_10 = 0
    min_DSC_10 = 1

    
    for fold in range(0, 4):
        
        file_coarse2fine = '/media/jionie/Disk1/results/coarse2fine_gan/fcn_vgg/test/fusion:X_Y_Z_' + str(fold) + '/results.txt'
         
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
                
                if (s[0].replace(' ', '')=='Round4'):
                    
                    if((s[1].split(' '))[1]=='DSC'):
                        mean_coarse2fine_4 += float((s[1].split(' '))[11])
                        if(float((s[1].split(' '))[11])>max_DSC_4):
                            max_DSC_4 = float((s[1].split(' '))[11])
                        if(float((s[1].split(' '))[11])<min_DSC_4):
                            min_DSC_4 = float((s[1].split(' '))[11])

                if (s[0].replace(' ', '')=='Round5'):
                   
                    if((s[1].split(' '))[1]=='DSC'):
                        mean_coarse2fine_5 += float((s[1].split(' '))[11])
                        if(float((s[1].split(' '))[11])>max_DSC_5):
                            max_DSC_5 = float((s[1].split(' '))[11])
                        if(float((s[1].split(' '))[11])<min_DSC_5):
                            min_DSC_5 = float((s[1].split(' '))[11])

                if (s[0].replace(' ', '')=='Round6'):
                    
                    if((s[1].split(' '))[1]=='DSC'):
                        mean_coarse2fine_6 += float((s[1].split(' '))[11])
                        if(float((s[1].split(' '))[11])>max_DSC_6):
                            max_DSC_6 = float((s[1].split(' '))[11])
                        if(float((s[1].split(' '))[11])<min_DSC_6):
                            min_DSC_6 = float((s[1].split(' '))[11])

                if (s[0].replace(' ', '')=='Round7'):
                    
                    if((s[1].split(' '))[1]=='DSC'):
                        mean_coarse2fine_7 += float((s[1].split(' '))[11])
                        if(float((s[1].split(' '))[11])>max_DSC_7):
                            max_DSC_7 = float((s[1].split(' '))[11])
                        if(float((s[1].split(' '))[11])<min_DSC_7):
                            min_DSC_7 = float((s[1].split(' '))[11])

                if (s[0].replace(' ', '')=='Round8'):
                    
                    if((s[1].split(' '))[1]=='DSC'):
                        mean_coarse2fine_8 += float((s[1].split(' '))[11])
                        if(float((s[1].split(' '))[11])>max_DSC_8):
                            max_DSC_8 = float((s[1].split(' '))[11])
                        if(float((s[1].split(' '))[11])<min_DSC_8):
                            min_DSC_8 = float((s[1].split(' '))[11])

                if (s[0].replace(' ', '')=='Round9'):
                    
                    if((s[1].split(' '))[1]=='DSC'):
                        mean_coarse2fine_9 += float((s[1].split(' '))[11])
                        if(float((s[1].split(' '))[11])>max_DSC_9):
                            max_DSC_9 = float((s[1].split(' '))[11])
                        if(float((s[1].split(' '))[11])<min_DSC_9):
                            min_DSC_9 = float((s[1].split(' '))[11])           

                if (s[0].replace(' ', '')=='Round10'):
                    
                    if((s[1].split(' '))[1]=='DSC'):
                        mean_coarse2fine_10 += float((s[1].split(' '))[11])
                        if(float((s[1].split(' '))[11])>max_DSC_10):
                            max_DSC_10 = float((s[1].split(' '))[11])
                        if(float((s[1].split(' '))[11])<min_DSC_10):
                            min_DSC_10 = float((s[1].split(' '))[11])


    mean_coarse2fine_1 /= total_samples
    mean_coarse2fine_2 /= total_samples
    mean_coarse2fine_3 /= total_samples
    mean_coarse2fine_4 /= total_samples
    mean_coarse2fine_5 /= total_samples
    mean_coarse2fine_6 /= total_samples
    mean_coarse2fine_7 /= total_samples
    mean_coarse2fine_8 /= total_samples
    mean_coarse2fine_9 /= total_samples
    mean_coarse2fine_10 /= total_samples

    output = open(result_file, 'a+')

    output.write('mean_coarse2fine_1: ' + str(mean_coarse2fine_1) + '\n')
    output.write('mean_coarse2fine_2: ' + str(mean_coarse2fine_2) + '\n')
    output.write('mean_coarse2fine_3: ' + str(mean_coarse2fine_3) + '\n')
    output.write('mean_coarse2fine_4: ' + str(mean_coarse2fine_4) + '\n')
    output.write('mean_coarse2fine_5: ' + str(mean_coarse2fine_5) + '\n')
    output.write('mean_coarse2fine_6: ' + str(mean_coarse2fine_6) + '\n')
    output.write('mean_coarse2fine_7: ' + str(mean_coarse2fine_7) + '\n')
    output.write('mean_coarse2fine_8: ' + str(mean_coarse2fine_8) + '\n')
    output.write('mean_coarse2fine_9: ' + str(mean_coarse2fine_9) + '\n')
    output.write('mean_coarse2fine_10: ' + str(mean_coarse2fine_10) + '\n')

    output.write('max_DSC_1: ' + str(max_DSC_1) + '\n')
    output.write('max_DSC_2: ' + str(max_DSC_2) + '\n')
    output.write('max_DSC_3: ' + str(max_DSC_3) + '\n')
    output.write('max_DSC_4: ' + str(max_DSC_4) + '\n')
    output.write('max_DSC_5: ' + str(max_DSC_5) + '\n')
    output.write('max_DSC_6: ' + str(max_DSC_6) + '\n')
    output.write('max_DSC_7: ' + str(max_DSC_7) + '\n')
    output.write('max_DSC_8: ' + str(max_DSC_8) + '\n')
    output.write('max_DSC_9: ' + str(max_DSC_9) + '\n')
    output.write('max_DSC_10: ' + str(max_DSC_10) + '\n')
    output.write('min_DSC_1: ' + str(min_DSC_1) + '\n')
    output.write('min_DSC_2: ' + str(min_DSC_2) + '\n')
    output.write('min_DSC_3: ' + str(min_DSC_3) + '\n')
    output.write('min_DSC_4: ' + str(min_DSC_4) + '\n')
    output.write('min_DSC_5: ' + str(min_DSC_5) + '\n')
    output.write('min_DSC_6: ' + str(min_DSC_6) + '\n')
    output.write('min_DSC_7: ' + str(min_DSC_7) + '\n')
    output.write('min_DSC_8: ' + str(min_DSC_8) + '\n')
    output.write('min_DSC_9: ' + str(min_DSC_9) + '\n')
    output.write('min_DSC_10: ' + str(min_DSC_10) + '\n')

    
    output.close()


    for fold in range(0, 4):

        file_coarse2fine = '/media/jionie/Disk1/results/coarse2fine_gan/fcn_vgg/test/fusion:X_Y_Z_' + str(fold) + '/results.txt'
         
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

                if (s[0].replace(' ', '')=='Round4'):
                   
                    if((s[1].split(' '))[1]=='DSC'):
                        deviation_coarse2fine_4 += (float((s[1].split(' '))[11]) - mean_coarse2fine_4)**2

                if (s[0].replace(' ', '')=='Round5'):
                    
                    if((s[1].split(' '))[1]=='DSC'):
                        deviation_coarse2fine_5 += (float((s[1].split(' '))[11]) - mean_coarse2fine_5)**2

                if (s[0].replace(' ', '')=='Round6'):

                    if((s[1].split(' '))[1]=='DSC'):
                        deviation_coarse2fine_6 += (float((s[1].split(' '))[11]) - mean_coarse2fine_6)**2

                if (s[0].replace(' ', '')=='Round7'):
                   
                    if((s[1].split(' '))[1]=='DSC'):
                        deviation_coarse2fine_7 += (float((s[1].split(' '))[11]) - mean_coarse2fine_7)**2

                if (s[0].replace(' ', '')=='Round8'):
                    
                    if((s[1].split(' '))[1]=='DSC'):
                        deviation_coarse2fine_8 += (float((s[1].split(' '))[11]) - mean_coarse2fine_8)**2

                if (s[0].replace(' ', '')=='Round9'):
                   
                    if((s[1].split(' '))[1]=='DSC'):
                        deviation_coarse2fine_9 += (float((s[1].split(' '))[11]) - mean_coarse2fine_9)**2

                if (s[0].replace(' ', '')=='Round10'):
                    
                    if((s[1].split(' '))[1]=='DSC'):
                        deviation_coarse2fine_10 += (float((s[1].split(' '))[11]) - mean_coarse2fine_10)**2



    deviation_coarse2fine_1 = (deviation_coarse2fine_1/total_samples)**0.5
    deviation_coarse2fine_2 = (deviation_coarse2fine_2/total_samples)**0.5
    deviation_coarse2fine_3 = (deviation_coarse2fine_3/total_samples)**0.5
    deviation_coarse2fine_4 = (deviation_coarse2fine_4/total_samples)**0.5
    deviation_coarse2fine_5 = (deviation_coarse2fine_5/total_samples)**0.5
    deviation_coarse2fine_6 = (deviation_coarse2fine_6/total_samples)**0.5
    deviation_coarse2fine_7 = (deviation_coarse2fine_7/total_samples)**0.5
    deviation_coarse2fine_8 = (deviation_coarse2fine_8/total_samples)**0.5
    deviation_coarse2fine_9 = (deviation_coarse2fine_9/total_samples)**0.5
    deviation_coarse2fine_10 = (deviation_coarse2fine_10/total_samples)**0.5

    output = open(result_file, 'a+')

    output.write('deviation_coarse2fine_1: ' + str(deviation_coarse2fine_1) + '\n')
    output.write('deviation_coarse2fine_2: ' + str(deviation_coarse2fine_2) + '\n')
    output.write('deviation_coarse2fine_3: ' + str(deviation_coarse2fine_3) + '\n')
    output.write('deviation_coarse2fine_4: ' + str(deviation_coarse2fine_4) + '\n')
    output.write('deviation_coarse2fine_5: ' + str(deviation_coarse2fine_5) + '\n')
    output.write('deviation_coarse2fine_6: ' + str(deviation_coarse2fine_6) + '\n')
    output.write('deviation_coarse2fine_7: ' + str(deviation_coarse2fine_7) + '\n')
    output.write('deviation_coarse2fine_8: ' + str(deviation_coarse2fine_8) + '\n')
    output.write('deviation_coarse2fine_9: ' + str(deviation_coarse2fine_9) + '\n')
    output.write('deviation_coarse2fine_10: ' + str(deviation_coarse2fine_10) + '\n')
    
    output.close()

    
                    
                
           
