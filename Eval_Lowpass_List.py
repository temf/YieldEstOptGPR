# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 08:59:43 2020

@author: Mona
"""


#import matlab.engine
from numpy import genfromtxt
import numpy as np
#import time




def eval_lowpass(self,sample_point,dB = False):
    
    sample_number = sample_point[0]+1
    freq_number = len(self.freqrange)
    
    
    
    
#    s = []
#    for i in range(len(sample_point)):
#        s.append(sample_point[i])
#    sample_point = s
    ##sample_point = [7.,7.,7.,1.,2.,1.]
    #print('s',s)
    #print('sample point', sample_point)
    #
#    f = []
#    for i in range(len(freq)):
#        f.append(freq[i])
#    freq = f
   # freq = [0,1,2,3,4,5,6,7]
   # print('f',f)
#    sample_point = matlab.double(sample_point)
#    freq = matlab.double(freq)
    #print(sample_point)
    #print(freq)
    
    
#    eng = matlab.engine.start_matlab()
    #print('blabla')
    #data=eng.CST_run(sample_point,freq)
    #data=eng.CST_run_wg(sample_point,freq,'wgc')
#    data=eng.CST(sample_point,freq,'lpf')
    
    #print(data) # freq_pos, freq, S complex???, SdB
    
    
    #QoI = genfromtxt('QoI_data_long.csv', delimiter=',',skip_header=1)
    QoI = self.QoI
    
    S_dB = QoI[(sample_number-1)*freq_number:(sample_number-1)*freq_number+freq_number,7]
    S_real = QoI[(sample_number-1)*freq_number:(sample_number-1)*freq_number+freq_number,4]
    S_imag = QoI[(sample_number-1)*freq_number:(sample_number-1)*freq_number+freq_number,5]
    
    S_dB = np.transpose([QoI[(sample_number-1)*freq_number:(sample_number-1)*freq_number+freq_number,7],QoI[(sample_number-1)*freq_number:(sample_number-1)*freq_number+freq_number,7]])
    S_real = np.transpose([QoI[(sample_number-1)*freq_number:(sample_number-1)*freq_number+freq_number,4],QoI[(sample_number-1)*freq_number:(sample_number-1)*freq_number+freq_number,4]])
    S_imag = np.transpose([QoI[(sample_number-1)*freq_number:(sample_number-1)*freq_number+freq_number,5],QoI[(sample_number-1)*freq_number:(sample_number-1)*freq_number+freq_number,5]])
    
    
    
    
    
    #print('S_real',S_real)
    #print('S_imag',S_imag)
    
#    if len(freq[0])==1:
#        if dB:
#            output = result_data[-1][3] 
#        else:
#            output_real = result_data[-1][4]
#            output_imag = result_data[-1][5]
#    else:
#        if dB:
#            output = np.transpose([np.transpose(result_data)[1][-len(freq[0]):], np.transpose(result_data)[3][-len(freq[0]):]])
#        else:
#            output_real = np.transpose([np.transpose(result_data)[1][-len(freq[0]):], np.transpose(result_data)[4][-len(freq[0]):]])
#            output_imag = np.transpose([np.transpose(result_data)[1][-len(freq[0]):], np.transpose(result_data)[5][-len(freq[0]):]])
 #   values=[]
 #   freq_indx=0
 #   freqrange = freq
 #   for k in range(len(data)):
 #       if data[k][0] == freqrange[freq_indx]:
 #           values.append(20*np.log10(data[k][1]))
 #           
 #           if freq_indx == len(freqrange)-1:
  #              
  #              break
  ##          else:
  #              freq_indx=freq_indx+1
    if dB:            
        return S_dB   
    else:
        return S_real, S_imag
    
    
    
    
    
    
    
    
    