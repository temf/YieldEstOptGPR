# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 08:59:43 2020

@author: Mona


Returns QoI data for Lowpass Filter from a csv file (S-Parameter values)
"""



import numpy as np


def eval_lowpass(self,sample_point,dB = False):
    
    sample_number = sample_point[0]+1
    freq_number = len(self.freqrange)
    
    QoI = self.QoI
    
    S_dB = QoI[(sample_number-1)*freq_number:(sample_number-1)*freq_number+freq_number,7]
    S_real = QoI[(sample_number-1)*freq_number:(sample_number-1)*freq_number+freq_number,4]
    S_imag = QoI[(sample_number-1)*freq_number:(sample_number-1)*freq_number+freq_number,5]
    
    S_dB = np.transpose([QoI[(sample_number-1)*freq_number:(sample_number-1)*freq_number+freq_number,1],QoI[(sample_number-1)*freq_number:(sample_number-1)*freq_number+freq_number,7]])
    S_real = np.transpose([QoI[(sample_number-1)*freq_number:(sample_number-1)*freq_number+freq_number,1],QoI[(sample_number-1)*freq_number:(sample_number-1)*freq_number+freq_number,4]])
    S_imag = np.transpose([QoI[(sample_number-1)*freq_number:(sample_number-1)*freq_number+freq_number,1],QoI[(sample_number-1)*freq_number:(sample_number-1)*freq_number+freq_number,5]])
    
    if dB:            
        return S_dB   
    else:
        return S_real, S_imag
    
    
    
    
    
    
    
    
    