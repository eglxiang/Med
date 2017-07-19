# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 13:03:45 2017

@author: apezeshk
"""
#######################################
# This part reads patches from a folder (pos/neg/fp/...) & displays a number of them
# to make sure you are extracting the right stuff! 
#######################################
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import h5py

patchDirectory = '/diskStation/LIDC/36368/fp_cases/cnn_36368_20170613130504/'
#patchDirectory = '/diskStation/LIDC/36368/pos_36368'
patchFiles = os.listdir(patchDirectory)
plt.figure()
plt.ion()
for i in range(0,50):
    currentPatchFile = os.path.join(patchDirectory, patchFiles[i])
    currentPatch = np.load(currentPatchFile)
    currentPatch = currentPatch.astype('int16') #bc some dicom cases are uint, u have to first read as int16; otherwise you get the max 65535 situation!
    #currentPatch = currentPatch.astype('float32')
    for j in range(0,8):
        plt.imshow(currentPatch[:,:,j], cmap='gray')
        tit ='fpnum: ' + str(i) + ', slc: ' + str(j);
        plt.title(tit)
        plt.show()
        time.sleep(0.15)
        #_ = raw_input('blah')
        plt.pause(0.0001)
        
        
#######################################
# This part reads a saved training file; you can then specify which of
# train/test/val sets you want to view to make sure you are including the right stuff!
#######################################
training_filename = '/home/apezeshk/Codes/DeepMed/36, 36, 8_1_0.5_discrim36368_cnn_36368_20161209111319.hdf5'
if os.path.exists(training_filename):
    with h5py.File(training_filename, 'r') as data_set:
        tmp_train_set = data_set.get('train_set')  # Reading list of patients and test file paths
        train_set = np.array(tmp_train_set)
        tmp_train_label = data_set.get('train_label')  # Reading list of patients and test file paths
        train_label = np.array(tmp_train_label)
        tmp_test_set = data_set.get('test_set')  # Reading list of patients and test file paths
        test_set = np.array(tmp_test_set)
        tmp_test_label = data_set.get('test_label')  # Reading list of patients and test file paths
        test_label = np.array(tmp_test_label)
        tmp_val_set = data_set.get('val_set')  # Reading list of patients and test file paths
        val_set = np.array(tmp_val_set)
        tmp_val_label = data_set.get('val_label')  # Reading list of patients and test file paths
        val_label = np.array(tmp_val_label)
        tr_len_pos = len(np.where(train_label==1)[0])
        tr_len_neg = len(np.where(train_label==0)[0])
        
plt.figure()
plt.ion()
for i in range(231,250):    
    currentPatch = val_set[i,0,:,:,:]
    #currentPatch = currentPatch.astype('int16') #bc some dicom cases are uint, u have to first read as int16; otherwise you get the max 65535 situation!
    #currentPatch = currentPatch.astype('float32')
    for j in range(0,8):
        plt.imshow(currentPatch[:,:,j], cmap='gray')
        tit ='fpnum: ' + str(i) + ', slc: ' + str(j);
        plt.title(tit)
        plt.show()
        time.sleep(0.12)
        #_ = raw_input('blah')
        plt.pause(0.0001)
        
