# this script reads in the list of one fold in cross validation
from __future__ import print_function

import sys
import time
import numpy as np
import theano
from theano.compile.debugmode import DebugMode
import theano.tensor as T
from lasagne.layers import dnn
import lasagne
import matplotlib.pyplot as plt
import matplotlib
import errno
import json
#import h5py
import tables

import os
import csv
import SupportFuncs
import pdb

# input parameters
folderpath = '/media/shamidian/sda2/ws1/folds_list_indexing'
numFolds = 10
# pos neg samples have been generated beforehand

inputParamsLoadData = {}
inputParamsLoadData['input_shape'] = '44, 44, 12'
inputParamsLoadData['learning_rate'] = '0.02' #for nonlinearityFC sigmoid (relu), use values around 0.04 (0.02)
inputParamsLoadData['momentum'] = '0.9'
inputParamsLoadData['num_epochs'] = '22'
inputParamsLoadData['batch_size'] = '100'
inputParamsLoadData['noduleCaseFilterParams'] = 'NumberOfObservers,>=,3;IURatio,>,0.0;LUNA'#'NumberOfObservers,>=,3;IURatio,>,0.0;SliceThicknessDicom,<,3'
inputParamsLoadData['train_set_size'] = '50000'#'700000'
inputParamsLoadData['test_set_size'] = '500'
inputParamsLoadData['positive_set_ratio'] = '0.12'
inputParamsLoadData['dropout'] = '0        .75'
inputParamsLoadData['nonlinearityToUse'] = 'relu' #this is for the conv layers 
inputParamsLoadData['nonlinearityToUseFC'] = 'relu' #this is for the FC layers after the convs
inputParamsLoadData['numberOfLayers'] = 3 #2 or 3; number of conv layers
inputParamsLoadData['numberOfFCUnits'] = 1000 #number of units in first fully connected layer; originally 64
inputParamsLoadData['numberOfFCLayers'] = 3 #2 or 3; number of FC layers (including the softmax layer)
inputParamsLoadData['numberOfConvFilts'] = 32 #number of filters in the conv layers; originally 32
inputParamsLoadData['filterSizeTable'] = np.array(((5,5,3), (5,5,3), (3,3,1))) #one row per conv layer, so number of rows should match number of conv layers!
inputParamsLoadData['augmentationRegularFlag'] = 1 #this will only add augmentation like flips, 90 degree rotation,...
inputParamsLoadData['augmentationTransformFlag'] = 1 #this will add augmentations from combo transformations (rotation, shear, size scaling)
inputParamsLoadData['weightInitToUse'] ='He' #weight initialization; either 'normal' or 'He' (for HeNormal)
inputParamsLoadData['lrDecayFlag'] = 1 #1 for using learning rate decay, 0 for constant learning rate throughout training
inputParamsLoadData['biasInitVal'] = 0.0 #doesn't help; idea was to use bias init 1 when applying relu but it was worse!
inputParamsLoadData['fp_per_case'] = '0'#set to '-1' or '0' if you dont want false positive; implemented as fp per pos, NOT fp per case! Not used if phase=='discrim'
inputParamsLoadData['phase'] = 'screen'#'screen' or 'discrim'; changes way train set is built
inputParamsLoadData['discrim_shape'] = '36, 36, 8' #Only used if phase==discrim; specifies patch size for discrimination (necessary to know which subfolder it is in)
inputParamsLoadData['fp_model_to_use'] = 'cnn_36368_20170502171727'#only used if fp_per_case>0 or phase=='discrim'; defines model (therefore subdirectory) fp's were extracted from
inputParamsLoadData['pos_test_size'] = '100'#ignored if -1; overrides the pos to neg ratio with certain pos number in test set (s.t. it won't be tied to % selected for training set)
#total test set size which includes the negatives comes from the subtract of total test size-posetive
#No need to change the following line, it will remove space/comma from input_shape and generate data_path accordingly!
inputParamsLoadData['data_path'] = os.path.join('/diskStation/LIDC/', 
                                    ((inputParamsLoadData['input_shape']).replace(' ','')).replace(',',''))

if phase == 'screen':
    if noduleCaseFilterParams == '':
        if fp_per_case == '0': #make different filenames per fp_per_case s.t. original & supplemented train set are maintained
            training_filename = os.path.join('./',input_shape+'_'+str(augmentationRegularFlag)+str(augmentationTransformFlag)+'_'+str(positive_set_ratio)+'.hdf5')
        else:
            training_filename = os.path.join('./',input_shape+'_'+str(augmentationRegularFlag)+str(augmentationTransformFlag)+'_'+str(positive_set_ratio)
                +'_fp' + fp_model_to_use + '_' + fp_per_case +'.hdf5')
    else:
        if fp_per_case == '0': #make different filenames per fp_per_case s.t. original & supplemented train set are maintained
            training_filename = os.path.join('./',input_shape+'_'+str(augmentationRegularFlag)+str(augmentationTransformFlag)+'_'+str(positive_set_ratio)+'_filt.hdf5')
        else:
            training_filename = os.path.join('./',input_shape+'_'+str(augmentationRegularFlag)+str(augmentationTransformFlag)+'_'+str(positive_set_ratio)
                +'_fp' + fp_model_to_use + '_' + fp_per_case +'_filt.hdf5')
elif phase == 'discrim':
    if noduleCaseFilterParams == '':
        training_filename = os.path.join('./',input_shape+'_'+str(augmentationRegularFlag)+str(augmentationTransformFlag)+'_'+str(positive_set_ratio)
            +'_discrim' + discrim_shape.replace(' ','').replace(',','') + '_' + fp_model_to_use + '.hdf5')
    else: 
        training_filename = os.path.join('./',input_shape+'_'+str(augmentationRegularFlag)+str(augmentationTransformFlag)+'_'+str(positive_set_ratio)
            +'_discrim' + discrim_shape.replace(' ','').replace(',','') + '_' + fp_model_to_use + '_filt.hdf5')

# read the list directory
lst_name = os.listdir(folderpath)
# make sure no temporary files in the directory
if len(lst_name) == numFolds:
	train_x,train_y,test_x,test_y=[],[],[],[]
	# cross validation
	for k in range(0,numFolds): #the k-th fold witholds the k-th as test set
		# stacking the testing samples
		test_name =lst_name[k]
		filepath = folderpath + '/' + test_name
		with open(filepath, 'rb') as csvfile:
			reader = csv.reader(csvfile)
			for line in reader:
				test_x.append(line[0].replace('/','_'))
				test_y.append(line[1])

		# stacking the training samples
		for i in lst_name:
			if i != test_name:
				# train set
				filepath = folderpath + '/' + i
				with open(filepath, 'rb') as csvfile:
					reader = csv.reader(csvfile)
					for line in reader:
						train_x.append(line[0].replace('/','_'))
						train_y.append(line[1])

		# print the num of training/testing samples
		numTrainSamples=len(train_x)
		print('numTrainSamples=' + str(numTrainSamples))
		numTestSamples =len(test_x)
		print('numTestSamples=' + str(numTestSamples))

		# read correpsonding training set
		# set the name for finding the data
        tr_len_pos,tr_len_neg,ts_len_pos,ts_len_neg = SupportFuncs.load_data(inputParamsLoadData)    	    

		# make sure only with those in the 9 folds
		# training with tr_len_pos, tr_len_neg, ts_len_pos, ts_len_neg
		training_file_handle = tables.open_file(training_filename, mode='r')   #file closed after both train/test done                                 
	    train_set = training_file_handle.root.train_set
	    train_label = training_file_handle.root.train_label
	    test_set = training_file_handle.root.test_set
	    test_label = training_file_handle.root.test_label
	    val_set = training_file_handle.root.val_set
	    val_label = training_file_handle.root.val_label

	    tr_len_pos = len(np.where(train_label[:]==1)[0]); ts_len_pos = len(np.where(test_label[:]==1)[0])
	    tr_len_neg = train_set.shape[0] - tr_len_pos; ts_len_neg = test_set.shape[0] - ts_len_pos
	    print("Train set number of positives:" + str(tr_len_pos))
	    print("Train set number of negatives:" + str(tr_len_neg))
	    if nonlinearityToUse == 'relu':
	        nonLinearity = lasagne.nonlinearities.rectify        
	    elif nonlinearityToUse == 'tanh':
	        nonLinearity = lasagne.nonlinearities.tanh        
	    elif nonlinearityToUse == 'sigmoid':
	        nonLinearity = lasagne.nonlinearities.sigmoid        
	    else:
	        raise Exception(
	            'nonlinearityToUse: Unsupported nonlinearity type has been selected for the network, retry with a supported one!')

	    if nonlinearityToUseFC == 'relu':
	        nonLinearityFC = lasagne.nonlinearities.rectify        
	    elif nonlinearityToUseFC == 'tanh':
	        nonLinearityFC = lasagne.nonlinearities.tanh        
	    elif nonlinearityToUseFC == 'sigmoid':
	        nonLinearityFC = lasagne.nonlinearities.sigmoid        
	    else:
	        raise Exception(
	            'nonlinearityToUseFC: Unsupported nonlinearity type has been selected for the network, retry with a supported one!')
	    
	    biasInit = lasagne.init.Constant(biasInitVal) #for relu use biasInit=1 s.t. inputs to relu are positive in beginning


		# evaluation


else:
	print('Not 10 folders!\n')

# call in positive_negative generating
#for i in [1:numItems]:
#for item in x:


#csvfile.close()