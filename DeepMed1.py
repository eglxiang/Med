# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 11:06:19 2016

@author: apezeshk
"""

from __future__ import print_function

import sys
import os
import time
import numpy as np
import theano
from theano.compile.debugmode import DebugMode
import SupportFuncs
import theano.tensor as T
from lasagne.layers import dnn
import lasagne
import matplotlib.pyplot as plt
import csv
import matplotlib
import errno
import json
#import h5py
import tables

localFlag = 1 #if 1, reads from the parameter settings defined on top, otherwise reads the csv config file
LIDC_PATH= '/diskStation/LIDC/'
test_path = './test_test'  # positive test data
result_path = './results'  # csv file results
figures_path = './figures'
model_path = './models'  # folder where trained models are saved

random_state = 1234
np.random.seed(random_state)
lasagne.layers.noise._srng = lasagne.layers.noise.RandomStreams(random_state)

########################
######Input Params######
inputParamsConfigLocal = {}
inputParamsConfigLocal['input_shape'] = '36, 36, 8'
inputParamsConfigLocal['learning_rate'] = '0.02' #for nonlinearityFC sigmoid (relu), use values around 0.04 (0.02)
inputParamsConfigLocal['momentum'] = '0.9'
inputParamsConfigLocal['num_epochs'] = '22'
inputParamsConfigLocal['batch_size'] = '100'
inputParamsConfigLocal['noduleCaseFilterParams'] = 'NumberOfObservers,>=,3;IURatio,>,0.0;LUNA'#'NumberOfObservers,>=,3;IURatio,>,0.0;SliceThicknessDicom,<,3'
inputParamsConfigLocal['train_set_size'] = '700000'
inputParamsConfigLocal['test_set_size'] = '500'
inputParamsConfigLocal['positive_set_ratio'] = '0.12'
inputParamsConfigLocal['dropout'] = '0.75'
inputParamsConfigLocal['nonlinearityToUse'] = 'relu' #this is for the conv layers 
inputParamsConfigLocal['nonlinearityToUseFC'] = 'relu' #this is for the FC layers after the convs
inputParamsConfigLocal['numberOfLayers'] = 3 #2 or 3; number of conv layers
inputParamsConfigLocal['numberOfFCUnits'] = 1000 #number of units in first fully connected layer; originally 64
inputParamsConfigLocal['numberOfFCLayers'] = 3 #2 or 3; number of FC layers (including the softmax layer)
inputParamsConfigLocal['numberOfConvFilts'] = 32 #number of filters in the conv layers; originally 32
inputParamsConfigLocal['filterSizeTable'] = np.array(((5,5,3), (5,5,3), (3,3,1))) #one row per conv layer, so number of rows should match number of conv layers!
inputParamsConfigLocal['augmentationRegularFlag'] = 1 #this will only add augmentation like flips, 90 degree rotation,...
inputParamsConfigLocal['augmentationTransformFlag'] = 1 #this will add augmentations from combo transformations (rotation, shear, size scaling)
inputParamsConfigLocal['weightInitToUse'] ='He' #weight initialization; either 'normal' or 'He' (for HeNormal)
inputParamsConfigLocal['lrDecayFlag'] = 1 #1 for using learning rate decay, 0 for constant learning rate throughout training
inputParamsConfigLocal['biasInitVal'] = 0.0 #doesn't help; idea was to use bias init 1 when applying relu but it was worse!
inputParamsConfigLocal['fp_per_case'] = '50'#set to '-1' or '0' if you dont want false positive; implemented as fp per pos, NOT fp per case! Not used if phase=='discrim'
inputParamsConfigLocal['phase'] = 'screen'#'screen' or 'discrim'; changes way train set is built
inputParamsConfigLocal['discrim_shape'] = '36, 36, 8'#Only used if phase==discrim; specifies patch size for discrimination (necessary to know which subfolder it is in)
inputParamsConfigLocal['fp_model_to_use'] = 'cnn_36368_20170502171727'#only used if fp_per_case>0 or phase=='discrim'; defines model (therefore subdirectory) fp's were extracted from
inputParamsConfigLocal['pos_test_size'] = '100'#ignored if -1; overrides the pos to neg ratio with certain pos number in test set (s.t. it won't be tied to % selected for training set)
#total test set size which includes the negatives comes from the subtract of total test size-posetive
#No need to change the following line, it will remove space/comma from input_shape and generate data_path accordingly!
inputParamsConfigLocal['data_path'] = os.path.join('/diskStation/LIDC/', 
                                    ((inputParamsConfigLocal['input_shape']).replace(' ','')).replace(',',''))
######Input Params######
########################

#n_layers = 3
#weight_init = lasagne.init.Normal() #moving this done so that He initialization with proper gain can be used
epoch_det = {}  # storyin the epoc results including val acc,loss, and training loss
all_val_accuracy = []  # Validation_accuracy list is used for the plot
all_val_loss = []  # Validation_loss list is used for the plot
all_val_AUC = []  # list of AUC for validation set across each epoch
training_loss = []  # Training_loss list is used for the plot
csv_config=[] #if localFlag==0, code will read network parameters from a csv and store that in csv_config
# print(os.path.join('./figures', experiment_id + '.png'))


# To make sure a direcotry exist
def Make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


#def Pathcheck(path):
#    if os.path.isdir(os.path.join('../result', path)):
#        print(path, "exist")
#    else:
#        Make_sure_path_exists(path)
#        print(path, "Just created and is ready to use")


# Deleting all the cases with nan value
#def Rem_nan(path):
#    for cases in os.listdir(path):
#        case_mat = np.load(os.path.join(path, cases))
#        if np.amax(case_mat) == np.amin(case_mat):
#            print(cases)
#            plt.imsave("./norm/" + str(cases[:-4]), case_mat[:, :, 8], cmap='gray')
#            os.remove(os.path.join(path, cases))


#def Norm(matrix):
#    output = ((matrix - np.amin(matrix)) / (np.amax(matrix) - np.amin(matrix)))
#    return output


# Gets input files size and all the parameterest that are used for thea model and save them as a row in csv files named as
# input files size

def Save_model(result_path, experi_id, input_shape, n_layers, batch_size, n_epoch, momentum, n_filter, training_loss,
               data_size, test_size, learning_rate, Totall_Accuracy, Test_AUC, sts, augmentationRegularFlag, augmentationTransformFlag, nonlinearityToUse,
               dropout,tr_len_pos,tr_len_neg):
    parameters = [result_path, experi_id, input_shape, n_layers, batch_size, n_epoch, momentum, n_filter, training_loss,
                  data_size, test_size, learning_rate, Totall_Accuracy, Test_AUC, sts, augmentationRegularFlag, augmentationTransformFlag,
                  nonlinearityToUse, dropout,tr_len_pos,tr_len_neg]
    header = ['Result_path', 'Experiment_id', 'Input_shape', 'N_layers', 'Batch_size', 'n_epoch', 'Momentum',
              'Learning_rate', 'n_filter', 'TrainingSet', 'TestSet', 'Training_loss', 'Total_accuracy', 'Test_AUC',
              'std', 'augmentationRegularFlag', 'augmentationTransformFlag', 'nonlinearityToUse', 'dropout','tr_num_pos','tr_num_neg']

    if os.path.isfile(os.path.join(result_path, str(input_shape.replace(',',''))) + str(
            ".csv")):
        writer = csv.writer(open(
            os.path.join(result_path, str(input_shape.replace(',',''))) + str(".csv"),
            'a'))
        writer.writerow(parameters)
    else:
        writer = csv.writer(open(
            os.path.join(result_path, str(input_shape.replace(',',''))) + str(".csv"),
            'wb'))
        writer.writerow(header)
        writer.writerow(parameters)


def Build_3dcnn(init_norm=lasagne.init.Normal(), inputParamsNetwork=dict(n_layer=2,shape=[10,10,10],dropout=0.1, nonLinearity=lasagne.nonlinearities.rectify, 
                nonLinearityFC=lasagne.nonlinearities.rectify, biasInit=lasagne.init.Constant(0.0)), input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    dropout = inputParamsNetwork['dropout']
    filterSizes = inputParamsNetwork['filterSizeTable']
    numberOfFCUnits = inputParamsNetwork['numberOfFCUnits']
    numberOfConvFilts = inputParamsNetwork['numberOfConvFilts']
    numberOfFCLayers = inputParamsNetwork['numberOfFCLayers']
    #    print(dropout)
    network = lasagne.layers.InputLayer(shape=(None,1,int(inputParamsNetwork['shape'].split(',')[0]),int(inputParamsNetwork['shape'].split(',')[1]),int(inputParamsNetwork['shape'].split(',')[2])),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    # network=lasagne.layers.dnn.Conv3DDNNLayer(network,num_filters=32,filter_size=(3,3,4),
    #                                           stride=(1, 1, 1),pad=1,
    #                                           nonlinearity=lasagne.nonlinearities.rectify,
    #                                           W=lasagne.init.GlorotUniform()
    #                                           )
    # network=lasagne.layers.dnn.MaxPool3DDNNLayer(network, pool_size=(2,2,2),stride=(2,2,2))
    if inputParamsNetwork['n_layer'] == 2 :

        network = lasagne.layers.dnn.Conv3DDNNLayer(network, num_filters=numberOfConvFilts, pad='same', filter_size=filterSizes[0,:],
                                                    stride=(1, 1, 1),
                                                    nonlinearity=inputParamsNetwork['nonLinearity'],
                                                    W=init_norm,
                                                    b=inputParamsNetwork['biasInit'],
                                                    flip_filters=False
                                                    )

        network = lasagne.layers.dnn.MaxPool3DDNNLayer(network, pool_size=(2, 2, 1))

        network = lasagne.layers.dnn.Conv3DDNNLayer(network, num_filters=numberOfConvFilts, pad='same', filter_size=filterSizes[1,:],
                                                stride=(1, 1, 1),
                                                nonlinearity=inputParamsNetwork['nonLinearity'],
                                                W=init_norm,
                                                b=inputParamsNetwork['biasInit'],
                                                )
        network = lasagne.layers.dnn.MaxPool3DDNNLayer(network, pool_size=(2, 2, 2))
    else:
        network = lasagne.layers.dnn.Conv3DDNNLayer(network, num_filters=numberOfConvFilts, pad='same', filter_size=filterSizes[0,:],
                                                    stride=(1, 1, 1),
                                                    nonlinearity=inputParamsNetwork['nonLinearity'],
                                                    W=init_norm,
                                                    b=inputParamsNetwork['biasInit'],
                                                    flip_filters=False
                                                    )

        network = lasagne.layers.dnn.MaxPool3DDNNLayer(network, pool_size=(2, 2, 1))

        network = lasagne.layers.dnn.Conv3DDNNLayer(network, num_filters=numberOfConvFilts, pad='same', filter_size=filterSizes[1,:],
                                                    stride=(1, 1, 1),
                                                    nonlinearity=inputParamsNetwork['nonLinearity'],
                                                    W=init_norm,
                                                    b=inputParamsNetwork['biasInit'],
                                                    )
        network = lasagne.layers.dnn.MaxPool3DDNNLayer(network, pool_size=(2, 2, 2))

        network = lasagne.layers.dnn.Conv3DDNNLayer(network, num_filters=numberOfConvFilts, pad='same', filter_size=filterSizes[2,:],
                                                    stride=(1, 1, 1),
                                                    nonlinearity=inputParamsNetwork['nonLinearity'],
                                                    W=init_norm,
                                                    b=inputParamsNetwork['biasInit'],
                                                    )


        # network = lasagne.layers.dnn.MaxPool3DDNNLayer(network, pool_size=(2, 2, 1))



    # network=lasagne.layers.PadLayer(network,width=[(0,1),(0,1)], batch_ndim=3)
    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:

    # network = lasagne.layers.dnn.MaxPool3DDNNLayer(network, pool_size=(2, 2, 2))


    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=dropout),
        num_units=numberOfFCUnits,
        nonlinearity=inputParamsNetwork['nonLinearityFC'])
        
    if numberOfFCLayers == 3:
        network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=dropout),
        num_units=numberOfFCUnits,
        nonlinearity=inputParamsNetwork['nonLinearityFC'])

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=dropout),
        num_units=2,
        nonlinearity=lasagne.nonlinearities.softmax)
    # network=lasagne.layers.DenseLayer(network, num_units=2, nonlinearity=None)

    return network


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.
# Notice that this function returns only mini-batches of size `batch_size`.
# If the size of the data is not a multiple of `batch_size`, it will not
# return the last (remaining) mini-batch.


def Iterate_minibatches(inputs, targets, batch_size, shuffle=False):
    assert len(inputs) == len(targets)    
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        #if passing inputs var as regular np array, can use the commented yield.
        #but for memory mapped inputs var should do this yield
        yield inputs[excerpt, :,:,:,:], targets[excerpt] 
        #yield inputs[excerpt], targets[excerpt]

        # def main(num_epochs=50,learning_rate=0.1):
        # Load the dataset pos_train_path,neg_train_path,pos_test_paths,neg_test_path


def Main(inputParamsConfig):
    #input_shape,learning_rate, momentum, num_epochs, batchsize, data_path, train_set_size, test_set_size,
#         positive_set_ratio, dropout,nonlinearityToUse,augmentation
    experiment_id = str(time.strftime("%Y%m%d%H%M%S"))
    input_shape = inputParamsConfig['input_shape']
    learning_rate = inputParamsConfig['learning_rate']
    momentum = inputParamsConfig['momentum']
    num_epochs = inputParamsConfig['num_epochs']
    batch_size = inputParamsConfig['batch_size']
    noduleCaseFilterParams = inputParamsConfig['noduleCaseFilterParams']
    data_path = inputParamsConfig['data_path']
    train_set_size = inputParamsConfig['train_set_size']
    test_set_size = inputParamsConfig['test_set_size']
    positive_set_ratio = inputParamsConfig['positive_set_ratio']
    dropout = inputParamsConfig['dropout']
    nonlinearityToUse = inputParamsConfig['nonlinearityToUse']
    nonlinearityToUseFC = inputParamsConfig['nonlinearityToUseFC']
    numberOfLayers = inputParamsConfig['numberOfLayers']
    numberOfFCUnits = inputParamsConfig['numberOfFCUnits']
    numberOfFCLayers = inputParamsConfig['numberOfFCLayers']
    numberOfConvFilts = inputParamsConfig['numberOfConvFilts']
    filterSizeTable = inputParamsConfig['filterSizeTable']
    augmentationRegularFlag = inputParamsConfig['augmentationRegularFlag']
    augmentationTransformFlag = inputParamsConfig['augmentationTransformFlag']
    weightInitToUse = inputParamsConfig['weightInitToUse']
    lrDecayFlag = inputParamsConfig['lrDecayFlag']
    biasInitVal = inputParamsConfig['biasInitVal']
    fp_per_case = inputParamsConfig['fp_per_case']
    phase = inputParamsConfig['phase']
    discrim_shape = inputParamsConfigLocal['discrim_shape']
    
    pos_test_size = inputParamsConfig['pos_test_size']
    fp_model_to_use = inputParamsConfig['fp_model_to_use']
    
    print(
        " Learning rate: '%s' , momentum: '%s',  num_epochs: '%s'  ,batch size: '%s'  ,data_path: '%s',Train Set Size: '%s' ,Test set Size: '%s' ,Positive set Ratio '%s' , dropout: '%s', nonlinearityToUse: '%s',augmentationRegularFlag: '%s',augmentationTransformFlag: '%s',number of layers: '%s', pos_test_size: '%s'" % (
        str(learning_rate), str(momentum), str(num_epochs), str(batch_size), data_path, str(train_set_size),
        str(test_set_size), str(positive_set_ratio), str(dropout), str(nonlinearityToUse), str(augmentationRegularFlag),
        str(augmentationTransformFlag), str(numberOfLayers), str(pos_test_size)))
    print(" Phase: '%s', Num FC Layers: '%s', Num FC Units: '%s', Number of ConvFilters: '%s'" % (str(phase), str(numberOfFCLayers), str(numberOfFCUnits), str(numberOfConvFilts)))
    num_epochs=int(num_epochs)
    batch_size=int(batch_size)
    
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
    # We save the created train and test set of size X and posetive ration r to reduce the overhead in running the pipline
    if os.path.exists(training_filename):
        print ("Training file already exists, reading it...")
#        with h5py.File(training_filename, 'r') as data_set:
#            tmp_train_set = data_set.get('train_set')  # Reading list of patients and test file paths
#            train_set = np.array(tmp_train_set)
#            tmp_train_label = data_set.get('train_label')  # Reading list of patients and test file paths
#            train_label = np.array(tmp_train_label)
#            tmp_test_set = data_set.get('test_set')  # Reading list of patients and test file paths
#            test_set = np.array(tmp_test_set)
#            tmp_test_label = data_set.get('test_label')  # Reading list of patients and test file paths
#            test_label = np.array(tmp_test_label)
#            tmp_val_set = data_set.get('val_set')  # Reading list of patients and test file paths
#            val_set = np.array(tmp_val_set)
#            tmp_val_label = data_set.get('val_label')  # Reading list of patients and test file paths
#            val_label = np.array(tmp_val_label)
#            tr_len_pos = len(np.where(train_label==1)[0])
#            tr_len_neg = len(np.where(train_label==0)[0])
    else:
        inputParamsLoadData = {}
        inputParamsLoadData['data_path'] = data_path
        inputParamsLoadData['input_shape'] = input_shape
        inputParamsLoadData['train_set_size'] = int(train_set_size)
        inputParamsLoadData['test_set_size'] = int(test_set_size)
        inputParamsLoadData['augmentationRegularFlag'] = int(augmentationRegularFlag)
        inputParamsLoadData['augmentationTransformFlag'] = int(augmentationTransformFlag)
        inputParamsLoadData['fp_per_case'] = int(fp_per_case)
        inputParamsLoadData['pos_test_size'] = int(pos_test_size)
        inputParamsLoadData['positive_set_ratio'] = float(positive_set_ratio)
        inputParamsLoadData['fp_model_to_use'] = fp_model_to_use
        inputParamsLoadData['phase'] = phase
        inputParamsLoadData['discrim_shape'] = discrim_shape
        inputParamsLoadData['noduleCaseFilterParams'] = noduleCaseFilterParams
        inputParamsLoadData['training_filename'] = training_filename
        
        tr_len_pos,tr_len_neg, ts_len_pos,ts_len_neg = SupportFuncs.load_data(inputParamsLoadData)    	    
       # train_set, train_label, test_set, test_label, val_set, val_label = SupportFuncs.load_data(data_path, int(train_set_size),
        #                                                                                      int(test_set_size),
         #                                                                                     int(augmentationFlag),float(positive_set_ratio))
#        with h5py.File(training_filename, 'w') as data_set:
#            #Write the dataset to a h5py file
#            data_set.create_dataset('train_set', data=train_set)
#            data_set.create_dataset('train_label', data=train_label)
#            data_set.create_dataset('test_set', data=test_set)
#            data_set.create_dataset('test_label', data=test_label)
#            data_set.create_dataset('val_set', data=val_set)
#            data_set.create_dataset('val_label', data=val_label)
                                     
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

    if weightInitToUse == 'normal': #according to documentation, different gains should be used depending on nonlinearity
        weight_init = lasagne.init.Normal()
    elif weightInitToUse == 'He':
        if nonlinearityToUse=='relu':
            gainToUse = np.sqrt(2)            
        else:
            gainToUse = 1
        
        weight_init = lasagne.init.HeNormal(gain=gainToUse)
    else:
        raise Exception(
            'weightInitToUse: Unsupported weight initialization type has been selected, retry with a supported one!')
            
    if lrDecayFlag==1: #if learning rate should be updated, then it has to be a shared variable
        learning_rate = theano.shared(np.array(learning_rate, dtype=theano.config.floatX))
        decayRate = 0.5
    else:
        learning_rate = float(learning_rate)
            
    dtensor5 = T.TensorType('float32', (False,) * 5)
    input_var = dtensor5('inputs')
    target_var = T.ivector('targets')

    inputParamsNetwork = dict(n_layer=numberOfLayers, shape=input_shape,dropout=float(dropout), nonLinearity=nonLinearity,
                              biasInit = biasInit, filterSizeTable = filterSizeTable, numberOfFCLayers=numberOfFCLayers,
                              numberOfFCUnits=numberOfFCUnits, numberOfConvFilts=numberOfConvFilts, 
                              nonLinearityFC=nonLinearityFC)
    network = Build_3dcnn(weight_init, inputParamsNetwork, input_var)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # loss=np.mean(loss)
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=learning_rate, momentum=float(momentum))

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    # test_loss = test_loss.mean()
    test_loss = test_loss.mean()

    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)  # mode='DebugMode'

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc, test_prediction])  # ,mode='DebugMode')

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(int(num_epochs)):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in Iterate_minibatches(train_set, train_label, int(batch_size), shuffle=True):
            inputs, targets = batch
            inputs = np.float32(inputs)
            train_err += train_fn(inputs, targets)
            train_batches += 1
            
        print('learning_rate = ' + str(learning_rate.get_value()))
        if lrDecayFlag == 1: #only update learning_rate if lrDecayFlag==1
            if ((epoch+1) % 12) == 0:
                learning_rate.set_value(decayRate * learning_rate.get_value())
            
        

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        all_val_pred = np.empty((0, 2),
                                dtype=float)  # initialize; array n_samplesx2 for the 2 class predictions for all validation samples
        all_val_labels = np.empty((0, 1),
                                  dtype=float)  # initialize; array n_samplesx1 for labels of all validation samples
        for batch in Iterate_minibatches(val_set, val_label, int(batch_size), shuffle=False):
            inputs, targets = batch
            inputs = np.float32(inputs)
            err, acc, val_pred = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1
            all_val_pred = np.vstack((all_val_pred, val_pred))
            all_val_labels = np.append(all_val_labels, targets)

        val_AUC, val_varAUC = SupportFuncs.Pred2AUC(all_val_pred, all_val_labels)
        # Then we print the results for this epoch:

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))
        print("  validation AUC: " + str(val_AUC) + ", std: " + str(np.sqrt(val_varAUC)))
        epoch_det[epoch + 1] = {'all_val_accuracy': (val_acc / val_batches), "all_val_loss": (val_err / val_batches),
                                "training_loss": (train_err / train_batches)}
        all_val_accuracy.append(val_acc / val_batches)
        all_val_loss.append(val_err / val_batches)
        all_val_AUC.append(val_AUC)
        training_loss.append(train_err / train_batches)

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    all_test_pred = np.empty((0, 2),
                             dtype=float)  # initialize; array n_samplesx2 for the 2 class predictions for all test samples
    all_test_labels = np.empty((0, 1), dtype=float)  # initialize; array n_samplesx1 for labels of all test samples
    for batch in Iterate_minibatches(test_set, test_label, int(batch_size), shuffle=False):
        inputs, targets = batch
        inputs = np.float32(inputs)
        err, acc, test_pred = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
        all_test_pred = np.vstack((all_test_pred, test_pred))
        all_test_labels = np.append(all_test_labels, targets)

    test_AUC, test_varAUC = SupportFuncs.Pred2AUC(all_test_pred, all_test_labels)
    
    ##########################################
    training_file_handle.close()
    ##########################################

    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))
    print("test AUC: " + str(test_AUC) + ", std: " + str(np.sqrt(test_varAUC)))
    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    tmp = input_shape.replace(' ','') #get rid of space and comma in input shape
    tmp = tmp.replace(',','')    
    filenameModel = os.path.join(model_path, 'cnn_' + tmp + '_' + experiment_id)
    filenameSamples = filenameModel + '_samples'
    np.savez(filenameModel, *lasagne.layers.get_all_param_values(network))
    np.savez(filenameSamples, inputs=inputs, targets=targets, err=err, acc=acc, training_loss=training_loss,
             test_pred=test_pred, all_test_pred=all_test_pred,all_test_labels=all_test_labels,
             inputParamsConfig=inputParamsConfig, tr_len_pos = tr_len_pos, tr_len_neg = tr_len_neg,
             all_val_loss=all_val_loss, all_val_accuracy=all_val_accuracy, test_AUC=test_AUC, test_varAUC=test_varAUC)
    if not os.path.exists('./figures'):
        os.makedirs('./figures')
    fig = plt.figure()
    # plt.plot(training_loss,'r',val_accuracy,'g',all_val_loss,'b')
    plt.plot(training_loss, 'r', label='Training_loss=' + str("%.6f" % training_loss[num_epochs - 1]))
    plt.plot(all_val_loss, 'r--', label='Val_loss=' + str("%.3f" % all_val_loss[num_epochs - 1]))
    plt.plot(all_val_accuracy, 'g', label='Val_accuracy=' + str("%.3f" % all_val_accuracy[num_epochs - 1]))
    plt.annotate(str("%.3f" % all_val_accuracy[num_epochs - 1]), xy=(num_epochs - 1, all_val_accuracy[num_epochs - 1]),
                 xytext=(num_epochs - 70, 0.6),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    plt.annotate(str("%.6f" % training_loss[num_epochs - 1]), xy=(num_epochs - 1, training_loss[num_epochs - 1]),
                 xytext=(num_epochs - 70, 0.3),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    plt.ylabel('Training loss and Validation accuracy')
    plt.xlabel('Number of Epochs')
    plt.title('Accuracy and Loss Changes')
    plt.legend(fontsize=13, loc=10)
    try:
        fig.savefig(os.path.join(figures_path, experiment_id))  # save the figure to file
    except:
        Make_sure_path_exists(figures_path)

    plt.close(fig)
    plt.show()
    # save_model(result_path, experiment_id, str(input_shape), n_layers, int(batchsize), num_epochs, momentum, learning_rate, 2
    #            , len(train_set), len(test_set), (test_err / test_batches), (test_acc / test_batches), test_AUC[0],
    #            np.sqrt(test_varAUC),augmentation)

    plt.close(fig)
    # plt.show()
    Save_model(result_path, experiment_id, str(input_shape), numberOfLayers, int(batch_size), num_epochs, momentum,
               inputParamsConfig['learning_rate'], 2
               , tr_len_pos+tr_len_neg, ts_len_pos+ts_len_neg, (test_err / test_batches), (test_acc / test_batches), test_AUC[0],
               np.sqrt(test_varAUC), augmentationRegularFlag, augmentationTransformFlag, nonlinearityToUse, dropout,tr_len_pos,tr_len_neg)

if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a CNN on LIDC using Lasagne.")
        print("Usage: [ CNN MODEL [EPOCHS]]" % sys.argv[3])
        print()
        print("       'cnn' for a Convolutional Neural Network (CNN).")
        print("EPOCHS: number of training epochs to perform (default: 500)")
    else:
        if localFlag==1: #in this case ignore the config csv, and use the parameters set out in the beginning for local run
            inputParamsConfigAll = inputParamsConfigLocal
            Main(inputParamsConfigAll)

        else: #in this case ignore run batch (sequentiaDeepMed_gpu1.pyl) mode where config is read from csv file
            done_cases={}#Contains all the
            inputParamsConfigAll={}
            if os.path.exists('./succesfull_cases.json'):
                with open('./succesfull_cases.json', 'r') as f:
                    done_cases=json.load(f)
            with open(os.path.join('./',"param0.csv"), 'rU') as csvfile:
                timeReader = csv.reader(csvfile)
                for row in timeReader:
                    csv_config.append(row)
            # writer = csv.writer(open(os.path.join(LIDC_PATH,"param.csv"), 'wb'))

            for row in csv_config[1:]:
                if ''.join(row) not in done_cases or done_cases[''.join(row)]==0: #This condition calls the main pipeline only for new cases and unsuccessful cases
                    print(row)
                    done_cases[''.join(row)] = 1
                    with open('./succesfull_cases.json', 'w') as f:
                        json.dump(done_cases, f)
                     #We make it equal to one to stop multiple running on the same parameters
                    for i in range(0,len(csv_config[0])): #for each element in header, put corresponding value in current row in config
                        inputParamsConfigAll[csv_config[0][i]] = row[i]

                    try:
                        Main(inputParamsConfigAll)

                    except:
                        done_cases[''.join(row)] = 0# We make it equal to zero if it was unsucceful try
                        with open('./succesfull_cases.json', 'w') as f:
                            json.dump(done_cases, f)
                        print ("oops error")
                else:
                    print(row ," are used once and you can find the result from ./result directory")

