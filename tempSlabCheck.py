# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 18:40:31 2016

@author: apezeshk
"""
import numpy as np
from matplotlib import pyplot as plt
import os
import h5py
import json
import SupportFuncs
import theano.tensor as T
from lasagne.layers import dnn
import lasagne
import theano
import glob 

############################################

############################################
def Build_3dcnn(init_norm=lasagne.init.Normal(), inputParamsNetwork=dict(n_layer=2,shape=[10,10,10],dropout=0.1, nonLinearity=lasagne.nonlinearities.rectify,
                biasInit=lasagne.init.Constant(0.0)), input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    dropout = inputParamsNetwork['dropout']
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

        network = lasagne.layers.dnn.Conv3DDNNLayer(network, num_filters=32, pad='same', filter_size=(5, 5, 3),
                                                    stride=(1, 1, 1),
                                                    nonlinearity=inputParamsNetwork['nonLinearity'],
                                                    W=init_norm,
                                                    b=inputParamsNetwork['biasInit'],
                                                    flip_filters=False
                                                    )

        network = lasagne.layers.dnn.MaxPool3DDNNLayer(network, pool_size=(2, 2, 1))

        network = lasagne.layers.dnn.Conv3DDNNLayer(network, num_filters=32, pad='same', filter_size=(5, 5, 3),
                                                stride=(1, 1, 1),
                                                nonlinearity=inputParamsNetwork['nonLinearity'],
                                                W=init_norm,
                                                b=inputParamsNetwork['biasInit'],
                                                )
        network = lasagne.layers.dnn.MaxPool3DDNNLayer(network, pool_size=(2, 2, 2))
    else:
        network = lasagne.layers.dnn.Conv3DDNNLayer(network, num_filters=32, pad='same', filter_size=(5, 5, 3),
                                                    stride=(1, 1, 1),
                                                    nonlinearity=inputParamsNetwork['nonLinearity'],
                                                    W=init_norm,
                                                    b=inputParamsNetwork['biasInit'],
                                                    flip_filters=False
                                                    )

        network = lasagne.layers.dnn.MaxPool3DDNNLayer(network, pool_size=(2, 2, 1))

        network = lasagne.layers.dnn.Conv3DDNNLayer(network, num_filters=32, pad='same', filter_size=(5, 5, 3),
                                                    stride=(1, 1, 1),
                                                    nonlinearity=inputParamsNetwork['nonLinearity'],
                                                    W=init_norm,
                                                    b=inputParamsNetwork['biasInit'],
                                                    )
        network = lasagne.layers.dnn.MaxPool3DDNNLayer(network, pool_size=(2, 2, 2))

        network = lasagne.layers.dnn.Conv3DDNNLayer(network, num_filters=32, pad='same', filter_size=(3, 3, 1),
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
        num_units=64,
        nonlinearity=lasagne.nonlinearities.sigmoid)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=dropout),
        num_units=2,
        nonlinearity=lasagne.nonlinearities.softmax)
    # network=lasagne.layers.DenseLayer(network, num_units=2, nonlinearity=None)

    return network

############################################
############################################    
############################################
    
    
if 0:
    asd = np.load('/diskStation/LIDC/28288/pos_28288/p0045_20000101_s3000547_3.npy').astype('float32')
    #asd = np.load('/diskStation/LIDC/28288/neg_smp_0_28288/p0045_20000101_s3000547_1.npy')
    asd = (asd - asd.min())/(asd.max() - asd.min())
    
    
    #plt.ion()
    for i in range(0,8):
        plt.figure(i)   # create a new figure
        plt.imshow(asd[:,:,i], cmap = 'gray') 
    #    plt.draw()    # show the figure, non-blocking
    #    
    #    _ = raw_input("Press [enter] to continue.") # wait for input from the user
    #    plt.close(i)    # close the figure to show the next one.
        
        
    for i in range(0,test_pred_full_volume_softmax0.shape[2]):
        plt.figure(i)   # create a new figure
        plt.subplot(121)
        plt.imshow(test_pred_full_volume_softmax0[:,:,i], cmap = 'gray') 
        
        noduleMaskResizeBin = noduleMaskResize>0.5
        noduleMaskResizeBin = noduleMaskResizeBin.astype('int')
        plt.subplot(122)
        plt.imshow(noduleMaskResizeBin[:,:,i], cmap = 'gray') 
    
    
else:  
#loads a presaved network; runs + or - examples in test set, or those from a particular patient against model
   ########################
   ######Input Params######
    minmaxFilePath = '/home/apezeshk/Codes/DeepMed/max_min.json' #file containing the min/max of each case; used for normalizing
    #minmaxFilePath = '/diskStation/temp/new_changes/max_min.json' #file containing the min/max of each case; used for normalizing
    pathSavedNetwork = '/home/apezeshk/Codes/DeepMed/models/cnn_36368_20160914121137.npz'
    #pathSavedNetwork = '/diskStation/temp/cnn_36368_20160817161531.npz'
    runFlag = 'pos' #'pos' to run only positive patches, 'neg' to only run negative patches
    patientMode = 'p0012' #'all' to run for all patients in test set, or only +/- of specific patient (e.g. 'p0012')
    testFileHdf5 = '/diskStation/temp/test_500_0.3_28288 .hdf5' #Only used if patientMode=='all'; paths to +/- test samples
    masterPatchFolder = '/diskStation/LIDC/' #master folder containing extracted patches; Only used if patientMode!='all'
    sardarImplementationFlag = 0
    
    with np.load(pathSavedNetwork) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]    
        
    dtensor5 = T.TensorType('float32', (False,) * 5)
    input_var = dtensor5('inputs')
    target_var = T.ivector('targets')
    ########################
    ########################
    ########################
    #####Network Params#####
    inputParamsConfigLocal = {}
    inputParamsConfigLocal['input_shape'] = '36, 36, 8'
    inputParamsConfigLocal['learning_rate'] = '0.05'
    inputParamsConfigLocal['momentum'] = '0.9'
    inputParamsConfigLocal['num_epochs'] = '50'
    inputParamsConfigLocal['batch_size'] = '100'
    inputParamsConfigLocal['train_set_size'] = '60000'
    inputParamsConfigLocal['test_set_size'] = '500'
    inputParamsConfigLocal['positive_set_ratio'] = '0.5'
    inputParamsConfigLocal['dropout'] = '0.1'
    inputParamsConfigLocal['nonlinearityToUse'] = 'relu'
    inputParamsConfigLocal['numberOfLayers'] = 3
    inputParamsConfigLocal['augmentationFlag'] = 1
    inputParamsConfigLocal['weightInitToUse'] ='He' #weight initialization; either 'normal' or 'He' (for HeNormal)
    inputParamsConfigLocal['lrDecayFlag'] = 1 #1 for using learning rate decay, 0 for constant learning rate throughout training
    inputParamsConfigLocal['biasInitVal'] = 0.0 #doesn't help; idea was to use bias init 1 when applying relu but it was worse!
    #No need to change the following line, it will remove space/comma from input_shape and generate data_path accordingly!
    inputParamsConfigLocal['data_path'] = os.path.join('/diskStation/LIDC/', 
                                        ((inputParamsConfigLocal['input_shape']).replace(' ','')).replace(',',''))
    ######Input Params######
    ########################
    
    inputParamsConfigAll = inputParamsConfigLocal
    #experiment_id = str(time.strftime("%Y%m%d%H%M%S"))
    input_shape = inputParamsConfigAll['input_shape']
    learning_rate = inputParamsConfigAll['learning_rate']
    momentum = inputParamsConfigAll['momentum']
    num_epochs = inputParamsConfigAll['num_epochs']
    batch_size = inputParamsConfigAll['batch_size']
    data_path = inputParamsConfigAll['data_path']
    train_set_size = inputParamsConfigAll['train_set_size']
    test_set_size = inputParamsConfigAll['test_set_size']
    positive_set_ratio = inputParamsConfigAll['positive_set_ratio']
    dropout = inputParamsConfigAll['dropout']
    nonlinearityToUse = inputParamsConfigAll['nonlinearityToUse']
    numberOfLayers = inputParamsConfigAll['numberOfLayers']
    augmentationFlag = inputParamsConfigAll['augmentationFlag']
    weightInitToUse = inputParamsConfigAll['weightInitToUse']
    lrDecayFlag = inputParamsConfigAll['lrDecayFlag']
    biasInitVal = inputParamsConfigAll['biasInitVal']
    if nonlinearityToUse == 'relu':
        nonLinearity = lasagne.nonlinearities.rectify
    elif nonlinearityToUse == 'tanh':
        nonLinearity = lasagne.nonlinearities.tanh
    elif nonlinearityToUse == 'sigmoid':
        nonLinearity = lasagne.nonlinearities.sigmoid
    else:
        raise Exception(
            'nonlinearityToUse: Unsupported nonlinearity type has been selected for the network, retry with a supported one!')
    
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
    #inputParamsNetwork = dict(shape=input_shape,dropout=float(dropout), nonLinearity=nonLinearity)
    imSize = input_shape.rsplit(',')
    imSize = [int(i) for i in imSize] #strip the commas and convert to list of int
    inputParamsNetwork = dict(n_layer=numberOfLayers, shape=input_shape,dropout=float(dropout), nonLinearity=nonLinearity,
                              biasInit = biasInit)
    ####End Input Params####
    ########################    
    #load the saved network parameters into network_cnn
    ########################    
    network_cnn = Build_3dcnn(weight_init, inputParamsNetwork, input_var)
    lasagne.layers.set_all_param_values(network_cnn, param_values)
    
    test_prediction = lasagne.layers.get_output(network_cnn, deterministic=True)
    val_fn = theano.function([input_var], [test_prediction])  # ,mode='DebugMode')
    ########################    
    ########################    
    ########################    
    #set up the input/labels to be pushed into saved network
    #First get the list of test cases
    #patient_id=[]
    #with h5py.File(os.path.join('/diskStation/temp/28288param/test_500_0.5_28288 .hdf5'), 'r') as hf:
    if patientMode=='all':
        with h5py.File(os.path.join(testFileHdf5), 'r') as hf:
            print('List of arrays in this file: \n', hf.keys())
            tmp_test_paths = hf.get('Test_set')  # Reading list of patients and test file paths
            pos_test_paths = np.array(tmp_test_paths) #full paths to all positive test patches
            tmp_test_paths = hf.get('neg_test_set')
            neg_test_paths = np.array(tmp_test_paths) #full paths to all negative test patches
    #        patient_in_test = hf.get('Patient_lables')
    #        for items in patient_in_test:
    #            patient_id.append(items)          
    else:
        tmp_path = (input_shape.replace(',','')).replace(' ','')
        pos_patch_folder = os.path.join(masterPatchFolder, tmp_path, ('pos_' + tmp_path))
        neg_patch_folder = os.path.join(masterPatchFolder, tmp_path, ('neg_smp_0_' + tmp_path))
        pos_test_paths = np.array(glob.glob(os.path.join(pos_patch_folder, (patientMode + '*'))))#find all patient* files
        neg_test_paths = np.array(glob.glob(os.path.join(neg_patch_folder, (patientMode + '*'))))
       
    if runFlag=='pos':
        num_test = pos_test_paths.shape[0]
        test_data = np.zeros((num_test,1,imSize[0], imSize[1], imSize[2]),dtype='float32') #array in proper format for input to network
        test_labels = np.ones((num_test,),dtype = 'float32') #labels for the data array     
        test_paths = pos_test_paths
    elif runFlag=='neg':
        num_test = neg_test_paths.shape[0]
        test_data = np.zeros((num_test,1,imSize[0], imSize[1], imSize[2]),dtype='float32') #array in proper format for input to network
        test_labels = np.zeros((num_test,),dtype = 'float32') #labels for the data array                
        test_paths = neg_test_paths
    
    
    with open(minmaxFilePath) as json_data:
        minmaxAll = json.load(json_data)
        
    for i in range(0,num_test):
        currentPath = test_paths[i]
        currentCase = os.path.basename(currentPath)
        currentCase = currentCase.split('_')[0:3]
        currentCase = "_".join(currentCase)
        
        currentMinMax = minmaxAll[currentCase].split('/')
        currentMax = np.float32(currentMinMax[0])
        currentMin = np.float32(currentMinMax[1])
        
        if sardarImplementationFlag==0:            
            currentPatch = (np.load(currentPath)).astype('int32') #converting uint16 to int32/float32 directly is problematic
            currentPatch = currentPatch.astype('float32')
            currentPatch = (currentPatch - currentMin)/(currentMax - currentMin)
        else:
            currentPatch = (np.load(currentPath)).astype('int16') #should convert uint16 to int16 then float32 to avoid typecasting problem
            currentPatch = currentPatch.astype('float32')
            currentPatch = (currentPatch - currentMax)/(currentMax - currentMin)
        
        test_data[i,0,:,:,:] = currentPatch
        
    test_pred = val_fn(test_data)
    test_pred = test_pred[0]
        
    test_acc = np.mean(np.equal(np.argmax(test_pred, axis=1), test_labels),
                      dtype=float)
                      
        
    
    
    
    
    
    
    
    
    
