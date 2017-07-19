# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 11:34:17 2016

@author: apezeshk
"""
import numpy as np
import lasagne
from lasagne.layers import dnn
import theano.tensor as T
import theano
import time
import scipy.io as sio
import os
import scipy.misc
import SupportFuncs
from scipy import ndimage as nd
import h5py
import json
import test_MaskFCN_lessMaxPool
from scipy.ndimage import measurements as Nd_measure
########################
########################
    # Optionally, you could now dump the network weights to a file like this:
sardarImplementationFlag = 0 #this one does the wrong normalization according to Sardar
#pathSavedNetwork = '/home/apezeshk/Codes/DeepMed/models/cnn_36368_20160817161531.npz'
#pathSavedSamples = '/home/apezeshk/Codes/DeepMed/models/cnn_36368_20160817161531_samples.npz'
#pathSavedNetwork = '/home/apezeshk/Codes/DeepMed/models/cnn_36368_20160914121137.npz'
#pathSavedSamples = '/home/apezeshk/Codes/DeepMed/models/cnn_36368_20160914121137_samples.npz'
#pathSavedNetwork = '/home/apezeshk/Codes/DeepMed/models/cnn_36368_20160921102657.npz'
#pathSavedSamples = '/home/apezeshk/Codes/DeepMed/models/cnn_36368_20160921102657_samples.npz'
#pathSavedNetwork = '/home/apezeshk/Codes/DeepMed/models/cnn_36368_20160921114711.npz'
#pathSavedSamples = '/home/apezeshk/Codes/DeepMed/models/cnn_36368_20160921114711_samples.npz'

#these are the new models; created after updating the test file (s.t. it has 250 +s in it instead of percentage based)
#as well as all the changes with how the noduleMask is processed for tagging, etc
#pathSavedNetwork = '/home/apezeshk/Codes/DeepMed/models/cnn_36368_20161207174913.npz'
#pathSavedSamples = '/home/apezeshk/Codes/DeepMed/models/cnn_36368_20161207174913_samples.npz'
pathSavedNetwork = '/home/apezeshk/Codes/DeepMed/models/cnn_36368_20170614145024.npz'
pathSavedSamples = '/home/apezeshk/Codes/DeepMed/models/cnn_36368_20170614145024_samples.npz'


#test_filename = '/home/apezeshk/Codes/DeepMed/test_500_-1_LIDC.hdf5'
test_filename = '/home/apezeshk/Codes/DeepMed/test_500_100_36368_filt.hdf5'

masterFolderLidc = '/raida/apezeshk/lung_dicom_dir'
#masterFolderLungInterior = '/raida/apezeshk/LIDC_v2/015_lung_only'
masterFolderLungInterior = '/raida/apezeshk/LIDC_v2/037_part_to_lung'
input_3D_npy = '/diskStation/LIDC/LIDC_NUMPY_3d'
score_map_path='/diskStation/LIDC/36368/score_map/' #master folder for score maps; add '/test' for test purporses if needed
#this value is set to 16 bc going thru FCN messes up edges of subvolumes, so need sufficient overlap (based on architecture)
#Value of 16 works for filter size (in z direction) 3>3>MaxPool>3(or 1)>4(FC layer w no padding);
z_depth = 16
opMode = 'full_fcn' #options are 'full_fcn' where both of FC layers become convolutional, or 'partial_fcn' where only next to lasy layer is made convolutional
minmaxFilePath = '/home/apezeshk/Codes/DeepMed/max_min.json' #file containing the min/max of each case; used for normalizing
runMode = 'test' #'train' s.t. FCN is applied to cases not in test (i.e. train cases), 'test' to run on test cases
cutPointFlag = 1 #if 0 uses hand-set coordinates, if 1 uses the regular partitioning procedure
# for resizeFCNFlag, if 1, resizes output of FCN to full size & saves that (this results in problems, because of multiple
# chopping of edges from the pad=0 in fully connected layer, compared to if you had passed the entire volume in single pass);
# If 0, will use another similar FCN to process the nodule mask in same way of chopping the volume, so that the nodules
# appear in exact correct spots as where they fall with the FCN output of CT case. This option saves the un-resized FCN output
# as well as the nodule mask that has been processed thru FCN (s.t. AUC, etc. can be calculated)
resizeFCNFlag = 0 
tagNoduleMaskFlag = 1 #if 1, tags the nodules in nodule mask with unique nodule tags s.t. they can be identified later; only works with resizeFCNFlag=0!
########################
#creating the output directory
########################
outDirectory = os.path.join(score_map_path, str(pathSavedNetwork.split('/')[-1][:-4]), runMode)
if not os.path.exists(outDirectory): #make subdirectory with model name if needed
    os.makedirs(outDirectory)

random_state = 1234
np.random.seed(random_state)
lasagne.layers.noise._srng = lasagne.layers.noise.RandomStreams(random_state)

########################
######Input Params######
savedSamplesFile = np.load(pathSavedSamples)
inputParamsConfigLocal = savedSamplesFile['inputParamsConfig'].item() #saved dict becomes an object, so need to turn back into dict!
test_samples = savedSamplesFile['inputs'] #the test samples/labels were saved in a particular order, so they can be loaded in same order
test_labels = savedSamplesFile['targets']
test_pred_orig = savedSamplesFile['test_pred']
#inputParamsConfigLocal = {}
#inputParamsConfigLocal['input_shape'] = '36, 36, 8'
#inputParamsConfigLocal['learning_rate'] = '0.05'
#inputParamsConfigLocal['momentum'] = '0.9'
#inputParamsConfigLocal['num_epochs'] = '1'
#inputParamsConfigLocal['batch_size'] = '1'
#inputParamsConfigLocal['data_path'] = '/diskStation/LIDC/36368/'
#inputParamsConfigLocal['train_set_size'] = '60000'
#inputParamsConfigLocal['test_set_size'] = '500'
#inputParamsConfigLocal['positive_set_ratio'] = '0.3'
#inputParamsConfigLocal['dropout'] = '0.1'
#inputParamsConfigLocal['nonlinearityToUse'] = 'relu'
#inputParamsConfigLocal['numberOfLayers'] = 3
#inputParamsConfigLocal['augmentationFlag'] = 1
#inputParamsConfigLocal['weightInitToUse'] ='He' #weight initialization; either 'normal' or 'He' (for HeNormal)
#inputParamsConfigLocal['lrDecayFlag'] = 1 #1 for using learning rate decay, 0 for constant learning rate throughout training
#inputParamsConfigLocal['biasInitVal'] = 0.0 #1 for using learning rate decay, 0 for constant learning rate throughout training
######Input Params######
########################

#n_layers = 3
weight_init = lasagne.init.Normal() #we now to He, but since everything is being loaded from pre-trained model is ok!!
epoch_det = {}  # storyin the epoc results including val acc,loss, and training loss
all_val_accuracy = []  # Validation_accuracy list is used for the plot
all_val_loss = []  # Validation_loss list is used for the plot
all_val_AUC = []  # list of AUC for validation set across each epoch
training_loss = []  # Training_loss list is used for the plot
csv_config=[] #if localFlag==0, code will read network parameters from a csv and store that in csv_config
experiment_id = str(time.strftime("%Y%m%d%H%M%S"))  # using the current date and time for the experiment name

patient_id = [] #will contain list of patient id's (e.g. p0034) in the test set
filteredCases = []
with h5py.File(test_filename, 'r') as hf:
    print('List of arrays in this file: \n', hf.keys())
    tmp_test_paths = hf.get('Test_set')  # Reading list of patients and test file paths
    pos_test_paths = np.array(tmp_test_paths)
    patient_in_test = hf.get('Patient_labels') #list with elements like 'p0034'
    filteredCaseList = hf.get('filteredCaseList') #list with elements like 'p0023_2000123_s324'; all cases that fit filter criteria
    for items in patient_in_test:
        patient_id.append(items)
    for items in filteredCaseList:
        filteredCases.append(items)
        
def Path_create(file_name): #takes something like p0023_2000123_s324 and returns p0023/2000123/s324
    spl_dir=file_name[:].replace('_','/')
    return spl_dir

   
#score_map_path_files=os.listdir(outDirectory)
#for item_ind in range(len(score_map_path_files)):
#    score_map_path_files[item_ind]=score_map_path_files[item_ind].split('_')[0]

def Build_3dfcn(init_norm=weight_init, inputParamsNetwork=dict(shape=[10,10,10],dropout=0.1, nonLinearity=lasagne.nonlinearities.rectify,
                nonLinearityFC=lasagne.nonlinearities.rectify), input_var=None):
    dropout = inputParamsNetwork['dropout']
    filterSizes = inputParamsNetwork['filterSizeTable']
    numberOfFCUnits = inputParamsNetwork['numberOfFCUnits']
    numberOfFCLayers = inputParamsNetwork['numberOfFCLayers']
    numberOfConvFilts = inputParamsNetwork['numberOfConvFilts']
    
    #    print(dropout)
    network = lasagne.layers.InputLayer(shape=(None,1,int(inputParamsNetwork['shape'].split(',')[0]),int(inputParamsNetwork['shape'].split(',')[1]),int(inputParamsNetwork['shape'].split(',')[2])),
                                        input_var=input_var)
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

        network = lasagne.layers.dnn.MaxPool3DDNNLayer(network, pool_size=(1, 1, 1))

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
                                                    

    #This is the layer that substitutes the fully connected layer according to current patch size & architecture
#    network = lasagne.layers.dnn.Conv3DDNNLayer(network, num_filters=numberOfFCUnits, pad=0, filter_size=(9, 9, 4),
#                                    stride=(1, 1, 1),
#                                    nonlinearity=lasagne.nonlinearities.sigmoid,
#                                    W=init_norm,
#                                    b=inputParamsNetwork['biasInit'],
#                                    )
    network = lasagne.layers.dnn.Conv3DDNNLayer(lasagne.layers.dropout(network, p=dropout), num_filters=numberOfFCUnits, pad=0, filter_size=(18, 18, 4),
                                    stride=(1, 1, 1),
                                    nonlinearity=inputParamsNetwork['nonLinearityFC'],
                                    W=init_norm,
                                    b=inputParamsNetwork['biasInit'],
                                    )
                                    
    if numberOfFCLayers==3:
        network = lasagne.layers.dnn.Conv3DDNNLayer(lasagne.layers.dropout(network, p=dropout), num_filters=numberOfFCUnits, 
                                pad=0, filter_size=(1, 1, 1), stride=(1, 1, 1),
                                nonlinearity=inputParamsNetwork['nonLinearityFC'],
                                W=init_norm,
                                b=inputParamsNetwork['biasInit'],
                                )

    if 0:
    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    # Use this option to check and make sure the output from FC at previous layer of original CNN and fully convolutional 
    # counterpart in FCN give the same result
        network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=dropout),
            num_units=2,
            nonlinearity=lasagne.nonlinearities.softmax)
    else:
    # Use this option to complete the conversion to FCN (i.e. no more dense layers)
    # will do the softmax later, it is giving error if we use it here for some reason
        network = lasagne.layers.dnn.Conv3DDNNLayer(lasagne.layers.dropout(network, p=dropout), num_filters=2, pad=0, filter_size=(1, 1, 1),
                                                stride=(1, 1, 1),
                                                nonlinearity=None,
                                                W=init_norm,
                                                )
        
    return network
    


    
inputParamsConfigAll = inputParamsConfigLocal
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
nonlinearityToUseFC = inputParamsConfigAll['nonlinearityToUseFC']
#augmentationFlag = inputParamsConfigAll['augmentationFlag']
numberOfLayers = inputParamsConfigAll['numberOfLayers']
numberOfFCUnits = inputParamsConfigAll['numberOfFCUnits']
numberOfFCLayers = inputParamsConfigAll['numberOfFCLayers']
numberOfConvFilts = inputParamsConfigAll['numberOfConvFilts']
filterSizeTable = np.array(((5,5,3), (5,5,3), (3,3,1)))  #inputParamsConfigAll['filterSizeTable']
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
        
if nonlinearityToUseFC == 'relu':
    nonLinearityFC = lasagne.nonlinearities.rectify        
elif nonlinearityToUseFC == 'tanh':
    nonLinearityFC = lasagne.nonlinearities.tanh        
elif nonlinearityToUseFC == 'sigmoid':
    nonLinearityFC = lasagne.nonlinearities.sigmoid        
else:
    raise Exception(
          'nonlinearityToUseFC: Unsupported nonlinearity type has been selected for the network, retry with a supported one!')
          
dtensor5 = T.TensorType('float32', (False,) * 5)
input_var = dtensor5('inputs')
target_var = T.ivector('targets')


biasInit = lasagne.init.Constant(biasInitVal) #for relu use biasInit=1 s.t. inputs to relu are positive in beginning
inputParamsNetwork = dict(n_layer=numberOfLayers, shape=input_shape,dropout=float(dropout), nonLinearity=nonLinearity,
                          biasInit = biasInit, filterSizeTable = filterSizeTable, numberOfFCLayers = numberOfFCLayers,
                          numberOfFCUnits = numberOfFCUnits, numberOfConvFilts=numberOfConvFilts,
                          nonLinearityFC = nonLinearityFC)

##############################
##############################
# And load them again later on like this:
with np.load(pathSavedNetwork) as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]

#Reshape the FC layer of saved CNN into FCN form    
#param_values[4] = param_values[4].reshape((64,32,7,7,5))
if numberOfLayers == 2:    
    W4_new = np.zeros((numberOfFCUnits,numberOfConvFilts,9,9,4)).astype('float32')
    for i in range(0,param_values[4].shape[1]): #weights for each node in FC layer form the columns
        current_node_weights = param_values[4][:,i]
        W4_new[i, :, :, :, :] = np.reshape(current_node_weights, (numberOfConvFilts,7,7,4), order = 'C')
        
    param_values[4] = W4_new
    fcnLayerFilterSize = W4_new.shape[2:] #e.g. from (64,32,9,9,4) to (9,9,4)
    
    if opMode == 'full_fcn':
        if numberOfFCLayers == 2:
            W6_new = np.zeros((2,numberOfFCUnits,1,1,1)).astype('float32')
            for i in range(0,param_values[6].shape[1]): #weights for each node in FC layer form the columns
                current_node_weights = param_values[6][:,i]
                W6_new[i, :, :, :, :] = np.reshape(current_node_weights, (numberOfFCUnits,1,1,1), order = 'C')
                
            param_values[6] = W6_new
            
        elif numberOfFCLayers == 3:
            W6_new = np.zeros((numberOfFCUnits,numberOfFCUnits,1,1,1)).astype('float32')
            for i in range(0,param_values[6].shape[1]): #weights for each node in FC layer form the columns
                current_node_weights = param_values[6][:,i]
                W6_new[i, :, :, :, :] = np.reshape(current_node_weights, (numberOfFCUnits,1,1,1), order = 'C')
                
            param_values[6] = W6_new
    
            W8_new = np.zeros((2,numberOfFCUnits,1,1,1)).astype('float32')
            for i in range(0,param_values[8].shape[1]): #weights for each node in FC layer form the columns
                current_node_weights = param_values[8][:,i]
                W8_new[i, :, :, :, :] = np.reshape(current_node_weights, (numberOfFCUnits,1,1,1), order = 'C')
                
            param_values[8] = W8_new
        
        
elif numberOfLayers == 3:
    W6_new = np.zeros((numberOfFCUnits,numberOfConvFilts,18,18,4)).astype('float32')
    for i in range(0,param_values[6].shape[1]): #weights for each node in FC layer form the columns
        current_node_weights = param_values[6][:,i]
        W6_new[i, :, :, :, :] = np.reshape(current_node_weights, (numberOfConvFilts,18,18,4), order = 'C')
        
    param_values[6] = W6_new
    fcnLayerFilterSize = W6_new.shape[2:] #e.g. from (64,32,9,9,4) to (9,9,4)
    
    if opMode == 'full_fcn':
        if numberOfFCLayers==2:
            W8_new = np.zeros((2,numberOfFCUnits,1,1,1)).astype('float32')
            for i in range(0,param_values[8].shape[1]): #weights for each node in FC layer form the columns
                current_node_weights = param_values[8][:,i]
                W8_new[i, :, :, :, :] = np.reshape(current_node_weights, (numberOfFCUnits,1,1,1), order = 'C')
                
            param_values[8] = W8_new
            
        elif numberOfFCLayers == 3:
            W8_new = np.zeros((numberOfFCUnits,numberOfFCUnits,1,1,1)).astype('float32')
            for i in range(0,param_values[8].shape[1]): #weights for each node in FC layer form the columns
                current_node_weights = param_values[8][:,i]
                W8_new[i, :, :, :, :] = np.reshape(current_node_weights, (numberOfFCUnits,1,1,1), order = 'C')
                
            param_values[8] = W8_new
    
            W10_new = np.zeros((2,numberOfFCUnits,1,1,1)).astype('float32')
            for i in range(0,param_values[10].shape[1]): #weights for each node in FC layer form the columns
                current_node_weights = param_values[10][:,i]
                W10_new[i, :, :, :, :] = np.reshape(current_node_weights, (numberOfFCUnits,1,1,1), order = 'C')
                
            param_values[10] = W10_new
        
        

#savedSamplesFile = np.load(pathSavedSamples)
#test_samples = savedSamplesFile['inputs'] #the test samples/labels were saved in a particular order, so they can be loaded in same order
#test_labels = savedSamplesFile['targets']
#test_pred_orig = savedSamplesFile['test_pred']

network_fcn = Build_3dfcn(weight_init, inputParamsNetwork, input_var)
lasagne.layers.set_all_param_values(network_fcn, param_values)

##############################
##############################
if opMode == 'partial_fcn':
    prediction = lasagne.layers.get_output(network_fcn)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # loss=np.mean(loss)
    # We could add some weight decay as well here, see lasagne.regularization.
    
    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network_fcn, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=float(learning_rate), momentum=float(momentum))
    
    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network_fcn,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network_fcn, deterministic=True)
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
    
    err_test_fcn, acc_test_fcn, test_pred_fcn = val_fn(test_samples, test_labels) #test_pred_fcn should be identical to test_pred_orig
    
elif opMode == 'full_fcn':
    chopVolumeFlag = 1
    test_prediction = lasagne.layers.get_output(network_fcn, deterministic=True)
    val_fn = theano.function([input_var], [test_prediction])  # ,mode='DebugMode')
    vol_scores_allVol = np.empty((0, 2))
    vol_labels_allVol = []
    all_cases = sorted(os.listdir(input_3D_npy))
    #all_cases = ['p0006_20000101_s3000556.npy']
    #all_cases= ['p0012_20000101_s3000561.npy']

    caseCounter = 1.0
    for currentCaseName in all_cases: #e.g. p0949_20000101_s3180.npy
        print 'Percent of cases running: ' + str(caseCounter/len(all_cases))
        caseCounter = caseCounter + 1
        
        if runMode=='train':
            if currentCaseName.split('_')[0] in patient_id:# if runMode='train', skip test cases
                continue
            if currentCaseName[:-4] not in filteredCases: #the case should also be one that matches filter criteria
                continue
        elif runMode=='test':
            if currentCaseName.split('_')[0] not in patient_id:# if runMode='test', skip train cases
                continue
        else:
            raise ValueError('Invalid runMode, runMode should be either train or test!')                 
            
        #To skip cases that have already been processed, uncomment line below
        if os.path.exists(os.path.join(outDirectory, str(currentCaseName[:-4])+'.mat')):
            continue
        
        # full_volume_path = '/diskStation/LIDC/LIDC_NUMPY_3d/p0012_20000101_s3000561.npy'
        sub_vol_one = []
        try:
            full_volume_path=os.path.join(input_3D_npy, currentCaseName)
            full_volume = np.load(full_volume_path)
            # full_volume = full_volume.astype('float32')
            full_volume = full_volume.astype('int16')
            full_volume = full_volume.astype('float32')
            if sardarImplementationFlag == 0:
                #full_volume = (full_volume - full_volume.min()) / (full_volume.max() - full_volume.min())
                with open(minmaxFilePath) as json_data:
                    minmaxAll = json.load(json_data)
                currentPath = full_volume_path[:-4] #-4 so that the '.npy' is removed
                currentCase = os.path.basename(currentPath)
                currentCase = currentCase.split('_')[0:3]
                currentCase = "_".join(currentCase)            
                currentMinMax = minmaxAll[currentCase].split('/')
                currentMax = np.float32(currentMinMax[0])
                currentMin = np.float32(currentMinMax[1])
                full_volume = (full_volume - currentMin)/(currentMax - currentMin)
            else:
                full_volume = (full_volume - full_volume.max()) / (full_volume.max() - full_volume.min())
                
            full_volume = full_volume.reshape((1, 1, 512, 512, full_volume.shape[2]))
            if cutPointFlag == 1:
                xCutPoints = [0, 512]
                yCutPoints = [0, 512]
                tmpFlag = 0
                zCutPoints = [0]
                zStep = 28
                while tmpFlag != 7321:  # to make the loop end, set tmpFlag=7321; otherwise hold prev slice number in it
                    currentZCut = tmpFlag + zStep
                    if currentZCut > full_volume.shape[4]:
                        currentZCut = full_volume.shape[4]
                        zCutPoints.append(currentZCut)
                        tmpFlag = 7321
                    else:
                        tmpFlag = currentZCut - z_depth  # this is amount of overlap between consecutive chops in z direction
                        zCutPoints.append(currentZCut)
                        zCutPoints.append(tmpFlag)
            z_size=[]
            x_size=[]
            y_size=[]
            first_cube_flag=0
            vol_scores_currentVol = np.empty((0, 2))
            score_mat=np.zeros(())
            vol_labels_currentVol = []
            #This part is for when last two subvolumes should be changed due to small number of slices in last subvolume
            # we take from one subvolume by 20 and add it to the other
            if (zCutPoints[-1]-zCutPoints[-2])<=16:
                zCutPoints[-3]=zCutPoints[-3]-20
                zCutPoints[-2] = zCutPoints[-2] - 20

            mat_dir = os.path.join(masterFolderLidc, Path_create(os.path.basename(full_volume_path))[:-4])
            mat_name = 'uniqueStats_' + os.path.basename(full_volume_path)[:-4] + '.mat'
            uniqueStatsData = sio.loadmat(os.path.join(mat_dir, mat_name))
            if tagNoduleMaskFlag == 1: #NOTE: only defining noduleMask to get its size; if tagging nodules, that will be done later!
                noduleMask = uniqueStatsData['allMaxRadiologistMsk']                
            elif tagNoduleMaskFlag == 0:
                noduleMask = uniqueStatsData['allMaxRadiologistMsk']
                
            for i in range(0, len(xCutPoints) / 2):
                for j in range(0, len(yCutPoints) / 2):
                    for k in range(0, len(zCutPoints) / 2):
                        xStart = xCutPoints[2 * i]
                        xEnd = xCutPoints[2 * i + 1]
                        yStart = yCutPoints[2 * j]
                        yEnd = yCutPoints[2 * j + 1]
                        zStart = zCutPoints[2 * k]
                        zEnd = zCutPoints[2 * k + 1]
                        print(xStart, xEnd - 1, yStart, yEnd - 1, zStart, zEnd - 1)
                        asd = full_volume[0, 0, xStart:xEnd, yStart:yEnd, zStart:zEnd]
                        asd = asd.reshape((1, 1, asd.shape[0], asd.shape[1], asd.shape[2])) #put subvolume in 5D form for input to FCN
                        test_pred_full_volume = val_fn(asd)
                        test_pred_full_volume = test_pred_full_volume[0]
                        test_pred_full_volume_softmax0 = np.exp(test_pred_full_volume[0, 0, :, :, :]) / (
                        np.exp(test_pred_full_volume[0, 0, :, :, :]) + np.exp(test_pred_full_volume[0, 1, :, :, :]))
                        test_pred_full_volume_softmax1 = np.exp(test_pred_full_volume[0, 1, :, :, :]) / (
                        np.exp(test_pred_full_volume[0, 0, :, :, :]) + np.exp(test_pred_full_volume[0, 1, :, :, :]))

                        tmp_sub_vol=test_pred_full_volume_softmax1

                        if xStart==xCutPoints[0] and yStart==yCutPoints[0]:
                            #NOTE: when u split the volume N times, the difference in size due to 0 padding in last layer
                            # is repeated N times also! So whereas if u passed the entire volume with first fully connected
                            # layer (converted to fully convolutional) of size (9,9,4) you would get -4+1=-3 as many slices,
                            # if you split the volume in 2 and pass each subvolume, you get another round of -3 slices in 
                            # the end!!!
                            try:#This part adds the sub volumes back to back and overwrites the bad slice with the correct one
                                sub_vol_one=np.concatenate((sub_vol_one[:,:,:-2],tmp_sub_vol[:,:,3:]),axis=2) #
                            except :
                                sub_vol_one=tmp_sub_vol

            if resizeFCNFlag == 1:            
                # Interpolate the small volume to the original size and save it as a mat file to be used in matlab
                # for non_max supression; Note that if anything other than order==0 is used when doing the interpolation,
                # then values of adjacent voxels affect value of interpolated voxel, so e.g. a high probability voxel
                # in low res can get smudged together with adjacent low probabilities, and max probability will go down!
                dsfactor = [w / float(g) for w, g in zip(noduleMask.shape, sub_vol_one.shape)]
                original_sized_volume = nd.interpolation.zoom(sub_vol_one, zoom=dsfactor,order=0) #changed from 1 to 0, to get nearest neighbor (replication) instead of interpolation
                sio.savemat(outDirectory+'/'+str(full_volume_path.split('/')[-1][:-4])+'.mat', {'scoreMap': original_sized_volume})
            elif resizeFCNFlag == 0:                
                ##########################
                ##Pass the nodule mask thru a similar FCN network to get it to matching size as sub_vol_one;
                ##This option was added bc chopping the volume results in clipping of edges each time due to 
                ##fully connected layer using pad=0; This resulted in problems when expanding output to full size,
                ##and shifting of the detected nodule location relative to where it was expected to be. When
                ##nodule mask is passed thru similar FCN however, the two outputs can be matched against each other easily.
                inputParamsNoduleMaskFCN = {}
                inputParamsNoduleMaskFCN['currentCaseName'] = currentCaseName #sth like p0012_20000101_s3000561.npy
                inputParamsNoduleMaskFCN['input_3D_npy'] = input_3D_npy
                inputParamsNoduleMaskFCN['masterFolderLidc'] = masterFolderLidc
                inputParamsNoduleMaskFCN['cutPointFlag'] = cutPointFlag
                inputParamsNoduleMaskFCN['z_depth'] = z_depth
                inputParamsNoduleMaskFCN['fcnLayerFilterSize'] = fcnLayerFilterSize
                inputParamsNoduleMaskFCN['tagNoduleMaskFlag'] = tagNoduleMaskFlag
                inputParamsNoduleMaskFCN['remapFlag'] = 1 #only used if tagNoduleMaskFlag==1
                
                noduleMaskFCN, noduleMaskFCN_lessThan3mm = test_MaskFCN_lessMaxPool.CreateNoduleMaskFCN(inputParamsConfigAll, inputParamsNoduleMaskFCN)
                if (sub_vol_one.shape[0] != noduleMaskFCN.shape[0]) or (sub_vol_one.shape[1] != noduleMaskFCN.shape[1]) or (sub_vol_one.shape[2] != noduleMaskFCN.shape[2]):
                    raise ValueError('Something in architecture of primary and nodule FCNs is different, the outputs have different size!')
                    
                inputParamsLungInteriorMaskFCN = {}
                inputParamsLungInteriorMaskFCN['currentCaseName'] = currentCaseName #sth like p0012_20000101_s3000561.npy                
                inputParamsLungInteriorMaskFCN['masterFolderLidc'] = masterFolderLidc                
                inputParamsLungInteriorMaskFCN['masterFolderLungInterior'] = masterFolderLungInterior
                inputParamsLungInteriorMaskFCN['cutPointFlag'] = cutPointFlag
                inputParamsLungInteriorMaskFCN['z_depth'] = z_depth
                inputParamsLungInteriorMaskFCN['fcnLayerFilterSize'] = fcnLayerFilterSize
                
                lungInteriorMaskFCN = test_MaskFCN_lessMaxPool.CreateLungInteriorMaskFCN(inputParamsConfigAll, inputParamsLungInteriorMaskFCN)
                if (sub_vol_one.shape[0] != lungInteriorMaskFCN.shape[0]) or (sub_vol_one.shape[1] != lungInteriorMaskFCN.shape[1]) or (sub_vol_one.shape[2] != lungInteriorMaskFCN.shape[2]):
                    raise ValueError('Something in architecture of primary and lung interior FCNs is different, the outputs have different size!')
                    
                sio.savemat(outDirectory+'/'+str(full_volume_path.split('/')[-1][:-4])+'.mat', 
                            {'scoreMap': sub_vol_one, 'noduleMask':noduleMaskFCN, 
                            'noduleMask_lessThan3mm':noduleMaskFCN_lessThan3mm,
                            'lungInteriorMask':lungInteriorMaskFCN})
                ##########################

            
            

        except ValueError as errOccured:
            print ("case %s got error, z cut points were: " %str(currentCaseName),zCutPoints)
            print errOccured.args
        except KeyboardInterrupt:
            print('Manual keyboard interrupt, aborting!')
            break
            
        except:            
            print ("case %s got error, z cut points were: " %str(currentCaseName),zCutPoints)

#        else:
#            print("case %s is in the test set and cannot be used" %str(currentCaseName.split('_')[0]))
    
    