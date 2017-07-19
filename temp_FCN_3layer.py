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
import json
########################
########################
    # Optionally, you could now dump the network weights to a file like this:
sardarImplementationFlag = 0 #this one does the wrong normalization according to Sardar
#pathSavedNetwork = '/home/apezeshk/Codes/DeepMed/models/cnn_36368_20160914121137.npz'#positive_set_ratio=0.5;
#pathSavedSamples = '/home/apezeshk/Codes/DeepMed/models/cnn_36368_20160914121137_samples.npz'
#pathSavedNetwork = '/home/apezeshk/Codes/DeepMed/models/cnn_36368_20160921102657.npz' #positive_set_ratio=0.5;1 fp per '+' sample added in this
#pathSavedSamples = '/home/apezeshk/Codes/DeepMed/models/cnn_36368_20160921102657_samples.npz'
pathSavedNetwork = '/home/apezeshk/Codes/DeepMed/models/cnn_36368_20160921114711.npz'#positive_set_ratio=0.75;3 fp per '+' sample added in this
pathSavedSamples = '/home/apezeshk/Codes/DeepMed/models/cnn_36368_20160921114711_samples.npz'



#pathSavedNetwork = '/diskStation/temp/cnn_36368_20160817161531.npz'
#pathSavedSamples = '/diskStation/temp/cnn_36368_20160817161531_samples.npz'
masterFolderLidc = '/raida/apezeshk/lung_dicom_dir'
minmaxFilePath = '/home/apezeshk/Codes/DeepMed/max_min.json' #file containing the min/max of each case; used for normalizing
opMode = 'full_fcn' #options are 'full_fcn' where both of FC layers become convolutional, or 'partial_fcn' where only next to lasy layer is made convolutional
cutPointFlag = 0 #if 0 uses hand-set coordinates, if 1 uses the regular partitioning procedure
#np.savez(pathSavedNetwork, *lasagne.layers.get_all_param_values(network))
#np.savez(pathSavedSamples, inputs=inputs, targets=targets, err=err, acc=acc, test_pred=test_pred)
########################
########################


random_state = 1234
np.random.seed(random_state)
lasagne.layers.noise._srng = lasagne.layers.noise.RandomStreams(random_state)

########################
######Input Params######
inputParamsConfigLocal = {}
inputParamsConfigLocal['input_shape'] = '36, 36, 8'
inputParamsConfigLocal['learning_rate'] = '0.05'
inputParamsConfigLocal['momentum'] = '0.9'
inputParamsConfigLocal['num_epochs'] = '1'
inputParamsConfigLocal['batch_size'] = '1'
inputParamsConfigLocal['data_path'] = '/diskStation/LIDC/36368/'
inputParamsConfigLocal['train_set_size'] = '60000'
inputParamsConfigLocal['test_set_size'] = '500'
inputParamsConfigLocal['positive_set_ratio'] = '0.3'
inputParamsConfigLocal['dropout'] = '0.1'
inputParamsConfigLocal['nonlinearityToUse'] = 'relu'
inputParamsConfigLocal['numberOfLayers'] = 3
inputParamsConfigLocal['augmentationFlag'] = 1
inputParamsConfigLocal['weightInitToUse'] ='He' #weight initialization; either 'normal' or 'He' (for HeNormal)
inputParamsConfigLocal['lrDecayFlag'] = 1 #1 for using learning rate decay, 0 for constant learning rate throughout training
inputParamsConfigLocal['biasInitVal'] = 0.0 #1 for using learning rate decay, 0 for constant learning rate throughout training
######Input Params######
########################

weight_init = lasagne.init.Normal()
epoch_det = {}  # storyin the epoc results including val acc,loss, and training loss
all_val_accuracy = []  # Validation_accuracy list is used for the plot
all_val_loss = []  # Validation_loss list is used for the plot
all_val_AUC = []  # list of AUC for validation set across each epoch
training_loss = []  # Training_loss list is used for the plot
csv_config=[] #if localFlag==0, code will read network parameters from a csv and store that in csv_config
experiment_id = str(time.strftime("%Y%m%d%H%M%S"))  # using the current date and time for the experiment name


def Path_create(file_name):
    spl_dir=file_name[:].replace('_','/')
    return spl_dir


def Build_3dfcn(init_norm=weight_init, inputParamsNetwork=dict(shape=[10,10,10],dropout=0.1, nonLinearity=lasagne.nonlinearities.rectify),
                input_var=None):
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
                                                    

    #This is the layer that substitutes the fully connected layer according to current patch size & architecture
    network = lasagne.layers.dnn.Conv3DDNNLayer(lasagne.layers.dropout(network, p=dropout), num_filters=64, pad=0, filter_size=(9, 9, 4),
                                    stride=(1, 1, 1),
                                    nonlinearity=lasagne.nonlinearities.sigmoid,
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
augmentationFlag = inputParamsConfigAll['augmentationFlag']
numberOfLayers = inputParamsConfigAll['numberOfLayers']
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
dtensor5 = T.TensorType('float32', (False,) * 5)
input_var = dtensor5('inputs')
target_var = T.ivector('targets')


biasInit = lasagne.init.Constant(biasInitVal) #for relu use biasInit=1 s.t. inputs to relu are positive in beginning
inputParamsNetwork = dict(n_layer=numberOfLayers, shape=input_shape,dropout=float(dropout), nonLinearity=nonLinearity,
                          biasInit = biasInit)

##############################
##############################
# And load them again later on like this:
with np.load(pathSavedNetwork) as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]

#Reshape the FC layer of saved CNN into FCN form    
#param_values[4] = param_values[4].reshape((64,32,7,7,5))
if numberOfLayers == 2:    
    W4_new = np.zeros((64,32,7,7,4)).astype('float32')
    for i in range(0,param_values[4].shape[1]): #weights for each node in FC layer form the columns
        current_node_weights = param_values[4][:,i]
        W4_new[i, :, :, :, :] = np.reshape(current_node_weights, (32,7,7,4), order = 'C')
        
    param_values[4] = W4_new
    
    if opMode == 'full_fcn':
        W6_new = np.zeros((2,64,1,1,1)).astype('float32')
        for i in range(0,param_values[6].shape[1]): #weights for each node in FC layer form the columns
            current_node_weights = param_values[6][:,i]
            W6_new[i, :, :, :, :] = np.reshape(current_node_weights, (64,1,1,1), order = 'C')
            
        param_values[6] = W6_new
        
        
elif numberOfLayers == 3:
    W6_new = np.zeros((64,32,9,9,4)).astype('float32')
    for i in range(0,param_values[6].shape[1]): #weights for each node in FC layer form the columns
        current_node_weights = param_values[6][:,i]
        W6_new[i, :, :, :, :] = np.reshape(current_node_weights, (32,9,9,4), order = 'C')
        
    param_values[6] = W6_new
    
    if opMode == 'full_fcn':
        W8_new = np.zeros((2,64,1,1,1)).astype('float32')
        for i in range(0,param_values[8].shape[1]): #weights for each node in FC layer form the columns
            current_node_weights = param_values[8][:,i]
            W8_new[i, :, :, :, :] = np.reshape(current_node_weights, (64,1,1,1), order = 'C')
            
        param_values[8] = W8_new
        

savedSamplesFile = np.load(pathSavedSamples)
test_samples = savedSamplesFile['inputs'] #the test samples/labels were saved in a particular order, so they can be loaded in same order
test_labels = savedSamplesFile['targets']
test_pred_orig = savedSamplesFile['test_pred']

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
    test_pred_fcn = val_fn(test_samples)
    test_pred_fcn = test_pred_fcn[0]
    test_pred_fcn = test_pred_fcn.squeeze()
    test_pred_fcn_softmax0 = np.exp(test_pred_fcn[:,0])/(np.exp(test_pred_fcn[:,0])+ np.exp(test_pred_fcn[:,1]))
    test_pred_fcn_softmax1 = np.exp(test_pred_fcn[:,1])/(np.exp(test_pred_fcn[:,0])+ np.exp(test_pred_fcn[:,1]))
    test_pred_fcn_softmax = np.column_stack((test_pred_fcn_softmax0,test_pred_fcn_softmax1)) #this should be identical to test_pred_orig
    
    vol_scores_allVol = np.empty((0,2))
    vol_labels_allVol = []
    
    #full_volume_path = '/diskStation/LIDC/LIDC_NUMPY_3d/p0148_20000101_s3.npy'
    full_volume_path = '/diskStation/LIDC/LIDC_NUMPY_3d/p0012_20000101_s3000561.npy'
    #full_volume_path = '/diskStation/LIDC/LIDC_NUMPY_3d/p0006_20000101_s3000556.npy'
    
    full_volume = np.load(full_volume_path)
    #full_volume = full_volume.astype('float32')
    full_volume = full_volume.astype('int16')
    full_volume = full_volume.astype('float32')
    if sardarImplementationFlag==0:        
        #full_volume = (full_volume - full_volume.min())/(full_volume.max() - full_volume.min())
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
        full_volume = (full_volume - full_volume.max())/(full_volume.max() - full_volume.min())
        
    full_volume = full_volume.reshape((1,1,512,512,full_volume.shape[2]))    

    if cutPointFlag==1:    
        xCutPoints = [0, 200, 170, 370, 340, 512]
        yCutPoints = [0, 200, 170, 370, 340, 512]
        tmpFlag = 0
        zCutPoints = [0]
        zStep = 80
        while tmpFlag!=7321: #to make the loop end, set tmpFlag=7321; otherwise hold prev slice number in it
            currentZCut = tmpFlag + zStep
            if currentZCut > full_volume.shape[4]:                
                currentZCut = full_volume.shape[4]
                zCutPoints.append(currentZCut)
                tmpFlag = 7321
            else:                
                tmpFlag = currentZCut - 12 #this is amount of overlap between consecutive chops in z direction    
                zCutPoints.append(currentZCut)
                zCutPoints.append(tmpFlag)
            
    else:
        xCutPoints = [0, 512]
        yCutPoints = [0, 512]
        zCutPoints = [58, 66]#[60, 68] #put in the actual slice numbers, 1 will be subtracted below
        zCutPoints[:] = [x - 1 for x in zCutPoints]#actual slices are these slices + 1; so subtract from actual slice numbers
#        xCutPoints = [230, 258]
#        yCutPoints = [308, 336]
#        zCutPoints = [43, 51]
        
    vol_scores_currentVol = np.empty((0,2))
    vol_labels_currentVol = []
    
    for i in range(0,len(xCutPoints)/2):
        for j in range(0,len(yCutPoints)/2):
            for k in range(0,len(zCutPoints)/2):   
#    for i in [2]:
#        for j in [1]:
#            for k in [3]:              
                xStart = xCutPoints[2*i]
                xEnd = xCutPoints[2*i+1]
                yStart = yCutPoints[2*j]
                yEnd = yCutPoints[2*j+1]
                zStart = zCutPoints[2*k]
                zEnd = zCutPoints[2*k+1]
                print(xStart, xEnd-1, yStart, yEnd-1, zStart, zEnd-1)                
                
                asd = full_volume[0,0,xStart:xEnd,yStart:yEnd,zStart:zEnd]
                asd = asd.reshape((1, 1, asd.shape[0], asd.shape[1], asd.shape[2]))
        
                test_pred_full_volume = val_fn(asd)
                test_pred_full_volume = test_pred_full_volume[0]
                test_pred_full_volume_softmax0 = np.exp(test_pred_full_volume[0,0,:,:,:])/(np.exp(test_pred_full_volume[0,0,:,:,:])+ np.exp(test_pred_full_volume[0,1,:,:,:]))
                test_pred_full_volume_softmax1 = np.exp(test_pred_full_volume[0,1,:,:,:])/(np.exp(test_pred_full_volume[0,0,:,:,:])+ np.exp(test_pred_full_volume[0,1,:,:,:]))
                
                #uniqueStatsPath = '/raida/apezeshk/lung_dicom_dir/p0614/20000101/s30983/uniqueStats_p0614_20000101_s30983.mat'
                #uniqueStatsData = sio.loadmat(uniqueStatsPath)
                mat_dir = os.path.join(masterFolderLidc, Path_create(os.path.basename(full_volume_path))[:-4])
                mat_name = 'uniqueStats_' + os.path.basename(full_volume_path)[:-4] + '.mat'
                uniqueStatsData = sio.loadmat(os.path.join(mat_dir, mat_name))
                
                noduleMask = uniqueStatsData['allMaxRadiologistMsk']            
                noduleMaskCrop = noduleMask[xStart:xEnd,yStart:yEnd,zStart:zEnd]       
                #noduleMaskCrop = noduleMask[xStart+1:xEnd+1,yStart+1:yEnd+1,zStart+1:zEnd+1] #testing whether mask should be shifted 
                         
            #    noduleMaskResize = np.zeros((test_pred_full_volume_softmax0.shape[0], test_pred_full_volume_softmax0.shape[1], test_pred_full_volume_softmax0.shape[2]))
            #    for i in range(0, test_pred_full_volume_softmax0.shape[2]):
            #        noduleMaskResize[:,:,i] = scipy.misc.imresize(noduleMaskCrop[:,:,i], (test_pred_full_volume_softmax0.shape[0], test_pred_full_volume_softmax0.shape[1]))
                dsfactor = [w/float(g) for w,g in zip(test_pred_full_volume_softmax0.shape, noduleMaskCrop.shape)]
                noduleMaskResize = nd.interpolation.zoom(noduleMaskCrop.astype('float'), zoom=dsfactor) #reducing size destroys smaller nodules!    
                
                vol_scores0 = test_pred_full_volume_softmax0.flatten()
                vol_scores1 = test_pred_full_volume_softmax1.flatten()
                vol_labels = noduleMaskResize.flatten()
                vol_labels = vol_labels>0.5 #threshold the interpolated nodule mask to make it binary
                vol_labels = vol_labels.astype('int')
                
                vol_scores_01 = np.hstack((vol_scores0.reshape((len(vol_scores0),1)), vol_scores1.reshape((len(vol_scores1),1))))
                vol_scores_currentVol = np.concatenate((vol_scores_currentVol, vol_scores_01), axis=0)
                vol_labels_currentVol = np.concatenate((vol_labels_currentVol, vol_labels), axis=0)
                
    vol_scores_allVol = np.concatenate((vol_scores_allVol, vol_scores_currentVol), axis=0)
    vol_labels_allVol = np.concatenate((vol_labels_allVol, vol_labels_currentVol), axis=0)
                
    test_AUC, test_varAUC = SupportFuncs.Pred2AUC(vol_scores_allVol, vol_labels_allVol)
    print test_AUC
    
    
    
 
    
    
    