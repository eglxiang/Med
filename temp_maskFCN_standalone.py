# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 14:26:20 2016

@author: apezeshk
"""
import numpy as np
import lasagne
from lasagne.layers import dnn
import theano.tensor as T
import theano
import scipy.io as sio
import os

def Path_create(file_name): #takes something like p0023_2000123_s324 and returns p0023/2000123/s324
    spl_dir=file_name[:].replace('_','/')
    return spl_dir
    
    
def Build_3dfcn_mask(init_norm, inputParamsNetwork, input_var=None):
    dropout = inputParamsNetwork['dropout']
    #    print(dropout)
    network = lasagne.layers.InputLayer(shape=(None,1,int(inputParamsNetwork['shape'].split(',')[0]),int(inputParamsNetwork['shape'].split(',')[1]),int(inputParamsNetwork['shape'].split(',')[2])),
                                        input_var=input_var)
    if inputParamsNetwork['n_layer'] == 2 :

        network = lasagne.layers.dnn.Conv3DDNNLayer(network, num_filters=1, pad='same', filter_size=(1, 1, 1),
                                                    stride=(1, 1, 1),
                                                    nonlinearity=inputParamsNetwork['nonLinearity'],
                                                    W=init_norm,
                                                    b=inputParamsNetwork['biasInit'],
                                                    flip_filters=False
                                                    )

        network = lasagne.layers.dnn.MaxPool3DDNNLayer(network, pool_size=(2, 2, 1))

        network = lasagne.layers.dnn.Conv3DDNNLayer(network, num_filters=1, pad='same', filter_size=(1, 1, 1),
                                                stride=(1, 1, 1),
                                                nonlinearity=inputParamsNetwork['nonLinearity'],
                                                W=init_norm,
                                                b=inputParamsNetwork['biasInit'],
                                                )
        network = lasagne.layers.dnn.MaxPool3DDNNLayer(network, pool_size=(2, 2, 2))
                
    else:
        network = lasagne.layers.dnn.Conv3DDNNLayer(network, num_filters=1, pad='same', filter_size=(1, 1, 1),
                                                    stride=(1, 1, 1),
                                                    nonlinearity=inputParamsNetwork['nonLinearity'],
                                                    W=init_norm,
                                                    b=inputParamsNetwork['biasInit'],
                                                    flip_filters=False
                                                    )

        network = lasagne.layers.dnn.MaxPool3DDNNLayer(network, pool_size=(2, 2, 1))

        network = lasagne.layers.dnn.Conv3DDNNLayer(network, num_filters=1, pad='same', filter_size=(1, 1, 1),
                                                    stride=(1, 1, 1),
                                                    nonlinearity=inputParamsNetwork['nonLinearity'],
                                                    W=init_norm,
                                                    b=inputParamsNetwork['biasInit'],
                                                    )
        network = lasagne.layers.dnn.MaxPool3DDNNLayer(network, pool_size=(2, 2, 2))

        network = lasagne.layers.dnn.Conv3DDNNLayer(network, num_filters=1, pad='same', filter_size=(1, 1, 1),
                                                    stride=(1, 1, 1),
                                                    nonlinearity=inputParamsNetwork['nonLinearity'],
                                                    W=init_norm,
                                                    b=inputParamsNetwork['biasInit'],
                                                    )
                                                    

    #This is the layer that substitutes the fully connected layer according to current patch size & architecture
    network = lasagne.layers.dnn.Conv3DDNNLayer(network, num_filters=1, pad=0, filter_size=(9, 9, 4),
                                    stride=(1, 1, 1),
                                    nonlinearity=inputParamsNetwork['nonLinearity'],
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
        network = lasagne.layers.dnn.Conv3DDNNLayer(network, num_filters=1, pad=0, filter_size=(1, 1, 1),
                                                stride=(1, 1, 1),
                                                nonlinearity=inputParamsNetwork['nonLinearity'],
                                                W=init_norm,
                                                )
        
    return network
    
    
##############################
##############################
# And load them again later on like this:
pathSavedNetwork = '/home/apezeshk/Codes/DeepMed/models/cnn_36368_20160921114711.npz'
pathSavedSamples = '/home/apezeshk/Codes/DeepMed/models/cnn_36368_20160921114711_samples.npz'    
currentCaseName = 'p0012_20000101_s3000561.npy'

input_3D_npy = '/diskStation/LIDC/LIDC_NUMPY_3d'
masterFolderLidc = '/raida/apezeshk/lung_dicom_dir'
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

weight_init = lasagne.init.Normal() #we now use He, but since everything is being loaded this is ok!!
biasInit = lasagne.init.Constant(biasInitVal) #for relu use biasInit=1 s.t. inputs to relu are positive in beginning

nonLinearity = lasagne.nonlinearities.linear #use linear since u just want propagation of mask thru model
inputParamsNetwork = dict(n_layer=numberOfLayers, shape=input_shape,dropout=float(dropout), nonLinearity=nonLinearity,
                          biasInit = biasInit)
                          
dtensor5 = T.TensorType('float32', (False,) * 5)
input_var = dtensor5('inputs')
network_fcn_mask = Build_3dfcn_mask(weight_init, inputParamsNetwork, input_var)
param_values_fcn_default = lasagne.layers.get_all_param_values(network_fcn_mask) #just so to get the fully connected dimension
######Input Params######
########################


    
#with np.load(pathSavedNetwork) as f:
#    param_values_fullnetwork = [f['arr_%d' % i] for i in range(len(f.files))]


W0 = np.ones((1,1,1,1,1)).astype('float32')
b0 = np.zeros((1,)).astype('float32')

W2 = np.ones((1,1,1,1,1)).astype('float32')
b2 = np.zeros((1,)).astype('float32')

if numberOfLayers == 2:    
    W4 = np.zeros(np.shape(param_values_fcn_default[4])[2:]).astype('float32') #get the filter shape of first fully connected layer in original network
    current_filt_shape = W4.shape
        
    W4[int(np.floor(current_filt_shape[0]/2.0)), int(np.floor(current_filt_shape[1]/2.0)), int(np.floor(current_filt_shape[2]/2.0)-1)] = 1
    W4[int(np.floor(current_filt_shape[0]/2.0)), int(np.floor(current_filt_shape[1]/2.0)), int(np.floor(current_filt_shape[2]/2.0))] = 1
    W4 = W4 * 0.5 #this is so that the output range will not change (since instead of delta fn, 2 entries are equal to 1)
    W4 = np.reshape(W4, (1,1,current_filt_shape[0],current_filt_shape[1],current_filt_shape[2])) #make it 5-tuple
    b4 = np.zeros((1,)).astype('float32')
    
    W6 = np.ones((1,1,1,1,1)).astype('float32')            
    b6 = np.zeros((1,)).astype('float32')
    param_values_mask = []
    param_values_mask.extend([W0, b0, W2, b2, W4, b4, W6, b6])      
        
elif numberOfLayers == 3:
    W4 = np.ones((1,1,1,1,1)).astype('float32')
    b4 = np.zeros((1,)).astype('float32')
    
    W6 = np.zeros(np.shape(param_values_fcn_default[6])[2:]).astype('float32') #get the filter shape of first fully connected layer in original network
    current_filt_shape = W6.shape
    
    # When fully connected layer has even size in z direction, e.g. (9,9,4), we can't have a delta function as filter
    # So using a filter with same size, with two 0.5s in it in 2nd and 3rd indices as next best thing!   
    W6[int(np.floor(current_filt_shape[0]/2.0)), int(np.floor(current_filt_shape[1]/2.0)), int(np.floor(current_filt_shape[2]/2.0)-1)] = 1
    W6[int(np.floor(current_filt_shape[0]/2.0)), int(np.floor(current_filt_shape[1]/2.0)), int(np.floor(current_filt_shape[2]/2.0))] = 1
    W6 = W6 * 0.5 #this is so that the output range will not change (since instead of delta fn, 2 entries are equal to 1)
    W6 = np.reshape(W6, (1,1,current_filt_shape[0],current_filt_shape[1],current_filt_shape[2])) #make it 5-tuple
    b6 = np.zeros((1,)).astype('float32')      
    
    W8 = np.ones((1,1,1,1,1)).astype('float32')
    b8 = np.zeros((1,)).astype('float32')      
    param_values_mask = []
    param_values_mask.extend([W0, b0, W2, b2, W4, b4, W6, b6, W8, b8])
      

lasagne.layers.set_all_param_values(network_fcn_mask, param_values_mask) #load the model with the weights/biases

mask_prediction = lasagne.layers.get_output(network_fcn_mask, deterministic=True)
val_fn = theano.function([input_var], [mask_prediction])  # ,mode='DebugMode')


################################################################################
######Now load the nodule mask, and shove it into the network
################################################################################

full_volume_path=os.path.join(input_3D_npy, currentCaseName)

full_mask_path = os.path.join(masterFolderLidc, Path_create(os.path.basename(full_volume_path))[:-4])
mat_name = 'uniqueStats_' + os.path.basename(full_volume_path)[:-4] + '.mat'
uniqueStatsData = sio.loadmat(os.path.join(full_mask_path, mat_name))
full_mask = uniqueStatsData['allMaxRadiologistMsk']
full_mask = full_mask.astype('float32')
#MAKE SURE THE TYPE FOR NODULE MASK IS RIGHT IN BELOW; DO U HAVE TO CONVERT TO INT16 THEN FLOAT32?!!



chopVolumeFlag = 1
cutPointFlag = 1
z_depth = 8
   
sub_vol_one = []
      
full_mask = full_mask.reshape((1, 1, 512, 512, full_mask.shape[2]))
if cutPointFlag == 1:
    xCutPoints = [0, 512]
    yCutPoints = [0, 512]
    tmpFlag = 0
    zCutPoints = [0]
    zStep = 80
    while tmpFlag != 7321:  # to make the loop end, set tmpFlag=7321; otherwise hold prev slice number in it
        currentZCut = tmpFlag + zStep
        if currentZCut > full_mask.shape[4]:
            currentZCut = full_mask.shape[4]
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
#this part is for the cases that last two slices should be changed if you we wanna to FCN( they got small z
# we take from one cube by 20 and add it to another cube
if (zCutPoints[-1]-zCutPoints[-2])<=10:
    zCutPoints[-3]=zCutPoints[-3]-20
    zCutPoints[-2] = zCutPoints[-2] - 20

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
            asd = full_mask[0, 0, xStart:xEnd, yStart:yEnd, zStart:zEnd]
            asd = asd.reshape((1, 1, asd.shape[0], asd.shape[1], asd.shape[2])) #put subvolume in 5D form for input to FCN
            test_pred_full_mask = val_fn(asd)
            test_pred_full_mask = test_pred_full_mask[0]
#            test_pred_full_mask_softmax0 = np.exp(test_pred_full_mask[0, 0, :, :, :]) / (
#            np.exp(test_pred_full_mask[0, 0, :, :, :]) + np.exp(test_pred_full_mask[0, 1, :, :, :]))
#            test_pred_full_mask_softmax1 = np.exp(test_pred_full_mask[0, 1, :, :, :]) / (
#            np.exp(test_pred_full_mask[0, 0, :, :, :]) + np.exp(test_pred_full_mask[0, 1, :, :, :]))

            #tmp_sub_vol=test_pred_full_mask_softmax1
            tmp_sub_vol = test_pred_full_mask.squeeze() #go from e.g. (1,1,120,120,25) to (120,120,25)

            if xStart==xCutPoints[0] and yStart==yCutPoints[0]:
                #NOTE: when u split the volume N times, the difference in size due to 0 padding in last layer
                # is repeated N times also! So whereas if u passed the entire volume with first fully connected
                # layer (converted to fully convolutional) of size (9,9,4) you would get -4+1=-3 as many slices,
                # if you split the volume in 2 and pass each subvolume, you get another round of -3 slices in 
                # the end!!!
                try:#This part adds the sub volumes back to back and overwrites the bad slice with the correct one
                    sub_vol_one=np.concatenate((sub_vol_one[:,:,:-2],tmp_sub_vol[:,:,2:]),axis=2) #I set the concatination margin to 2 since we have a one max pool for Z and last 2 slices are not correctly convolved
                except:
                    sub_vol_one=tmp_sub_vol
                    
sub_vol_one_bin = (sub_vol_one>0.0).astype('int') #convert to binary; it originally has 0.5 values due to z direction elongation in fully connected layer filter
