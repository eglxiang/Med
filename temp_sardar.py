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
import scipy.misc
import os
import SupportFuncs
from scipy import ndimage as nd
import h5py
########################
########################
    # Optionally, you could now dump the network weights to a file like this:
input_3D_npy='/diskStation/LIDC/LIDC_NUMPY_3d'
pathSavedNetwork = '/home/apezeshk/Codes/DeepMed/temp/cnn_model.npz'
pathSavedSamples = '/home/apezeshk/Codes/DeepMed/temp/cnn_model_samples.npz'
mat_file='/raida/apezeshk/lung_dicom_dir/'
mat_pre='uniqueStats_'
opMode = 'full_fcn' #options are 'full_fcn' where both of FC layers become convolutional, or 'partial_fcn' where only next to lasy layer is made convolutional
# np.savez(pathSavedNetwork, *lasagne.layers.get_all_param_values(network))
# np.savez(pathSavedSamples, inputs=inputs, targets=targets, err=err, acc=acc, test_pred=test_pred)
########################
########################
clip_rate=[3,3,2]
output_sizes = [249,249,70]
M_layer=1
random_state = 1234
np.random.seed(random_state)
lasagne.layers.noise._srng = lasagne.layers.noise.RandomStreams(random_state)

#################### loading cases in test
#patient_id=[]
#with h5py.File(os.path.join('/home/shamidian/Summer2016/DeepMed/test_500_0.5_303010 .hdf5'), 'r') as hf:
#    print('List of arrays in this file: \n', hf.keys())
#    tmp_test_paths = hf.get('Test_set')  # Reading list of patients and test file paths
#    pos_test_paths = np.array(tmp_test_paths)
#    patient_in_test = hf.get('Patient_lables')
#    for items in patient_in_test:
#        patient_id.append(items)


########################
######Input Params######
inputParamsConfigLocal = {}
inputParamsConfigLocal['input_shape'] = '30, 30, 10'
inputParamsConfigLocal['learning_rate'] = '0.1'
inputParamsConfigLocal['momentum'] = '0.9'
inputParamsConfigLocal['num_epochs'] = '10'
inputParamsConfigLocal['batch_size'] = '200'
inputParamsConfigLocal['data_path'] = '/diskStation/LIDC/303010/'
inputParamsConfigLocal['train_set_size'] = '3000'
inputParamsConfigLocal['test_set_size'] = '500'
inputParamsConfigLocal['positive_set_ratio'] = '0.5'
inputParamsConfigLocal['dropout'] = '0.1'
inputParamsConfigLocal['nonlinearityToUse'] = 'sigmoid'
inputParamsConfigLocal['augmentationFlag'] = 0
######Input Params######
########################

n_layers = 3
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


    network = lasagne.layers.dnn.Conv3DDNNLayer(network, num_filters=32, pad='same', filter_size=(5, 5, 3),
                                                stride=(1, 1, 1),
                                                nonlinearity=inputParamsNetwork['nonLinearity'],
                                                W=init_norm,
                                                flip_filters=False
                                                )

    network = lasagne.layers.dnn.MaxPool3DDNNLayer(network, pool_size=(2, 2, 1))

    network = lasagne.layers.dnn.Conv3DDNNLayer(network, num_filters=32, pad='same', filter_size=(5, 5, 3),
                                                stride=(1, 1, 1),
                                                nonlinearity=inputParamsNetwork['nonLinearity'],
                                                W=init_norm,
                                                )
    # network=lasagne.layers.PadLayer(network,width=[(0,1),(0,1)], batch_ndim=3)
    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:

    network = lasagne.layers.dnn.MaxPool3DDNNLayer(network, pool_size=(2, 2, 2))
    # A fully-connected layer of 256 units with 50% dropout on its inputs:
#    network = lasagne.layers.DenseLayer(
#        lasagne.layers.dropout(network, p=dropout),
#        num_units=64,
#        nonlinearity=lasagne.nonlinearities.sigmoid)
    network = lasagne.layers.dnn.Conv3DDNNLayer(network, num_filters=64, pad=0, filter_size=(7, 7, 5),
                                                stride=(1, 1, 1),
                                                nonlinearity=inputParamsNetwork['nonLinearity'],
                                                W=init_norm,
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

inputParamsNetwork = dict(shape=input_shape,dropout=float(dropout), nonLinearity=nonLinearity)

##############################
##############################
# And load them again later on like this:
with np.load(pathSavedNetwork) as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]

#Reshape the FC layer of saved CNN into FCN form    
#param_values[4] = param_values[4].reshape((64,32,7,7,5))
W4_new = np.zeros((64,32,7,7,5)).astype('float32')
for i in range(0,param_values[4].shape[1]): #weights for each node in FC layer form the columns
    current_node_weights = param_values[4][:,i]
    W4_new[i, :, :, :, :] = np.reshape(current_node_weights, (32,7,7,5), order = 'C')
    
param_values[4] = W4_new

if opMode == 'full_fcn':
    W6_new = np.zeros((2,64,1,1,1)).astype('float32')
    for i in range(0,param_values[6].shape[1]): #weights for each node in FC layer form the columns
        current_node_weights = param_values[6][:,i]
        W6_new[i, :, :, :, :] = np.reshape(current_node_weights, (64,1,1,1), order = 'C')
        
    param_values[6] = W6_new

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
    test_prediction = lasagne.layers.get_output(network_fcn, deterministic=True)
    val_fn = theano.function([input_var], [test_prediction])  # ,mode='DebugMode')
    test_pred_fcn = val_fn(test_samples)
    test_pred_fcn = test_pred_fcn[0]
    test_pred_fcn = test_pred_fcn.squeeze()
    test_pred_fcn_softmax0 = np.exp(test_pred_fcn[:,0])/(np.exp(test_pred_fcn[:,0])+ np.exp(test_pred_fcn[:,1]))
    test_pred_fcn_softmax1 = np.exp(test_pred_fcn[:,1])/(np.exp(test_pred_fcn[:,0])+ np.exp(test_pred_fcn[:,1]))
    test_pred_fcn_softmax = np.column_stack((test_pred_fcn_softmax0,test_pred_fcn_softmax1)) #this should be identical to test_pred_orig
    #all_cases= os.listdir(input_3D_npy)
    all_cases = ['p0614_20000101_s30983.npy']
    dim0_score_start_pos = []
    dim0_score_end_pos = []
    dim0_start_pos = []
    dim0_end_pos = []
    patch_size=30,30,10
    case_flag = 0
    total_vol_score_tmp = []
    vol_labels_tmp = []
    for cases in all_cases:
      #  if cases.split('_')[0] in patient_id:
        full_volume = np.load(os.path.join(input_3D_npy,cases))
        full_volume = full_volume.astype('float32')
        full_volume = (full_volume - full_volume.min()) / (full_volume.max() - full_volume.min())
        input_size=full_volume.shape
        in_height, in_width, in_time = input_size
        for part in range(clip_rate[0]):
            dim0_score_start_pos.append(1 + part * output_sizes[0] / clip_rate[0])
            dim0_score_end_pos.append((part + 1) * output_sizes[0] / clip_rate[0])
            dim0_start_pos.append(2 * M_layer * (1 + part * output_sizes[0] / clip_rate[0] - 1) + 1)
            dim0_end_pos.append(2 * M_layer * ((part + 1) * output_sizes[0] / clip_rate[0] - 1) + patch_size[0])
        dim0_pos = zip(dim0_start_pos, dim0_end_pos)
        dim0_score_pos = zip(dim0_score_start_pos, dim0_score_end_pos)

        dim1_score_start_pos = []
        dim1_score_end_pos = []
        dim1_start_pos = []
        dim1_end_pos = []
        for part in range(clip_rate[1]):
            dim1_score_start_pos.append(1 + part * output_sizes[1] / clip_rate[1])
            dim1_score_end_pos.append((part + 1) * output_sizes[1] / clip_rate[1])
            dim1_start_pos.append(2 * M_layer * (1 + part * output_sizes[1] / clip_rate[1] - 1) + 1)
            dim1_end_pos.append(2 * M_layer * ((part + 1) * output_sizes[1] / clip_rate[1] - 1) + patch_size[1])
        dim1_pos = zip(dim1_start_pos, dim1_end_pos)
        dim1_score_pos = zip(dim1_score_start_pos, dim1_score_end_pos)

        dim2_score_start_pos = []
        dim2_score_end_pos = []
        dim2_start_pos = []
        dim2_end_pos = []
        for part in range(clip_rate[2]):
            dim2_score_start_pos.append(1 + part * output_sizes[2] / clip_rate[2])
            dim2_score_end_pos.append((part + 1) * output_sizes[2] / clip_rate[2])
            dim2_start_pos.append(2 * M_layer * (1 + part * output_sizes[2] / clip_rate[2] - 1) + 1)
            dim2_end_pos.append(2 * M_layer * ((part + 1) * output_sizes[2] / clip_rate[2] - 1) + patch_size[2])
        dim2_pos = zip(dim2_start_pos, dim2_end_pos)
        dim2_score_pos = zip(dim2_score_start_pos, dim2_score_end_pos)

        score_mask = np.zeros((2, output_sizes[0], output_sizes[1], output_sizes[2]))

        data_set =full_volume
        for dim0 in range(clip_rate[0]):
            for dim1 in range(clip_rate[1]):
                for dim2 in range(clip_rate[2]):
                    # sys.stdout.write('
                    # .')
                    print(dim0_pos[dim0][0] - 1, dim0_pos[dim0][1], dim1_pos[dim1][0] - 1, dim1_pos[dim1][1],
                          dim2_pos[dim2][0] - 1, dim2_pos[dim2][1])
                    smaller_data = data_set[dim0_pos[dim0][0]-1:dim0_pos[dim0][1],dim1_pos[dim1][0]-1:dim1_pos[dim1][1],dim2_pos[dim2][0]-1:dim2_pos[dim2][1]]

                    smaller_data = smaller_data.reshape((1, 1, smaller_data.shape[0], smaller_data.shape[1], smaller_data.shape[2]))
                    test_pred_full_volume = val_fn(smaller_data)
                    test_pred_full_volume = test_pred_full_volume[0]
                    test_pred_full_volume_softmax0 = np.exp(test_pred_full_volume[0, 0, :, :, :]) / (
                    np.exp(test_pred_full_volume[0, 0, :, :, :]) + np.exp(test_pred_full_volume[0, 1, :, :, :]))
                    test_pred_full_volume_softmax1 = np.exp(test_pred_full_volume[0, 1, :, :, :]) / (
                    np.exp(test_pred_full_volume[0, 0, :, :, :]) + np.exp(test_pred_full_volume[0, 1, :, :, :]))
                    mat_dir = mat_file + Path_create(cases)[:-4]
                    mat_name = mat_pre + cases[:-4] + ".mat"
                    if os.path.exists(mat_dir + '/' + mat_name):
                        noduleData = sio.loadmat(mat_dir + '/' + mat_name)

                    noduleMask = noduleData['allMaxRadiologistMsk']
                    noduleMaskCrop = noduleMask[dim0_pos[dim0][0]-1:dim0_pos[dim0][1],dim1_pos[dim1][0]-1:dim1_pos[dim1][1],dim2_pos[dim2][0]-1:dim2_pos[dim2][1]]
                    # noduleMaskResize = np.zeros((test_pred_full_volume_softmax0.shape[0],test_pred_full_volume_softmax0.shape[1],test_pred_full_volume_softmax0.shape[2]))
                    # for i in range(0,test_pred_full_volume_softmax0.shape[2]):
                    #     noduleMaskResize[:,:,i] =  scipy.misc.imresize(noduleMaskCrop[:,:,i], (test_pred_full_volume_softmax0.shape[0], test_pred_full_volume_softmax0.shape[1]))
                    # noduleMaskResize = np.resize(noduleMask, (
                    # test_pred_full_volume_softmax0.shape[0], test_pred_full_volume_softmax0.shape[1],
                    # test_pred_full_volume_softmax0.shape[2]))
                    dsfactor = [w / float(g) for w, g in
                                zip(test_pred_full_volume_softmax0.shape, noduleMaskCrop.shape)]
                    noduleMaskResize = nd.interpolation.zoom(noduleMaskCrop.astype('float'), zoom=dsfactor)

                    noduleMaskResize = noduleMaskResize > 0.5
                    noduleMaskResize = noduleMaskResize.astype('int')
                    print len(np.where(noduleMaskResize==1)[0])
                    vol_scores_zero = test_pred_full_volume_softmax0.flatten()
                    vol_scores_one = test_pred_full_volume_softmax1.flatten()
                    vol_scores_zero=vol_scores_zero.reshape(len(vol_scores_zero),1)
                    vol_scores_one = vol_scores_one.reshape(len(vol_scores_one),1)
                    if case_flag ==1:
                        vol_score= np.concatenate((vol_scores_zero, vol_scores_one), axis=1)
                        total_vol_score_tmp = np.vstack((total_vol_score_tmp,vol_score))
                        vol_scores_zoronone = np.zeros
                        vol_labels = noduleMaskResize.flatten()
                        vol_labels_tmp = np.concatenate((vol_labels_tmp, vol_labels), axis=0)
                    else:
                        total_vol_score_tmp = np.concatenate((vol_scores_zero, vol_scores_one), axis=1)
                        vol_scores_zoronone = np.zeros
                        vol_labels_tmp = noduleMaskResize.flatten()
                        case_flag = 1



    test_AUC, test_varAUC = SupportFuncs.Pred2AUC(total_vol_score_tmp, vol_labels_tmp.astype('int'))
    print test_AUC

    