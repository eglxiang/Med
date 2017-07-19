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
    
    
def Build_3dfcn_mask(init_norm, inputParamsNetwork, fcnLayerFilterSize, input_var=None):
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

        network = lasagne.layers.dnn.MaxPool3DDNNLayer(network, pool_size=(1, 1, 1))

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
    network = lasagne.layers.dnn.Conv3DDNNLayer(network, num_filters=1, pad=0, filter_size=fcnLayerFilterSize,
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
################################################################################################################
# FilterNodules: (based on SupportFuncs.FilterNodulesCases; but tags returned by this are actual tags, not adjusted)
# Applies filter parameters to elements in a uniqueStats file, and returns a list of nodule tags that fit the 
# filter criteria for that case.
# Inputs:
#    filterParams: string with list of filtering criteria. Different comparisons delimited using ';', whereas
# the csv header, comparison type, and value are delimited with ','. E.g.:
# filterParams = 'NumberOfObservers,>=,2;IURatio,>=,0.2;SliceThicknessDicom,>=,2;SliceThicknessDicom,<=,3'
# Outputs:
#    filteredNoduleList: list contains actual tag numbers (i.e. matching what is in the uniqueStats file) for
# nodules in a case that fit the filtering criteria.
################################################################################################################
def FilterNodules(uniqueStatsData, filterParams):        
    uniqueStats = uniqueStatsData['uniqueStats']
    filterParamsSplit = filterParams.split(';')
    filteredNoduleTagList = [] #initialize; will contain list of all unique nodules that fit the filter; each element sth like p0001_20000101_s3000566_0 (the _0 is adjusted tag number; see below for why correction is done)
    
    numObjects = uniqueStats.shape[0]
    for i in range(0, numObjects): #object tags start from 1, so numObjects has to be incremented to get right nums
        currentTag = int(uniqueStats[i]['Tag'][0][0][0]) 
                
        currentNoduleFlag = True #initialize; if it is set to 0 for any filter condition, that row is ignored        
        for j in range(len(filterParamsSplit)):
            currentFilterParams = filterParamsSplit[j].split(',') 
            if currentFilterParams[0] != 'LUNA': #LUNA in filterParams is for case selection, so skip that if present
                currentNoduleCheck = eval(str(uniqueStats[i][currentFilterParams[0]][0][0][0])+currentFilterParams[1]+currentFilterParams[2])
                currentNoduleFlag = currentNoduleFlag and currentNoduleCheck
            
        if currentNoduleFlag == True:
            filteredNoduleTagList.append(currentTag)            
                    
    return filteredNoduleTagList
    
##############################
def TagNoduleMask(uniqueStatsData, inputParamsTagNodule):
    # In order to be able to tell which nodule is being detected later, tag each connected object in 
    # nodule mask with the corresponding unqiue nodule tag number from uniqeStats; this way the tag
    # number stays even after being processed thru FCN, and we can tell which nodules been detected,
    # as well as refer to uniqueStats for stats of that nodule (e.g. size,...)
#    uniquePath = '/raida/apezeshk/lung_dicom_dir/p0962/20000101/s5350/uniqueStats_p0962_20000101_s5350.mat'
#    uniqueStatsData = sio.loadmat(uniquePath)
    uniqueStats = uniqueStatsData['uniqueStats']
    noduleMaskOrig = uniqueStatsData['allMaxRadiologistMsk']
    noduleMaskTagged = np.zeros(noduleMaskOrig.shape, dtype='float32') #this will hold the tagged >3mm nodules 
    smallNoduleMaskTagged = np.zeros(noduleMaskOrig.shape, dtype='float32') #this will hold the tagged <3mm nodules     
    
    remapFlag = inputParamsTagNodule['remapFlag']       
    #uniqueStatsPath = inputParamsTagNodule['uniqueStatsPath']
    noduleCaseFilterParams = inputParamsTagNodule['noduleCaseFilterParams']
    if noduleCaseFilterParams == '': #if no filter is being applied, just separate the output masks according to size of nodules
        noduleCaseFilterParams = 'avgPrincipalAxisLength,!=,0'
    
    filteredNoduleTagList = FilterNodules(uniqueStatsData, noduleCaseFilterParams)
    numObjects = uniqueStats.shape[0]
    tagMappingTable = {} #will contain mapping of tags in noduleMaskTagged to actual nodule tag numbers
    for i in range(0, numObjects): #object tags start from 1, so numObjects has to be incremented to get right nums
        if uniqueStats[i]['Tag'][0][0][0] == -1: #cases (e.g. p0441) with no nodules in them have a single entry with Tag=-1 & no avgMsk field, so skip
            continue
        
        currentSubMask = uniqueStats[i]['avgMsk'][0].astype('float32') #avgMsk is uint8
        if len(currentSubMask.shape) == 2: #if it is single slice mask, it has to be reshaped
            currentSubMask = currentSubMask.reshape((currentSubMask.shape[0], currentSubMask.shape[1], 1))
            
        currentMinZ = uniqueStats[i]['minZ'][0][0][0] - 1 #-1 bc these are real slice numbers but we need to start from 0
        #currentMaxZ = uniqueStats[i]['maxZ'][0][0][0] - 1 
        #currentAvgPrincipalAxisLength = uniqueStats[i]['avgPrincipalAxisLength'][0][0][0]
        
        #need to add the 1 again bc python doesn't include last index!!!
        #i is incremented bc loop is from index 0
        if remapFlag == 0:
            #remapFlag==1 should be used always; see the comment for that condition below!!!
            #Line below is obsolete, overwrites slices in consecutive loops if >1 nodule on a slice!
            #noduleMaskTagged[:,:,currentMinZ:(currentMaxZ+1)] = currentSubMask * (i+1) #set the mask to corresponding nodule tag number            
            currentRow, currentCol, currentSlc = np.where(currentSubMask==1)
            #currentSubMask is a 512x512x(number of slices for current nodule), so need to adjust for starting slice number of nodule to get index in full-size volume
            currentSlc = currentSlc + currentMinZ 
            
#            if currentAvgPrincipalAxisLength>0:
#                noduleMaskTagged[currentRow, currentCol, currentSlc] = i+1
#            elif currentAvgPrincipalAxisLength==0:
#                smallNoduleMaskTagged[currentRow, currentCol, currentSlc] = i+1
            if (i+1) in filteredNoduleTagList:
                noduleMaskTagged[currentRow, currentCol, currentSlc] = i+1
            else:
                smallNoduleMaskTagged[currentRow, currentCol, currentSlc] = i+1
                
        elif remapFlag == 1:
            #set the mask to corresponding nodule tag number PLUS a number that makes it indivisible by 2;
            #This is done s.t. when last fully connected layer has even size and thereby has non-delta function
            #form, the halving of tag values at edges of nodules won't make them have same tag as another valid
            #tag number (e.g. edge of nodule tag 4 value gets halved to 2, which is same as nodule tag 2, so it 
            #would be misidentified)
            #Line below is obsolete, overwrites slices in consecutive loops if >1 nodule on a slice!
            #noduleMaskTagged[:,:,currentMinZ:(currentMaxZ+1)] = currentSubMask * (i+1.5) 
            currentRow, currentCol, currentSlc = np.where(currentSubMask==1)
            #currentSubMask is a 512x512x(number of slices for current nodule), so need to adjust for starting slice number of nodule to get index in full-size volume
            currentSlc = currentSlc + currentMinZ 
            #Small nodules sometimes are detected as <3mm by one radiologist & >3mm by another, resulting
            #in different tag numbers even though they are the same nodule; to avoid confusion in numbering,
            #i.e. overwriting of a nodule's numbers by different tags, separate the small nodule mask (<3mm nodules)
            #from >3mm nodule mask.
#            if currentAvgPrincipalAxisLength>0:
#                noduleMaskTagged[currentRow, currentCol, currentSlc] = i+1.5
#            elif currentAvgPrincipalAxisLength==0:
#                smallNoduleMaskTagged[currentRow, currentCol, currentSlc] = i+1.5
            if (i+1) in filteredNoduleTagList:
                noduleMaskTagged[currentRow, currentCol, currentSlc] = i+1.5
            else:
                smallNoduleMaskTagged[currentRow, currentCol, currentSlc] = i+1.5
      
    if remapFlag == 0:
        tagMappingTable = {i+1:i+1 for i in range(0, numObjects)}
    elif remapFlag == 1:
        #map both the modified tag, and its halved value to proper tag number
        tagMappingTable = {i+1.5:i+1 for i in range(0, numObjects)}
        tagMappingTable.update({(i+1.5)/2.0:i+1 for i in range(0, numObjects)})
    return noduleMaskTagged, smallNoduleMaskTagged, tagMappingTable
########################################################
# Obsolete way TagNoduleMask was done; problem with it was that fragmented nodules (that have disjoint parts) could not
# be correctly matched against the nodule tag number for the fragment parts. So replaced with new implementation!
#def TagNoduleMask(uniqueStatsData):
#    uniqueStats = uniqueStatsData['uniqueStats']
#    noduleMaskOrig = uniqueStatsData['allMaxRadiologistMsk']
#    noduleMaskLabeled, numObjects = Nd_measure.label(noduleMaskOrig, np.ones((3,3,3))) #use this structuring element for 26 connectivity
#    noduleMaskTagged = np.zeros(noduleMaskLabeled.shape, dtype='int16')
#    casePath = uniqueStats[0]['CasePath'][0][0]
#    numTagged = 0 #initialize; keeps track of how many objects are tagged for error check later
#    #Now match each connected component with the correct nodule tag number from uniqueStats
#    for i in range(1, numObjects+1): #objects labeled starting with 1, so numObjects has to be incremented to get right nums
#        currentMatch = 0 #keep track of whether current label (object) matched a unique nodule tag
#        currentRow, currentCol, currentSlc = np.where(noduleMaskLabeled==i)
#        #x,y are col and row, respectively, relative to 0; z and slc are actual slice number (i.e. starting from 1)
#        currentCentroid = np.array((np.mean(currentRow), np.mean(currentCol), 1+np.mean(currentSlc))) 
#        for j in range(0, uniqueStats.shape[0]): #shape is e.g. (5,1) for case with 5 nodules
#            currentUniqueAvgCentroidX = uniqueStats[j]['avgCentroidX'][0][0]
#            currentUniqueAvgCentroidY = uniqueStats[j]['avgCentroidY'][0][0]
#            currentUniqueAvgCentroidZ = uniqueStats[j]['avgCentroidZ'][0][0]
#            #see comment above about why x,y get swapped
#            currentDist = (currentCentroid[0]-currentUniqueAvgCentroidY)**2
#            + (currentCentroid[1]-currentUniqueAvgCentroidX)**2
#            + (currentCentroid[2]-currentUniqueAvgCentroidZ)**2 
#            currentDist = np.sqrt(currentDist)
#            if currentDist <= 3: #kind of arbitrary distance threshold to match against a unique nodule
#                noduleMaskTagged[(currentRow, currentCol, currentSlc)] = j
#                numTagged = numTagged + 1
#                currentMatch = 1
#                
#        if currentMatch == 0:
#            print 'label ' + str(i) + ' did not match any tags in ' + casePath
#            
#    if numTagged<numObjects:
#        print 'Tagging error: not all nodules were correctly tagged in nodule mask! case: ' + casePath
#        raise Exception(
#        'Tagging error: not all nodules were correctly tagged in nodule mask! case: ' + casePath)
#        
#    return noduleMaskTagged
######################################################################################3
    
##################################
# This function takes the mat file containing the nodule mask for an entire  
# case, and passes it thru the FCN to create the corresponding processed mask. It
# also has option to return two different masks, one for >3mm nodules, and another
# for <3mm ones, so that the two sets of nodules can be differentiated from one another later on.
##################################
def CreateNoduleMaskFCN(inputParamsConfigLocal, inputParamsNoduleMaskFCN):    
    # And load them again later on like this:
#    pathSavedNetwork = '/home/apezeshk/Codes/DeepMed/models/cnn_36368_20160921114711.npz'
#    pathSavedSamples = '/home/apezeshk/Codes/DeepMed/models/cnn_36368_20160921114711_samples.npz'  
    #currentCaseName = 'p0012_20000101_s3000561.npy'
    currentCaseName = inputParamsNoduleMaskFCN['currentCaseName'] #sth like p0012_20000101_s3000561.npy
    input_3D_npy = inputParamsNoduleMaskFCN['input_3D_npy']
    masterFolderLidc = inputParamsNoduleMaskFCN['masterFolderLidc']
    cutPointFlag = inputParamsNoduleMaskFCN['cutPointFlag']
    z_depth = inputParamsNoduleMaskFCN['z_depth']
    fcnLayerFilterSize = inputParamsNoduleMaskFCN['fcnLayerFilterSize']
    tagNoduleMaskFlag = inputParamsNoduleMaskFCN['tagNoduleMaskFlag']
    remapFlag = inputParamsNoduleMaskFCN['remapFlag']
    
    noduleCaseFilterParams = inputParamsConfigLocal['noduleCaseFilterParams']
    # Further below where the fully connected layer filter is being constructed, the
    # way it is defined expects dimensions 1,2 of fcnLayerFilterSize to be odd, and dimension 3 even;
    # If you have to pass fcnLayerFilterSize that doesn't fit this, then you should write a separate definition for 
    # how that filter is defined. Idea being that ideally you want to define a delta function to convolve, but when
    # that is not possible due to even dimension, you will have to think of something else!
    if (np.mod(fcnLayerFilterSize[0], 2) != 1) or (np.mod(fcnLayerFilterSize[1], 2) != 1) or (np.mod(fcnLayerFilterSize[2], 2) != 0):
        print "fcnLayerFilterSize[0:1] should technically be odd, but choosing not to error out!"
        #raise ValueError('MaskFCN>>CreateNoduleMaskFCN>>fcnLayerFilterSize: expected dimensions 1,2 to be odd, dimension 3 to be even!')
        
#    input_3D_npy = '/diskStation/LIDC/LIDC_NUMPY_3d'
#    masterFolderLidc = '/raida/apezeshk/lung_dicom_dir'
    ########################
    ######Input Params######
#    inputParamsConfigLocal = {}
#    inputParamsConfigLocal['input_shape'] = '36, 36, 8'
#    inputParamsConfigLocal['learning_rate'] = '0.05'
#    inputParamsConfigLocal['momentum'] = '0.9'
#    inputParamsConfigLocal['num_epochs'] = '1'
#    inputParamsConfigLocal['batch_size'] = '1'
#    inputParamsConfigLocal['data_path'] = '/diskStation/LIDC/36368/'
#    inputParamsConfigLocal['train_set_size'] = '60000'
#    inputParamsConfigLocal['test_set_size'] = '500'
#    inputParamsConfigLocal['positive_set_ratio'] = '0.3'
#    inputParamsConfigLocal['dropout'] = '0.1'
#    inputParamsConfigLocal['nonlinearityToUse'] = 'relu'
#    inputParamsConfigLocal['numberOfLayers'] = 3
#    inputParamsConfigLocal['augmentationFlag'] = 1
#    inputParamsConfigLocal['weightInitToUse'] ='He' #weight initialization; either 'normal' or 'He' (for HeNormal)
#    inputParamsConfigLocal['lrDecayFlag'] = 1 #1 for using learning rate decay, 0 for constant learning rate throughout training
#    inputParamsConfigLocal['biasInitVal'] = 0.0 #1 for using learning rate decay, 0 for constant learning rate throughout training
    
    
    inputParamsConfigAll = inputParamsConfigLocal
    input_shape = inputParamsConfigAll['input_shape']
    #learning_rate = inputParamsConfigAll['learning_rate']
    #momentum = inputParamsConfigAll['momentum']
    #num_epochs = inputParamsConfigAll['num_epochs']
    #batch_size = inputParamsConfigAll['batch_size']
    #data_path = inputParamsConfigAll['data_path']
    #train_set_size = inputParamsConfigAll['train_set_size']
    #test_set_size = inputParamsConfigAll['test_set_size']
    #positive_set_ratio = inputParamsConfigAll['positive_set_ratio']
    dropout = inputParamsConfigAll['dropout']
    #nonlinearityToUse = inputParamsConfigAll['nonlinearityToUse']
    #augmentationFlag = inputParamsConfigAll['augmentationFlag']
    numberOfLayers = inputParamsConfigAll['numberOfLayers']
    biasInitVal = inputParamsConfigAll['biasInitVal']
    
    weight_init = lasagne.init.Normal() #we now use He, but since everything is being loaded later this is ok!!
    biasInit = lasagne.init.Constant(biasInitVal) #for relu use biasInit=1 s.t. inputs to relu are positive in beginning
    
    nonLinearity = lasagne.nonlinearities.linear #use linear since u just want propagation of mask thru model
    inputParamsNetwork = dict(n_layer=numberOfLayers, shape=input_shape,dropout=float(dropout), nonLinearity=nonLinearity,
                              biasInit = biasInit)
                              
    dtensor5 = T.TensorType('float32', (False,) * 5)
    input_var = dtensor5('inputs')
    network_fcn_mask = Build_3dfcn_mask(weight_init, inputParamsNetwork, fcnLayerFilterSize, input_var)
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
    uniqueStatsPath = os.path.join(full_mask_path, mat_name)
    uniqueStatsData = sio.loadmat(uniqueStatsPath)
    if tagNoduleMaskFlag == 0:
        full_mask = uniqueStatsData['allMaxRadiologistMsk'] #this returns uint8
        fake_mask = np.zeros(np.shape(full_mask)) #just so that in the end we can return the two outputs, even tho this is useless
        maskList = [full_mask, fake_mask]
    elif tagNoduleMaskFlag== 1:
        inputParamsTagNodule = {}
        inputParamsTagNodule['remapFlag'] = remapFlag
        inputParamsTagNodule['uniqueStatsPath'] = uniqueStatsPath
        inputParamsTagNodule['noduleCaseFilterParams'] = noduleCaseFilterParams
        largerThan3mm_full_mask, lessThan3mm_full_mask, tagMappingTable = TagNoduleMask(uniqueStatsData, inputParamsTagNodule) #this returns float32
        maskList = [largerThan3mm_full_mask, lessThan3mm_full_mask]
        
    
    #MAKE SURE THE TYPE FOR NODULE MASK IS RIGHT IN BELOW; DO U HAVE TO CONVERT TO INT16 THEN FLOAT32?!!
    loopCounter = 0 #just so you know which output is being produced in the iteration, so that you can assign it
    for currentMask in maskList:
        loopCounter = loopCounter + 1
        currentMask = currentMask.astype('float32')           
    #    chopVolumeFlag = 1
    #    cutPointFlag = 1
    #    z_depth = 8           
        sub_vol_one = []
              
        currentMask = currentMask.reshape((1, 1, 512, 512, currentMask.shape[2]))
        if cutPointFlag == 1:
            xCutPoints = [0, 512]
            yCutPoints = [0, 512]
            tmpFlag = 0
            zCutPoints = [0]
            zStep = 80
            while tmpFlag != 7321:  # to make the loop end, set tmpFlag=7321; otherwise hold prev slice number in it
                currentZCut = tmpFlag + zStep
                if currentZCut > currentMask.shape[4]:
                    currentZCut = currentMask.shape[4]
                    zCutPoints.append(currentZCut)
                    tmpFlag = 7321
                else:
                    tmpFlag = currentZCut - z_depth  # this is amount of overlap between consecutive chops in z direction
                    zCutPoints.append(currentZCut)
                    zCutPoints.append(tmpFlag)
    #    z_size=[]
    #    x_size=[]
    #    y_size=[]
    #    first_cube_flag=0
    #    vol_scores_currentVol = np.empty((0, 2))
    #    score_mat=np.zeros(())
    #    vol_labels_currentVol = []
        #this part is for the cases that last two slices should be changed if you we wanna to FCN( they got small z
        # we take from one cube by 20 and add it to another cube
        if (zCutPoints[-1]-zCutPoints[-2])<=16:
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
                    asd = currentMask[0, 0, xStart:xEnd, yStart:yEnd, zStart:zEnd]
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
                            sub_vol_one=np.concatenate((sub_vol_one[:,:,:-2],tmp_sub_vol[:,:,3:]),axis=2) #I set the concatination margin to 2 since we have a one max pool for Z and last 2 slices are not correctly convolved
                        except:
                            sub_vol_one=tmp_sub_vol
                            
        if tagNoduleMaskFlag == 0:
            sub_vol_one_fin = (sub_vol_one>0.0).astype('int') #convert to binary; it originally has 0.5 values due to z direction elongation in fully connected layer filter
        elif tagNoduleMaskFlag == 1 and remapFlag == 1:
            #In this mode, the half values due to z direction elongation will have to be corrected by remapping numbers to tags
            #So use tagMappingTable to map each value (or half-value) to proper tag number according to look up table; 
            #e.g. map both values 0.75 & 1.5 to tag 1
            dictKeys = tagMappingTable.keys()
            sub_vol_one_fin = np.zeros(np.shape(sub_vol_one)).astype('int')
            for currentKey in dictKeys:
                currentMappedTag = tagMappingTable[currentKey] #we need to map currentKey to currentMappedTag
                currentRow, currentCol, currentSlc = np.where(sub_vol_one==currentKey)
                sub_vol_one_fin[currentRow, currentCol, currentSlc] = currentMappedTag
                
        elif tagNoduleMaskFlag == 1 and remapFlag == 0:
            #doesn't make sense to do anything but create an error in this condition; the results will just be wrong otherwise!
            raise ValueError('MaskFCN>>CreateNoduleMaskFCN>>remapFlag: only tagNoduleMaskFlag == 1 and remapFlag == 1 implemented!')
            
        if loopCounter == 1:
            sub_vol_one_fin_fcn1 = sub_vol_one_fin.copy()
        elif loopCounter == 2:
            sub_vol_one_fin_fcn2 = sub_vol_one_fin.copy()
        
    return sub_vol_one_fin_fcn1, sub_vol_one_fin_fcn2


##################################
# This function takes the mat file containing the mask of the interior region of lung for 
# a case, and passes it thru the FCN to create the corresponding processed mask.
# NOTE: IF THE LUNG INTERIOR MASK DOES NOT EXIST, IT WON'T ERROR OUT, BUT WILL RETURN AN ALL ZEROS ARRAY
# PASSED THRU THE FCN! THIS WAS DONE BC SAVING OF NODULE MASKS IS DONE IN SAME STRUCTURE AS THIS, SO WE DON'T
# WANT TO ERROR OUT!
##################################
def CreateLungInteriorMaskFCN(inputParamsConfigLocal, inputParamsLungInteriorMaskFCN):     
#    pathSavedNetwork = '/home/apezeshk/Codes/DeepMed/models/cnn_36368_20160921114711.npz'
#    pathSavedSamples = '/home/apezeshk/Codes/DeepMed/models/cnn_36368_20160921114711_samples.npz'  
    #currentCaseName = 'p0012_20000101_s3000561.npy'
    currentCaseName = inputParamsLungInteriorMaskFCN['currentCaseName'] #sth like p0012_20000101_s3000561.npy
    #input_3D_npy = inputParamsLungInteriorMaskFCN['input_3D_npy']
    masterFolderLidc = inputParamsLungInteriorMaskFCN['masterFolderLidc']
    masterFolderLungInterior = inputParamsLungInteriorMaskFCN['masterFolderLungInterior']
    cutPointFlag = inputParamsLungInteriorMaskFCN['cutPointFlag']
    z_depth = inputParamsLungInteriorMaskFCN['z_depth']
    fcnLayerFilterSize = inputParamsLungInteriorMaskFCN['fcnLayerFilterSize']
    #tagNoduleMaskFlag = inputParamsLungInteriorMaskFCN['tagNoduleMaskFlag'] #we don't need to tag anything for lung interior
    #remapFlag = inputParamsLungInteriorMaskFCN['remapFlag']
    # Further below where the fully connected layer filter is being constructed, the
    # way it is defined expects dimensions 1,2 of fcnLayerFilterSize to be odd, and dimension 3 even;
    # If you have to pass fcnLayerFilterSize that doesn't fit this, then you should write a separate definition for 
    # how that filter is defined. Idea being that ideally you want to define a delta function to convolve, but when
    # that is not possible due to even dimension, you will have to think of something else!
    if (np.mod(fcnLayerFilterSize[0], 2) != 1) or (np.mod(fcnLayerFilterSize[1], 2) != 1) or (np.mod(fcnLayerFilterSize[2], 2) != 0):
        print "fcnLayerFilterSize[0:1] should technically be odd, but choosing not to error out!"
        #raise ValueError('MaskFCN>>CreateNoduleMaskFCN>>fcnLayerFilterSize: expected dimensions 1,2 to be odd, dimension 3 to be even!')
#    input_3D_npy = '/diskStation/LIDC/LIDC_NUMPY_3d'
#    masterFolderLidc = '/raida/apezeshk/lung_dicom_dir'
    ########################
    ######Input Params######
#    inputParamsConfigLocal = {}
#    inputParamsConfigLocal['input_shape'] = '36, 36, 8'
#    inputParamsConfigLocal['learning_rate'] = '0.05'
#    inputParamsConfigLocal['momentum'] = '0.9'
#    inputParamsConfigLocal['num_epochs'] = '1'
#    inputParamsConfigLocal['batch_size'] = '1'
#    inputParamsConfigLocal['data_path'] = '/diskStation/LIDC/36368/'
#    inputParamsConfigLocal['train_set_size'] = '60000'
#    inputParamsConfigLocal['test_set_size'] = '500'
#    inputParamsConfigLocal['positive_set_ratio'] = '0.3'
#    inputParamsConfigLocal['dropout'] = '0.1'
#    inputParamsConfigLocal['nonlinearityToUse'] = 'relu'
#    inputParamsConfigLocal['numberOfLayers'] = 3
#    inputParamsConfigLocal['augmentationFlag'] = 1
#    inputParamsConfigLocal['weightInitToUse'] ='He' #weight initialization; either 'normal' or 'He' (for HeNormal)
#    inputParamsConfigLocal['lrDecayFlag'] = 1 #1 for using learning rate decay, 0 for constant learning rate throughout training
#    inputParamsConfigLocal['biasInitVal'] = 0.0 #1 for using learning rate decay, 0 for constant learning rate throughout training
    
    
    inputParamsConfigAll = inputParamsConfigLocal
    input_shape = inputParamsConfigAll['input_shape']
    #learning_rate = inputParamsConfigAll['learning_rate']
    #momentum = inputParamsConfigAll['momentum']
    #num_epochs = inputParamsConfigAll['num_epochs']
    #batch_size = inputParamsConfigAll['batch_size']
    #data_path = inputParamsConfigAll['data_path']
    #train_set_size = inputParamsConfigAll['train_set_size']
    #test_set_size = inputParamsConfigAll['test_set_size']
    #positive_set_ratio = inputParamsConfigAll['positive_set_ratio']
    dropout = inputParamsConfigAll['dropout']
    #nonlinearityToUse = inputParamsConfigAll['nonlinearityToUse']
    #augmentationFlag = inputParamsConfigAll['augmentationFlag']
    numberOfLayers = inputParamsConfigAll['numberOfLayers']
    biasInitVal = inputParamsConfigAll['biasInitVal']
    
    weight_init = lasagne.init.Normal() #we now use He, but since everything is being loaded later this is ok!!
    biasInit = lasagne.init.Constant(biasInitVal) #for relu use biasInit=1 s.t. inputs to relu are positive in beginning
    
    nonLinearity = lasagne.nonlinearities.linear #use linear since u just want propagation of mask thru model
    inputParamsNetwork = dict(n_layer=numberOfLayers, shape=input_shape,dropout=float(dropout), nonLinearity=nonLinearity,
                              biasInit = biasInit)
                              
    dtensor5 = T.TensorType('float32', (False,) * 5)
    input_var = dtensor5('inputs')
    network_fcn_mask = Build_3dfcn_mask(weight_init, inputParamsNetwork, fcnLayerFilterSize, input_var)
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
    ######Now load the lung interior mask, and shove it into the network
    ################################################################################
    
    #full_volume_path=os.path.join(input_3D_npy, currentCaseName)
    
    full_mask_path = os.path.join(masterFolderLungInterior, Path_create(currentCaseName)[:-4])
    mat_name = 'lungInterior_' + currentCaseName[:-4] + '.mat'
    if os.path.isfile(os.path.join(full_mask_path, mat_name)):
        lungInteriorData = sio.loadmat(os.path.join(full_mask_path, mat_name))  
        full_mask = lungInteriorData['currentFullVolBin'] #this returns uint8
    else: #read the corresponding unique mask, s.t. you will have proper size fake mask
        uniqueMask_path = os.path.join(masterFolderLidc, Path_create(currentCaseName)[:-4])
        tmp_name = 'uniqueStats_' + currentCaseName[:-4] + '.mat'
        uniqueStatsData = sio.loadmat(os.path.join(uniqueMask_path, tmp_name))    
        unique_mask = uniqueStatsData['allMaxRadiologistMsk'] #this returns uint8
        full_mask = np.zeros(np.shape(unique_mask))           
    
    #MAKE SURE THE TYPE FOR LUNG INTERIOR MASK IS RIGHT IN BELOW; DO U HAVE TO CONVERT TO INT16 THEN FLOAT32?!!    
    currentMask = full_mask.astype('float32')           
#    chopVolumeFlag = 1
#    cutPointFlag = 1
#    z_depth = 8           
    sub_vol_one = []
          
    currentMask = currentMask.reshape((1, 1, 512, 512, currentMask.shape[2]))
    if cutPointFlag == 1:
        xCutPoints = [0, 512]
        yCutPoints = [0, 512]
        tmpFlag = 0
        zCutPoints = [0]
        zStep = 80
        while tmpFlag != 7321:  # to make the loop end, set tmpFlag=7321; otherwise hold prev slice number in it
            currentZCut = tmpFlag + zStep
            if currentZCut > currentMask.shape[4]:
                currentZCut = currentMask.shape[4]
                zCutPoints.append(currentZCut)
                tmpFlag = 7321
            else:
                tmpFlag = currentZCut - z_depth  # this is amount of overlap between consecutive chops in z direction
                zCutPoints.append(currentZCut)
                zCutPoints.append(tmpFlag)
#    z_size=[]
#    x_size=[]
#    y_size=[]
#    first_cube_flag=0
#    vol_scores_currentVol = np.empty((0, 2))
#    score_mat=np.zeros(())
#    vol_labels_currentVol = []
    #this part is for the cases that last two slices should be changed if you we wanna to FCN( they got small z
    # we take from one cube by 20 and add it to another cube
    if (zCutPoints[-1]-zCutPoints[-2])<=16:
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
                asd = currentMask[0, 0, xStart:xEnd, yStart:yEnd, zStart:zEnd]
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
                        sub_vol_one=np.concatenate((sub_vol_one[:,:,:-2],tmp_sub_vol[:,:,3:]),axis=2) #I set the concatination margin to 2 since we have a one max pool for Z and last 2 slices are not correctly convolved
                    except:
                        sub_vol_one=tmp_sub_vol
                        
    
    sub_vol_one_fin = (sub_vol_one>0.0).astype('int') #convert to binary; it originally has 0.5 values due to z direction elongation in fully connected layer filter
    
        
    return sub_vol_one_fin