# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 12:41:27 2016

@author: apezeshk
"""
import numpy as np
import os
import re
import h5py
import tables
import json
import math
import csv
#import pdb


###############################################################################################
# EROCcov:
#This implements Adam's EROCCov function in IQModelo; The helper is exactly same as
#that from the matlab code:
# Computes U-statistic estimates of the area under ROC, LROC, or 
# EROC curves, and the corresponding Sen-type covariance matrix
# estimate for q imaging scenarios (e.g. q readers), m class 1 ratings, n class2 ratings,
# as described in the following
# references:
#
# E. R. DeLong, D. M. DeLong, and D. L. Clarke-Pearson, "Comparing the
# areas under two or more correlated receiver operating characteristic
# curves: A nonparametric approach," Biometrics, vol. 44, no. 3,
# pp. 837-845, Sept. 1988.  
#
# A. Wunderlich, F. Noo, �A nonparametric procedure for comparing the areas 
# under correlated LROC curves,� IEEE Transactions on Medical Imaging, 
# vol. 31, no. 11, pp. 2050-2061, Nov. 2012.
#
# A. Wunderlich, B. Goossens, �Nonparametric EROC analysis for observer 
# performance evaluation on joint detection and estimation tasks,� Proc. of 
# SPIE Medical Imaging Conference, Vol. 9037, 90370F, Feb. 2014.
# 
# Inputs:
#   X (q x m matrix of class 1 ratings)
#   Y (q x n matrix of class 2 ratings)
#   U (q x n matrix of utilities for class 2 images) 
# Note: For ROC analysis, set U = ones(q,n).  For LROC analysis, each entry 
# of U is one if the lesion was localized correctly, and zero otherwise.  
# For EROC analysis, U is a defined by a utility function for parameter 
# estimation, as described in the third reference above.     
# 
# Outputs: 
#   AUC (q x 1 vector of AUC estimates)
#   S (q x q covariance matrix) 
###############################################################################################
def EROCcov(Xin, Yin, Uin):
    X = Xin.copy()
    Y = Yin.copy()
    U= Uin.copy()
    #Check inputs, if taking inputVar.shape shows (numberOfElement,), make their shape (1,numberOfElements)
    #This is to accomodate situation where a zero dimension array containing scores is passed for class0 and class1
    #for a single reader/classifier (rather than multiple readers/classifiers);
    if len(X.shape)==1:
        X = np.zeros((1,len(X)))
        X[0,] = Xin.copy()
        
    if len(Y.shape)==1:
        Y = np.zeros((1,len(Y)))
        Y[0,] = Yin.copy()
        
    if len(U.shape)==1:
        U = np.zeros((1,len(U)))
        U[0,] = Uin.copy()
        
    if X.shape[0] != Y.shape[0]:
        raise ValueError('X and Y should have the same number of rows!')
        
    q = X.shape[0]
    m = X.shape[1]
    n = Y.shape[1]
    AUC = np.zeros((q,))
    V10 = np.zeros((q,m))
    V01 = np.zeros((q,n))
    S10 = np.zeros((q,q))
    S01 = np.zeros((q,q))
    S= np.zeros((q,q))
    for k in range(q):
        x=X[k,:]  
        y=Y[k,:]
        u=U[k,:]
        #compute utility-success matrix for modality k
        US = np.zeros((m,n))
        for i in range(m):
            #Second term is needed for categorical ratings.
            #For continuous-valued ratings, it has no effect.
            US[i,]=u * (y > x[i]) + u*(.5)*(y==x[i])
           
        AUC[k] = np.mean(US)
        V10[k,]=np.sum(US,axis=1)/float(n) #compute the structural components
        V01[k,]=np.sum(US,axis=0)/float(m)
        
    S10 = np.cov(V10) #In np.cov, each row is a single variable; This is opposite of matlab, hence the transposition in matlab version
    S01 = np.cov(V01)
    S=S10/float(m)+S01/float(n)           
    return AUC, S
    

################################################################################################################
# Pred2AUC:
#Given the prediction matrix for each class, and labels for a set of samples, formats the inputs in 
# the form necessary for EROCcov, and returns the AUC as well as std of AUC.
# 
# Inputs:
#   test_pred (mx2 matrix of classifiers scores, one row for each sample and two columns for scores of belonging to class 1 or 2)
#   test_labels ((m,) array of labels for each sample in test_pred
# m is total number of  samples (+ and -) for which test scores are provided
# Outputs: 
#   aucEROC: AUC of classifier for the test samples
#   sEROC: variance of AUC
################################################################################################################
def Pred2AUC(test_pred, test_labels):
    ind0 = np.where(test_labels==0)[0] #np.where returns a tuple, so do the [0] indexing to get correct length
    ind1 = np.where(test_labels==1)[0] 
    x0BaseLine = np.zeros((1,len(ind0))) #
    x1BaseLine = np.zeros((1,len(ind1)))
    x0BaseLine[0,:] = test_pred[ind0,1].copy()
    x1BaseLine[0,:] = test_pred[ind1,1].copy()
    uBaseLine = np.ones((1,len(ind1)))
    aucEROC, sEROC = EROCcov(x0BaseLine,x1BaseLine,uBaseLine)
    return aucEROC, sEROC
    
################################################################################################################
# FilterNodulesCases:
# Takes filter parameters based on elements in master uniqueStats file, and returns two lists.
# Inputs:
#    filterParams: string with list of filtering criteria. Different comparisons delimited using ';', whereas
# the csv header, comparison type, and value are delimited with ','. If one of the fields starts with LUNA,
# then the filtering of cases will be according to LUNA (instead of e.g sliceThickness), and filtering of nodules
# will be based on the other filtering parameters. Example for regular usage:
# filterParams = 'NumberOfObservers,>=,2;IURatio,>=,0.2;SliceThicknessDicom,>=,2;SliceThicknessDicom,<=,3'
# Example for when you want to restrict to LUNA cases:
# filterParams = 'NumberOfObservers,>=,2;IURatio,>=,0.0;LUNA'
# Outputs:
#    filteredNoduleList: list contains elements like p0001_20000101_s3000566_0 where the '0' is unique nodule tag-1 (reason for -1 is
# that when creating the positive patches, they are saved with index starting from 0); So list of positive patches can be 
# compared against this list to see whether to include that positive and associated augmented samples or not;
#    filteredCaseList: contains elements like p0001_20000101_s3000566; some cases may not have any nodules that fit the filter
# criteria for nodules, but if their sliceThickness is suitable, they can still be used for their negatives. So the two
# lists often contain different number of unique CT cases!!!
################################################################################################################
def FilterNodulesCases(filterParams):    
    allUniqueStatsFile = '/raida/apezeshk/lung_dicom_dir/LIDCuniqueStats_03142017.csv'
    masterLIDCFolder = '/raida/apezeshk/lung_dicom_dir/'
    origLIDCcaseNameFile = '/raida/apezeshk/lung_dicom_dir/origCaseNamesLIDC.csv'
    #filterParams = 'NumberOfObservers,>=,2;IURatio,>=,0.2;SliceThicknessDicom,>=,2;SliceThicknessDicom,<=,3'
    filteredNoduleList = [] #initialize; will contain list of all unique nodules that fit the filter; each element sth like p0001_20000101_s3000566_0 (the _0 is adjusted tag number; see below for why correction is done)
    filteredCaseList = [] #initialize; will contain list of all cases with nodules that fit the filter; each element sth like p0001_20000101_s3000566
    filterParamsSplit = filterParams.split(';')
    
    if 'LUNA' in filterParams:
        with open(origLIDCcaseNameFile, 'rb') as f:
            reader = csv.reader(f)
            caseListMatchedLUNA = list(reader)    
        caseListMatchedLUNA = [sublist[0] for sublist in caseListMatchedLUNA if sublist[1]!='None'] #above returns a list of lists; this returns rows that matched LUNA
        caseListMatchedLUNA = [x.replace('/','_') for x in caseListMatchedLUNA] #go from p0001/20000101/s3000566 to p0001_20000101_s3000566
            
    with open(allUniqueStatsFile, "rb") as infile:
        reader = csv.DictReader(infile)        
        for row in reader:        
            currentCasePath = row['CasePath']
            currentTag = int(row['Tag']) - 1 #do this correction bc tag number in saved pos cases is per row index, so starts from 0
            
            currentBaseCase = currentCasePath.replace(masterLIDCFolder,'') #get sth like p0001/20000101/s3000566
            currentBaseCase = currentBaseCase.replace('/','_') #get sth. like p0001_20000101_s3000566
            currentNoduleFlag = True #initialize; if it is set to 0 for any filter condition, that row is ignored
            currentCaseFlag = True
            for i in range(len(filterParamsSplit)):
                currentFilterParams = filterParamsSplit[i].split(',')   
                if currentFilterParams[0] == 'LUNA':
                    #in this situation, every case/nodule has to be from cases in LUNA, so check against that list
                    currentCaseCheck = currentBaseCase in caseListMatchedLUNA
                    currentCaseFlag = currentCaseFlag and currentCaseCheck
                    currentNoduleFlag = currentNoduleFlag and currentCaseCheck
                else:                        
                    currentNoduleCheck = eval(row[currentFilterParams[0]]+currentFilterParams[1]+currentFilterParams[2])
                    currentNoduleFlag = currentNoduleFlag and currentNoduleCheck
                    
                    #we don't really need to check case suitability based on slice thickness for each row, but whatever
                    if currentFilterParams[0]== 'SliceThicknessDicom' or currentFilterParams[0]== 'SliceThickness':
                        currentCaseCheck = eval(row[currentFilterParams[0]]+currentFilterParams[1]+currentFilterParams[2])
                        currentCaseFlag = currentCaseFlag and currentCaseCheck
                
            if currentNoduleFlag == True:#currentCaseFlag being True is implied, since that one is just based on sliceThickness whereas currentNoduleFlag is based on every selected filter
                filteredNoduleList.append(currentBaseCase+'_'+str(currentTag))
            
            if currentCaseFlag == True: #to avoid duplicate patient ids, only add to list if not already in there
                if currentBaseCase not in filteredCaseList:
                    filteredCaseList.append(currentBaseCase)
                    
    return filteredNoduleList, filteredCaseList

################################################################################################################
#Gen_data_discrim
# >>>Difference with Gen_data is that this func is used for the discrimination phase, which is trained on fp's from
# a screening model, and not the randomly extracted negatives. So this func will not need to create a test set (it will
# refer to same test set as was used for screening phase), and will only take negs from the fp's of a screening model.
#input number 5,
#data_path is the root directory containg all the posetive and negative and aumented data for the small x-y-z shape samples
#trainset is number of instants you want in your train set, test set: size of your train set, pos_percent: the posetive split ratio in your test and train,
# augmentation=None generates the train set with no augmentation and augmentatio=1 adds posetive augmented cases to the trainset, number of actual pos instants=number of aumented instants in train set
#out put: pos_train_path,neg_train_path,pos_test_paths,neg_test_path are lists containing the full path of the cases that are being used for creating the train and test sets
#example:
#input gen_data(data_path='/diskstation/LIDC/303010/',40,10,0.2,1), train set size=40, test_set size=10, pos/neg ratio=0.2, augemntation=yes
#***directories in data_path should start eaither with neg_, pos_ or pos_aug prefix
##############################################################################################################
#def Gen_data_discrim(data_path, train_set_size, test_set_size, pos_percent, augmentation, pos_test_size, fp_model_to_use, discrim_shape, noduleCaseFilterParams):
def Gen_data_discrim(inputParamsLoadData):
    data_path = inputParamsLoadData['data_path']
    train_set_size = inputParamsLoadData['train_set_size']
    test_set_size = inputParamsLoadData['test_set_size']
    augmentationRegularFlag = inputParamsLoadData['augmentationRegularFlag']
    augmentationTransformFlag = inputParamsLoadData['augmentationTransformFlag']
    #fp_per_c = inputParamsLoadData['fp_per_case'] #not used if phase=='discrim'
    pos_test_size = inputParamsLoadData['pos_test_size']
    pos_percent = inputParamsLoadData['positive_set_ratio']
    fp_model_to_use = inputParamsLoadData['fp_model_to_use']
    #phase = inputParamsLoadData['phase']
    discrim_shape = inputParamsLoadData['discrim_shape']
    noduleCaseFilterParams = inputParamsLoadData['noduleCaseFilterParams']    
    
    #neg_dir_list = []
    neg_fp_dir_list=[]
    pos_dir_list = []
    pos_aug_dir_list = []
    pos_aug_aux_dir_list = []
    #fp_list=[]
    pos_train_instants = int(train_set_size * pos_percent) #theoretical max # of +'s, actual number based on aug flags, etc may be less
    neg_train_instants= int(train_set_size * (1 - pos_percent))

    directory_lists = os.listdir(data_path)
    patch_size=data_path.split('/')[-1] #this is something like '36368'
    for dirc in directory_lists:  # This loop adds the posetive and negative directory list into the neg_dir_list and pos_dir_list
#        if dirc[:3] == 'neg':#directories in data_path should start either with neg_ or pos_ or pos_aug prefix
#            neg_dir_list.append(dirc)
#        if dirc[:7]=='neg_smp':
#            neg_test_dir_list.append(dirc)
        if dirc[:9] == 'pos_aug_0':
            pos_aug_dir_list.append(dirc)
        if dirc[:11] == 'pos_aug_aux':
            pos_aug_aux_dir_list.append(dirc)
#        if dirc[:2]=='fp':
#            fp_list.append(os.path.join(dirc,fp_model_to_use)) #fp's are in subdir corresponding to model used for them
        if dirc[:2] == 'fp': #this folder will be fp's from train set of screening, so none of them are from test set!
            #neg_fp_dir_list.append(os.path.join(dirc, 'discrim', fp_model_to_use, discrim_shape.replace(' ','').replace(',',''))) #original way, where discrim fp's are written to a folder associated with a patch size in case you wanna extract different patch sizes based on same model
            neg_fp_dir_list.append(os.path.join(dirc, fp_model_to_use))
        elif dirc[:10] == 'pos_'+patch_size:
            pos_dir_list.append(dirc)
            
    if noduleCaseFilterParams != '':
        filteredNoduleList, filteredCaseList = FilterNodulesCases(noduleCaseFilterParams)

    #get full paths to all available negative patches
    all_neg_file_path = Get_file_paths(data_path, neg_fp_dir_list,
                                      len(os.listdir(os.path.join(data_path, neg_fp_dir_list[0]))))
    if noduleCaseFilterParams != '':    
        #as long as patiend id is valid per filteredCaseList, neg patches from that case can be used even
        #if none of its positives make the cut
        #elements in all_neg_file_path are e.g. '/diskStation/LIDC/36368/neg_smp_0_36368/p0822_20000101_s30035_11_484_159_46_6.npy'
        #So below first get the filename (without path), then extract the patient identifier and match againset the case list
        tmp_list = [x for x in all_neg_file_path if '_'.join(os.path.basename(x).split('_')[0:3]) in filteredCaseList]
        all_neg_file_path = tmp_list
                    
#    negative_stat = {}
#    negative_stat = Neg_sample_count(negative_stat, all_neg_file_path)
#    if fp_per_c>0:
#        fp_train_path=Get_file_paths(data_path, fp_list, len(os.listdir(os.path.join(data_path, fp_list[0]))))
    
    #get full paths to all available positive patches (excluding aug versions of them, so this will give count of originals)
    pos_file_path = Get_file_paths(data_path, pos_dir_list, len(os.listdir(os.path.join(data_path, pos_dir_list[0]))))#
    
    if noduleCaseFilterParams != '':    
        #if nodule is valid per filteredNoduleList, keep it otherwise let it go; 
        #elements in pos_file_path are e.g. '/diskStation/LIDC/36368/pos_36368/p0099_20000101_s3000655_2.npy'
        tmp_list = [x for x in pos_file_path if os.path.basename(x)[:-4] in filteredNoduleList]
        pos_file_path = tmp_list
    #all_aug_original_pos = Get_file_paths(data_path, pos_aug_dir_list,  len(os.listdir(os.path.join(data_path, pos_aug_dir_list[0]))))
    #all_aug_original_pos.extend(pos_file_path)#Adding pos and aug_pos into one


    pos_n_neg_test_dic = {} #this dicts contains all the record about the test samples
    #org_pos_case_test=[]
    pos_test_paths = []
    pos_train_path = []
    neg_train_path = []
    neg_test_path = []
    #pos_test_list=[] #list of patients in test set
    #Check if test set with current selected parameters exists. If so, include only those patients that are NOT present
    #in test set for creating training set. If test set does not exist, first create it and then use same logic to 
    #produce the training set.
    if noduleCaseFilterParams == '':
        testfilePath = os.path.join('./','test_'+str(test_set_size)+'_'+str(pos_test_size)+'_'+os.path.basename(data_path)+'.hdf5')
    else:
        testfilePath = os.path.join('./','test_'+str(test_set_size)+'_'+str(pos_test_size)+'_'+os.path.basename(data_path)+'_filt.hdf5')
        
    if os.path.exists(testfilePath):#Checks if that test set previously made by that ratio
        with h5py.File(testfilePath, 'r') as hf:
            print('List of arrays in this file: \n', hf.keys())
            tmp_test_paths = hf.get('Test_set') #Reading list of patients and test file paths
            pos_test_paths=np.array(tmp_test_paths) #full paths to '+' test patches
            patient_in_test = hf.get('Patient_labels') #elements in patient labels are the patient id's like 'p0400'
            tmp_neg_test = hf.get('neg_test_set')
            neg_test_path = np.array(tmp_neg_test)
            for patient in np.array(patient_in_test): #patient is like p0001; so the keys to pos_n_neg_test_dic dic are both this patient number & full paths to neg patches
                pos_n_neg_test_dic[patient] = 1
            for neg_path in neg_test_path:#Also adds the negative file path to the pos_n_neg_test_dic to make the search faster in next steps
                pos_n_neg_test_dic[neg_path] = 1
    else: #generate an error when in discriminate mode, test set for both screening and discrimination phases should be same
        raise Exception(
            'Test set with designated properties does not exist! Test set should be same as one used in screening...')
        ####if test hdf5 doesn't exist, generate test set       
#        for cases in pos_file_path:  # TEST POS
#            if len(pos_test_paths) < pos_test_instants:
#                patient_id = re.split("[/_]", cases)[-4]  # -4 is because after spliting the path string 04 gives the patient id
#                pos_test_paths.append(cases)
#                org_pos_case_test.append(cases)
#                pos_n_neg_test_dic[cases]=1
#                for last_patient in pos_file_path:#adding all the noduls for that patient
#                    if re.split("[/_]", last_patient)[-4]==patient_id and len(pos_test_paths) < pos_test_instants:
#                        if last_patient not in pos_n_neg_test_dic:
#                            pos_test_paths.append(last_patient)
#                pos_n_neg_test_dic[patient_id] = 1
#                pos_test_list.append(patient_id)
#
#        num_neg_per_case=math.ceil(neg_test_instants/len(org_pos_case_test)) #How many neg we need for each positive
#        for cases in org_pos_case_test:
#            tmp_cases=("_".join((cases.split('/')[-1]).split('_')[:4]))[:-4]# output exp case name: p0400_20000101_s3000013_1
#            try:
#                neg_for_spec_case_test=np.random.choice(negative_stat[tmp_cases], int(num_neg_per_case)+10, replace=False)#added 10 is just to make sure we get enough neg for each case
#            except:
#                neg_for_spec_case_test=np.random.choice(negative_stat[tmp_cases], negative_stat[tmp_cases]-1, replace=False)
#            for inds in neg_for_spec_case_test:#because naming previously started with zero but we dont have zero here
#                if inds == 0:
#                    continue
#                case_name_to_save = negative_stat[cases.replace("pos_", "neg_smp_0_")[:-4] +'_'+ str(inds) + '.npy']
#                if len(neg_test_path) < neg_test_instants:
#                    neg_test_path.append(case_name_to_save)
#
#
#        with h5py.File(os.path.join('./', 'test_' + str(test_set_size) + '_' + str(pos_test_size) +'_'+os.path.basename(data_path)+'.hdf5'), 'w') as hf:
#            hf.create_dataset('Test_set', data=pos_test_paths)
#            hf.create_dataset('Patient_labels', data=pos_test_list)
#            hf.create_dataset('neg_test_set', data=neg_test_path)
        ####End of test set creation, or reading from existing test set#########
      
    if augmentationRegularFlag==1 or augmentationTransformFlag==1:#for the dataset with augmentation
        #aug_regular/transform_filenames are lists with filenames (not full paths)
        #aug_file_path is full path to augmentation patches
        aug_regular_filenames = os.listdir(os.path.join(data_path,pos_aug_dir_list[0]))
        aug_transform_filenames = os.listdir(os.path.join(data_path,pos_aug_aux_dir_list[0]))
        if augmentationRegularFlag==1 and augmentationTransformFlag!=1:
            aug_file_path = [os.path.join(data_path,pos_aug_dir_list[0],x) for x in aug_regular_filenames]
        elif augmentationRegularFlag!=1 and augmentationTransformFlag==1:
            aug_file_path = [os.path.join(data_path,pos_aug_aux_dir_list[0],x) for x in aug_transform_filenames]
        elif augmentationRegularFlag==1 and augmentationTransformFlag==1:
            aug_file_path = [os.path.join(data_path,pos_aug_dir_list[0],x) for x in aug_regular_filenames]
            aug_file_path = aug_file_path + [os.path.join(data_path,pos_aug_aux_dir_list[0],x) for x in aug_transform_filenames]
            
        if noduleCaseFilterParams != '':    
            #if nodule is valid per filteredNoduleList, keep it otherwise let it go; 
            #elements in aug_file_path are e.g. 'p0305_20000101_s2_6_r31.npy'
            tmp_list = [x for x in aug_file_path if '_'.join(os.path.basename(x).split('_')[0:4]) in filteredNoduleList]
            aug_file_path = tmp_list
            
#        all_aug_path_dics={}#stores path to all pos aug patches
#        for all_aug_cases in aug_file_path:
#            all_aug_path_dics[all_aug_cases]=1
        # aug_file_path=Get_file_paths(data_path,pos_aug_dir_list,entire_aug_cases)#Contains the aug file path
#        if fp_per_c >0:
#            fpFlag = np.zeros((len(fp_train_path),), dtype=bool) #for each fp in fp folder, keeps track of whether it has been added to train set already; used to avoid duplicate fp's in train set
            
        for cases in pos_file_path:#cases here means original '+' patch (excluding aug +'s) , and not case! TRAIN POS checking if patient case is not in test and adding that to the train test path
            patient_id = re.split("[/_]", cases)[-4] #this gives e.g. p0570
            if (len(pos_train_path)<pos_train_instants) and (patient_id not in pos_n_neg_test_dic): # Just to check that train and test set dont have any overlaps
                pos_train_path.append(cases) #this is actually adding the current patch from a case, 'cases' is a misnomer here
                pos_train_path=Aug_append(cases,pos_train_path,aug_file_path)
                  # TRAIN NEG PATH

#                if len(neg_train_path) < neg_train_instants:#
#                    # for cases in pos_file_path:#this posetive because we are making negative per each negative case # making the list of paths for training and test negatives
#                        case_tmp = ("_".join((cases.split('/')[-1]).split('_')[:4]))[:-4] #this is sth like 'p0570_20000101_s0_5' (so has the index number at end)
#                        try:
#                            neg_for_spec_case=np.random.choice(negative_stat[case_tmp], negative_stat[case_tmp]-1, replace=False)
#                        except:
#                            neg_for_spec_case=np.random.choice(negative_stat[case_tmp], negative_stat[case_tmp]-1, replace=False)
#
#
#                        for inds in neg_for_spec_case:#because naming previously started with zero but we dont have zero here
#                            if inds == 0:
#                                continue
#                            case_name_to_save = negative_stat[cases.replace("pos_", "neg_smp_0_")[:-4] +'_'+ str(inds) + '.npy']
#                            neg_train_path.append(case_name_to_save)
#                if len(neg_train_path) < neg_train_instants:
#                    temp_set_test_fp_neg = set(fp_train_path)
#                    fp_count=0#number of added fp to negative set
#                    tmp_counter = -1 #initialize; keeps track of index of fp_sample within the whole list;                    
#                    for fp_sample in temp_set_test_fp_neg:
#                        tmp_counter += 1
#                        if fp_count<=fp_per_c and len(neg_train_path) < neg_train_instants and (fpFlag[tmp_counter]!=1):
#                            #comparison below is between strings like this 'p0001_s123123_432'
#                            if "_".join(cases.split('/')[-1].split('_')[:3])=="_".join(fp_sample.split('/')[-1].split('_')[:3]):
#                                neg_train_path.append(fp_sample)
#                                fpFlag[tmp_counter] = 1 #if adding an fp, set its flag to 1 s.t. it won't get added again
#                                fp_count+=1
    else:
        for cases in pos_file_path:#TRAIN POS checking if patient case is not in test and adding taht to the train test path
            patient_id = re.split("[/_]", cases)[-4]
            if len(pos_train_path)<pos_train_instants and patient_id not in pos_n_neg_test_dic: # Just to check that train and test set dont have any overlaps
                pos_train_path.append(cases)

    #Unlike Gen_data, here just keep adding the fp's until the max number of negatives is obtained    
    #This is bc of 2 things: 1) negs in the fp folder are all from screening so have no overlap with test set
    #and 2) the negs in fp folders don't have the naming convention corresponding to '+' samples (which was the
    #case for random negs before), so they can just be accumulated until max is achieved.
    for cases in all_neg_file_path:  # TRAIN NEG PATH
            if (len(neg_train_path) < neg_train_instants) and (cases not in pos_n_neg_test_dic):
                #if cases not in pos_n_neg_test_dic:
                    # making the list of paths for traning and test negatives
                neg_train_path.append(cases)

    return pos_train_path,neg_train_path,pos_test_paths,neg_test_path

################################################################################################################
#Gen_data
#input number 5,
#data_path is the root directory containg all the posetive and negative and aumented data for the small x-y-z shape samples
#trainset is number of instants you want in your train set, test set: size of your train set, pos_percent: the posetive split ratio in your test and train,
# augmentation=None generates the train set with no augmentation and augmentatio=1 adds posetive augmented cases to the trainset, number of actual pos instants=number of aumented instants in train set
#out put: pos_train_path,neg_train_path,pos_test_paths,neg_test_path are lists containing the full path of the cases that are being used for creating the train and test sets
#example:
#input gen_data(data_path='/diskstation/LIDC/303010/',40,10,0.2,1), train set size=40, test_set size=10, pos/neg ratio=0.2, augemntation=yes
#***directories in data_path should start eaither with neg_, pos_ or pos_aug prefix
##############################################################################################################
#def Gen_data(data_path,train_set_size,test_set_size,pos_percent,augmentation,fp_per_c,pos_test_size, fp_model_to_use, noduleCaseFilterParams):
def Gen_data(inputParamsLoadData,train_x,test_x):
    data_path = inputParamsLoadData['data_path']
    train_set_size = inputParamsLoadData['train_set_size']
    test_set_size = inputParamsLoadData['test_set_size']
    augmentationRegularFlag = inputParamsLoadData['augmentationRegularFlag']
    augmentationTransformFlag = inputParamsLoadData['augmentationTransformFlag']
    fp_per_c = inputParamsLoadData['fp_per_case'] #not used if phase=='discrim'
    pos_test_size = inputParamsLoadData['pos_test_size']
    pos_percent = inputParamsLoadData['positive_set_ratio']
    fp_model_to_use = inputParamsLoadData['fp_model_to_use']
    #phase = inputParamsLoadData['phase']
    #discrim_shape = inputParamsLoadData['discrim_shape']
    noduleCaseFilterParams = inputParamsLoadData['noduleCaseFilterParams']    
    
    #neg_dir_list = []
    neg_test_dir_list=[] #despite its name, it is just the directory for all neg samples
    pos_dir_list = []
    pos_aug_dir_list = []
    pos_aug_aux_dir_list = []
    fp_list=[]
    pos_train_instants = int(train_set_size * pos_percent) #theoretical max # of +'s, actual number based on aug flags, etc may be less
    
    if pos_test_size==-1:
        pos_test_instants = int(test_set_size * pos_percent)
        neg_test_instants = int(test_set_size * (1 - pos_percent))
    else:
        pos_test_instants=pos_test_size
        neg_test_instants = test_set_size-pos_test_instants
    
    neg_train_instants= int(train_set_size * (1 - pos_percent))
    
    print "Theoretical max num positives: " + str(pos_train_instants)
    print "Theoretical max num negatives: " + str(neg_train_instants)
#################################################################################
###### Old way of defining # neg samples; It was complicated, and in the end better to
###### fix the total number of neg's based on pos_percent, and then see effect of
###### different augmentations since available number of pos's will increase up to 
###### the cap based on pos_percent and augmentation flags.
#    #neg_train_instants is theoretical max count of neg samples to include in training based on numbers passed to the fn.
#    #So actual number of neg samples in training could be less, depending on availability of actual samples!
#    if (fp_per_c == 0 or fp_per_c == -1) and augmentation==0:
#        neg_train_instants= int(train_set_size * (1 - pos_percent))
#    if (fp_per_c == 0 or fp_per_c == -1) and augmentation == 1:
#    #multiplying by 8 beacuse in aug we need more neg for pos augs as well
#        neg_train_instants=(int(train_set_size * (1 - pos_percent))*8)
#    if (fp_per_c != 0 and fp_per_c != -1) and augmentation == 1:
#        # The first method for neg_train_instants below adds too many negatives, & ratio of + to - gets severely distorted;
#        # This is bc the max theoretical # -'s is based on max theoretical # of +'s, which is
#        # basically never achievable; so too many -'s per each + get added!
#        #neg_train_instants=(int(train_set_size * (1 - pos_percent))*8)+(fp_per_c*pos_train_instants) 
#        neg_train_instants=(int(train_set_size * (1 - pos_percent))*8)
#    if (fp_per_c != 0 and fp_per_c != -1) and augmentation==0:
#        print ("You should enter the correct FP( 0 or 1 ) and Augmentation 0 or 1 for dataset generator")
#    # neg_to_pos_ratio=int((neg_train_instants/pos_train_instants)) #Number of the negative cases that we should pick for each posetive
#################################################################################

    directory_lists = os.listdir(data_path)
    patch_size=data_path.split('/')[-1] #this is something like '36368'
    for dirc in directory_lists:  # This loop adds the posetive and negative directory list into the neg_dir_list and pos_dir_list
#        if dirc[:3] == 'neg':#directories in data_path should start either with neg_ or pos_ or pos_aug prefix
#            neg_dir_list.append(dirc)
        if dirc[:7]=='neg_smp':
            neg_test_dir_list.append(dirc) #despite its name, it is the directory for all neg samples
        if dirc[:9] == 'pos_aug_0':
            pos_aug_dir_list.append(dirc)
        if dirc[:11] == 'pos_aug_aux':
            pos_aug_aux_dir_list.append(dirc)
        if dirc[:2]=='fp':
            fp_list.append(os.path.join(dirc,fp_model_to_use)) #fp's are in subdir corresponding to model used for them
            #fp_list.append(os.path.join(dirc, 'discrim', fp_model_to_use, '36368')) #this is for when u wanna add fp's from a discrim model since they have more samples in them
        elif dirc[:10] == 'pos_'+patch_size:
            pos_dir_list.append(dirc)

    #get full paths to all available negative patches
    if noduleCaseFilterParams != '':
        filteredNoduleList, filteredCaseList = FilterNodulesCases(noduleCaseFilterParams)
    
    #full paths to all neg patches; 
    all_neg_file_path = Get_file_paths(data_path,neg_test_dir_list,len(os.listdir(os.path.join(data_path, neg_test_dir_list[0]))),train_x,test_x,'test')
    
    if noduleCaseFilterParams != '':    
        #as long as patiend id is valid per filteredCaseList, neg patches from that case can be used even
        #if none of its positives make the cut
        #elements in all_neg_file_path are e.g. '/diskStation/LIDC/36368/neg_smp_0_36368/p0822_20000101_s30035_11_484_159_46_6.npy'
        tmp_list = [x for x in all_neg_file_path if '_'.join(os.path.basename(x).split('_')[0:3]) in filteredCaseList]
        all_neg_file_path = tmp_list
                                          
    negative_stat = {}
    negative_stat = Neg_sample_count(negative_stat, all_neg_file_path)
    if fp_per_c>0:
        fp_train_path=Get_file_paths(data_path, fp_list, len(os.listdir(os.path.join(data_path, fp_list[0]))),train_x,test_x,'train')
        if noduleCaseFilterParams != '':    
            #as long as patiend id is valid per filteredCaseList, neg patches from that case can be used even
            #if none of its positives make the cut
            #elements in fp_train_path are e.g. '/diskStation/LIDC/36368/fp_cases/cnn_36368_20161209111319/p0896_20000101_s3000057_352_128_86.npy'
            tmp_list = [x for x in fp_train_path if '_'.join(os.path.basename(x).split('_')[0:3]) in filteredCaseList]
            fp_train_path = tmp_list
    
    #get full paths to all available positive patches (excluding aug versions of them, so this will give count of originals)
    pos_file_path = Get_file_paths(data_path, pos_dir_list, len(os.listdir(os.path.join(data_path, pos_dir_list[0]))),train_x,test_x,'train')#whole positive case
    
    if noduleCaseFilterParams != '':
        #if nodule is valid per filteredNoduleList, keep it otherwise let it go; 
        #elements in pos_file_path are e.g. '/diskStation/LIDC/36368/pos_36368/p0099_20000101_s3000655_2.npy'
        tmp_list = [x for x in pos_file_path if os.path.basename(x)[:-4] in filteredNoduleList]
        pos_file_path = tmp_list
    #all_aug_original_pos = Get_file_paths(data_path, pos_aug_dir_list,  len(os.listdir(os.path.join(data_path, pos_aug_dir_list[0]))))
    #all_aug_original_pos.extend(pos_file_path)#Adding pos and aug_pos into one


    pos_n_neg_test_dic = {} #this dicts contains all the record about the test samples
    org_pos_case_test=[]
    pos_test_paths = []
    pos_train_path = []
    neg_train_path = []
    neg_test_path = []
    pos_test_list=[] #list of patients in test set
    #Check if test set with current selected parameters exists. If so, include only those patients that are NOT present
    #in test set for creating training set. If test set does not exist, first create it and then use same logic to 
    #produce the training set.
    if noduleCaseFilterParams == '':
        testfilePath = os.path.join('./','test_'+str(test_set_size)+'_'+str(pos_test_size)+'_'+os.path.basename(data_path)+'.hdf5')
    else:
        testfilePath = os.path.join('./','test_'+str(test_set_size)+'_'+str(pos_test_size)+'_'+os.path.basename(data_path)+'_filt.hdf5')
        
    if os.path.exists(testfilePath):#Checks if that test set previously made by that ratio
        with h5py.File(testfilePath, 'r') as hf:
            print('List of arrays in this file: \n', hf.keys())
            tmp_test_paths = hf.get('Test_set') #Reading list of patients and test file paths
            pos_test_paths=np.array(tmp_test_paths) #full paths to '+' test patches
            patient_in_test = hf.get('Patient_labels') #elements in patient labels are like 'p0400'
            tmp_neg_test = hf.get('neg_test_set')
            neg_test_path = np.array(tmp_neg_test)
            for patient in np.array(patient_in_test): #patient is like p0001; so the keys to pos_n_neg_test_dic dic are both this patient number & full paths to neg patches
                pos_n_neg_test_dic[patient] = 1
            for neg_path in neg_test_path:#Also adds the negative file path to the pos_n_neg_test_dic to make the search faster in next steps
                pos_n_neg_test_dic[neg_path] = 1
    else: #if test hdf5 doesn't exist, generate test set       
        for cases in pos_file_path:  # TEST POS
            if len(pos_test_paths) < pos_test_instants:
                patient_id = re.split("[/_]", cases)[-4]  # -4 is because after spliting the path string 04 gives the patient id
                pos_test_paths.append(cases)
                org_pos_case_test.append(cases)
                pos_n_neg_test_dic[cases]=1
                for last_patient in pos_file_path:  # adding all the noduls for that patient
                    if re.split("[/_]", last_patient)[-4]==patient_id and len(pos_test_paths) < pos_test_instants:
                        if last_patient not in pos_n_neg_test_dic:
                            pos_test_paths.append(last_patient)
                pos_n_neg_test_dic[patient_id] = 1
                pos_test_list.append(patient_id)

        num_neg_per_case=math.ceil(neg_test_instants/len(org_pos_case_test)) #How many neg we need for each positive
        for cases in org_pos_case_test:
            tmp_cases=("_".join((cases.split('/')[-1]).split('_')[:4]))[:-4]# output exp case name: p0400_20000101_s3000013_1
            try:
                #np.random.choice(a,b) selects b samples with uniform probability from arange(0,a);
                #negative_stat[tmp_cases] gives how many neg patches were extracted for a given positive patch
                #(remember that naming of neg patches is based on a particular pos patch, even though now we don't
                #rely on a pos patch z location to extract neg patches, which was the original reason behind the naming
                #convention)
                neg_for_spec_case_test=np.random.choice(negative_stat[tmp_cases], int(num_neg_per_case)+10, replace=False)#added 10 is just to make sure we get enough neg for each case
            except:
                neg_for_spec_case_test=np.random.choice(negative_stat[tmp_cases], negative_stat[tmp_cases]-1, replace=False)
            for inds in neg_for_spec_case_test:#because naming previously started with zero but we dont have zero here
                if inds == 0:
                    continue
                case_name_to_save = negative_stat[cases.replace("pos_", "neg_smp_0_")[:-4] +'_'+ str(inds) + '.npy']
                if len(neg_test_path) < neg_test_instants:
                    neg_test_path.append(case_name_to_save)


        with h5py.File(testfilePath, 'w') as hf:
            hf.create_dataset('Test_set', data=pos_test_paths)
            hf.create_dataset('Patient_labels', data=pos_test_list)
            hf.create_dataset('neg_test_set', data=neg_test_path)
            if noduleCaseFilterParams != '':
                hf.create_dataset('filteredNoduleList', data=filteredNoduleList)
                hf.create_dataset('filteredCaseList', data=filteredCaseList)
               
        ####End of test set creation, or reading from existing test set#########

    if augmentationRegularFlag==1 or augmentationTransformFlag==1:#for the dataset with augmentation
        #aug_regular/transform_filenames are lists with filenames (not full paths)
        #aug_file_path is full path to augmentation patches
        aug_regular_filenames = os.listdir(os.path.join(data_path,pos_aug_dir_list[0]))
        aug_transform_filenames = os.listdir(os.path.join(data_path,pos_aug_aux_dir_list[0]))
        if augmentationRegularFlag==1 and augmentationTransformFlag!=1:
            aug_file_path = [os.path.join(data_path,pos_aug_dir_list[0],x) for x in aug_regular_filenames]
        elif augmentationRegularFlag!=1 and augmentationTransformFlag==1:
            aug_file_path = [os.path.join(data_path,pos_aug_aux_dir_list[0],x) for x in aug_transform_filenames]
        elif augmentationRegularFlag==1 and augmentationTransformFlag==1:
            aug_file_path = [os.path.join(data_path,pos_aug_dir_list[0],x) for x in aug_regular_filenames]
            aug_file_path = aug_file_path + [os.path.join(data_path,pos_aug_aux_dir_list[0],x) for x in aug_transform_filenames]
            
        if noduleCaseFilterParams != '':    
            #if nodule is valid per filteredNoduleList, keep it otherwise let it go; 
            #elements in aug_file_path are e.g. 'p0305_20000101_s2_6_r31.npy'
            tmp_list = [x for x in aug_file_path if '_'.join(os.path.basename(x).split('_')[0:4]) in filteredNoduleList]
            aug_file_path = tmp_list
        
#        all_aug_path_dics={}#stores path to all pos aug patches
#        for all_aug_cases in aug_file_path:
#            all_aug_path_dics[all_aug_cases]=1
        # aug_file_path=Get_file_paths(data_path,pos_aug_dir_list,entire_aug_cases)#Contains the aug file path
        fp_count_total = 0 #initialize; will keep track of total number of fp's added to train set
        if fp_per_c >0:            
            fpFlag = np.zeros((len(fp_train_path),), dtype=bool) #for each fp in fp folder, keeps track of whether it has been added to train set already; used to avoid duplicate fp's in train set
            
        for cases in pos_file_path:#cases here means original '+' patch (excluding aug +'s) , and not case! TRAIN POS checking if patient case is not in test and adding that to the train test path
            patient_id = re.split("[/_]", cases)[-4] #this gives e.g. p0570
            if (len(pos_train_path)<pos_train_instants) and (patient_id not in pos_n_neg_test_dic): # Just to check that train and test set dont have any overlaps
                pos_train_path.append(cases) #this is actually adding the current patch from a case, 'cases' is a misnomer here
                pos_train_path=Aug_append(cases, pos_train_path, aug_file_path)
                  # TRAIN NEG PATH

                if len(neg_train_path) < neg_train_instants:#                    
                    case_tmp = ("_".join((cases.split('/')[-1]).split('_')[:4]))[:-4] #this is sth like 'p0570_20000101_s0_5' (so has the index number at end)
                    try:
                        neg_for_spec_case=np.random.choice(negative_stat[case_tmp], negative_stat[case_tmp]-1, replace=False)
                    except:
                        neg_for_spec_case=np.random.choice(negative_stat[case_tmp], negative_stat[case_tmp]-1, replace=False)


                    for inds in neg_for_spec_case:#because naming previously started with zero but we dont have zero here
                        if inds == 0:
                            continue
                        case_name_to_save = negative_stat[cases.replace("pos_", "neg_smp_0_")[:-4] +'_'+ str(inds) + '.npy']
                        neg_train_path.append(case_name_to_save)
                if len(neg_train_path) < neg_train_instants and fp_per_c >0:
                    temp_set_test_fp_neg = set(fp_train_path)
                    fp_count=0#number of fp's added to negative set for current pos patch 
                    tmp_counter = -1 #initialize; keeps track of index of fp_sample within the whole list;                    
                    for fp_sample in temp_set_test_fp_neg:
                        tmp_counter += 1
                        if fp_count<=fp_per_c and len(neg_train_path) < neg_train_instants and (fpFlag[tmp_counter]!=1):
                            #comparison below is between strings like this 'p0001_s123123_432'
                            if "_".join(cases.split('/')[-1].split('_')[:3])=="_".join(fp_sample.split('/')[-1].split('_')[:3]):
                                neg_train_path.append(fp_sample)
                                fpFlag[tmp_counter] = 1 #if adding an fp, set its flag to 1 s.t. it won't get added again
                                fp_count+=1
                                fp_count_total+=1
        
        print "Total number of fp added to train set: " + str(fp_count_total)
        #after adding neg patches corresponding to pos patches that match filter criteria, we may still be short
        #(note that neg patches are named according to pos patch tag number)
        #So keep adding neg patches until we get to neg_train_instants                                
        if len(neg_train_path) < neg_train_instants: 
            for negPatch in all_neg_file_path:
                if len(neg_train_path) < neg_train_instants:
                    if negPatch not in pos_n_neg_test_dic and negPatch not in neg_train_path:
                        # making the list of paths for traning and test negatives
                        neg_train_path.append(cases)
    else:
        for cases in pos_file_path:#TRAIN POS checking if patient case is not in test and adding that to the train test path
            patient_id = re.split("[/_]", cases)[-4]
            if len(pos_train_path)<pos_train_instants and patient_id not in pos_n_neg_test_dic: # Just to check that train and test set dont have any overlaps
                pos_train_path.append(cases)
        for cases in all_neg_file_path:  # TRAIN NEG PATH
            if len(neg_train_path) < neg_train_instants:
                if cases not in pos_n_neg_test_dic:
                    # making the list of paths for traning and test negatives
                    neg_train_path.append(cases)

    return pos_train_path,neg_train_path,pos_test_paths,neg_test_path

#########################################################################################################
#norm
#This function gets the input numpy matrix and returns the normalized matrix using (X-min(X))/(Max(X)-Min(X))
#########################################################################################################
# def norm(matrix):
#     output=((matrix-np.amin(matrix))/(np.amax(matrix)-np.amin(matrix)))
#     return output
#




#########################################################################################################
#get_file_paths
#This method returns list that contains number of the requested cases from pos_neg_direc_list
#inputs:
#directory:the root path etc. '/diskstation/LIDC/303010/'
#pos_neg_direc_list:
#total_cases: number of the cases to be added into the list for example
#get_file_paths(data_path,pos_aug_dir_list,500) creats 500 sample into a list from the pos_sug list
########################################################################################################
def Get_file_paths(directory,pos_neg_direc_list,total_cases,train_x,test_x,indicator):
    file_paths = []
    list_pso_neg_subdir=[]# List which will store all of the full filepaths.
    list_pso_neg_subdir_chosen=[]
    for sub_dir in pos_neg_direc_list:
        list_pso_neg_subdir.extend(os.listdir(os.path.join(directory, sub_dir)))
    if indicator == 'train':
        for prefix in train_x:
            for filename in list_pso_neg_subdir:
                if prefix in filename:
                    list_pso_neg_subdir_chosen.append(filename)
        #prefix = iter(train_x)
        #filename = iter(list_pso_neg_subdir)
        #for i in range(len(train_x)):
        #    for j in range(len(list_pso_neg_subdir)):
        #        char_prefix = prefix.next()
        #        char_filename = filename.next()
        #        if char_prefix in char_filename:
        #            list_pso_neg_subdir_chosen.append(char_filename)
    if indicator == 'test':
        for prefix in test_x:
            for filename in list_pso_neg_subdir:
                if prefix in filename:
                    list_pso_neg_subdir_chosen.append(filename)
    size_of_dir = len(list_pso_neg_subdir_chosen)
    # neeed to see if we really need random.choice
    candidate_neg=np.random.choice(size_of_dir, size_of_dir, replace=False)
    for i in candidate_neg:
        file_paths.append(os.path.join(os.path.join(directory,pos_neg_direc_list[0]), list_pso_neg_subdir_chosen[int(i)]))

    return file_paths#returns a list containing the paths

##############################################################################################################
#mat_generate_from_path
#Input1 list of lists, where each list contains paths from the input data,
# Input 2 list of strings, identifying contents of each list in Input1
#norm_flag is for the times that you want to normalize by whole volumes max and min if you set it to zero it will
#use the max and min of the small volume
#output1: 5D numpy array [#numberofsamples][channel][xshape][yshape][zshape]
#for example for a train set of size 50 this function will return a numpy array with following shape[50,1,30,30,10] since input data size is 30X30X10
#output2: is a vector reprentetive of lables for the sample example we will have a 1D vector with associated lables[0-1]
#output type is set to float and unit8 to be used in our CNN network
# ##############################################################################################################
def mat_generate_from_path(data_list, name_list, inputParamsMatGen, norm_flag=1,input_3d_npy='/diskStation/LIDC/LIDC_NUMPY_3d'):
    input_shape = inputParamsMatGen['input_shape'] #e.g '36, 36 , 8'
    training_filename = inputParamsMatGen['training_filename']
    for i in range(len(name_list)):
        if name_list[i] == 'train_pos':
            tr_len_pos = len(data_list[i])
            tr_ind_pos = i #maintains the index associated with this variable in data_list (which is a list of lists)
        elif name_list[i] == 'train_neg':
            tr_len_neg = len(data_list[i])
            tr_ind_neg = i
        elif name_list[i] == 'test_pos':
            ts_len_pos = len(data_list[i])
            ts_ind_pos = i
        elif name_list[i] == 'test_neg':
            ts_len_neg = len(data_list[i])
            ts_ind_neg = i
        else:
            raise ValueError('Illdefined data set type!')
            
    tr_len_total = tr_len_pos + tr_len_neg
    ts_len_total = ts_len_pos + ts_len_neg    
    #sample pos path: /diskStation/LIDC/36368/pos_36368/p0099_20000101_s3000655_2.npy'
    patch_shape = input_shape.replace(' ','') #remove space, then split into individual number    
    patch_shape = [int(x) for x in patch_shape.split(',')] #e.g. [36,36,8]
    #pdb.set_trace()
    with tables.open_file(training_filename, mode='w') as training_file_pointer:
        #training_file_pointer = tables.open_file(training_filename, mode='w')
        filters = tables.Filters(complevel=5, complib='blosc')
        train_set_storage = training_file_pointer.create_earray(training_file_pointer.root, 'train_set', tables.Atom.from_dtype(np.dtype('float32')),
                                              shape=(0, 1, patch_shape[0],patch_shape[1],patch_shape[2]), filters=filters, expectedrows=tr_len_total)
        train_label_storage = training_file_pointer.create_earray(training_file_pointer.root, 'train_label', tables.Atom.from_dtype(np.dtype('uint8')),
                                              shape=(0,), filters=filters, expectedrows=tr_len_total)
        test_set_storage = training_file_pointer.create_earray(training_file_pointer.root, 'test_set', tables.Atom.from_dtype(np.dtype('float32')),
                                              shape=(0, 1, patch_shape[0],patch_shape[1],patch_shape[2]), filters=filters, expectedrows=ts_len_total)
        test_label_storage = training_file_pointer.create_earray(training_file_pointer.root, 'test_label', tables.Atom.from_dtype(np.dtype('uint8')),
                                              shape=(0,), filters=filters, expectedrows=ts_len_total)
        val_set_storage = training_file_pointer.create_earray(training_file_pointer.root, 'val_set', tables.Atom.from_dtype(np.dtype('float32')),
                                              shape=(0, 1, patch_shape[0],patch_shape[1],patch_shape[2]), filters=filters, expectedrows=ts_len_total)
        val_label_storage = training_file_pointer.create_earray(training_file_pointer.root, 'val_label', tables.Atom.from_dtype(np.dtype('uint8')),
                                              shape=(0,), filters=filters, expectedrows=ts_len_total) 
        if norm_flag==1:
            def norm_mx_mn(matrix, mx, mn):#internal normaalization having max and min of large volume
                output = ((matrix - mn) / (mx - mn))
                return output
    
            if os.path.exists('./max_min.json'):
                with open('./max_min.json', 'r') as maxminfile:
                    mx_mn_cases = json.load(maxminfile)
            else:
                Find_max_min(input_3d_npy)
                with open('./max_min.json', 'r') as maxminfile:
                    mx_mn_cases = json.load(maxminfile)
            
            #first read the lists containing training +/- paths & write them incrementally, then do the same for test +/- paths
            for ind_pos, ind_neg in [[tr_ind_pos, tr_ind_neg], [ts_ind_pos, ts_ind_neg]]: 
                t_pos = data_list[ind_pos] #training/test pos paths
                t_neg = data_list[ind_neg] #training/test neg paths
                current_data_mode = name_list[ind_pos].split('_')[0] #'train' or 'test'; s.t. you know what variables to write to
                #t_size = len(t_pos) + len(t_neg)
                #data_shape = re.split("[/_]", t_pos[0])[-5] #figure out patch shape based on one of the sample paths
                #t_label = np.zeros(shape=t_size)
                #t_set = np.zeros(shape=(
                #                t_size, 1, int(data_shape[:2]), int(data_shape[2:4]), int(data_shape[4:])))
                #indecies = 0
                
                #t_pos and t_neg processed separately bc label is defined differently for them
                for items in t_pos:
                    maincase=items.split('/')[-1]
                    maincase = maincase.split('_')[:3]
                    patientcase=maincase[0] + "_" + maincase[1] + "_" + maincase[2]
                    newarray = (np.load(items)).astype('int16')
                    current_patch = norm_mx_mn(newarray.astype('float32'),float(mx_mn_cases[patientcase].split('/')[0]),
                                               float(mx_mn_cases[patientcase].split('/')[1]))
                    current_patch = current_patch.astype(np.float32)
                    #the reshape is necessary s.t. the append operation works correctly without rank error
                    current_patch = np.reshape(current_patch, (1,1,patch_shape[0],patch_shape[1],patch_shape[2]))
                    current_label = np.uint8(1);  current_label = np.reshape(current_label, (1,))
                    if current_data_mode == 'train':
                        train_set_storage.append(current_patch)
                        train_label_storage.append(current_label)
                    elif current_data_mode == 'test':
                        test_set_storage.append(current_patch)
                        test_label_storage.append(current_label)
                        val_set_storage.append(current_patch) #setting val=test data
                        val_label_storage.append(current_label)
    #                t_set[indecies][0][:, :, :] = norm_mx_mn(newarray.astype('float32'),float(mx_mn_cases[patientcase].split('/')[0]),float(mx_mn_cases[patientcase].split('/')[1]))
    #                t_label[indecies] = 1
    #                indecies += 1
     
                for items in t_neg:
                    maincase=items.split('/')[-1]
                    maincase = maincase.split('_')[:3]
                    patientcase=maincase[0] + "_" + maincase[1] + "_" + maincase[2]
                    newarray = (np.load(items)).astype('int16')
                    current_patch = norm_mx_mn(newarray.astype('float32'), float(mx_mn_cases[patientcase].split('/')[0]),
                                                             float(mx_mn_cases[patientcase].split('/')[1]))
                    current_patch = current_patch.astype(np.float32)
                    #the reshape is necessary s.t. the append operation works correctly without rank error
                    current_patch = np.reshape(current_patch, (1,1,patch_shape[0],patch_shape[1],patch_shape[2]))
                    current_label = np.uint8(0); current_label = np.reshape(current_label, (1,))
                    if current_data_mode == 'train':
                        train_set_storage.append(current_patch)
                        train_label_storage.append(current_label)
                    elif current_data_mode == 'test':
                        test_set_storage.append(current_patch)
                        test_label_storage.append(current_label)
                        val_set_storage.append(current_patch) #setting val=test data
                        val_label_storage.append(current_label)                           
    #                t_set[indecies][0][:, :, :] = norm_mx_mn(newarray.astype('float32'), float(mx_mn_cases[patientcase].split('/')[0]),
    #                                                         float(mx_mn_cases[patientcase].split('/')[1]))
    #                # t_set[indecies][0][:, :, :] = newarray
    #                t_label[indecies] = 0
    #                indecies += 1

#    return np.float32(t_set), t_label.astype(np.uint8),len(t_pos),len(t_neg) #MAKE SURE YOU FLOAT32 T_SET & T_LABLE!!!
    #training_file_pointer.close()
    
    return tr_len_pos, tr_len_neg, ts_len_pos, ts_len_neg
##############################################################################################################
#mat_generate_from_path
#input data_path and it removes all the cases that cause nan after normalization
#*****************This method should be called for each dataset once before running the network************
##############################################################################################################
def rem_nan(path):
    for dir in os.listdir(path):
        for cases in os.listdir(os.path.join(path,dir)):
            if cases[:3]!='pos': # Only scanning the negative cases
                case_mat = np.load(os.path.join(os.path.join(path,dir),cases))
                if np.amax(case_mat) == np.amin(case_mat):
                    print (cases)
                    os.remove(os.path.join(os.path.join(path,dir),cases))
# rem_nan('/diskstation/LIDC/404010/')
##############################################################################################################
#Load _data
#Input 1 : data_path is the explicit address of the numpy patches in our example its located in diskstation server diskstation
# data_path='/diskstation/LIDC/303010/'
#input 2:size of train set
#input 3:size of Test set
#input 4:Ratio of posetive in train and test it's set to 0.5 by default but it should be changed if there is not enough posetive samples
##############################################################################################################
def load_data(inputParamsLoadData,train_x,test_x):
    print("Writing data...")
    
#    data_path = inputParamsLoadData['data_path']
#    train_set_size = inputParamsLoadData['train_set_size']
#    test_set_size = inputParamsLoadData['test_set_size']
#    augmentationRegularFlag = inputParamsLoadData['augmentationRegularFlag']
#    augmentationTransformFlag = inputParamsLoadData['augmentationTransformFlag']
#    fp_per_c = inputParamsLoadData['fp_per_case'] #not used if phase=='discrim'
#    pos_test_size = inputParamsLoadData['pos_test_size']
#    positive_set_ratio = inputParamsLoadData['positive_set_ratio']
#    fp_model_to_use = inputParamsLoadData['fp_model_to_use']
    input_shape = inputParamsLoadData['input_shape'] #used to determine patch shape for train/test pre-allocation
    phase = inputParamsLoadData['phase']
    training_filename = inputParamsLoadData['training_filename']
#    discrim_shape = inputParamsLoadData['discrim_shape']
#    noduleCaseFilterParams = inputParamsLoadData['noduleCaseFilterParams']
    
    if phase == 'screen':
        #train_pos,train_neg,test_pos,test_neg=Gen_data(data_path,train_set_size,test_set_size,positive_set_ratio,augmentation,fp_per_c,pos_test_size, fp_model_to_use, noduleCaseFilterParams)
        train_pos,train_neg,test_pos,test_neg=Gen_data(inputParamsLoadData,train_x,test_x)
    elif phase == 'discrim':
        #train_pos,train_neg,test_pos,test_neg=Gen_data_discrim(data_path, train_set_size, test_set_size, positive_set_ratio, augmentation, pos_test_size, fp_model_to_use, discrim_shape, noduleCaseFilterParams)
        train_pos,train_neg,test_pos,test_neg=Gen_data_discrim(inputParamsLoadData)
    
    #mat_generate_from_path takes the lists containing the full paths to train/test data, 
    #writes the training file into hdf5, and returns the lengths of each of the arrays
    #contained in that file
    inputParamsMatGen = {}
    inputParamsMatGen['input_shape'] = input_shape
    inputParamsMatGen['training_filename'] = training_filename
    tr_len_pos, tr_len_neg, ts_len_pos, ts_len_neg = mat_generate_from_path([train_pos,train_neg, test_pos, test_neg], 
                                                                            ['train_pos', 'train_neg', 'test_pos', 'test_neg'],
                                                                            inputParamsMatGen)
    #te_set,te_label = mat_generate_from_path(test_pos,test_neg)
    #va_set, va_label = te_set, te_label
        
    print("Writing data is completed")


    return tr_len_pos,tr_len_neg, ts_len_pos, ts_len_neg  
   
#############################################################################################################
#Aug_append
#This function gets the full path to a positive patch, and appends full paths to corresponding
#augmented patches to a list; in the end all augmented patches corresponding to each positive patch will
#be in the list;
#NOTE: This function needs to be changed if any changes happen to naming convention of aug patches!!!
##############################################################################################################
def Aug_append(case,cases_list,aug_path_list):
    #case is full path to a positive patch
    #cases_list will hold list of all full paths to positive patches (including augmentations); The loop from which
    #   this function is called from, will incrementally update it to include all positive patches (including aug patches).
    #aug_path_list is list of full paths to the augmented patches
    for aug_path in aug_path_list:
        #if pos patch identifier ("case" is e.g. p0003_20000101_s3000611_0.npy) matches that of aug patch, append the aug patch
        if os.path.basename(aug_path)[:-8]==case.split('/')[-1][:-4]: 
            cases_list.append(aug_path)
    return cases_list
##Below is old way it was done##
#def Aug_append(case,cases_list,aug_path_list):
#    #case is full path to a positive patch
#    #cases_list will hold list of all full paths to positive patches (including augmentations); The loop from which
#    #   this function is called from, will incrementally update it to include all positive patches (including aug patches).
#    #aug_path_list is list of filenames (without full path) of the augmented patches
#    for key, value in aug_path_list.iteritems():
#        #if pos patch identifier ("case" is e.g. p0003_20000101_s3000611_0.npy) matches that of aug patch, append the aug patch
#        if key[:-8]==case.split('/')[-1][:-4]: 
#            cases_list.append(os.path.join(os.path.dirname(os.path.dirname(case))+'/pos_aug_0_'+case.split('/')[-3],key))
#    return cases_list
   
##############################################################################################################
#index_mapping
#this function finds all the max and min of the cases and dumps those values in a json file
#
############################################################################################################
# This is how max/min were found originally; the new form uses a threshold to set outliers
# to values based on max/min of volume without the outliers; Reason was some cases had
# noise bands with extreme values, or extensive metal artifacts;  If u want to use this version, 
# just comment out the new implementation and uncomment the old!
#
#def Find_max_min(input_3d_npy):
#    with open('./max_min.json', 'w') as f:
#        max_min={}
#        for cases in os.listdir(input_3d_npy):
#                temp=np.load(os.path.join(input_3d_npy,cases))
#                temp=temp.astype('int16')
#                max_min[str(cases)[:-4]]=str(np.max(temp))+'/'+str(np.min(temp))
#        json.dump(max_min,f)
############################################################################################################## 
def Find_max_min(input_3d_npy):
    threshMax = 3000 #values above this will be set to max of volume if those values are removed
    threshMin = -3000#values below this will be set to min of volume if those values are removed
    with open('./max_min.json', 'w') as f:
        max_min={}
        for cases in os.listdir(input_3d_npy):
                currentVol=np.load(os.path.join(input_3d_npy,cases))
                currentVol=currentVol.astype('int16')

                tmp = np.zeros(currentVol.shape, dtype=np.int16)
                tmp[currentVol<threshMax] = currentVol[currentVol<threshMax] #zero out outliers, and find new max from rest of volume
                max_mod = tmp.max()
                tmp = np.zeros(currentVol.shape, dtype=np.int16) #now do same as above, but for negative outliers 
                tmp[currentVol>threshMin] = currentVol[currentVol>threshMin]
                min_mod = tmp.min()
                currentVol[currentVol>=threshMax] = max_mod
                currentVol[currentVol<=threshMin] = min_mod                
                
                max_min[str(cases)[:-4]]=str(np.max(currentVol))+'/'+str(np.min(currentVol))
        json.dump(max_min,f)

##############################################################################################################
#Neg_sample_count
# This function reads the list containing all the negative patches and returns a hash table that contains
#the patch name (sth like 'p0822_20000101_s30035_11') and number of the created negative cases for that specific sample;
#Note that each negative patch is created based on some positive patch, and numbering as in the example reflects that.
#The keys of the hash table fall into 2 groups: keys based on patch names (without path), as well as the full paths of
#patches with their coordinates chopped off; for first group, values show the number of neg patches with that id, for
#second group of keys, values are the full path of a negative patch including the coordinates. E.g. for file with path
#'/diskStation/LIDC/36368/neg_smp_0_36368/p0822_20000101_s30035_11_484_159_46_6.npy', the corresponding
#key (of second type) is '/diskStation/LIDC/36368/neg_smp_0_36368/p0822_20000101_s30035_11_6.npy'
############################################################################################################
def Neg_sample_count(neg_smp_count_htable,all_neg_list):
    for items in all_neg_list:
        neg_smp_count_htable[os.path.join(os.path.dirname(items), 
                                          ("_".join((items.split('/')[-1]).split('_')[:4])+'_'+
                                          "_".join((items.split('/')[-1]).split('_')[-1:])[:-4])+'.npy')]=items #saves the actual name of each file for the corresponding negative file which has X Y X in the file name and we shouldn't have
        case_id= "_".join((items.split('/')[-1]).split('_')[:4]) #gives sth like 'p0822_20000101_s30035_11'
        if case_id in neg_smp_count_htable:
            tmp_value=neg_smp_count_htable[case_id]
            neg_smp_count_htable[case_id]=tmp_value+1
        else:
            neg_smp_count_htable[case_id]=1
    return neg_smp_count_htable






# Find_max_min('/diskStation/LIDC/LIDC_NUMPY_3d')

