# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 13:12:16 2017

@author: apezeshk
"""
#Has two parts: 
#First part uses nodule filtering criteria on a master unique stats file to find
#out how many cases and nodules we will end up with using those criteria; note that for number
#resulting cases, only sliceThickness filter is applied (bc normal cases can still be used for their negatives);

#Second part uses the LUNA case list and THAT PART'S OWN FILTER PARAMETERS to see how many cases we get 
#(we should be able to get same number of cases as in LUNA, i.e. 888), as well as how many nodules. We actually
#end up with slightly less number of nodules compared to LUNA, bc of how the single point detected nodules (i.e. those
#thought to be <3mm by one radiologist) are assigned their own separate tag, even though they may have been detected 
#as >3mm by another radiologist. So when you apply the NumberOfObservers filter criteria, the numbers won't come up the same.
import csv
allUniqueStatsDict = {}
#allUniqueStatsFile = '/raida/apezeshk/lung_dicom_dir/LIDCuniqueStats_04242014.csv'
#filterParams = 'NumberOfObservers,>=,2;IURatio,>=,0.2;SliceThicknessDicom,>=,2;SliceThicknessDicom,<=,3'
allUniqueStatsFile = '/raida/apezeshk/LIDCuniqueStats_03142017.csv'
filterParams = 'NumberOfObservers,>=,3;IURatio,>,0;SliceThicknessDicom,<,3'
#filterParams = 'NumberOfObservers,>=,2;IURatio,>=,0.2;SliceThicknessDicom,>=,1.5;SliceThicknessDicom,<=,3'
masterLIDCFolder = '/raida/apezeshk/lung_dicom_dir/'

#with open(allUniqueStatsFile, "rb") as infile:
#    reader = csv.reader(infile)
#    headers = next(reader)[0:]
#    for row in reader:
#        allUniqueStatsDict[row[0]] = {key: value for key, value in zip(headers, row[1:])}
filterParamsSplit = filterParams.split(';')
masterNoduleList = [] #initialize; will contain list of all unique nodules that fit the filter; each element sth like p0001_20000101_s3000566_0 (the _0 is adjusted tag number; see below for why correction is done)
masterCaseList = [] #initialize; will contain list of all cases with nodules that fit the filter; each element sth like #initialize; will contain list of all unique nodules that fit the filter; each element sth like p0001_20000101_s3000566
with open(allUniqueStatsFile, "rb") as infile:
    reader = csv.DictReader(infile)        
    for row in reader:        
        currentCasePath = row['CasePath']
        currentTag = int(row['Tag']) - 1 #do this correction bc tag number in saved pos cases is per row index, so starts from 0
#        currentSliceThickness = float(row['SliceThicknessDicom'])
#        currentIURatio = float(row['IURatio'])
#        currentNumberOfObservers = int(row['NumberOfObservers'])
        
        currentBaseCase = currentCasePath.replace(masterLIDCFolder,'') #get sth like p0001/20000101/s3000566
        currentBaseCase = currentBaseCase.replace('/','_') #get sth. like p0001_20000101_s3000566
        currentNoduleFlag = True #initialize; if it is set to 0 for any filter condition, that row is ignored
        currentCaseFlag = True
        for i in range(len(filterParamsSplit)):
            currentFilterParams = filterParamsSplit[i].split(',')            
            currentNoduleCheck = eval(row[currentFilterParams[0]]+currentFilterParams[1]+currentFilterParams[2])
            currentNoduleFlag = currentNoduleFlag and currentNoduleCheck
            
            #we don't really need to check case suitability based on slice thickness for each row, but whatever
            if currentFilterParams[0]== 'SliceThicknessDicom' or currentFilterParams[0]== 'SliceThickness':
                currentCaseCheck = eval(row[currentFilterParams[0]]+currentFilterParams[1]+currentFilterParams[2])
                currentCaseFlag = currentCaseFlag and currentCaseCheck
            
        if currentNoduleFlag == True: #currentCaseFlag being True is implied, since that one is just based on sliceThickness whereas currentNoduleFlag is based on every selected filter
            masterNoduleList.append(currentBaseCase+'_'+str(currentTag))
        
        if currentCaseFlag == True:
            if currentBaseCase not in masterCaseList:
                masterCaseList.append(currentBaseCase)
                
                
                
#############################################################
##### This part uses the cases that match the LUNA challenge 
##### and counts how many unique nodules we get from that list
##### compared to number in the LUNA paper
origLIDCcaseNameFile = '/raida/apezeshk/temp/origCaseNamesLIDC.csv'
masterNoduleListLUNA = [] #initialize; will contain list of all unique nodules that we have from LUNA cases; each element sth like p0001_20000101_s3000566_0 (the _0 is adjusted tag number; see below for why correction is done)
masterCaseListLUNA = [] #initialize; will contain list of all cases that matched LUNA; each element sth like #initialize; will contain list of all unique nodules that fit the filter; each element sth like p0001_20000101_s3000566
filterParamsLUNA = 'NumberOfObservers,>=,3;IURatio,>,0' #this is in effect the criteria for LUNA; case list (including according to slice thickness) already filtered by LUNA
filterParamsSplitLUNA = filterParamsLUNA.split(';')
masterCaseListLUNA = [] #keeps track of which LUNA cases we were able to match against; its count should match count of caseListMatchedLUNA!!!

with open(origLIDCcaseNameFile, 'rb') as f:
    reader = csv.reader(f)
    caseListMatchedLUNA = list(reader)
    
caseListMatchedLUNA = [sublist[0] for sublist in caseListMatchedLUNA if sublist[1]!='None'] #above returns a list of lists; this returns rows that matched LUNA

with open(allUniqueStatsFile, "rb") as infile:
    reader = csv.DictReader(infile)        
    for row in reader:        
        currentCasePath = row['CasePath']
        currentTag = int(row['Tag']) - 1 #do this correction bc tag number in saved pos cases is per row index, so starts from 0
#        currentSliceThickness = float(row['SliceThicknessDicom'])
#        currentIURatio = float(row['IURatio'])
#        currentNumberOfObservers = int(row['NumberOfObservers'])
        
        currentBaseCase = currentCasePath.replace(masterLIDCFolder,'') #get sth like p0001/20000101/s3000566
        if currentBaseCase in caseListMatchedLUNA:                            
            if currentBaseCase not in masterCaseListLUNA:
                masterCaseListLUNA.append(currentBaseCase)
                
            currentBaseCase = currentBaseCase.replace('/','_') #get sth. like p0001_20000101_s3000566
            currentNoduleFlag = True #initialize; if it is set to 0 for any filter condition, that row is ignored
            
            for i in range(len(filterParamsSplitLUNA)):
                currentFilterParams = filterParamsSplitLUNA[i].split(',')            
                currentNoduleCheck = eval(row[currentFilterParams[0]]+currentFilterParams[1]+currentFilterParams[2])
                currentNoduleFlag = currentNoduleFlag and currentNoduleCheck         
                                
            if currentNoduleFlag == True:
                masterNoduleListLUNA.append(currentBaseCase+'_'+str(currentTag))
                
nonMatchingLUNA = [val for val in caseListMatchedLUNA if val not in masterCaseListLUNA]            
print 'Cases that matched LUNA but were not in unique stats csv: '
print nonMatchingLUNA
                
                
############################################
####Testing how the SupportFuncs.FilterNodulesCases works
############################################
#filterParams = 'NumberOfObservers,>=,3;IURatio,>,0.0;LUNA'                
#allUniqueStatsFile = '/raida/apezeshk/lung_dicom_dir/LIDCuniqueStats_03142017.csv'
#masterLIDCFolder = '/raida/apezeshk/lung_dicom_dir/'
#origLIDCcaseNameFile = '/raida/apezeshk/lung_dicom_dir/origCaseNamesLIDC.csv'
##filterParams = 'NumberOfObservers,>=,2;IURatio,>=,0.2;SliceThicknessDicom,>=,2;SliceThicknessDicom,<=,3'
#filteredNoduleList = [] #initialize; will contain list of all unique nodules that fit the filter; each element sth like p0001_20000101_s3000566_0 (the _0 is adjusted tag number; see below for why correction is done)
#filteredCaseList = [] #initialize; will contain list of all cases with nodules that fit the filter; each element sth like p0001
#filterParamsSplit = filterParams.split(';')
#
#if 'LUNA' in filterParams:
#    with open(origLIDCcaseNameFile, 'rb') as f:
#        reader = csv.reader(f)
#        caseListMatchedLUNA = list(reader)    
#    caseListMatchedLUNA = [sublist[0] for sublist in caseListMatchedLUNA if sublist[1]!='None'] #above returns a list of lists; this returns rows that matched LUNA
#    caseListMatchedLUNA = [x.replace('/','_') for x in caseListMatchedLUNA] #go from p0001/20000101/s3000566 to p0001_20000101_s3000566
#        
#with open(allUniqueStatsFile, "rb") as infile:
#    reader = csv.DictReader(infile)        
#    for row in reader:        
#        currentCasePath = row['CasePath']
#        currentTag = int(row['Tag']) - 1 #do this correction bc tag number in saved pos cases is per row index, so starts from 0
#        
#        currentBaseCase = currentCasePath.replace(masterLIDCFolder,'') #get sth like p0001/20000101/s3000566
#        currentBaseCase = currentBaseCase.replace('/','_') #get sth. like p0001_20000101_s3000566
#        currentNoduleFlag = True #initialize; if it is set to 0 for any filter condition, that row is ignored
#        currentCaseFlag = True
#        for i in range(len(filterParamsSplit)):
#            currentFilterParams = filterParamsSplit[i].split(',')   
#            if currentFilterParams[0] == 'LUNA':
#                #in this situation, every case/nodule has to be from cases in LUNA, so check against that list
#                currentCaseCheck = currentBaseCase in caseListMatchedLUNA
#                currentCaseFlag = currentCaseFlag and currentCaseCheck
#                currentNoduleFlag = currentNoduleFlag and currentCaseCheck
#            else:                        
#                currentNoduleCheck = eval(row[currentFilterParams[0]]+currentFilterParams[1]+currentFilterParams[2])
#                currentNoduleFlag = currentNoduleFlag and currentNoduleCheck
#                
#                #we don't really need to check case suitability based on slice thickness for each row, but whatever
#                if currentFilterParams[0]== 'SliceThicknessDicom' or currentFilterParams[0]== 'SliceThickness':
#                    currentCaseCheck = eval(row[currentFilterParams[0]]+currentFilterParams[1]+currentFilterParams[2])
#                    currentCaseFlag = currentCaseFlag and currentCaseCheck
#            
#        if currentNoduleFlag == True:#currentCaseFlag being True is implied, since that one is just based on sliceThickness whereas currentNoduleFlag is based on every selected filter
#            filteredNoduleList.append(currentBaseCase+'_'+str(currentTag))
#        
#        if currentCaseFlag == True: #to avoid duplicate patient ids, only add to list if not already in there
#            if currentBaseCase not in filteredCaseList:
#                filteredCaseList.append(currentBaseCase)              
        
        
        
            
            
        