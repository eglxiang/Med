# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 11:10:47 2017

@author: apezeshk
"""
import os 
import dicom
import csv
# Uses the list of case names from the LUNA16 competition to find out what the original case names
# of our LIDC cases are. Matching is according to seriesUID. Note that in the end we get some cases
# that don't match bc there were multiple scans for that patient (e.g. p0159 where only one of the 
# scans matches) or p0078 which for some reason we have multiple copies of. See the output csv to see
# what i mean.
pathdicom = '/raida/apezeshk/lung_dicom_dir/'
csvCasesLUNA = '/raida/apezeshk/temp/LUNA16_evaluationScript/evaluationScript/annotations/seriesuids.csv'
outFile = '/raida/apezeshk/temp/origCaseNamesLIDC.csv'

tmp = ''
lstFilesDCM = []
for dirName, subdirList, fileList in os.walk(pathdicom):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            if tmp!=dirName:
                tmp=dirName
                lstFilesDCM.append(dirName)
                break #once the case path has been added, no need to repeat procedure for every file
     

with open(csvCasesLUNA, 'rb') as f:
    reader = csv.reader(f)
    caseListLUNA = list(reader)
    
caseListLUNA = [val for sublist in caseListLUNA for val in sublist] #above returns a list of lists; this flattens it
     
caseNameMatchList = []  
matchingIndices = [] #keeps track of which indices of LUNA list were found 
nonMatchCounter = 0   
matchCounter = 0
for currentCaseFullPath in sorted(lstFilesDCM):
    currentAllFiles = os.listdir(currentCaseFullPath)
    for currentFile in currentAllFiles:
        if currentFile.endswith('.dcm'):
            currentDicom = dicom.read_file(os.path.join(currentCaseFullPath, currentFile))
            currentCaseName = currentCaseFullPath.replace(pathdicom, '') #drop the LIDC path, s.t. case name components remain
            if currentDicom.SeriesInstanceUID in caseListLUNA:                
                currentOrigName = currentDicom.SeriesInstanceUID
                caseNameMatchList.append([currentCaseName, currentOrigName])
                matchingIndices.append(caseListLUNA.index(currentDicom.SeriesInstanceUID))
                matchCounter = matchCounter + 1
            else:
                caseNameMatchList.append([currentCaseName, 'None'])
                nonMatchCounter = nonMatchCounter + 1
                    
            break #once seriesUID of one dcm has been read, no need to repeat for every file
            
with open(outFile, "wb") as f:
    writer = csv.writer(f)
    writer.writerows(caseNameMatchList)
    
#indices of elementes in LUNA list that didn't match any of our cases
# p0107 & p0085 & p0146 & p0267 have slice thickness<3mm but aren't matching LUNA cases; not sure if that's
# bc of missing/inconsistently spaced slices or something else!
indNonMatchingLUNA = [i for i in range(0,len(caseListLUNA)) if i not in matchingIndices]
  
casesMissingUniqueStats = [] #find out of the matching cases, which ones don't have a uniqueStats file       
for listItem in caseNameMatchList:
    if listItem[1]!='None':
        currentCasePath = os.path.join(pathdicom, listItem[0])
        currentMatName = 'uniqueStats_' + listItem[0].replace('/', '_') + '.mat'
        if not os.path.isfile(os.path.join(currentCasePath, currentMatName)):
            casesMissingUniqueStats.append(listItem[0])
 
## apparently we are missing some LIDC cases (indNonMatchingLUNA is not empty);
## so find out which patient ids these correspond to >>>ACTUALLY this doesn't
## show the missing ones (this shows p0238 & p0585 which don't exist in LIDC);
## Found the actual missing ones by searching using advanced option on LIDC
## website for the missing seriesInstanceUIDs from above           
#patientIDs = [x.replace(pathdicom,'') for x in sorted(lstFilesDCM)]
#patientIDs = [x[0:5] for x in patientIDs]

#for pID in ["p%04d" % (i+1) for i in range(1012)]:
#      if pID not in patientIDs:
#          print pID
#         
  
##############################################################################
# Use below to rename the downloaded LIDC cases into the format needed; First
# you need to create the folder/subfolder structure in targetEndFolder!!!
##############################################################################
##sourceFolder = '/raida/apezeshk/temp/LIDC-IDRI-0332/1.3.6.1.4.1.14519.5.2.1.6279.6001.299122687641741853427119259207/1.3.6.1.4.1.14519.5.2.1.6279.6001.159996104466052855396410079250'
##targetEndFolder = 'p0332/20000101/s30078/'
#sourceFolder = '/raida/apezeshk/temp/LIDC-IDRI-1012/1.3.6.1.4.1.14519.5.2.1.6279.6001.676549258486738448212921834668/1.3.6.1.4.1.14519.5.2.1.6279.6001.153646219551578201092527860224'
#targetEndFolder = 'p1012/20000101/s32231/'
#lidcMasterFolder = '/raida/apezeshk/lung_dicom_dir/'
#for filename in os.listdir(sourceFolder):
#    if filename.endswith(".dcm"):
#        fileparts = targetEndFolder.replace('/','_')
#        newFilename = fileparts+filename[-7:]
#        os.rename(os.path.join(sourceFolder, filename), os.path.join(lidcMasterFolder, targetEndFolder, newFilename))
#    elif filename.endswith(".xml"):
#        os.rename(os.path.join(sourceFolder, filename), os.path.join(lidcMasterFolder, targetEndFolder, filename))
        

          
          

            

            
            
        
    