import numpy as np
import os
import sys
import scipy.io as sio
import operator
import math
from matplotlib import pyplot as plt
import time

#######################################################################
#This code generates the false positive patches from the score cands that are created using octave with 'patch' option
#You should first get the score map files using the the score map generator code then apply the octave file to get the candidates
#then you should apply this code to generates false positive patches
#
########################################################################
#masterFolderLidc = '/raida/apezeshk/lung_dicom_dir'
input_3D_npy = '/diskStation/LIDC/LIDC_NUMPY_3d'
score_map_cand_folder='/raida/apezeshk/temp/score_map_post_proc/cnn_36368_20170113154332/test/nodule'
output_fp_save='/diskStation/LIDC/'
resizeFlag = 0 #0/1; based on whether saved scoreMap was resized to full-size; when 0, centerInds need to be mapped to full-size coordinates for cropping
#for each of x,y,z dims, e.g. x, mapping below defines full-size x_f = ax+b where x is coordinate in FCN proc-ed score map
#For 3-layer architecture & (9,9,4) in FC layer, e.g. x_f is in [4*(x+ floor((9-1)/2), 4*(x+ floor((9-1)/2) + 3]
#which is [4x+16, 4x+19]; so we add 1 so that we move closer to actual center of the associated 4x4 mapped full-size patch
coordinateMapTable = np.array(((4,17), (4,17), (2,1))) #Used only if resizeFlag==0; HAS TO BE CHANGED BASED ON ARCHITECTURE!
patch_size = np.array((36,36,8), dtype='int') #Only used if phase == 'discrim'
phase = 'screen' #'screen' or 'discrim'; see comments on next lines for how it works
#if 'screen', uses same patch size as screening model; if 'discrim', uses provided patch size and also writes
#to subfolder associated with that patch size
uniform_pick = False #if True, equal number of samples taken from each bin of histogram of fp scores; if False, number of samples in each bin proportional to its frequency
total_fp_per_case=35 #This is approximately total number of fp's from each case; These will be spread evenly from different bins of score range, so number per bin is small
# safe_margin isn't really in addition to total_fp_per_case; it is just something added s.t. if fp extraction fails
# it will repeat the cycle until number of fp's per bin samples are extracted and saved!
safe_margin=10 #adding more cases just in case it was close to edge and we needed extra
#minMaxVals = np.array((0.9,1)) #centerInds with scores in this range will be sampled for fp extraction in bins of 0.1


inputParamsFp_Generate = {}
#inputParamsFp_Generate['minMaxVals'] = minMaxVals
inputParamsFp_Generate['resizeFlag'] = resizeFlag
inputParamsFp_Generate['score_map_cand_folder'] = score_map_cand_folder
inputParamsFp_Generate['total_fp_per_case'] = total_fp_per_case
inputParamsFp_Generate['safe_margin'] = safe_margin
inputParamsFp_Generate['coordinateMapTable'] = coordinateMapTable
inputParamsFp_Generate['phase'] = phase
inputParamsFp_Generate['patch_size'] = patch_size


#Gets center and creates the sub volume as the patch size with half patch size viggle
def crop_3d(xcen,ycen,zcen,input_np,patch_size):
    x_viggle = patch_size[0] / 2
    yviggla = patch_size[1] / 2
    zviggla = patch_size[2] / 2
    ArrayDicom = np.zeros(patch_size, dtype=float)
    ArrayDicom[:,:,:]=input_np[(int(xcen)-int(x_viggle)):int(xcen)+int(x_viggle),(int(ycen)-int(yviggla)):(int(ycen)+int(yviggla)),(int(zcen)-int(zviggla)):(int(zcen)+int(zviggla))]
    return ArrayDicom



#minMaxVals = inputParamsFp_Generate['minMaxVals'] 
resizeFlag = inputParamsFp_Generate['resizeFlag']
score_map_cand_folder = inputParamsFp_Generate['score_map_cand_folder']
total_fp_per_case = inputParamsFp_Generate['total_fp_per_case']
safe_margin = inputParamsFp_Generate['safe_margin']
coordinateMapTable = inputParamsFp_Generate['coordinateMapTable']
patch_size = inputParamsFp_Generate['patch_size'] #designates subfolder where patches will be written to if phase==discrim
phase = inputParamsFp_Generate['phase']

cand_lis = sorted(os.listdir(score_map_cand_folder))    
if 'PostProcAUC.mat' in cand_lis: #we just want the score map mat files to be in the list
    cand_lis.remove('PostProcAUC.mat')
    
modelName = score_map_cand_folder.split('/')[-3] #folder structure has be in certain way, s.t. this gives model name!
tmp = sio.loadmat(os.path.join(score_map_cand_folder, cand_lis[0])) #just to get patch_size
patch_size_screening = [int(tmp.get('patchSize')[0][0]), int(tmp.get('patchSize')[0][1]),
                  int(tmp.get('patchSize')[0][2])] #this is the patch size used in the screening model; for saveFolder
                  
if phase == 'screen':
    saveFolder = os.path.join(output_fp_save, str(patch_size_screening[0]) + str(patch_size_screening[1]) + str(patch_size_screening[2])
                                         ,'fp_cases', modelName)
elif phase == 'discrim':
    saveFolder = os.path.join(output_fp_save, str(patch_size_screening[0]) + str(patch_size_screening[1]) + str(patch_size_screening[2])
                                         ,'fp_cases', phase, modelName
                                         , str(patch_size[0]) + str(patch_size[1]) + str(patch_size[2]))
                                     
 
    
    
    
cands='p0024_20000101_s3000557.mat'
co_ordinates = sio.loadmat(os.path.join(score_map_cand_folder, cands))

co_ordinate_dict={}#
#reading center_score_map and
#mat_content=co_ordinates.get('center_score_map')
for cord_ind in range(co_ordinates.get('centerInds').shape[0]-1):
    #In Below: str(co_ordinates.get('centerInds')[cord_ind]) is something like '[ 83.   3.   1.]', so skip the brackets
    #so the dict ends up something like this: ' 73.   1.   1.': (0.97319018840789795, 0)
    currentCenterInds = co_ordinates.get('centerInds')[cord_ind] - 1 #-1 to go from matlab to python indexing!
    if resizeFlag == 0: #if coordinates need to be remapped to full size (resizeFlag=0), do the mapping!
        currentCenterInds[0] = coordinateMapTable[0,0]*currentCenterInds[0] + coordinateMapTable[0,1]
        currentCenterInds[1] = coordinateMapTable[1,0]*currentCenterInds[1] + coordinateMapTable[1,1]
        currentCenterInds[2] = coordinateMapTable[2,0]*currentCenterInds[2] + coordinateMapTable[2,1]
        
    co_ordinate_dict[str(currentCenterInds)[1:-1]]=co_ordinates.get('currentCaseScores')[0][cord_ind],co_ordinates.get('currentCaseLabels')[0][cord_ind].astype('int')
#Below dict is sth like ' 73.   1.   1.': (0.97319018840789795, 0), so v[1]==0 , means take only label 0 points
temp_fp_dict = dict((k, v) for k, v in co_ordinate_dict.iteritems() if v[1] ==0)
# temp_tp_dict = dict((k, v) for k, v in co_ordinate_dict.iteritems() if v[1] == 1)
picked_fp=[]

try: #finds range of scores associated with centerInds
    max_val=temp_fp_dict[max(temp_fp_dict.iteritems(), key=operator.itemgetter(1))[0]][0]
    min_val=temp_fp_dict[min(temp_fp_dict.iteritems(), key=operator.itemgetter(1))[0]][0]
except: #Not sure why above would fail ever!!!
    max_val=1 #range of scores we want for the fp's (e.g. if we only want fp's w scores bw 0.9-1)
    min_val=0.9
    print ("case %s got error in max min: " %cands, )            

difer=max_val-min_val
numBins = 10 #number of intervals; so number of actual points u need in linspace is this plus 1 for histogram fn
histoBins = np.linspace(0, 1, numBins+1) #array with equal size bins 0:0.1:1 including 1
if uniform_pick == True:
    #the bins are each 0.1 wide below; pick same number of samples in each bin
    #number_per_each=(total_fp_per_case)/int(math.ceil(difer/0.1))#calculates number of the samples that we want from each score bin in range
    number_per_each=np.ones((1, numBins), dtype=float) * (total_fp_per_case)/int(math.ceil(difer/0.1))#calculates number of the samples that we want from each score bin in range
    number_per_each = number_per_each.astype(int) #just to make it integer
elif uniform_pick == False:
    #pick samples in each bin according to frequency in score data; this uses freq among + and -, not just -'s
    currentAllScores = co_ordinates.get('currentCaseScores')[0]
    scoreHist, binEdges = np.histogram(currentAllScores, bins=histoBins)
    scoreHist = scoreHist.astype(float) / len(currentAllScores) #normalize the counts according to total count of scores to get prob
    number_per_each=np.ones((1, numBins), dtype=float) * scoreHist * total_fp_per_case#calculates number of the samples that we want from each score bin in range
    number_per_each = number_per_each.astype(int) #just to make it integer
    
#for ind_c in range(0,int(math.ceil(difer/0.1))): 
for ind_c in range(0,numBins): 
    #For each score bin (e.g. 0.8-0.9), sort the fp's whose scores fall within that range, and select
    #with uniform probability from the sorted list
    #Each not_uni_dic element is sth like ' 233.  433.   99.': (0.10173878073692322, 0)
    #not_uni_dic=dict((k, v) for k, v in temp_fp_dict.iteritems() if min_val+(0.1)*ind_c <=  v[0] <= min_val+(0.1)*ind_c+0.1 )
    print ind_c
    print ind_c
    not_uni_dic=dict((k, v) for k, v in temp_fp_dict.iteritems() if 0.0+(0.1)*ind_c <=  v[0] <= 0.0+(0.1)*ind_c+0.1 )
    if not bool(not_uni_dic): #if no values fall within that range, dict will be empty and bool(dict) evaluates to false so we skip loop;
        continue
    
    #each element in sorted_fp is something like ('  85.  317.   95.', (0.10016164928674698, 0))
    sorted_fp = sorted((not_uni_dic.iteritems()), key=operator.itemgetter(1))
                
    #due to bad lung interior masks, sometimes too few points exist, and random.choice 
    #produces an error that you want to select e.g. 10 points out of 5 points; just skip such instances
    currentNumberToSelect = number_per_each[0,ind_c]+safe_margin
    if currentNumberToSelect>len(sorted_fp): 
        continue
    
    rand_picked_instances = np.random.choice(len(sorted_fp), currentNumberToSelect, replace=False)

        
    picked_fp=[]
    #Now loop thru the indices of the fp's; if the patch extraction works, save the patch; 
    #otherwise the "except" will handle that iteration of the loop s.t. program won't blow up.                
    for items_ind in rand_picked_instances: 
        if len(picked_fp) < number_per_each[0,ind_c]: #continue until number_per_each samples in each score bin are extracted
            fp_x=int(sorted_fp[items_ind][0].split()[0][:-1])-1
            fp_y=int(sorted_fp[items_ind][0].split()[1][:-1])-1
            fp_z=int(sorted_fp[items_ind][0].split()[2][:-1])-1
            try:
                np_cube = np.load(os.path.join(input_3D_npy, cands[:-4] + '.npy'))
                selected_fp_sub_volum=crop_3d(fp_x,fp_y,fp_z,np_cube,patch_size)
                picked_fp.append(sorted_fp[items_ind])   
                np.save(os.path.join('/raida/apezeshk/temp/scoreMapTemp/'
                                         , str(cands)[:-4] + '_' + str(fp_x) + '_' + str(fp_y) + '_' + str(fp_z)), selected_fp_sub_volum)
                      
            except KeyboardInterrupt:
                print('Manual keyboard interrupt, aborting!')
                sys.exit(0)             
            except:
                print str(fp_x)+'_'+str(fp_y)+'_'+str(fp_z), "case: ",cands
                
                

patchDirectory = '/raida/apezeshk/temp/scoreMapTemp'
#patchDirectory = '/diskStation/LIDC/36368/pos_36368'
patchFiles = os.listdir(patchDirectory)
plt.figure()
plt.ion()
for i in range(0,len(patchFiles)):
    currentPatchFile = os.path.join(patchDirectory, patchFiles[i])
    if currentPatchFile[-4:] == '.mat':
        continue
    
    currentPatch = np.load(currentPatchFile)
    currentPatch = currentPatch.astype('int16') #bc some dicom cases are uint, u have to first read as int16; otherwise you get the max 65535 situation!
    #currentPatch = currentPatch.astype('float32')
    for j in range(0,8):
        plt.imshow(currentPatch[:,:,j], cmap='gray')
        tit ='fpnum: ' + str(i) + ', slc: ' + str(j);
        plt.title(tit)
        plt.show()
        time.sleep(0.15)
        #_ = raw_input('blah')
        plt.pause(0.0001)


