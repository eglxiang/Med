import numpy as np
import os
import sys
import scipy.io as sio
import operator
import math

#######################################################################
#This code generates the false positive patches from the score cands that are created using octave with 'patch' option
#You should first get the score map files using the the score map generator code then apply the octave file to get the candidates
#then you should apply this code to generates false positive patches
#
########################################################################
#masterFolderLidc = '/raida/apezeshk/lung_dicom_dir'
numpy_master_case_path = '/diskStation/LIDC/LIDC_NUMPY_3d'
score_map_cand_folder = '/raida/apezeshk/temp/score_map_post_proc/cnn_36368_20170613130504/train/patch'
output_fp_save = '/diskStation/LIDC/'
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
threshScore = 0.95 # any centerInds with score less than this will be suppressed; Use 0 if you want it to have no effect
numHistoBins = 15 #number of histogram bins to consider for fp's bw threshScore:1 for proportional sampling; only used if uniform_pick==False
total_fp_per_case = 1650 #This is approximately total number of fp's from each case; These will be spread uniformly or proportionally based on choice of "uniform_pick"
# safe_margin isn't really in addition to total_fp_per_case; it is just something added s.t. if fp extraction fails
# it will repeat the cycle until number of fp's per bin samples are extracted and saved!
safe_margin=2 #adding more cases just in case it was close to edge and we needed extra
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


def Fp_Generate(inputParamsFp_Generate, uniform_pick=True):
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
                                         
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)
     
    caseCounter = 0.0
    
    #tempFlag = 0
    for cands in cand_lis:        
        print 'Percent of cases running: ' + str(caseCounter/len(cand_lis)) + ', Case Running: ' + cands              
        
        caseCounter = caseCounter+1
        
        input_3d_npy = np.load(os.path.join(numpy_master_case_path, cands[:-4] + '.npy'))
        input_3d_npy = input_3d_npy.astype('int16') #for cases that are uint16
#        if cands=='p0584_20000101_s30968.mat':
#            tempFlag = 1
#            
#        if tempFlag == 0:
#            continue
        
        # cands='p0623_20000101_s32127.mat'
        co_ordinates = sio.loadmat(os.path.join(score_map_cand_folder, cands))
        
        co_ordinate_dict={}#
        #reading center_score_map and
        #mat_content=co_ordinates.get('center_score_map')
        for cord_ind in range(co_ordinates.get('centerInds').shape[0]):
            #In Below: str(co_ordinates.get('centerInds')[cord_ind]) is something like '[ 83.   3.   1.]', so skip the brackets
            #so the dict ends up something like this: ' 73.   1.   1.': (0.97319018840789795, 0)
            currentCenterInds = co_ordinates.get('centerInds')[cord_ind] - 1 #-1 to go from matlab to python indexing!
            if resizeFlag == 0: #if coordinates need to be remapped to full size (resizeFlag=0), do the mapping!
                currentCenterInds[0] = coordinateMapTable[0,0]*currentCenterInds[0] + coordinateMapTable[0,1]
                currentCenterInds[1] = coordinateMapTable[1,0]*currentCenterInds[1] + coordinateMapTable[1,1]
                currentCenterInds[2] = coordinateMapTable[2,0]*currentCenterInds[2] + coordinateMapTable[2,1]
                
            co_ordinate_dict[str(currentCenterInds)[1:-1]]=co_ordinates.get('currentCaseScores')[0][cord_ind],co_ordinates.get('currentCaseLabels')[0][cord_ind].astype('int')
        #Below dict is sth like ' 73.   1.   1.': (0.97319018840789795, 0), so v[1]==0 , means take only label 0 points
        temp_fp_dict = dict((k, v) for k, v in co_ordinate_dict.iteritems() if (v[1] ==0 and v[0]>=threshScore))
        #temp_fp_dict = dict((k, v) for k, v in co_ordinate_dict.iteritems() if v[1] ==0)
        # temp_tp_dict = dict((k, v) for k, v in co_ordinate_dict.iteritems() if v[1] == 1)
        picked_fp=[]

        try: #finds range of scores associated with centerInds
            max_val=temp_fp_dict[max(temp_fp_dict.iteritems(), key=operator.itemgetter(1))[0]][0]
            min_val=temp_fp_dict[min(temp_fp_dict.iteritems(), key=operator.itemgetter(1))[0]][0]
        except: #Not sure why above would fail ever!!!
            max_val=1 #range of scores we want for the fp's (e.g. if we only want fp's w scores bw 0.9-1); used only if uniform_pick == True
            min_val=0.9
            print ("case %s got error in max min: " %cands, )            
        
        difer=max_val-min_val
        #numHistoBins = 10 #number of intervals; so number of actual points u need in linspace is this plus 1 for histogram fn
        histoBins = np.linspace(threshScore, 1, numHistoBins+1) #array with equal size bins within threshScore:1 including 1
        if uniform_pick == True:
            #the bins are each 0.1 wide below; pick same number of samples in each bin
            #number_per_each=(total_fp_per_case)/int(math.ceil(difer/0.1))#calculates number of the samples that we want from each score bin in range
            number_per_each=np.ones((1, numHistoBins), dtype=float) * (total_fp_per_case)/int(math.ceil(difer/0.1))#calculates number of the samples that we want from each score bin in range
            number_per_each = number_per_each.astype(int) #just to make it integer
        elif uniform_pick == False:
            #pick samples in each bin according to frequency in score data; this uses just -'s
            currentAllScores = co_ordinates.get('currentCaseScores')[0]
            currentAllLabels = co_ordinates.get('currentCaseLabels')[0]
            currentScoresClass0 = currentAllScores[np.where(currentAllLabels==0)]
            currentScoresClass0_thresh = currentScoresClass0[np.where(currentScoresClass0>=threshScore)]
            #scoreHist, binEdges = np.histogram(currentAllScores, bins=histoBins)
            scoreHist, binEdges = np.histogram(currentScoresClass0_thresh, bins=histoBins)
            scoreHist = scoreHist.astype(float) / len(currentScoresClass0_thresh) #normalize the counts according to total count of scores to get prob
            number_per_each=np.ones((1, numHistoBins), dtype=float) * scoreHist * total_fp_per_case#calculates number of the samples that we want from each score bin in range
            number_per_each = number_per_each.astype(int) #just to make it integer
            
        #for ind_c in range(0,int(math.ceil(difer/0.1))): 
        for ind_c in range(0,numHistoBins): 
            #For each score bin (e.g. 0.8-0.9), sort the fp's whose scores fall within that range, and select
            #with uniform probability from the sorted list
            #Each not_uni_dic element is sth like ' 233.  433.   99.': (0.10173878073692322, 0)
            #not_uni_dic=dict((k, v) for k, v in temp_fp_dict.iteritems() if min_val+(0.1)*ind_c <=  v[0] <= min_val+(0.1)*ind_c+0.1 )
            not_uni_dic=dict((k, v) for k, v in temp_fp_dict.iteritems() if histoBins[ind_c] <=  v[0] <= histoBins[ind_c+1] )
            if not bool(not_uni_dic): #if no values fall within that range, dict will be empty and bool(dict) evaluates to false so we skip loop;
                continue
            
            #each element in sorted_fp is something like ('  85.  317.   95.', (0.10016164928674698, 0))
            sorted_fp = sorted((not_uni_dic.iteritems()), key=operator.itemgetter(1))
                        
            #due to bad lung interior masks, sometimes too few points exist, and random.choice 
            #produces an error that you want to select e.g. 10 points out of 5 points; just skip such instances
            currentNumberToSelect = number_per_each[0,ind_c]+safe_margin
            if currentNumberToSelect>len(sorted_fp): 
                currentNumberToSelect = len(sorted_fp)
            
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
                        #Before it was reading the full CT case in every iteration; took it out s.t. it's done once!
                        #np_cube = np.load(os.path.join(numpy_master_case_path, cands[:-4] + '.npy'))
                        #selected_fp_sub_volum=crop_3d(fp_x,fp_y,fp_z,np_cube,patch_size)
                        selected_fp_sub_volum=crop_3d(fp_x, fp_y, fp_z, input_3d_npy, patch_size)
                        picked_fp.append(sorted_fp[items_ind])
                        
                        np.save(
                            os.path.join(saveFolder
                                         , str(cands)[:-4] + '_' + str(fp_x) + '_' + str(fp_y) + '_' + str(fp_z)), selected_fp_sub_volum)

                    except KeyboardInterrupt:
                        print('Manual keyboard interrupt, aborting!')
                        sys.exit(0)                        
                    except:
                        print str(fp_x)+'_'+str(fp_y)+'_'+str(fp_z), "case: ",cands

Fp_Generate(inputParamsFp_Generate, uniform_pick)


