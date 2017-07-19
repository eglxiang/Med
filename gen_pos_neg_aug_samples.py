###################################################################################################
#Sardar Hamidian 07-1-2016
#Reading dicom files and creating 3D-numpy patches both positive and negative (<3mm nodules NOT extracted)
#if you want to add more negative you should change the margin rand  in line 179-183
#Input 3D numpy array of dicom files
#Output 3D small samples for feeding CNN model
#
###################################################################################################
import numpy as np
import os
import sys
import scipy.io as sio
from random import randint
#from skimage import io
from skimage import transform as tf
# Things to Do:
#X Change name of folders for output and output_aug so they reflect that they are for pos patches
#X Change folder naming programitcally based on patch_size
#X (Random per pos patch, so not known exactly) Put something reflecting how many negs are extracted per pos up here
#X Add new augmentation modes (contrast, shear, size, combo...); Perhaps should go in different folder s.t. effect of w/wo can 
#   be compared to see if there is any benefit from these augmentations.
#XShould we do .astype('int16') for cases that are uint16 right here? The patches do get read as int16
#   in SupportFuncs, but not sure if that fixes any potential issues (e.g. for augmentation patches) or
#   if int16 conversion should be done here. >>>Actually patches from crop_3d were written as float, so 
#   later in SupportFunc it is unnecessary for them to be read in as int16 and then float, they are already float;
#   but full volume was being read as is, and now i added conversion to int16 right when it is read;
#X Something to help avoid re-extraction of patches for cases we already processed?
#X Wrong implementation: pos_nodules_in_each_case is being iteratively updated to include all the nodules;
#   but the check of neg patches is done as this is still being loaded with new nodules; e.g. for first nodule,
#   intersection is only checked against 1st nodule, then in iteration for 2nd nodule negatives are checked for 
#   intersection against both 1st and 2nd nodule, and so on; So the info on ALL nodules should be first loaded, 
#   and only then the intersection should be checked!
#
patch_size = (44,44,12) #(36,36,8)
patch_str = ''.join([str(x) for x in patch_size]) #e.g. get '28288' from (28,28,8)
transFlag = 1 #whether to also augment w transformed nodules; 0 will only augment w flip/rotate nodules; if 1, the transformed nodules written in separate directory

pos_output_path = os.path.join('/diskStation/LIDC', patch_str, 'pos_' + patch_str)
neg_output_path = os.path.join('/diskStation/LIDC', patch_str, 'neg_smp_0_' + patch_str)
aug_output_path = os.path.join('/diskStation/LIDC', patch_str, 'pos_aug_0_' + patch_str)
aug_aux_output_path = os.path.join('/diskStation/LIDC', patch_str, 'pos_aug_aux_' + patch_str) #if transFlag==1, transformed nodules written to this

#pos_output_path = os.path.join('/diskStation/LIDC', patch_str, 'pos_' + patch_str + '_test')
#neg_output_path = os.path.join('/diskStation/LIDC', patch_str, 'neg_smp_0_' + patch_str + '_test')
#aug_output_path = os.path.join('/diskStation/LIDC', patch_str, 'pos_aug_0_' + patch_str + '_test')
#aug_aux_output_path = os.path.join('/diskStation/LIDC', patch_str, 'pos_aug_aux_' + patch_str + '_test') #if transFlag==1, transformed nodules written to this
numpy_master_case_path='/diskStation/LIDC/LIDC_NUMPY_3d'
lidcPath='/raida/apezeshk/lung_dicom_dir/'

mat_pre='uniqueStats_'
lidc_case_list=os.listdir(numpy_master_case_path)
# lidc_sample=['p0049_20000101_s3000627.npy']



#This is the nodule class and keeps all the necessary information about each nodule
class can_nudul(object):
    def __init__(self,case_id,x,y,z,x_size,y_size,z_size, avgCentroidX,avgCentroidY,avgCentroidZ,IURatio,ymarg=0,zmarg=0):
        self.case_id=case_id
        self.x=x #the way he passes the arguments, these 3 are minY, minX, minZ for that nodule in uniqueStats
        self.y=y
        self.z=z
        self.x_size= x_size #the way he passes the arguments, these 3 are maxY, maxX, maxZ for that nodule in uniqueStats
        self.y_size = y_size
        self.z_size = z_size
        self.avgCentroidX = avgCentroidX
        self.avgCentroidY = avgCentroidY
        self.avgCentroidZ = avgCentroidZ
        self.IURatio=IURatio #if it is zero means nodule is smaller than 3mm
    def cal_siz(self): #this caculates the size of the nodule
        weight=(self.x_size-self.x+1)
        height=(self.y_size-self.y+1)
        depth=(self.z_size-self.z+1)
        return (weight*height*depth)
    def volum_size(self):# This returns the volum wieght,heigh and depth
        return (self.x_size-self.x+1),(self.y_size-self.y+1),(self.z_size-self.z+1)

class can_nudul_pos_neg(object):#this is same as the other except it does not have the centroid info of the nodule
    def __init__(self,x,y,z,x_size,y_size,z_size,IURatio=0):

        self.x=x #the way he passes the arguments, these 3 are minY, minX, minZ for that nodule in uniqueStats
        self.y=y
        self.z=z
        self.x_size= x_size #the way he passes the arguments, these 3 are maxY, maxX, maxZ for that nodule in uniqueStats
        self.y_size = y_size
        self.z_size = z_size
        self.IURatio = IURatio
    def cal_siz(self): #this caculates the size of the nodule
        weight=(self.x_size-self.x+1)
        height=(self.y_size-self.y+1)
        depth=(self.z_size-self.z+1)
        return (weight*height*depth)
    def volum_size(self):# This returns the volum wieght,heigh and depth
        return (self.x_size-self.x+1),(self.y_size-self.y+1),(self.z_size-self.z+1)
def path_creat(file_name):
    spl_dir=file_name[:].replace('_','/')
    return spl_dir
    
#def pick_from_volum(input_array,can_nudul):
#    x=can_nudul.x
#    y=can_nudul.y
#    z=can_nudul.z


def crop_3d(xcen,ycen,zcen,input_np,x_viggle=patch_size[0]/2,yviggla=patch_size[1]/2,zviggla=patch_size[2]/2):
    ArrayDicom = np.zeros(patch_size, dtype=float)
    ArrayDicom[:,:,:]=input_np[(int(xcen)-int(x_viggle)):int(xcen)+int(x_viggle),(int(ycen)-int(yviggla)):(int(ycen)+int(yviggla)),(int(zcen)-int(zviggla)):(int(zcen)+int(zviggla))]
    return ArrayDicom

#########################################################################
#this function does the data augmentation with flipping & rotating
# Seven possible conditions can be generated here
#Number of rotation(1-3) Flip number(1-2)
#########################################################################
def aug_mat(input_3d,aug_type=None,NumberofRotation=None,flipnumber=None):
    if aug_type=='rotate':
        rot_mat=np.rot90(input_3d,NumberofRotation)
        return rot_mat
    elif aug_type=='flip' and flipnumber==1:
        flip_mat=np.fliplr(input_3d)
        return flip_mat
    elif aug_type=='flip' and flipnumber ==2:
        flip_mat=np.flipud(input_3d)
        return flip_mat
    elif aug_type=='both' and flipnumber==1:
        flip_rot=np.rot90(input_3d,NumberofRotation)
        flip_mat=np.fliplr(flip_rot)
        return flip_mat
    elif aug_type=='both' and flipnumber==2:
        flip_rot=np.rot90(input_3d,NumberofRotation)
        flip_mat=np.flipud(flip_rot)
        return flip_mat 
    elif aug_type=='both' and NumberofRotation==2 and flipnumber==1:
        flip_mat = np.fliplr(np.flipud(np.rot90(input_3d, NumberofRotation)))
        return flip_mat
    else:
        return input_3d
        
def save_aug_case(pth, matrix):
    np.save(pth + "_r11", aug_mat(matrix, 'rotate', 1, 1))
    np.save(pth + "_r31", aug_mat(matrix, 'rotate', 3, 1))
    np.save(pth + "_r21", aug_mat(matrix, 'rotate', 2, 1))
    np.save(pth + "_f11", aug_mat(matrix, 'flip', 1, 1))
    np.save(pth + "_f12", aug_mat(matrix, 'flip', 1, 2))
    np.save(pth + "_b11", aug_mat(matrix, 'both', 1, 1))
    np.save(pth + "_b12", aug_mat(matrix, 'both', 1, 2))
    np.save(pth + "_b21", aug_mat(matrix, 'both', 2, 1)) #NEW: added 4/26/2017
    
#########################################################################
#these functions do the data augmentation by applying various
#transformations (combo of rotation, size scaling, horizontal shear)
#########################################################################
#Takes THE RELEVANT SLICES, LOCATION OF NODULE, THEN APPL TRANSFORMATIONS WITH 
#DIFFERENT PARAMETERS, AND SAVE THE TRANSFORMED PATCHES;
def crop_relevantSlices(zcen, input_np, patchSize):
    #Returns slices of the ct that contain the nodule; number of slices returned
    #is dictated by number of slices of "patchSize"; NOTE that the output is float, same
    #situation as "crop_3d" fn.
    relevantSlices = np.zeros((input_np.shape[0], input_np.shape[1], patchSize[2]), dtype=float)
    zviggle = patchSize[2]/2
    relevantSlices[:,:,:]=input_np[:, :,(int(zcen)-int(zviggle)):(int(zcen)+int(zviggle))]
    return relevantSlices   
   
def Aug_trans(relevantSlices, aug_transParams):
    #Applies various transformations to full slices containing a nodule, then extracts the transformed nodule,
    #and writes the transformed nodules to an output directory. Transformations are combo of rotation, size scale, 
    #& horizontal shear;
    #Inputs: (the last 3 inputs listed below are within fields of aug_transParams)
    #   relevantSlices: full slices of ct containing a particular nodule, type float64, with same number of slices as patchSize[2]
    #   noduleCentroid: array containing centroid info of nodule (row,col,slice); used to locate it within relevantSlices
    #   patchSize: tuple containing patchSize info (3 elements, for height/width/slices)
    #   aug_transPath: pathname of folder to write the transformed nodules into
    #Outputs:
    #   Will write all the transformed patches (based on how many elements in transParamArray) to an output directory

    #Note: scikit.AffineTransform says rotation and shear are in radians, but if I give angle in degrees
    #for rotation (e.g. -90) it will do the correct rotation (i.e. not in radians!!!) For shear it doesn't make any sense
    #what is happening! It just applies horizontal shear, and it is not related to radians at all...
    transParamArray = np.array([[-60, 0.75, -0.15], 
                            [60, 1.25, 0.15], 
                            [-120, 0.8, -0.2], 
                            [120, 1.2, 0.2], #from your 2016 TMI paper, sans the contrast param; rotation/size scale/horizontal shear
                            [30, 1.15, 0.1],
                            [-30, 0.85, -0.1],
                            [-15, 0.9, -0.05],
                            [15, 1.1, 0.05]]) #and 4 new ones
                            
    noduleCentroid = aug_transParams['noduleCentroid']
    patchSize = aug_transParams['patchSize']
    aug_transPath = aug_transParams['aug_transPath']
    case_id = aug_transParams['case_id'] #this is the patient identifier + '_' + (noduleTag - 1)
    
    centerPoint = np.array((int(noduleCentroid[0]), int(noduleCentroid[1]))) #center point of nodule within the x/y plane: row,col
    #rectPos: 1st two elements are row/col of top left of bbox centered on nodule; is used to find the 
    #coordinates of bbox and thereby centerpoint of nodule after the transformation, so that patch can
    #be centered on correct location;
    rectPos = [int(centerPoint[0]-0.5*patchSize[0]), int(centerPoint[1]-0.5*patchSize[1]),
               patchSize[0], patchSize[1]] 
               
    array_int16 = np.zeros((2,2), dtype='int16') #just so that we can use its dtype to make sure relevantSlices also float64
    #centerPoint = np.array((int(rectPos[0]+.5*rectPos[2]), int(rectPos[1]+.5*rectPos[3])))
    
    for indParamArray in range(transParamArray.shape[0]):    
        angle = transParamArray[indParamArray, 0]
        scaleFactor = transParamArray[indParamArray, 1]
        shearFactor = transParamArray[indParamArray, 2]
        #scaleFactor = 1.0
        #shearFactor = 0.2
        #angle = 30
        
        #rectPos = [348, 296, 50, 50] #actual row/col of top left, and patchSize                
        for i in range(relevantSlices.shape[2]):
            #For each slice, apply the current transformation parameters to full slices
            currentSlice = relevantSlices[:,:,i]
            
            if relevantSlices.dtype == array_int16.dtype:
                #Rotation, etc. turn image into float and normalize to (0,1) if input is not float;
                #In that case, you need to switch back to correct scale so you will need to know min/max;
                #If image is already float, those operations will not affect image and it will retain its original range.
                imageMin = currentSlice.min()
                imageMax = currentSlice.max()
            
            rotateImage = tf.rotate(currentSlice, angle=angle, resize=True) #note: Unlike matlab version, rotate around center; otherwise output image may clip parts of image
            #rotateFake = tf.rotate(fakeImage, angle=angle, resize=True)
            
            #rotateImage = tf.rotate(relevantSlices, angle=angle, resize=True, center=(centerPoint[1], centerPoint[0])) #note: center for fn is in matlab image coordinates, not row/col!!
            #rotateFake = tf.rotate(fakeImage, angle=angle, resize=True, center=(centerPoint[1], centerPoint[0]))
            
            tfScale = tf.AffineTransform(scale=(1.0/scaleFactor, 1.0/scaleFactor)) #for some reason affine trans takes inverse of desired transformation as input
            scaleImage = tf.warp(rotateImage, tfScale, output_shape = (int(scaleFactor*rotateImage.shape[0]), int(scaleFactor*rotateImage.shape[1])))
            #scaleFake = tf.warp(rotateFake, tfScale, output_shape = (int(scaleFactor*rotateImage.shape[0]), int(scaleFactor*rotateImage.shape[1])))
            
            
            tfShear = tf.AffineTransform(shear = shearFactor)
            shearImage = tf.warp(scaleImage, tfShear)
            #shearFake = tf.warp(scaleFake, tfShear) #not using the output_size option, somehow the sheared image won't be centered in it
                
            if i==0: #TO MAKE THINGS RUN FASTER, calculate UPDATED CENTERPOINTNEW ONLY FOR SINGLE SLICE
                fakeImage = np.zeros((np.shape(currentSlice)[0], np.shape(currentSlice)[1]))
                fakeImage[rectPos[0]:(rectPos[0]+rectPos[2]), rectPos[1]:(rectPos[1]+rectPos[3])] = 1    
                rotateFake = tf.rotate(fakeImage, angle=angle, resize=True)
                scaleFake = tf.warp(rotateFake, tfScale, output_shape = (int(scaleFactor*rotateImage.shape[0]), int(scaleFactor*rotateImage.shape[1])))
                shearFake = tf.warp(scaleFake, tfShear) #not using the output_size option, somehow the sheared image won't be centered in it
                shearFake = shearFake.astype('bool')
                [row, col] = np.where(shearFake==1)
                rectPosNew = [min(row), min(col), max(row)-min(row)+1, max(col)-min(col)+1] #this defines the transformed box
                centerPointNew = np.array((int(rectPosNew[0]+.5*rectPosNew[2]), int(rectPosNew[1]+.5*rectPosNew[3]))) #find the center of the box
                
                #initialize output size in first iteration of loop
                procImage = np.zeros((shearFake.shape[0], shearFake.shape[1], relevantSlices.shape[2]), dtype = 'float64')
           
            procImage[:,:,i] = shearImage.copy() 
            if relevantSlices.dtype == array_int16.dtype:
                #>>>crop_3d fn returns a patch of type float, and a float is what gets written
                #out; so in the end float type is forced, but good to do the conversion back to original dtype
                #(bc rotation, etc result in normalized to 0,1 type float image) before that step for consistency
                procImage[:,:,i] = (imageMin + shearImage * (imageMax-imageMin)).astype('float64')
            
        cropTrans = np.zeros(patchSize, dtype=float) #this is important; bc crop_3d also does this, & vol is written as float
        cropTrans[:,:,:]=procImage[int(centerPointNew[0]-patchSize[0]/2):int(centerPointNew[0]+patchSize[0]/2), int(centerPointNew[1]-patchSize[1]/2):int(centerPointNew[1]+patchSize[1]/2),:]
        np.save(os.path.join(aug_transPath, case_id + '_m' + "%02d" % (indParamArray,)), cropTrans)
 
#########################################################################
#ensure_dir
#Creates direcotry if doesnt exist
#########################################################################
def ensure_dir(f):
    #d = os.path.dirname(f)
    if not os.path.exists(f):
        os.makedirs(f)
        
ensure_dir(pos_output_path), ensure_dir(neg_output_path)
ensure_dir(aug_output_path), ensure_dir(aug_aux_output_path)

def calculateintersect(cube1,cube2): #See comments in can_nodule above for how these args are actually defined
    x_overlap = max(0, min(cube1.x_size, cube2.x_size) - max(cube1.x, cube2.x))
    y_overlap = max(0, min(cube1.y_size, cube2.y_size) - max(cube1.y, cube2.y))
    z_overlap = max(0, min(cube1.z_size, cube2.z_size) - max(cube1.z, cube2.z))
    return abs(x_overlap * y_overlap * z_overlap)



#def path_creat(file_name):
#    spl_dir=file_name[:].replace('_','/')
#    return spl_dir
nudulsize={}
case_num=1
for case in lidc_case_list: #lidc_case_list has elements like 'p0049_20000101_s3000627.npy'
    pos_nodules_in_each_case=[]
    mat_dir=lidcPath+path_creat(case)[:-4]
    mat_name=mat_pre+case[:-4]+".mat"
    if os.path.exists(os.path.join(mat_dir, mat_name)):
        mat_contents = sio.loadmat(os.path.join(mat_dir, mat_name))
        oct_struct=mat_contents['uniqueStats']
        input_3d_npy = np.load(os.path.join(numpy_master_case_path, case))
        input_3d_npy = input_3d_npy.astype('int16') #for cases that are uint16
        
        for cases_ind in range(len(mat_contents["uniqueStats"])): #this is looping over nodules in uniqueStats 
            # print (oct_struct[cases_ind]["CasePath"][0][0].replace('/','_')[31:]+'_'+str(cases_ind) ), #creating unique is for pat
            case_id=oct_struct[cases_ind]["CasePath"][0][0].replace('/','_')[len(lidcPath):]+'_'+str(cases_ind)
            case_y= oct_struct[cases_ind]["minX"][0][0][0]
            case_x= oct_struct[cases_ind]["minY"][0][0][0]
            case_z= oct_struct[cases_ind]["minZ"][0][0][0]
            case_y_max= oct_struct[cases_ind]["maxX"][0][0][0]                                   # case_n=can_nudul(case_id,cases_ind)
            case_x_max= oct_struct[cases_ind]["maxY"][0][0][0]
            case_z_max= oct_struct[cases_ind]["maxZ"][0][0][0]
            case_y_avg= oct_struct[cases_ind]["avgCentroidX"][0][0][0] #Note that these are switched, e.g. case_Y_avg is avgCentroidX (bc the saved info is in matlab image coordinates)
            case_x_avg= oct_struct[cases_ind]["avgCentroidY"][0][0][0]
            case_z_avg= oct_struct[cases_ind]["avgCentroidZ"][0][0][0]
            case_IURatio=oct_struct[cases_ind]["IURatio"][0][0][0]
            my_nudule= can_nudul(case_id,case_x,case_y,case_z,case_x_max,case_y_max,case_z_max,case_x_avg,case_y_avg,case_z_avg,case_IURatio)
           
            #input_3d_npy = np.load(os.path.join(numpy_master_case_path, case))                        
            if my_nudule.IURatio == 0:
                print "<3mm lesion, will not extract!"

            if my_nudule.IURatio !=0:
                # NOTE: Up to and including SPIE, The commented block below had two problems, first: this was
                # within the loop adding each nodule info to pos_nodules_in_each_case; the negatives were then 
                # being extracted within same iteration, based on whether they intersected with current list of nodules
                #  (they should have been compared against info on ALL nodules); 2nd: if current pos patch could not
                # be extracted, the code would have printed an error, but written out an empty array anyway!!!
            
                #print my_nudule.IURatio
#                emty_arry = np.zeros(patch_size, dtype=float)
#                try:
#                    emty_arry[:, :, :] = crop_3d(case_x_avg, case_y_avg, case_z_avg, input_3d_npy)
#                except:
#                    print("case",case,"couldn't be made ")
#                np.save(pos_output_path  + case_id, emty_arry)#saving the nodule itself
#                save_aug_case(aug_output_path + case_id, emty_arry)
                pos_nodules_in_each_case.append(my_nudule)

        for currentNodInfo in pos_nodules_in_each_case:
            #for each nodule>3mm that was added to pos_nodules_in_each_case, extract the pos patch; 
            #Then use random x,y,z
            #coordinates to define a candidate neg patch; Check the candidate against every nodule coordinates
            #to make sure it has no overlap, if that condition is met extract and save the neg patch;
            #Note: Up to and including SPIE, this was using the avgCentroidZ for z slice, and then random x,y 
            emty_arry = np.zeros(patch_size, dtype=float)
            try:
                case_x_avg = currentNodInfo.avgCentroidX #row/col/slice of avg centroid
                case_y_avg = currentNodInfo.avgCentroidY
                case_z_avg = currentNodInfo.avgCentroidZ
                case_id = currentNodInfo.case_id
                emty_arry[:, :, :] = crop_3d(case_x_avg, case_y_avg, case_z_avg, input_3d_npy)
                np.save(os.path.join(pos_output_path, case_id), emty_arry)#saving the nodule itself
                save_aug_case(os.path.join(aug_output_path, case_id), emty_arry)
                
                if transFlag == 1:
                    relevantSlices = crop_relevantSlices(case_z_avg, input_3d_npy, patch_size)
                    aug_transParams = {}
                    aug_transParams['noduleCentroid'] = np.array((case_x_avg, case_y_avg, case_z_avg)) 
                    aug_transParams['patchSize'] = patch_size
                    aug_transParams['aug_transPath'] = aug_aux_output_path 
                    aug_transParams['case_id'] = case_id
                    Aug_trans(relevantSlices, aug_transParams)
            except KeyboardInterrupt:
                        print('Manual keyboard interrupt, aborting!')
                        sys.exit(0) 
            except:
                print("case",currentNodInfo.case_id,"couldn't be made ") #case_id combines patient identifier & nodule tag -1
                continue
            
            ind = 1
            #z = currentNodInfo.avgCentroidZ
            for z in xrange(randint(int(patch_size[2]/2), 30), input_3d_npy.shape[2]-int(patch_size[2]/2),randint(25,50)): 
                for y in xrange(randint(int(patch_size[1]/2),50), input_3d_npy.shape[1]-int(patch_size[1]/2), randint(50,150)):
                    for x in xrange(randint(int(patch_size[1]/2),50), input_3d_npy.shape[0]-int(patch_size[0]/2), randint(50,150)):
                        #window basically has the bbox of the candidate neg patch                    
                        window = can_nudul_pos_neg(x, y, z, x + patch_size[0], y + patch_size[1],
                                                   z + patch_size[2])
                        print x,y,z
                        #flag=False
                        intersection=0 #this is the overal intersection area; for each candidate '-', check against every positive in that case; if no overlap with any, extract.
                        for items in pos_nodules_in_each_case:
                            intersection=int(calculateintersect(window,items)+intersection)
                        if intersection==0:
                            neg_emty_arry=np.zeros(patch_size, dtype=float)
                            try:
                                neg_emty_arry[:, :, :] = crop_3d(x,y,z, input_3d_npy)
                                np.save(os.path.join(neg_output_path, case_id + '_' +str(x)+'_'+str(y)+'_'+str(z)+'_'+ str(ind)), neg_emty_arry)
                                ind += 1
                            except KeyboardInterrupt:
                                print('Manual keyboard interrupt, aborting!')
                                sys.exit(0) 
                            except:
                                print "Selected coordinates for negative patch cannot be cropped",x,y,z
            
#        try:            
#            ind = 1
#            z=case_z_avg
#            # for z in xrange(randint(0,40), input_3d_npy.shape[2]-int(patch_size[2]/2),randint(40,60)):  # this goes into each case and generates the negative cases
#            for y in xrange(randint(0,50), input_3d_npy.shape[1]-int(patch_size[1]/2), randint(50,200)):
#                for x in xrange(randint(0,100), input_3d_npy.shape[0]-int(patch_size[0]/2), randint(50,200)):
#                    window = can_nudul_pos_neg(x, y, z, x + patch_size[0], y + patch_size[1],
#                                               z + patch_size[2])
#                    print x,y,z
#                    flag=False
#                    intersection=0 #this is the overal intersection area; for each candidate '-', check against every positive in that case; if no overlap with any, extract.
#                    for items in pos_nodules_in_each_case:
#                        intersection=int(calculateintersect(window,items)+intersection)
#                    if intersection==0:
#                        neg_emty_arry=np.zeros(patch_size, dtype=float)
#                        try:
#                            neg_emty_arry[:, :, :] = crop_3d(x,y,z, input_3d_npy)
#                            np.save(neg_output_path + case_id + '_' +str(x)+'_'+str(y)+'_'+str(z)+'_'+ str(ind), neg_emty_arry)
#                            ind += 1
#                        except:
#                            print "selected coordinates wasnt match the input volume size to be croped",x,y,z
#                    else:
#                        print ("there is a overlap with posetive case")
#        except:
#            print case_id, "got error in negatives"
#            print sys.exc_info()


