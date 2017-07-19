###################################################################################################
#Sardar Hamidian 07-1-2016
#Reading dicom files and creating 3D-numpy array
#Input 3D numpy array of dicom files
#Output 3D small samples for feeding CNN model
# NOTE: Unlike gen_pos_neg_aug_samples, this doesn't check to see if the neg patches
# have overlap with the positive patches (there may be other differences too!!!)
###################################################################################################
import numpy as np
import os
import sys
import scipy.io as sio
from random import randint

output='/diskstation/LIDC/404010/pos_404010/'
neg_output='/diskstation/LIDC/404010/neg_aug_0_404010/'
output_aug='/diskstation/LIDC/404010/pos_aug_0_404010/'
neg_output_aug='/diskstation/LIDC/404010/neg_aug_1_404010/'
input='/diskstation/LIDC/LIDC_NUMPY_3d'
small_nodules='/diskstation/LIDC/404010/z_nodules/'
mat_file='/raida/apezeshk/lung_dicom_dir/'
mat_pre='uniqueStats_'
lidc_sample=os.listdir(input)

patch_size = (40,40,10)

#This is the nodule class and keeps all the neccery information about each nodule
class can_nudul(object):
    def __init__(self,case_id,x,y,z,x_size,y_size,z_size, avgCentroidX,avgCentroidY,avgCentroidZ,IURatio,ymarg=0,zmarg=0):
        self.case_id=case_id
        self.x=x
        self.y=y
        self.z=z
        self.x_size= x_size
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


def pick_from_volum(input_array,can_nudul):
    x=can_nudul.x
    y=can_nudul.y
    z=can_nudul.z


def crop_3d(xcen,ycen,zcen,input_np,x_viggle=patch_size[0]/2,yviggla=patch_size[1]/2,zviggla=patch_size[2]/2):
    ArrayDicom = np.zeros(patch_size, dtype=float)
    ArrayDicom[:,:,:]=input_np[(int(xcen)-int(x_viggle)):int(xcen)+int(x_viggle),(int(ycen)-int(yviggla)):(int(ycen)+int(yviggla)),(int(zcen)-int(zviggla)):(int(zcen)+int(zviggla))]
    return ArrayDicom

#this function does the data augmentation
# Seven possible conditions can be generated here
#Number of rotation(1-3) Flip number(1-2)
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
    else:
        return input_3d

#########################################################################
#ensure_dir
#Creates direcotry if doesnt exist
#########################################################################
def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)
ensure_dir(output), ensure_dir(neg_output),ensure_dir(output_aug),ensure_dir(neg_output_aug),ensure_dir(small_nodules)

def save_aug_case(pth, matrix):
    np.save(pth + "_r11", aug_mat(matrix, 'rotate', 1, 1))
    np.save(pth + "_r31", aug_mat(matrix, 'rotate', 3, 1))
    np.save(pth + "_r21", aug_mat(matrix, 'rotate', 2, 1))
    np.save(pth + "_f11", aug_mat(matrix, 'flip', 1, 1))
    np.save(pth + "_f12", aug_mat(matrix, 'flip', 1, 2))
    np.save(pth + "_b11", aug_mat(matrix, 'both', 1, 1))
    np.save(pth + "_b12", aug_mat(matrix, 'both', 1, 2))

def path_creat(file_name):
    spl_dir=file_name[:].replace('_','/')
    return spl_dir
nudulsize={}
case_num=1
for case in lidc_sample:
    mat_dir=mat_file+path_creat(case)[:-4]
    mat_name=mat_pre+case[:-4]+".mat"
    if os.path.exists(mat_dir+'/'+mat_name):
        mat_contents = sio.loadmat(mat_dir+'/'+mat_name)
        oct_struct=mat_contents['uniqueStats']
        for cases_ind in range(len(mat_contents["uniqueStats"])):
            # print (oct_struct[cases_ind]["CasePath"][0][0].replace('/','_')[31:]+'_'+str(cases_ind) ), #creating unique is for pat
            case_id=oct_struct[cases_ind]["CasePath"][0][0].replace('/','_')[31:]+'_'+str(cases_ind)
            case_x= oct_struct[cases_ind]["minX"][0][0][0]
            case_y= oct_struct[cases_ind]["minY"][0][0][0]
            case_z= oct_struct[cases_ind]["minZ"][0][0][0]
            case_x_max= oct_struct[cases_ind]["maxX"][0][0][0]                                   # case_n=can_nudul(case_id,cases_ind)
            case_y_max= oct_struct[cases_ind]["maxY"][0][0][0]
            case_z_max= oct_struct[cases_ind]["maxZ"][0][0][0]
            case_x_avg= oct_struct[cases_ind]["avgCentroidX"][0][0][0]                                   # case_n=can_nudul(case_id,cases_ind)
            case_y_avg= oct_struct[cases_ind]["avgCentroidY"][0][0][0]
            case_z_avg= oct_struct[cases_ind]["avgCentroidZ"][0][0][0]
            case_IURatio=oct_struct[cases_ind]["IURatio"][0][0][0]
            my_nudule= can_nudul(case_id,case_x,case_y,case_z,case_x_max,case_y_max,case_z_max,case_x_avg,case_y_avg,case_z_avg,case_IURatio)
            # print can_nudul.cal_siz(my_nudule)
            # print can_nudul.volum_size(my_nudule)
            # nudulsize[case_id]= can_nudul.cal_siz(my_nudule)
            inpu_3d_npy = np.load(input+'/'+case)
            # print my_nudule.IVRatio
            #This is for the nodules size = zero save that in anothe rdirector
            if my_nudule.IURatio != 0:
                emty_arry = np.zeros(patch_size, dtype=float)
                try:
                    emty_arry[:, :, :] = crop_3d(case_y_avg, case_x_avg, case_z_avg, inpu_3d_npy)
                    np.save(small_nodules + case_id, emty_arry)
                except:
                    print case_id, "got error"
                    print sys.exc_info()


            if my_nudule.IURatio !=0:
                print my_nudule.IURatio
                emty_arry = np.zeros(patch_size, dtype=float)
                neg_emty_arry = np.zeros(patch_size, dtype=float)
                try:

                    neg_emty_arry[:, :, :] = crop_3d(case_y_avg + randint(40, 60), case_x_avg + randint(40, 60),
                                                     case_z_avg + randint(10, 15), inpu_3d_npy)
                    emty_arry[:,:,:]=crop_3d(case_y_avg,case_x_avg,case_z_avg,inpu_3d_npy)#y and X are swiched to align the matlabe cordinate
                    np.save(output  + case_id, emty_arry)#saving the nodule itself
                    np.save(neg_output + case_id, neg_emty_arry)#saving the negative sample
                    save_aug_case(output_aug  + case_id, emty_arry)#using the aug function to flip and rotate and save the posetive
                    save_aug_case(neg_output_aug + case_id, neg_emty_arry)#using aug function for negative samples

                except:
                    print case_id, "got error"
                    print sys.exc_info()
# imgplot = plt.imsave("sardar", inpu_3d_npy[0:512,0:512,int(case_z_avg)])