# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 12:10:33 2017

@author: apezeshk
"""
from matplotlib import pyplot as plt
import numpy as np
#Use this to display a single slice of the original/transformed patch in one figure,
#and every slice of the original and the transformed copies in another figure. This can be used
#to verify the images look right both in terms of the transformations as well as the cropping
#s.t. the nodule is centered in the patch.
#The last few lines can also be used to verify that the simple rotation/flipped augmentations
#look alright, and that they are all different with one another.

#nodOrig = np.load('/diskStation/LIDC/36368/pos_36368_test/p0451_20000101_s3000315_0.npy')
nodOld =  np.load('/diskStation/LIDC/36368/pos_36368/p1012_20000101_s32231_0.npy').astype('int16')
nodTrans1 = np.load('/diskStation/LIDC/36368/pos_aug_aux_36368/p1012_20000101_s32231_0_m00.npy')
nodTrans2 = np.load('/diskStation/LIDC/36368/pos_aug_aux_36368/p1012_20000101_s32231_0_m01.npy')
nodTrans3 = np.load('/diskStation/LIDC/36368/pos_aug_aux_36368/p1012_20000101_s32231_0_m02.npy')
nodTrans4 = np.load('/diskStation/LIDC/36368/pos_aug_aux_36368/p1012_20000101_s32231_0_m03.npy')

nodTrans5 = np.load('/diskStation/LIDC/36368/pos_aug_aux_36368/p1012_20000101_s32231_0_m04.npy')
nodTrans6 = np.load('/diskStation/LIDC/36368/pos_aug_aux_36368/p1012_20000101_s32231_0_m05.npy')
nodTrans7 = np.load('/diskStation/LIDC/36368/pos_aug_aux_36368/p1012_20000101_s32231_0_m06.npy')
nodTrans8 = np.load('/diskStation/LIDC/36368/pos_aug_aux_36368/p1012_20000101_s32231_0_m07.npy')

i = 4; #slice number to be shown in the first figure
plt.figure(); 
plt.subplot(1,5,1); plt.imshow(nodOld[:,:,i], cmap = 'gray');
plt.subplot(1,5,2); plt.imshow(nodTrans1[:,:,i], cmap = 'gray');
plt.subplot(1,5,3); plt.imshow(nodTrans2[:,:,i], cmap = 'gray');
plt.subplot(1,5,4); plt.imshow(nodTrans3[:,:,i], cmap = 'gray');
plt.subplot(1,5,5); plt.imshow(nodTrans4[:,:,i], cmap = 'gray');

plt.figure()
plt.subplot(8,5,1); plt.imshow(nodOld[:,:,0], cmap = 'gray');
plt.subplot(8,5,6); plt.imshow(nodOld[:,:,1], cmap = 'gray');
plt.subplot(8,5,11); plt.imshow(nodOld[:,:,2], cmap = 'gray');
plt.subplot(8,5,16); plt.imshow(nodOld[:,:,3], cmap = 'gray');
plt.subplot(8,5,21); plt.imshow(nodOld[:,:,4], cmap = 'gray');
plt.subplot(8,5,26); plt.imshow(nodOld[:,:,5], cmap = 'gray');
plt.subplot(8,5,31); plt.imshow(nodOld[:,:,6], cmap = 'gray');
plt.subplot(8,5,36); plt.imshow(nodOld[:,:,7], cmap = 'gray');

plt.subplot(8,5,2); plt.imshow(nodTrans1[:,:,0], cmap = 'gray');
plt.subplot(8,5,7); plt.imshow(nodTrans1[:,:,1], cmap = 'gray');
plt.subplot(8,5,12); plt.imshow(nodTrans1[:,:,2], cmap = 'gray');
plt.subplot(8,5,17); plt.imshow(nodTrans1[:,:,3], cmap = 'gray');
plt.subplot(8,5,22); plt.imshow(nodTrans1[:,:,4], cmap = 'gray');
plt.subplot(8,5,27); plt.imshow(nodTrans1[:,:,5], cmap = 'gray');
plt.subplot(8,5,32); plt.imshow(nodTrans1[:,:,6], cmap = 'gray');
plt.subplot(8,5,37); plt.imshow(nodTrans1[:,:,7], cmap = 'gray');

plt.subplot(8,5,3); plt.imshow(nodTrans2[:,:,0], cmap = 'gray');
plt.subplot(8,5,8); plt.imshow(nodTrans2[:,:,1], cmap = 'gray');
plt.subplot(8,5,13); plt.imshow(nodTrans2[:,:,2], cmap = 'gray');
plt.subplot(8,5,18); plt.imshow(nodTrans2[:,:,3], cmap = 'gray');
plt.subplot(8,5,23); plt.imshow(nodTrans2[:,:,4], cmap = 'gray');
plt.subplot(8,5,28); plt.imshow(nodTrans2[:,:,5], cmap = 'gray');
plt.subplot(8,5,33); plt.imshow(nodTrans2[:,:,6], cmap = 'gray');
plt.subplot(8,5,38); plt.imshow(nodTrans2[:,:,7], cmap = 'gray');

plt.subplot(8,5,4); plt.imshow(nodTrans3[:,:,0], cmap = 'gray');
plt.subplot(8,5,9); plt.imshow(nodTrans3[:,:,1], cmap = 'gray');
plt.subplot(8,5,14); plt.imshow(nodTrans3[:,:,2], cmap = 'gray');
plt.subplot(8,5,19); plt.imshow(nodTrans3[:,:,3], cmap = 'gray');
plt.subplot(8,5,24); plt.imshow(nodTrans3[:,:,4], cmap = 'gray');
plt.subplot(8,5,29); plt.imshow(nodTrans3[:,:,5], cmap = 'gray');
plt.subplot(8,5,34); plt.imshow(nodTrans3[:,:,6], cmap = 'gray');
plt.subplot(8,5,39); plt.imshow(nodTrans3[:,:,7], cmap = 'gray');

plt.subplot(8,5,5); plt.imshow(nodTrans4[:,:,0], cmap = 'gray');
plt.subplot(8,5,10); plt.imshow(nodTrans4[:,:,1], cmap = 'gray');
plt.subplot(8,5,15); plt.imshow(nodTrans4[:,:,2], cmap = 'gray');
plt.subplot(8,5,20); plt.imshow(nodTrans4[:,:,3], cmap = 'gray');
plt.subplot(8,5,25); plt.imshow(nodTrans4[:,:,4], cmap = 'gray');
plt.subplot(8,5,30); plt.imshow(nodTrans4[:,:,5], cmap = 'gray');
plt.subplot(8,5,35); plt.imshow(nodTrans4[:,:,6], cmap = 'gray');
plt.subplot(8,5,40); plt.imshow(nodTrans4[:,:,7], cmap = 'gray');

plt.figure()
plt.subplot(8,5,1); plt.imshow(nodOld[:,:,0], cmap = 'gray');
plt.subplot(8,5,6); plt.imshow(nodOld[:,:,1], cmap = 'gray');
plt.subplot(8,5,11); plt.imshow(nodOld[:,:,2], cmap = 'gray');
plt.subplot(8,5,16); plt.imshow(nodOld[:,:,3], cmap = 'gray');
plt.subplot(8,5,21); plt.imshow(nodOld[:,:,4], cmap = 'gray');
plt.subplot(8,5,26); plt.imshow(nodOld[:,:,5], cmap = 'gray');
plt.subplot(8,5,31); plt.imshow(nodOld[:,:,6], cmap = 'gray');
plt.subplot(8,5,36); plt.imshow(nodOld[:,:,7], cmap = 'gray');

plt.subplot(8,5,2); plt.imshow(nodTrans5[:,:,0], cmap = 'gray');
plt.subplot(8,5,7); plt.imshow(nodTrans5[:,:,1], cmap = 'gray');
plt.subplot(8,5,12); plt.imshow(nodTrans5[:,:,2], cmap = 'gray');
plt.subplot(8,5,17); plt.imshow(nodTrans5[:,:,3], cmap = 'gray');
plt.subplot(8,5,22); plt.imshow(nodTrans5[:,:,4], cmap = 'gray');
plt.subplot(8,5,27); plt.imshow(nodTrans5[:,:,5], cmap = 'gray');
plt.subplot(8,5,32); plt.imshow(nodTrans5[:,:,6], cmap = 'gray');
plt.subplot(8,5,37); plt.imshow(nodTrans5[:,:,7], cmap = 'gray');

plt.subplot(8,5,3); plt.imshow(nodTrans6[:,:,0], cmap = 'gray');
plt.subplot(8,5,8); plt.imshow(nodTrans6[:,:,1], cmap = 'gray');
plt.subplot(8,5,13); plt.imshow(nodTrans6[:,:,2], cmap = 'gray');
plt.subplot(8,5,18); plt.imshow(nodTrans6[:,:,3], cmap = 'gray');
plt.subplot(8,5,23); plt.imshow(nodTrans6[:,:,4], cmap = 'gray');
plt.subplot(8,5,28); plt.imshow(nodTrans6[:,:,5], cmap = 'gray');
plt.subplot(8,5,33); plt.imshow(nodTrans6[:,:,6], cmap = 'gray');
plt.subplot(8,5,38); plt.imshow(nodTrans6[:,:,7], cmap = 'gray');

plt.subplot(8,5,4); plt.imshow(nodTrans7[:,:,0], cmap = 'gray');
plt.subplot(8,5,9); plt.imshow(nodTrans7[:,:,1], cmap = 'gray');
plt.subplot(8,5,14); plt.imshow(nodTrans7[:,:,2], cmap = 'gray');
plt.subplot(8,5,19); plt.imshow(nodTrans7[:,:,3], cmap = 'gray');
plt.subplot(8,5,24); plt.imshow(nodTrans7[:,:,4], cmap = 'gray');
plt.subplot(8,5,29); plt.imshow(nodTrans7[:,:,5], cmap = 'gray');
plt.subplot(8,5,34); plt.imshow(nodTrans7[:,:,6], cmap = 'gray');
plt.subplot(8,5,39); plt.imshow(nodTrans7[:,:,7], cmap = 'gray');

plt.subplot(8,5,5); plt.imshow(nodTrans8[:,:,0], cmap = 'gray');
plt.subplot(8,5,10); plt.imshow(nodTrans8[:,:,1], cmap = 'gray');
plt.subplot(8,5,15); plt.imshow(nodTrans8[:,:,2], cmap = 'gray');
plt.subplot(8,5,20); plt.imshow(nodTrans8[:,:,3], cmap = 'gray');
plt.subplot(8,5,25); plt.imshow(nodTrans8[:,:,4], cmap = 'gray');
plt.subplot(8,5,30); plt.imshow(nodTrans8[:,:,5], cmap = 'gray');
plt.subplot(8,5,35); plt.imshow(nodTrans8[:,:,6], cmap = 'gray');
plt.subplot(8,5,40); plt.imshow(nodTrans8[:,:,7], cmap = 'gray');

################################################################################
# This part is to display the regular augmentations (simple flip/rotations)
################################################################################
plt.figure;
nodSimple1 = np.load('/diskStation/LIDC/36368/pos_aug_0_36368/p1012_20000101_s32231_0_r11.npy')
nodSimple2 = np.load('/diskStation/LIDC/36368/pos_aug_0_36368/p1012_20000101_s32231_0_r21.npy')
nodSimple3 = np.load('/diskStation/LIDC/36368/pos_aug_0_36368/p1012_20000101_s32231_0_r31.npy')
nodSimple4 = np.load('/diskStation/LIDC/36368/pos_aug_0_36368/p1012_20000101_s32231_0_f11.npy')
nodSimple5 = np.load('/diskStation/LIDC/36368/pos_aug_0_36368/p1012_20000101_s32231_0_f12.npy')
nodSimple6 = np.load('/diskStation/LIDC/36368/pos_aug_0_36368/p1012_20000101_s32231_0_b11.npy')
nodSimple7 = np.load('/diskStation/LIDC/36368/pos_aug_0_36368/p1012_20000101_s32231_0_b12.npy')
nodSimple8 = np.load('/diskStation/LIDC/36368/pos_aug_0_36368/p1012_20000101_s32231_0_b21.npy')
plt.subplot(2,4,1); plt.imshow(nodSimple1[:,:,i], cmap = 'gray');
plt.subplot(2,4,2); plt.imshow(nodSimple2[:,:,i], cmap = 'gray');
plt.subplot(2,4,3); plt.imshow(nodSimple3[:,:,i], cmap = 'gray');
plt.subplot(2,4,4); plt.imshow(nodSimple4[:,:,i], cmap = 'gray');
plt.subplot(2,4,5); plt.imshow(nodSimple5[:,:,i], cmap = 'gray');
plt.subplot(2,4,6); plt.imshow(nodSimple6[:,:,i], cmap = 'gray');
plt.subplot(2,4,7); plt.imshow(nodSimple7[:,:,i], cmap = 'gray');
plt.subplot(2,4,8); plt.imshow(nodSimple8[:,:,i], cmap = 'gray');