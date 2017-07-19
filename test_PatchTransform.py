# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 14:54:03 2017

@author: apezeshk
"""
from skimage import io
from skimage import transform as tf
from matplotlib import pyplot as plt
import numpy as np
import time
# X NOTE THAT PROCS BELOW CHANGE IMAGE FROM UINT TO FLOAT & normalize to 0,1!!! SO GO BACK TO UINT OR WHATEVER TYPE IN THE END
#>>>Actually in gen_pos_neg_aug_samples.py, the crop_3d fn returns a patch of type float, and a float is what gets written
#out; so in the end float type is forced, but good to do the conversion before that step for consistency anyway

# X SHOULD I ALSO APPLY THE OUTPUTSIZE PARAM WHEN DOING WARP FOR SHEAR? (SIMILAR TO HOW IT IS DONE FOR ROTATE/SCALE)

# X TO MAKE THINGS RUN FASTER, DO THE CALCULATION OF THE UPDATED CENTERPOINTNEW ONLY FOR SINGLE SLICE, AND NOT
#WITHIN THE LOOP (IT WON'T CHANGE ACROSS SLICES, SO NO NEED TO REPEAT THE TRANSFORMATIONS!)
start_time = time.time()

#source = io.imread('/raida/apezeshk/TERMINATOR.jpg')
sourceCase = np.load('/diskStation/LIDC/LIDC_NUMPY_3d/p0001_20000101_s3000566.npy')
sourceCase = sourceCase.astype('int16')

source = sourceCase[:,:,39:46]

#Note: scikit.AffineTransform says rotation and shear are in radians, but if I give angle in degrees
#for rotation (e.g. -90) it will do the correct rotation (i.e. not in radians!!!); For shear it doesn't make any sense
#what is happening! It just applies horizontal shear, and it is not related to radians at all...
transParamArray = np.array([[-60, 0.75, -0.15], 
                            [60, 1.25, 0.15], 
                            [-120, 0.8, -0.2], 
                            [-15, 1.1, 0.05]]) #from your 2016 TMI paper, sans the contrast param; rotation/size scale/horizontal shear
indParamArray = 3 #pick which transformation to use based on index here

angle = transParamArray[indParamArray, 0]
scaleFactor = transParamArray[indParamArray, 1]
shearFactor = transParamArray[indParamArray, 2]
#scaleFactor = 1.0
#shearFactor = 0.2
#angle = 30

rectPos = [348, 296, 50, 50] #actual row/col of top left, and patchSize
    
#centerPoint = np.array((arnold.shape[0]/2, arnold.shape[1]/2))
centerPoint = np.array((int(rectPos[0]+.5*rectPos[2]), int(rectPos[1]+.5*rectPos[3]))) #point around which the image will be rotated; same as center of patch bbox

for i in range(source.shape[2]):
    currentSlice = source[:,:,i]
    imageMin = currentSlice.min()
    imageMax = currentSlice.max()
    
    rotateImage = tf.rotate(currentSlice, angle=angle, resize=True) #note: Unlike matlab version, rotate around center; otherwise output image may clip parts of image
    #rotateFake = tf.rotate(fakeImage, angle=angle, resize=True)
    
    #rotateImage = tf.rotate(source, angle=angle, resize=True, center=(centerPoint[1], centerPoint[0])) #note: center for fn is in matlab image coordinates, not row/col!!
    #rotateFake = tf.rotate(fakeImage, angle=angle, resize=True, center=(centerPoint[1], centerPoint[0]))
    
    tfScale = tf.AffineTransform(scale=(1.0/scaleFactor, 1.0/scaleFactor))
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
        procImage = np.zeros((shearFake.shape[0], shearFake.shape[1], source.shape[2]), dtype = 'int16')
       
    procImage[:,:,i] = (imageMin + shearImage * (imageMax-imageMin)).astype('int16')
    #sourceTrans = tf.warp(rotateImage, tform)
    
    elapsed_time = time.time() - start_time
    print 'Elapsed time: ' + str(elapsed_time)
    
    plt.figure()
    plt.imshow(procImage[:,:,i], cmap='gray')
    
plt.figure()
io.imshow(shearFake)



#scaleMat = np.array(([[scaleFactor, 0, 0],[0, scaleFactor, 0],[0, 0, 1]]))
#shearMat = np.array(([[1, 0, 0],[shearFactor, 1, 0],[0, 0, 1]]))
#tformMat = np.dot(scaleMat, shearMat) #shear transform multiplied by scale transform
#tform = tf.AffineTransform(matrix=tformMat)  