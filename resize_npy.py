# Author: Xiang Xiang
# eglxiang@gmail.com
# 06/29/2017

import os
import cv2
import glob
import numpy as np
from PIL import Image
from resizeimage import resizeimage
#import pdb

# reduce the size by the rate of 
alpha = 0.80 # (x,y)
beta = 0.70 # number of slices

#pdb.set_trace()

rootpath = '/diskStation/LIDC/'
readfolder = 'LIDC_NUMPY_3d/'
savefolder = 'LIDC_NUMPY_3d_resized/' + "{:.2f}".format(alpha) + '_' + "{:.2f}".format(alpha) + '_' + "{:.2f}".format(beta)
os.mkdir((rootpath+savefolder), 0755)
lst_name = os.listdir(rootpath + readfolder)

for i in lst_name:
	filename = rootpath + readfolder + i
	currentVol = np.load(filename)
	currentVol = currentVol.astype('int16')
	[height,width,number] = currentVol.shape
	h_new = int( alpha*height )
	w_new = int( alpha*width )
	n_new = int( beta*number )
	newVol = np.zeros([h_new,w_new,n_new],dtype=np.int16)
	for n in range(0,n_new):
		# select which slice
		n_old = int((n+1)/beta)-1
		currentSlice = currentVol[:,:,n_old]
		# resize one slice
		for h in range(0,h_new):
			h_old = int((h+1)/alpha)-1
			for w in range(0,w_new):
				w_old = int((w+1)/alpha)-1
				newVol[h,w,n] = currentVol[h_old,w_old,n_old]

	# save the resized volume
	savepath = rootpath + savefolder + '/' + i
	np.save(savepath,newVol)

	# make folder different