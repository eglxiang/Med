# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 13:55:48 2016

@author: apezeshk
"""
#Plots different channels of kernels in different layers in 3x8 grid (each column shows different channels of same kernel)
savedNetwork = lasagne.layers.get_all_params(network_fcn) #this gives list of network weight/biases
w0 = savedNetwork[0].eval() #weights in 1st conv layer
w1 = savedNetwork[2].eval() #weights in 2nd conv layer
for j in range(0,4):
    for i in range(0,8):
        plt.figure(j)    
        plt.subplot(3,8,1+i)
        plt.imshow(w1[j*8+i,0,:,:,0], cmap = 'gray')
        plt.axis('off')
        plt.subplot(3,8,9+i)
        plt.imshow(w1[j*8+i,0,:,:,1], cmap = 'gray')
        plt.axis('off')
        plt.subplot(3,8,17+i)
        plt.imshow(w1[j*8+i,0,:,:,2], cmap = 'gray')
        plt.axis('off')
        
    plt.tight_layout()
    plt.suptitle('Layer 2 features: each column shows different channels of same filter')
    

#this one does same as above, just different subplot format (8x3) where each row shows the channels of same kernel    
#for j in range(0,4):
#    for i in range(0,8):
#        plt.figure(j)    
#        plt.subplot(8,3,i*3+1)
#        plt.imshow(w0[j*8+i,0,:,:,0], cmap = 'gray')
#        plt.axis('off')
#        plt.subplot(8,3,i*3+2)
#        plt.imshow(w0[j*8+i,0,:,:,1], cmap = 'gray')
#        plt.axis('off')
#        plt.subplot(8,3,i*3+3)
#        plt.imshow(w0[j*8+i,0,:,:,2], cmap = 'gray')
#        plt.axis('off')
#        
#    plt.tight_layout()
    