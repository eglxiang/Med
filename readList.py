# this script reads in the list of one fold in cross validation
import os
import csv
import SupportFuncs
import pdb

# input parameters
folderpath = '/media/shamidian/sda2/ws1/folds_list_indexing'
numFolds = 10
# pos neg samples have been generated beforehand

# read the list directory
lst_name = os.listdir(folderpath)
# make sure no temporary files in the directory
if len(lst_name) == numFolds:
	train_x,train_y,test_x,test_y=[],[],[],[]
	# cross validation
	for k in range(0,numFolds): #the k-th fold witholds the k-th as test set
		# stacking the testing samples
		test_name =lst_name[k]
		filepath = folderpath + '/' + test_name
		with open(filepath, 'rb') as csvfile:
			reader = csv.reader(csvfile)
			for line in reader:
				test_x.append(line[0].replace('/','_'))
				test_y.append(line[1])

		# stacking the training samples
		for i in lst_name:
			if i != test_name:
				# train set
				filepath = folderpath + '/' + i
				with open(filepath, 'rb') as csvfile:
					reader = csv.reader(csvfile)
					for line in reader:
						train_x.append(line[0].replace('/','_'))
						train_y.append(line[1])

		# print the num of training/testing samples
		numTrainSamples=len(train_x)
		print('numTrainSamples=' + str(numTrainSamples))
		numTestSamples =len(test_x)
		print('numTestSamples=' + str(numTestSamples))

		# read correpsonding training set
		# set the name for finding the data
		load_data(inputParamsLoadData) 
		# search where is the data assigned
		
		# training

		# evaluation


else:
	print('Not 10 folders!\n')

# call in positive_negative generating
#for i in [1:numItems]:
#for item in x:


#csvfile.close()