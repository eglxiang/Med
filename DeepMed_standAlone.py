from __future__ import print_function

import sys
import os
import time
import numpy as np
import theano
from theano.compile.debugmode import DebugMode
import SupportFuncs
import theano.tensor as T
from lasagne.layers import dnn
import lasagne
import matplotlib.pyplot as plt
import csv
import matplotlib
import errno
import json

localFlag = 1 #if 1, reads from the parameter settings defined on top, otherwise reads the csv config file
LIDC_PATH= '/diskStation/LIDC/'
test_path = './test_test'  # positive test data
result_path = './results'  # csv file results
figures_path = './figures'

random_state = 1234
np.random.seed(random_state)
lasagne.layers.noise._srng = lasagne.layers.noise.RandomStreams(random_state)

########################
######Input Params######
inputParamsConfigLocal = {}
inputParamsConfigLocal['input_shape'] = '30, 30, 10'
inputParamsConfigLocal['learning_rate'] = '0.1'
inputParamsConfigLocal['momentum'] = '0.9'
inputParamsConfigLocal['num_epochs'] = '300'
inputParamsConfigLocal['batch_size'] = '100'
inputParamsConfigLocal['data_path'] = '/diskStation/LIDC/303010/'
inputParamsConfigLocal['train_set_size'] = '3000'
inputParamsConfigLocal['test_set_size'] = '500'
inputParamsConfigLocal['positive_set_ratio'] = '0.5'
inputParamsConfigLocal['dropout'] = '0.1'
inputParamsConfigLocal['nonlinearityToUse'] = 'sigmoid'
inputParamsConfigLocal['augmentationFlag'] = '1'
######Input Params######
########################

n_layers = 3
weight_init = lasagne.init.Normal()
epoch_det = {}  # storyin the epoc results including val acc,loss, and training loss
all_val_accuracy = []  # Validation_accuracy list is used for the plot
all_val_loss = []  # Validation_loss list is used for the plot
all_val_AUC = []  # list of AUC for validation set across each epoch
training_loss = []  # Training_loss list is used for the plot
csv_config=[] #if localFlag==0, code will read network parameters from a csv and store that in csv_config
experiment_id = str(time.strftime("%Y%m%d%H%M%S"))  # using the current date and time for the experiment name

print(os.path.join('./figures', experiment_id + '.png'))


# To make sure a direcotry exist
def Make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def Pathcheck(path):
    if os.path.isdir(os.path.join('../result', path)):
        print(path, "exist")
    else:
        Make_sure_path_exists(path)
        print(path, "Just created and is ready to use")


# Delting all the cases with nan value
def Rem_nan(path):
    for cases in os.listdir(path):
        case_mat = np.load(os.path.join(path, cases))
        if np.amax(case_mat) == np.amin(case_mat):
            print(cases)
            plt.imsave("./norm/" + str(cases[:-4]), case_mat[:, :, 8], cmap='gray')
            os.remove(os.path.join(path, cases))


def Norm(matrix):
    output = ((matrix - np.amin(matrix)) / (np.amax(matrix) - np.amin(matrix)))
    return output


# Gets input files size and all the parameterest that are used for thea model and save them as a row in csv files named as
# input files size

def Save_model(result_path, experi_id, input_shape, n_layers, batch_size, n_epoch, momentum, n_filter, training_loss,
               data_size, test_size, learning_rate, Totall_Accuracy, Test_AUC, sts,augmentationFlag):
    parameters = [result_path, experi_id, input_shape, n_layers, batch_size, n_epoch, momentum, n_filter, training_loss,
                  data_size, test_size, learning_rate, Totall_Accuracy, Test_AUC, sts,augmentationFlag]
    header = ['Result_path', 'Experiment_id', 'Input_shape', 'N_layers', 'Batch_size', 'n_epoch', 'Momentum',
              'Learning_rate', 'n_filter', 'TrainingSet', 'TestSet', 'Training_loss', 'Total_accuracy', 'Test_AUC',
              'std','augmentationFlag']

    if os.path.isfile(os.path.join(result_path, str(input_shape)) + str(
            ".csv")):
        writer = csv.writer(open(
            os.path.join(result_path, str(input_shape)) + str(".csv"),
            'a'))
        writer.writerow(parameters)
    else:
        writer = csv.writer(open(
            os.path.join(result_path, str(input_shape)) + str(".csv"),
            'wb'))

        writer.writerow(header)
        writer.writerow(parameters)


def Build_3dcnn(init_norm=weight_init, inputParamsNetwork=dict(shape=[10,10,10],dropout=0.1, nonLinearity=lasagne.nonlinearities.rectify),
                input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    dropout = inputParamsNetwork['dropout']
    #    print(dropout)
    network = lasagne.layers.InputLayer(shape=(None,1,int(inputParamsNetwork['shape'].split(',')[0]),int(inputParamsNetwork['shape'].split(',')[1]),int(inputParamsNetwork['shape'].split(',')[2])),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    # network=lasagne.layers.dnn.Conv3DDNNLayer(network,num_filters=32,filter_size=(3,3,4),
    #                                           stride=(1, 1, 1),pad=1,
    #                                           nonlinearity=lasagne.nonlinearities.rectify,
    #                                           W=lasagne.init.GlorotUniform()
    #                                           )
    # network=lasagne.layers.dnn.MaxPool3DDNNLayer(network, pool_size=(2,2,2),stride=(2,2,2))


    network = lasagne.layers.dnn.Conv3DDNNLayer(network, num_filters=32, pad='same', filter_size=(5, 5, 3),
                                                stride=(1, 1, 1),
                                                nonlinearity=inputParamsNetwork['nonLinearity'],
                                                W=init_norm,
                                                flip_filters=False
                                                )

    network = lasagne.layers.dnn.MaxPool3DDNNLayer(network, pool_size=(2, 2, 1))

    network = lasagne.layers.dnn.Conv3DDNNLayer(network, num_filters=32, pad='same', filter_size=(5, 5, 3),
                                                stride=(1, 1, 1),
                                                nonlinearity=inputParamsNetwork['nonLinearity'],
                                                W=init_norm,
                                                )
    # network=lasagne.layers.PadLayer(network,width=[(0,1),(0,1)], batch_ndim=3)
    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:

    network = lasagne.layers.dnn.MaxPool3DDNNLayer(network, pool_size=(2, 2, 2))
    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=dropout),
        num_units=64,
        nonlinearity=lasagne.nonlinearities.sigmoid)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=dropout),
        num_units=2,
        nonlinearity=lasagne.nonlinearities.softmax)
    # network=lasagne.layers.DenseLayer(network, num_units=2, nonlinearity=None)

    return network


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.
# Notice that this function returns only mini-batches of size `batch_size`.
# If the size of the data is not a multiple of `batch_size`, it will not
# return the last (remaining) mini-batch.


def Iterate_minibatches(inputs, targets, batch_size, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

        # def main(num_epochs=50,learning_rate=0.1):
        # Load the dataset pos_train_path,neg_train_path,pos_test_paths,neg_test_path


#def Main(inputParamsConfig):
#input_shape,learning_rate, momentum, num_epochs, batchsize, data_path, train_set_size, test_set_size,
#         positive_set_ratio, dropout,nonlinearityToUse,augmentation
if localFlag==1: #in this case ignore the config csv, and use the parameters set out in the beginning for local run
    inputParamsConfigAll = inputParamsConfigLocal
#    Main(inputParamsConfigAll)
else: #in this case ignore run batch (sequential) mode where config is read from csv file
    done_cases={}#Contains all the
    if os.path.exists('./succesfull_cases.json'):
        with open('./succesfull_cases.json', 'r') as f:
            done_cases=json.load(f)
    with open(os.path.join(LIDC_PATH,"param.csv"), 'rU') as csvfile:
        timeReader = csv.reader(csvfile)
        for row in timeReader:
            csv_config.append(row)
    # writer = csv.writer(open(os.path.join(LIDC_PATH,"param.csv"), 'wb'))
    
    for row in csv_config[1:]:
        if ''.join(row) not in done_cases or done_cases[''.join(row)]==0: #This condition calls the main pipeline only for new cases and unsuccessful cases
            print(row)
            try:
                done_cases[''.join(row)] = 1 #We make it equal to one to stop multiple running on the same parameters
                for i in range(0,len(csv_config[0])): #for each element in header, put corresponding value in current row in config
                    inputParamsConfigAll[csv_config[0][i]] = row[i]
                    #Main(*row)
#                    Main(inputParamsConfigAll)
                    
            except:
                done_cases[''.join(row)] = 0# We make it equal to zero if it was unsucceful try
                print ("oops error")
        else:
            print(row ," are used once and you can find the result from ./result direcotry")
    with open('./succesfull_cases.json', 'w') as f:
        json.dump(done_cases, f)




input_shape = inputParamsConfigAll['input_shape']
learning_rate = inputParamsConfigAll['learning_rate']
momentum = inputParamsConfigAll['momentum']
num_epochs = inputParamsConfigAll['num_epochs']
batch_size = inputParamsConfigAll['batch_size']
data_path = inputParamsConfigAll['data_path']
train_set_size = inputParamsConfigAll['train_set_size']
test_set_size = inputParamsConfigAll['test_set_size']
positive_set_ratio = inputParamsConfigAll['positive_set_ratio']
dropout = inputParamsConfigAll['dropout']
nonlinearityToUse = inputParamsConfigAll['nonlinearityToUse']
augmentationFlag = inputParamsConfigAll['augmentationFlag']


print (" Learning rate: '%s' , momentum: '%s',  num_epochs: '%s'  ,batch size: '%s'  ,data_path: '%s' ,Train Set Size: '%s' ,Test set Size: '%s' ,Positive set Ratio '%s' , dropout: '%s', nonlinearityToUse: '%s',augmentationFlag: '%s'" %(str(learning_rate),str(momentum),str(num_epochs), str(batch_size),data_path ,str(train_set_size) ,str(test_set_size) ,str(positive_set_ratio),str(dropout),str(nonlinearityToUse),str(augmentationFlag)))
num_epochs=int(num_epochs)
batch_size=int(batch_size)
train_set, train_label, test_set, test_label, val_set, val_label = SupportFuncs.load_data(data_path, int(train_set_size),
                                                                                          int(test_set_size),
                                                                                          int(augmentationFlag),float(positive_set_ratio))
if nonlinearityToUse == 'relu':
    nonLinearity = lasagne.nonlinearities.rectify
elif nonlinearityToUse == 'tanh':
    nonLinearity = lasagne.nonlinearities.tanh
elif nonlinearityToUse == 'sigmoid':
    nonLinearity = lasagne.nonlinearities.sigmoid
else:
    raise Exception(
        'nonlinearityToUse: Unsupported nonlinearity type has been selected for the network, retry with a supported one!')
dtensor5 = T.TensorType('float32', (False,) * 5)
input_var = dtensor5('inputs')
target_var = T.ivector('targets')

inputParamsNetwork = dict(shape=input_shape,dropout=float(dropout), nonLinearity=nonLinearity)
network = Build_3dcnn(weight_init, inputParamsNetwork, input_var)

# Create a loss expression for training, i.e., a scalar objective we want
# to minimize (for our multi-class problem, it is the cross-entropy loss):
prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean()
# loss=np.mean(loss)
# We could add some weight decay as well here, see lasagne.regularization.

# Create update expressions for training, i.e., how to modify the
# parameters at each training step. Here, we'll use Stochastic Gradient
# Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(
    loss, params, learning_rate=float(learning_rate), momentum=float(momentum))

# Create a loss expression for validation/testing. The crucial difference
# here is that we do a deterministic forward pass through the network,
# disabling dropout layers.
test_prediction = lasagne.layers.get_output(network, deterministic=True)
test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                        target_var)
# test_loss = test_loss.mean()
test_loss = test_loss.mean()

# As a bonus, also create an expression for the classification accuracy:
test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                  dtype=theano.config.floatX)

# Compile a function performing a training step on a mini-batch (by giving
# the updates dictionary) and returning the corresponding training loss:
train_fn = theano.function([input_var, target_var], loss, updates=updates)  # mode='DebugMode'

# Compile a second function computing the validation loss and accuracy:
val_fn = theano.function([input_var, target_var], [test_loss, test_acc, test_prediction])  # ,mode='DebugMode')

# Finally, launch the training loop.
print("Starting training...")
# We iterate over epochs:
for epoch in range(int(num_epochs)):
    # In each epoch, we do a full pass over the training data:
    train_err = 0
    train_batches = 0
    start_time = time.time()
    for batch in Iterate_minibatches(train_set, train_label, int(batch_size), shuffle=True):
        inputs, targets = batch
        inputs = np.float32(inputs)
        train_err += train_fn(inputs, targets)
        train_batches += 1

    # And a full pass over the validation data:
    val_err = 0
    val_acc = 0
    val_batches = 0
    all_val_pred = np.empty((0, 2),
                            dtype=float)  # initialize; array n_samplesx2 for the 2 class predictions for all validation samples
    all_val_labels = np.empty((0, 1),
                              dtype=float)  # initialize; array n_samplesx1 for labels of all validation samples
    for batch in Iterate_minibatches(val_set, val_label, int(batch_size), shuffle=False):
        inputs, targets = batch
        inputs = np.float32(inputs)
        err, acc, val_pred = val_fn(inputs, targets)
        val_err += err
        val_acc += acc
        val_batches += 1
        all_val_pred = np.vstack((all_val_pred, val_pred))
        all_val_labels = np.append(all_val_labels, targets)

    val_AUC, val_varAUC = SupportFuncs.Pred2AUC(all_val_pred, all_val_labels)
    # Then we print the results for this epoch:

    print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, num_epochs, time.time() - start_time))
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
    print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))
    print("  validation AUC: " + str(val_AUC) + ", std: " + str(np.sqrt(val_varAUC)))
    epoch_det[epoch + 1] = {'all_val_accuracy': (val_acc / val_batches), "all_val_loss": (val_err / val_batches),
                            "training_loss": (train_err / train_batches)}
    all_val_accuracy.append(val_acc / val_batches)
    all_val_loss.append(val_err / val_batches)
    all_val_AUC.append(val_AUC)
    training_loss.append(train_err / train_batches)

# After training, we compute and print the test error:
test_err = 0
test_acc = 0
test_batches = 0
all_test_pred = np.empty((0, 2),
                         dtype=float)  # initialize; array n_samplesx2 for the 2 class predictions for all test samples
all_test_labels = np.empty((0, 1), dtype=float)  # initialize; array n_samplesx1 for labels of all test samples
for batch in Iterate_minibatches(test_set, test_label, int(batch_size), shuffle=False):
    inputs, targets = batch
    inputs = np.float32(inputs)
    err, acc, test_pred = val_fn(inputs, targets)
    test_err += err
    test_acc += acc
    test_batches += 1
    all_test_pred = np.vstack((all_test_pred, test_pred))
    all_test_labels = np.append(all_test_labels, targets)

test_AUC, test_varAUC = SupportFuncs.Pred2AUC(all_test_pred, all_test_labels)

print("Final results:")
print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
print("  test accuracy:\t\t{:.2f} %".format(
    test_acc / test_batches * 100))
print("test AUC: " + str(test_AUC) + ", std: " + str(np.sqrt(test_varAUC)))
# Optionally, you could now dump the network weights to a file like this:
# np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
#
# And load them again later on like this:

if not os.path.exists('./figures'):
    os.makedirs('./figures')
fig = plt.figure()
# plt.plot(training_loss,'r',val_accuracy,'g',all_val_loss,'b')
plt.plot(training_loss, 'r', label='Training_loss=' + str("%.6f" % training_loss[num_epochs - 1]))
plt.plot(all_val_accuracy, 'g', label='Val_accuracy=' + str("%.3f" % all_val_accuracy[num_epochs - 1]))
plt.annotate(str("%.3f" % all_val_accuracy[num_epochs - 1]), xy=(num_epochs - 1, all_val_accuracy[num_epochs - 1]),
             xytext=(num_epochs - 70, 0.6),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.annotate(str("%.6f" % training_loss[num_epochs - 1]), xy=(num_epochs - 1, training_loss[num_epochs - 1]),
             xytext=(num_epochs - 70, 0.3),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.ylabel('Training loss and Validation accuracy')
plt.xlabel('Number of Epochs')
plt.title('Accuracy and Loss Changes')
plt.legend(fontsize=13, loc=10)
try:
    fig.savefig(os.path.join(figures_path, experiment_id))  # save the figure to file
except:
    Make_sure_path_exists(figures_path)

plt.close(fig)
plt.show()
Save_model(result_path, experiment_id, str(input_shape), n_layers, int(batch_size), num_epochs, momentum, learning_rate, 2
           , len(train_set), len(test_set), (test_err / test_batches), (test_acc / test_batches), test_AUC[0],
           np.sqrt(test_varAUC),augmentationFlag)



#if ('--help' in sys.argv) or ('-h' in sys.argv):
#    print("Trains a CNN on LIDC using Lasagne.")
#    print("Usage: [ CNN MODEL [EPOCHS]]" % sys.argv[3])
#    print()
#    print("       'cnn' for a Convolutional Neural Network (CNN).")
#    print("EPOCHS: number of training epochs to perform (default: 500)")
#else:
#if localFlag==1: #in this case ignore the config csv, and use the parameters set out in the beginning for local run
#    inputParamsConfigAll = inputParamsConfigLocal
#    Main(inputParamsConfigAll)
#else: #in this case ignore run batch (sequential) mode where config is read from csv file
#    done_cases={}#Contains all the
#    if os.path.exists('./succesfull_cases.json'):
#        with open('./succesfull_cases.json', 'r') as f:
#            done_cases=json.load(f)
#    with open(os.path.join(LIDC_PATH,"param.csv"), 'rU') as csvfile:
#        timeReader = csv.reader(csvfile)
#        for row in timeReader:
#            csv_config.append(row)
#    # writer = csv.writer(open(os.path.join(LIDC_PATH,"param.csv"), 'wb'))
#    
#    for row in csv_config[1:]:
#        if ''.join(row) not in done_cases or done_cases[''.join(row)]==0: #This condition calls the main pipeline only for new cases and unsuccessful cases
#            print(row)
#            try:
#                done_cases[''.join(row)] = 1 #We make it equal to one to stop multiple running on the same parameters
#                for i in range(0,len(csv_config[0])): #for each element in header, put corresponding value in current row in config
#                    inputParamsConfigAll[csv_config[0][i]] = row[i]
#                    #Main(*row)
#                    Main(inputParamsConfigAll)
#                    
#            except:
#                done_cases[''.join(row)] = 0# We make it equal to zero if it was unsucceful try
#                print ("oops error")
#        else:
#            print(row ," are used once and you can find the result from ./result direcotry")
#    with open('./succesfull_cases.json', 'w') as f:
#        json.dump(done_cases, f)


