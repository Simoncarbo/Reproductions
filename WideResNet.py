# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 17:42:53 2017

@author: sicarbonnell
Forked from Keras cifar10_cnn example

Reproduces Wide ResNets results on CIFAR-100
WN 28 - 10 with dropout 0.3

Converged to 19.05% error on test set. 18.85% is reported by original paper.
"""
import numpy as np
np.random.seed(0)
import pickle
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Input, merge, BatchNormalization, Activation, Dense, Dropout
from keras.layers import ZeroPadding2D, Convolution2D, GlobalAveragePooling2D
from keras.regularizers import l2
import keras.backend as K

from keras.datasets import cifar100
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

#==============================================================================
# Parameters
#==============================================================================
# files
history_file = 'WN28_10_history.p'
model_file = 'WN28_10.h5'

# parameters for WideResnet model
k = 10 # widening factor
N = 4 # number of blocks per stage. Depth = 6*N+4
dropout = 0.3

# optimization parameters
batch_size = 128
opt = SGD(lr=0.1, momentum=0.9, nesterov=True)
nb_epoch = 200
data_augmentation = True
weight_decay = 0.0005

def schedule(epoch):
    standard = 0.1
    if epoch <60:
        return standard
    elif epoch <120:
        return standard * 0.2
    elif epoch <160:
        return standard * 0.2**2
    else:
        return standard * 0.2**3

#==============================================================================
# Data importation
#==============================================================================
(X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode='fine')
X_train, X_test = np.float32(X_train), np.float32(X_test)

# input normalization
m = np.mean(X_train)
X_train -= m
X_test  -= m
st = np.std(X_train)
X_train /= st
X_test  /= st

# convert output to one-hot encoding
y_train = to_categorical(y_train)
y_test  = to_categorical(y_test)

#==============================================================================
# Model
#==============================================================================
def block(inp,nbfilters,dropout,weight_decay,subsample = (1,1)): 
    x = inp
    
    for i in [1,2]:
        x = BatchNormalization(mode = 0,axis = 1)(x)
        x = Activation('relu')(x)
        
        if dropout>0. and i==2:
            x = Dropout(dropout)(x)
        
        x = ZeroPadding2D((1,1))(x)
        if subsample is not None and i==1:
            x = Convolution2D(nbfilters,3,3,subsample=subsample,W_regularizer=l2(weight_decay))(x)
        else:
            x = Convolution2D(nbfilters,3,3,W_regularizer=l2(weight_decay))(x)
    
    if subsample==(1,1) and inp._keras_shape[1] == nbfilters: # checks for subsampling or change in nb of filters
        return merge([x,inp],mode = 'sum')
    else:
        return merge([x,Convolution2D(nbfilters,1,1,subsample = subsample,W_regularizer=l2(weight_decay))(inp)],mode = 'sum')

def stage(x, nbfilters, N, dropout, weight_decay, subsample = True):
    if subsample:
        x = block(x,nbfilters, dropout, weight_decay, subsample = (2,2))
    else: 
        x = block(x,nbfilters, dropout, weight_decay)
    for i in range(1,N):
        x = block(x,nbfilters, dropout, weight_decay)
    return x

# Wide ResNet for Cifar100
# Contains 3 stages
# arguments are lists with one parameter per stage
# input conv nbfilter is always 16
def WideResNet_C100(nbfilters,nbblocks,dropout,weight_decay):
    if K.image_dim_ordering() == 'th':
        input_model = Input(shape = (3,32,32))
    else:
        input_model = Input(shape = (32,32,3))
    
    # input convolution
    x = ZeroPadding2D((1,1))(input_model)
    x = Convolution2D(16, 3, 3,W_regularizer=l2(weight_decay))(x)
    
    # stage 1, 32x32
    x = stage(x,nbfilters[0],nbblocks[0], dropout, weight_decay, subsample = False)
    
    # stage 1, 16x16
    x = stage(x,nbfilters[1],nbblocks[1], dropout, weight_decay)
    
    # stage 1, 8x8
    x = stage(x,nbfilters[2],nbblocks[2], dropout, weight_decay)
    
    x = BatchNormalization(mode = 0,axis = 1)(x)
    x = Activation('relu')(x)
    
    x = GlobalAveragePooling2D()(x)
    x = Dense(100,W_regularizer=l2(weight_decay))(x)
    x = Activation('softmax')(x)
    
    return Model(input_model,x)
    
# create and compile model
model = WideResNet_C100([16*k,32*k,64*k],[N]*3,dropout,weight_decay)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

#==============================================================================
# Training
#==============================================================================
if not data_augmentation:
    print('Not using data augmentation.')
    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        nb_epoch=nb_epoch,
                        validation_data=(X_test,y_test),
                        shuffle=True,
                        callbacks = [LearningRateScheduler(schedule),
                                     ModelCheckpoint(model_file[:-3]+'_{epoch:02d}.h5', period=10)])
else:
    print('Using real-time data augmentation.')    
    datagen = ImageDataGenerator(width_shift_range=0.125,
                                 height_shift_range=0.125,
                                 fill_mode='reflect',
                                 horizontal_flip=True)

    history = model.fit_generator(datagen.flow(X_train, y_train,batch_size=batch_size),
                                  samples_per_epoch=X_train.shape[0],
                                  nb_epoch=nb_epoch,
                                  validation_data=(X_test,y_test),
                                  callbacks = [LearningRateScheduler(schedule),
                                               ModelCheckpoint(model_file[:-3]+'_{epoch:02d}.h5', period=10)])

# For some mysterious reason, not able to pickle History object directly
pickle.dump({'epoch' : history.epoch,'history' : history.history}, open(history_file, "wb" ) )
model.save(model_file)

plt.figure()
plt.plot(history.epoch,1-np.array(history.history['acc']),'b')
plt.plot(history.epoch,1-np.array(history.history['val_acc']),'r')
plt.legend(['train_error','test_error'])

print('final test error: '+str(1-history.history['val_acc'][-1]))