import numpy as np
import pretty_midi
import os
import csv
import math
import re
np.random.seed(42)
import tensorflow as tf
import keras
tf.set_random_seed(42)
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, Dropout
from keras.layers import TimeDistributed
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import pickle
import sys
import heapq
#import seaborn as sns
#from pylab import rcParams

def noteVal(y):
    val = y
    val = val.split('-', 1)[0]
    val = re.sub('Tied', '', val)
    return noteList.get(val.lower(), 0)

#noteList = {'whole':48,'half':24,'dotted half':16,'triplet':16,'quarter':12,'dotted quarter':8,'eighth':6,'dotted eighth':4, 
#            'sixteenth':3, 'dotted sixteenth':3, 'thirty second':2, 'triplet thirty second':1, 'sixty fourth':1}

noteList = {'whole':24,'half':12,'dotted half':8,'triplet':8,'quarter':6,'dotted quarter':4,'eighth':3,'dotted eighth':3,
            'sixteenth':2, 'dotted sixteenth':2, 'thirty second':1, 'triplet thirty second':1, 'sixty fourth':1}
    

# What should the format hold?
# Simplest format is note pitch held for a note value amount of time, not allowing multiple notes at once

###############################################################################################################################

MEMORY = 24
OFFSET = 1
FUTURE = 1

def itercount(val):
    return 1+(range(0, val - MEMORY, OFFSET))[-1]/OFFSET

D = np.load('../mnt/Downloads/DATASET.npz')
PianoRoll = (D['BarRoll']).item()

MINnote = 128
MAXnote = -1
TRsize = 0
NUMBEROFNOTES = set()

for label in PianoRoll:
    song = PianoRoll[label]
    locations = set((np.where(song))[0])
    NUMBEROFNOTES = NUMBEROFNOTES | locations
    MAXnote = np.maximum( MAXnote, max(locations))
    MINnote = np.minimum( MINnote, min(locations) )
    TRsize = TRsize + itercount(song.shape[1])

NOTEspace = int(MAXnote-MINnote+2)

print 'Training Data Size = '  + str(TRsize)
print 'MAXnote = ' + str(MAXnote)
print 'MINnote = ' + str(MINnote)
print 'Number of notes = ' + str(len(NUMBEROFNOTES))


smp = 0
files = 0
Input = np.zeros( (TRsize, MEMORY, NOTEspace), dtype='float32' )
Output = np.zeros( (TRsize, FUTURE*NOTEspace), dtype='float32')

for label in PianoRoll:
    song = PianoRoll[label]
    print 'Unzipping training label: ' + label + '(with augmentation)'
    files = files+1
    for idx in range(0, song.shape[1] - MEMORY, OFFSET):
        cnt = min(idx,song.shape[1]-MEMORY-FUTURE)
        for i in range(0,FUTURE):
            Output[smp, i*NOTEspace:(i+1)*NOTEspace] = song[MINnote-1:MAXnote+1,cnt+MEMORY+i]
        Input[smp, :, :] = (song[MINnote-1:MAXnote+1,cnt:cnt+MEMORY]).transpose()
        smp+=1

print 'Sucessfully unzipped ' +str(files)+ ' files'


#################################################################################################################

for f in os.listdir('checkpoints/'):
    os.remove(os.path.join('checkpoints/', f))

#from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K
from keras.utils import plot_model

def inv_MSE_hamming_distance(y_true, y_pred):
    return K.mean(K.abs(y_true) - K.abs(y_true - K.round(y_pred)), axis=-1)

# instantiate model
model = Sequential()

# input layer
model.add(LSTM(256, input_shape=(MEMORY, NOTEspace), return_sequences=True)) #, kernel_initializer='Zeros'))
model.add(Dropout(0.5))

# hidden layer 1
model.add(LSTM(256))
model.add(Dropout(0.5))

# output layer
model.add(Dense(FUTURE*NOTEspace))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['categorical_accuracy', inv_MSE_hamming_distance])

# checkpoint
filepath="checkpoints/weights-improvement-{epoch:02d}-{loss:.2f}.hdf5"
checkpointer = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False, mode='auto')

plot_model(model, to_file='model.png', show_shapes=True)

#############################################################################################################################

# fit the model
history = model.fit(Input, Output, validation_split=0.2, batch_size=128, epochs=20, shuffle=True, callbacks=[checkpointer]).history

#############################################################################################################################

import h5py

model_yaml = model.to_yaml()
with open("tmp/model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("model.hdf5")
print("Saved model to disk")

#############################################################################################################################

model = None

from keras.models import model_from_yaml

yaml_file = open('tmp/model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
model = model_from_yaml(loaded_model_yaml)
# load weights into new model
model.load_weights("model.hdf5") #model.hdf5")
print("Loaded model from disk")