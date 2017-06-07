from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.layers import Dropout
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import pandas as pd
import glob
import string

seqlen = 30

chars = sorted(set("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,!?-+()'/&\" "))
remchars = (set(chr(i) for i in range(0,254)).difference(chars))

blurbs = []
allFiles = glob.glob("kickstarter-data/*.csv")
frame = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    list_.append(df)
frame = pd.concat(list_)
del list_
del allFiles

frame["blurb"] = frame["blurb"].astype(str)
for blurb in frame["blurb"]:
    try:
        blurb.encode("ascii")
    except UnicodeEncodeError:
        continue
    else:
        blurb.encode("ascii")
        str = blurb.translate({ord(c): None for c in remchars})
        str += "\n"
        blurbs.append(str)
        if len(blurbs) >= 50000:
            break
del frame
chars.append("\n")
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
'''
def batches(blurbs, seqlen, batchsize, epochs):
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    nb_batches = (len(blurbs) - 1) // (batchsize * seqlen)
    for epoch in range(epochs):
        for batch in range(nb_batches):
            X = np.zeros((batchsize, seqlen, len(chars)), dtype=np.bool)
            y = np.zeros((batchsize, seqlen, len(chars)), dtype=np.bool)
            for t, char in enumerate(blurb):
                X[batch, t, char_indices[char]] = 1
                y[batch, t, char_indices[char + 1]] = 1
            yield X, y, epoch
'''
step = 1
sentences = []
next_chars = []
for text in blurbs:
    for i in range(0, len(text) - seqlen, step):
        sentences.append(text[i: i + seqlen])
        next_chars.append(text[i + seqlen])

#one hot encoding
X = np.zeros((len(sentences), seqlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# build the model: a triple LSTM
model = Sequential()
model.add(LSTM(128, input_shape=(seqlen, len(chars)), return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))
model.load_weights("3x128-LSTM-0.5-dropout/weights-1.793.hdf5")
model.compile(loss='categorical_crossentropy', optimizer='adam')

filepath = "3x128-LSTM-0.5-dropout/weights-{loss:.3f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(X, y,
          epochs=200,
          callbacks = callbacks_list)