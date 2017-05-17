from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import pandas as pd

text = ""
projects = pd.read_csv('most_backed.csv')
projects["blurb"] = projects["blurb"].astype(str)
for blurb in projects["blurb"]:
    text += blurb
text = text.lower()

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])

#one hot encoding
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model: a triple LSTM
model = Sequential()
model.add(LSTM(512, input_shape=(maxlen, len(chars)), return_sequences=True))
model.add(LSTM(512, return_sequences=True))
model.add(LSTM(512))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

model.load_weights("weights-1.581.hdf5")
model.compile(loss='categorical_crossentropy', optimizer='adam')

filepath = "weights-{loss:.3f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(X, y,
          validation_split=0.33,
          epochs=200,
          batch_size=128,
          callbacks = callbacks_list)