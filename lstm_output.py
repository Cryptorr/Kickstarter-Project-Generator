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

seqlen = 30

chars = sorted(set("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,!?-+()'/&\" "))
chars.append("\n")
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

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

model.load_weights("3x128-LSTM-0.5-dropout/weights-1.757.hdf5")
model.compile(loss='categorical_crossentropy', optimizer='adam')
#generate output
start_index = 0

generated = ''
sentence = 'Exploding kittens is a card ga'
generated += sentence
print('Seed: "' + sentence + '"')
sys.stdout.write(generated)

for i in range(150):
    x = np.zeros((1, seqlen, len(chars)))
    for t, char in enumerate(sentence):
        x[0, t, char_indices[char]] = 1.

    preds = model.predict(x, verbose=0)[0]
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    next_index = np.argmax(probas)
    next_char = indices_char[next_index]

    generated += next_char
    sentence = sentence[1:] + next_char

    sys.stdout.write(next_char)
    sys.stdout.flush()
    if (next_char == "\n"):
        break