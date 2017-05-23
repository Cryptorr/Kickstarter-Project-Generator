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

model.load_weights("weights-0.377.hdf5")
model.compile(loss='categorical_crossentropy', optimizer='adam')
#generate output
start_index = random.randint(0, len(text) - maxlen - 1)

generated = ''
sentence = text[start_index: start_index + maxlen]
generated += sentence
print('Seed: "' + sentence + '"')
sys.stdout.write(generated)

for i in range(200):
    x = np.zeros((1, maxlen, len(chars)))
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