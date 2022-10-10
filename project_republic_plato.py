#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re, string
from numpy import array
from pickle import dump
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
import numpy as np

def load_file(filename):
    f = open(filename, 'r')
    data = f.read()
    f.close()
    return data

def clean_doc(doc):
    doc = doc.replace('--',' ')
    tokens = doc.split()
    re_punt = re.compile(f"[{re.escape(string.punctuation)}]")
    tokens = [re_punt.sub('',w) for w in tokens]
    tokens = [w for w in tokens if w.isalpha()]
    tokens = [w.lower() for w in tokens]
    return tokens

def save_file(lines, filename):
    data = '\n'.join(lines)
    f = open(filename, 'w')
    f.write(data)
    f.close()
   
def define_model(vocab_size, seq_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 50, input_length=seq_length))
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model  

# Below code creates training data with 50 words input and 1 words output

# filename = "./republic.txt"
# doc = load_file(filename)
# print(f"Total words in document: {len(doc.split())}")
# tokens = clean_doc(doc)
# print(f"Total tokens after cleaning document: {len(tokens)}")
# print(f"Total unique words for vocab and model input: {len(set(tokens))}")

# length = 50 + 1
# sequences = list()
# for i in range(length, len(tokens)):
#     seq = tokens[i-length:i]
#     line = ' '.join(seq)
#     sequences.append(line)
# print(f"Total sequence of 50 words in and 1 words out: {len(sequences)}")
# out_filename = "./republic_seq.txt"
# save_file(sequences, out_filename)

in_filename = "./republic_seq.txt"
doc = load_file(in_filename)
lines = doc.split("\n")

t = Tokenizer()
t.fit_on_texts(lines)

sequences = t.texts_to_sequences(lines)
vocab_size = len(t.word_index) + 1

sequences = array(sequences)
X, y = sequences[:, :-1], sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)
seq_length = X.shape[1]

model = define_model(vocab_size, seq_length)
model.fit(X, y, epochs=100, batch_size = 128)
model.save('model.h5')
dump(t, open('tokenizer.pkl', 'wb'))




