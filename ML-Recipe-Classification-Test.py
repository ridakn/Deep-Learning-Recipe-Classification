#!/usr/bin/env python
# coding: utf-8


import csv
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout, Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import warnings
warnings.simplefilter("ignore", UserWarning)

# fix random seed for reproducibility
np.random.seed(7)

ROOT_PATH = './'


def load_data(path, name_of_data):
    data, labels = [], []
    
    if name_of_data == "test":
        
        with open(path + 'data/' + name_of_data + '.txt') as f1:
            reader1 = csv.reader(f1, delimiter = "\n")
            for line in reader1:
                data.append(line[0])
                
        return data
    
    else:
        
        with open(path + 'data/' + name_of_data + '.txt') as f1:
            reader1 = csv.reader(f1, delimiter = "\n")
            for line in reader1:
                data.append(line[0])

        with open(path + 'data/' + name_of_data +'.labels') as f2:
            reader2 = csv.reader(f2)
            for line in reader2:
                labels.append(int(line[0]))

        return data, labels


X_train, y_train = load_data(ROOT_PATH, "train")
X_test = load_data(ROOT_PATH, "test")

print("Train Records: ", len(X_train), len(y_train))
print("Test Records: ", len(X_test))

tokenizer = Tokenizer(num_words=10000, lower = True)
tokenizer.fit_on_texts(X_train)

X_test = tokenizer.texts_to_sequences(X_test)

# Adding 1 because of reserved 0 index
vocab_size = len(tokenizer.word_index) + 1

# truncate and pad input sequences
max_review_length = 1000

X_test = sequence.pad_sequences(X_test, maxlen=max_review_length, padding='post')

X_test = np.array(X_test)
print(X_test.dtype)
print(X_test.shape)

print("Vocab size =", vocab_size)

MODEL_INDEX = 35
modelname='PR2_' + str(MODEL_INDEX)

json_file = open(ROOT_PATH + 'Models/_' + modelname + '.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights(filepath = ROOT_PATH + 'Models/_' + modelname + '.hdf5')
print("Loaded " + modelname +" from disk")

opt = Adam(learning_rate=0.001)
# evaluate loaded model on test data
loaded_model.compile(loss= 'categorical_crossentropy',
              optimizer = opt, 
              metrics=['accuracy'])

predicted_labels = loaded_model.predict([X_test,X_test])

final_predictions = np.argmax(predicted_labels, axis=-1)
print(final_predictions.shape)
final_predictions = final_predictions + 1

with open(ROOT_PATH + "Results/" + modelname + "_Results.txt", 'w', newline='') as w1:

    writer = csv.writer(w1, delimiter=' ')

    for p in list(final_predictions):
        writer.writerow([p])
        
print("File saved at ", ROOT_PATH + "Results/" + modelname + "_Results.txt")
