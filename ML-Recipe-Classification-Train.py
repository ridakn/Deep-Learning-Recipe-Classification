# -*- coding: utf-8 -*-

import csv
import re
import numpy as np
from keras.utils.np_utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Bidirectional
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Dropout, concatenate
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

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
X_val, y_val = load_data(ROOT_PATH, "val")
X_test = load_data(ROOT_PATH, "test")

print("Train Records: ", len(X_train), len(y_train))
print("Val Records: ", len(X_val), len(y_val))
print("Test Records: ", len(X_test))

print(X_train[0])
print(y_train[0])

tokenizer = Tokenizer(num_words=10000, lower = True)
tokenizer.fit_on_texts(X_train)

# Adding 1 because of reserved 0 index
vocab_size = len(tokenizer.word_index) + 1
print("Vocab Size ", vocab_size)

X_train = tokenizer.texts_to_sequences(X_train)
X_val = tokenizer.texts_to_sequences(X_val)
X_test = tokenizer.texts_to_sequences(X_test)

print(X_train[0])

# truncate and pad input sequences
max_review_length = 1000
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length, padding='post')
X_val = sequence.pad_sequences(X_val, maxlen=max_review_length, padding='post')
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length, padding='post')

X_train = np.array(X_train)
X_val = np.array(X_val)
X_test = np.array(X_test)

y_train = np.array(y_train)
y_val = np.array(y_val)

print(X_train.dtype)
print(X_val.dtype)
print(X_test.dtype)

print(y_train.dtype)
print(y_val.dtype)

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)

print("Vocab size =", vocab_size)

y_train = y_train - 1
y_val = y_val - 1

train_labels = to_categorical(y_train, num_classes = 12)
val_labels = to_categorical(y_val, num_classes = 12)

print(y_train[0])
print(train_labels[0])

print(X_train.shape, train_labels.shape)
print(X_val.shape, val_labels.shape)

embeddings_index = {}
f = open(ROOT_PATH + 'glove.6B.50d.txt',encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Total %s word vectors in Glove 6B 50d.' % len(embeddings_index))

word_index = tokenizer.word_index
embedding_vecor_length = 50

print('Number of Unique Tokens',len(word_index))
embedding_matrix = np.random.random((len(word_index) + 1, embedding_vecor_length))

for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:

        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


MODEL_INDEX = 35

# create the model
embedding_vecor_length = 50

model1 = Sequential()
model1.add(Embedding(vocab_size, embedding_vecor_length, 
                    weights=[embedding_matrix],
                    input_length = max_review_length, 
                    trainable=True))

model1.add(Dropout(0.3))

model1.add(Bidirectional(GRU(embedding_vecor_length, return_sequences = True)))
model1.add(Bidirectional(GRU(embedding_vecor_length)))
model1.add(Dense(128, activation='relu'))

model2 = Sequential()
model2.add(Embedding(vocab_size, embedding_vecor_length, 
                    weights=[embedding_matrix],
                    input_length = max_review_length, 
                    trainable=True))

model2.add(Dropout(0.3))

model2.add(Bidirectional(GRU(embedding_vecor_length, return_sequences = True)))
model2.add(Bidirectional(GRU(embedding_vecor_length)))
model2.add(Dense(128, activation='relu'))

merged = concatenate([model1.output, model2.output])

output_layer = Dense(24, activation='relu')(merged)
output_layer = Dense(12, activation='softmax')(output_layer)

final_model = Model(inputs = [model1.input, model2.input], outputs = [output_layer])
final_model.summary()

opt = Adam(learning_rate=0.001)
final_model.compile(loss = 'categorical_crossentropy',
              optimizer = opt, 
              metrics=['accuracy'])

modelname='PR2_' + str(MODEL_INDEX)
model_json = final_model.to_json()

with open(ROOT_PATH + 'Models/_' + modelname + '.json', "w") as json_file:
    json_file.write(model_json)
    
    
# define early stopping callback
earlystop = EarlyStopping(monitor='val_loss', 
                          min_delta=0.001, 
                          patience=7, 
                          verbose=2, 
                          mode='auto', 
                          baseline=None, 
                          restore_best_weights=True)  

# define modelcheckpoint callback
checkpointer = ModelCheckpoint(filepath = ROOT_PATH + 'Models/_' + modelname + '.hdf5',
                               monitor='val_loss', 
                               save_best_only=True)

callbacks_list = [earlystop, 
                  checkpointer,
                  ReduceLROnPlateau()]

final_model.fit([X_train, X_train], train_labels, 
          validation_data=([X_val, X_val], val_labels), 
          epochs=100, 
          batch_size=50, 
          callbacks=callbacks_list)