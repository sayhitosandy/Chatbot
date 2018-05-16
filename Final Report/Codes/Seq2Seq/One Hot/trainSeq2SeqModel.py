# import pandas as pd
import numpy as np
import tensorflow as tf
import re
import time
# import nltk
import re
# from nltk.stem import PorterStemmer, WordNetLemmatizer
# from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import string
from random import shuffle
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import LSTM,RepeatVector,TimeDistributed,Dense,Activation,Input
from keras.models import Model
import pickle
import gensim
from gensim import corpora,models,similarities
import h5py
from keras.losses import categorical_crossentropy
import keras as ker

def perplexity(y_true, y_pred):
    cross_entropy = categorical_crossentropy(y_true, y_pred)
    perplexity = np.power(2.0, cross_entropy)
    return perplexity


LENGTH_THRESHOLD=(2,20)#actually its 21 ,since included 'EOS'
LENGTH_THRESHOLD_OFFSET=1#for the 'EOS'

EMBEDDING_VECTOR_SIZE=300

with open('QAPairs_good_embeddings.pickle', 'rb') as f:
    QAPairs_good_embeddings = pickle.load(f) 

with open('word2index.pickle', 'rb') as f:
    word2index = pickle.load(f) 


with open('embedding_matrix.pickle', 'rb') as f:
    embedding_matrix = pickle.load(f) 

print(QAPairs_good_embeddings[0][0])
print("\n\n")
print(QAPairs_good_embeddings[0][1])

Qs=[]
As=[]
for pair in QAPairs_good_embeddings:
	Qs.append(pair[0])
	As.append(pair[1])

Qs=np.array(Qs)
As=np.array(As)	

Q_tr,Q_te,A_tr,A_te=train_test_split(Qs,As,test_size=0.2,random_state=1)


HIDDEN_LAYER_SIZE=500
NUM_LAYERS_DECODER=3

model = Sequential()

# Creating encoder network
# myModel=Sequential()
model.add(LSTM(output_dim=EMBEDDING_VECTOR_SIZE,input_shape=Q_tr.shape[1:],return_sequences=True,init='glorot_normal',inner_init='glorot_normal',activation='sigmoid'))
# decoder networrk
model.add(LSTM(output_dim=EMBEDDING_VECTOR_SIZE,input_shape=Q_tr.shape[1:],return_sequences=True,init='glorot_normal',inner_init='glorot_normal',activation='sigmoid'))
model.add(LSTM(output_dim=EMBEDDING_VECTOR_SIZE,input_shape=Q_tr.shape[1:],return_sequences=True,init='glorot_normal',inner_init='glorot_normal',activation='sigmoid'))
model.add(LSTM(output_dim=EMBEDDING_VECTOR_SIZE,input_shape=Q_tr.shape[1:],return_sequences=True,init='glorot_normal',inner_init='glorot_normal',activation='sigmoid'))
model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=[perplexity])


model.fit(Q_tr,A_tr,nb_epoch=100,validation_data=(Q_te,A_te))
model.save('LSTM100.h5')
