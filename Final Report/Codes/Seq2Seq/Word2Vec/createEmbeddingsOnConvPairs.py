# import pandas as pd
import numpy as np
import tensorflow as tf
import re
import time
import nltk
import re
from sklearn.manifold import TSNE

# from nltk.stem import PorterStemmer, WordNetLemmatizer
# from nltk.tokenize import word_tokenize
import string
from random import shuffle
import keras.preprocessing.text
from keras.preprocessing.text import Tokenizer

import keras.preprocessing.sequence 	
import pickle
import gensim
from gensim import corpora,models,similarities
import matplotlib.pyplot as plt

LENGTH_THRESHOLD=(2,20)#actually its 21 ,since included 'EOS'
LENGTH_THRESHOLD_OFFSET=1#for the 'EOS'

EMBEDDING_VECTOR_SIZE=300

with open('QAPairs_good.pickle', 'rb') as f:
    QAPairs_good = pickle.load(f) 

print("Conv. pairs loaded!")
print (QAPairs_good[:2])

linesList=set([])

for pair in QAPairs_good:
	linesList.add(pair[0])
	linesList.add(pair[1])

print(len(linesList),len(QAPairs_good))

linesList=list(linesList)




linesList_tokenized=[keras.preprocessing.text.text_to_word_sequence(sent,
                                               filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                               lower=True,
                                               split=" ") for sent in linesList]
# linesList_tokenized=[nltk.word_tokenize(sent) for sent in linesList]

print (linesList_tokenized[1000:1002])

embeddingModel=gensim.models.Word2Vec(linesList_tokenized,min_count=1,size=EMBEDDING_VECTOR_SIZE)

embeddingModel.save('embeddingModel_'+str(EMBEDDING_VECTOR_SIZE))

print(embeddingModel.most_similar('you'))

# get word2index ditionary
MAX_NB_WORDS=9000

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(linesList)
sequences = tokenizer.texts_to_sequences(linesList)

word2index = tokenizer.word_index
print('Found %s unique tokens.' % len(word2index))

print(word2index['you'])


embedding_matrix = np.zeros((len(word2index) + 1, EMBEDDING_VECTOR_SIZE))

for word, i in word2index.items():
    embedding_vector = embeddingModel.wv[word]
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


with open('word2index.pickle','wb') as f:
	pickle.dump(word2index,f, protocol=pickle.HIGHEST_PROTOCOL)	


with open('embedding_matrix.pickle','wb') as f:
	pickle.dump(embedding_matrix,f, protocol=pickle.HIGHEST_PROTOCOL)	

# X = embeddingModel[embeddingModel.wv.vocab]
# print (X[:2])

# for visualization :

# tsne = TSNE(n_components=2)
# X_tsne = tsne.fit_transform(X)

# plt.scatter(X_tsne[:, 0], X_tsne[:, 1])

# for label, x, y in zip(embeddingModel.wv.vocab, X_tsne[:, 0], X_tsne[:, 1]):
#     plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    
# plt.show()