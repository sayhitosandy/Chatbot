# import pandas as pd
import numpy as np
import tensorflow as tf
import re
import time
# import nltk
import re
# from nltk.stem import PorterStemmer, WordNetLemmatizer
# from nltk.tokenize import word_tokenize
import string
from random import shuffle
import keras.preprocessing.text
import keras.preprocessing.sequence 	
import pickle
import gensim
from gensim import corpora,models,similarities

EMBEDDING_VECTOR_SIZE=300

embeddingModel=gensim.models.Word2Vec.load('embeddingModel_'+str(EMBEDDING_VECTOR_SIZE))

print("loaded embeddings successfully!")

with open('QAPairs_good.pickle', 'rb') as f:
    QAPairs_good = pickle.load(f) 

print("loaded QAPairs_good successfully!")

# print(embeddingModel.wv['vrdstart'])
# print(embeddingModel.wv['vrdend'])
# print(embeddingModel.wv['vrdpad'])

QAPairs_good_embeddings=[]

for pair in QAPairs_good:
	q=keras.preprocessing.text.text_to_word_sequence(pair[0],
                                               filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                               lower=True,
                                               split=" ")
	a=keras.preprocessing.text.text_to_word_sequence(pair[1],
                                           filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                           lower=True,
                                           split=" ")
	# print(q)
	q_embedded=np.array([embeddingModel.wv[token] for token in q])
	# print(q_embedded)
	# exit()
	a_embedded=np.array([embeddingModel.wv[token] for token in a])
	QAPairs_good_embeddings.append((q_embedded,a_embedded))


# print (len(QAPairs_good_embeddings[0][0]))
# print('\n')
# print (len(QAPairs_good_embeddings[0][1]))


with open('QAPairs_good_embeddings.pickle','wb') as f:
	pickle.dump(QAPairs_good_embeddings,f, protocol=pickle.HIGHEST_PROTOCOL)

print ("saved QA Embeddings  pickled successfully. File name is : 'QAPairs_good_embeddings.pickle' ")