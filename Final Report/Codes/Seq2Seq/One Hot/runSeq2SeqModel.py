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
	
import pickle
import gensim
from gensim import corpora,models,similarities


with open('QAPairs_good_embeddings.pickle', 'rb') as f:
    QAPairs_good_embeddings = pickle.load(f) 

print(QAPairs_good_embeddings[0][0])
print("\n\n")
print(QAPairs_good_embeddings[0][1])

Qs=[]
As=[]
for pair in QAPairs_good_embeddings:
	Qs.append(pair[0])
	As.append(pair[1])

Q_tr,Q_te,A_tr,A_te=train_test_split(Qs,As,test_size=0.2,random_state=1)

