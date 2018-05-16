import numpy as np
# import tensorflow as tf
import re
import time
# import nltk
import re
# from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
# from keras.models import Sequential
# from keras.layers.embeddings import Embedding
# from keras.layers import LSTM,RepeatVector,TimeDistributed,Dense,Activation,Input,GRU
# from keras.models import Model


# constant!!!
numLines=666576

lines=open('cornell movie-dialogs corpus/movie_lines.txt').read().split('\n')
conversations=open('cornell movie-dialogs corpus/movie_conversations.txt').read().split('\n')


# lines=lines[:10]
linesList=['' for x in range(numLines+10)]
# conversations=conversations[:10]

# make array of lines
# preprocess words here : possibly wordnet lemmatizer
# lemmatizerWordnet = WordNetLemmatizer()
max=-1
for line in lines:
	if line=='':
		break		
	tokens=line.split('+++$+++')
	lineID=tokens[0]
	# print lineID
	lineID=int(lineID.strip()[1:])
	# if lineID>max:
	# 	max=lineID
	# 	pass
	lineText=tokens[-1].lower()

	lineText=re.sub('[^A-Za-z.!?]+', ' ', lineText)
	# print lineID,
	# try:
	# 	lineText=" ".join([lemmatizerWordnet.lemmatize(i) for i in lineText.split()])
	# except Exception:
	# 	pass  
	linesList[lineID]=lineText
	# print linesList[lineID]
# print (linesList[:100])

currentAndNext=[]
linesList=linesList[:1000]



with open('TrainingSet_Hmm.txt','w') as f:

	for line in linesList:

		if len(line)>1:
			tokens=word_tokenize(line)

			for i in range(len(tokens)-1):
				currentAndNext.append((tokens[i],tokens[i+1]))
				f.write(tokens[i]+"\t"+tokens[i+1]+"\n")

			f.write('\n')

	f.close()


with open('TestingSet_Hmm.txt','w') as f:

	for line in linesList:

		if len(line)>1:
			tokens=word_tokenize(line)

			for i in range(len(tokens)-1):
				# currentAndNext.append((tokens[i],tokens[i+1]))
				f.write(tokens[i]+"\n")

			f.write('\n')
	

	f.close()