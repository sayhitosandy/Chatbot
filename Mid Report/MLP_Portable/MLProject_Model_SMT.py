# import pandas as pd
import numpy as np
import tensorflow as tf
import re
import time
import nltk
from nltk.align.ibm1.IBMModel1 import IBMModel
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import LSTM,RepeatVector,TimeDistributed,Dense,Activation,Input
from keras.models import Model


# constant!!!
numLines=666576

lines=open('cornell movie-dialogs corpus/movie_lines.txt').read().split('\n')
conversations=open('cornell movie-dialogs corpus/movie_conversations.txt').read().split('\n')


# lines=lines[:10]
linesList=['' for x in xrange(numLines+10)]
# conversations=conversations[:10]

# make array of lines
# preprocess words here : possibly wordnet lemmatizer
lemmatizerWordnet = WordNetLemmatizer()
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
# print linesList
QAPairs=[] #Question-Answer pairs
QAPairs_good=[]

print "lines loaded and formatted (numbers,punc removed except ?!.)."

for conversation in conversations:
	# print conversation
	bracketi=conversation.find('[')
	conversation=conversation[bracketi:]
	conversationIDs = re.sub('\W+','', conversation ).split('L')[1:]
	# print conversationIDs
	for i in xrange(len(conversationIDs)-1):
		curI=int(conversationIDs[i])
		nextI=int(conversationIDs[i+1])
		QAPairs.append((linesList[curI],linesList[nextI]))

# print len(QAPairs)

length_threshold=(2,20)#actually its 21 ,since included 'EOS'
length_threshold_offset=1
for qa in QAPairs:
	
	lq=len(qa[0].split())
	la=len(qa[1].split())

	if (la<=length_threshold[1] and la >= length_threshold[0]) and (lq<=length_threshold[1] and lq>= length_threshold[0]):
		QAPairs_good.append(qa)

# print QAPairs_good
print "QA_pairs filtered, good pairs remain."


# QAPairs_good is the now refined dataset which we'll work upon
#  can tokenize instead of split here...
wordsToCountDict={}
ctr=0


#####################3

# TO REDUCE MEMORY LIMIT .USE ONLY 100 SAMPLES OF QA"
QAPairs_good=QAPairs_good[:100]

########################
for qa in QAPairs_good:
	# print qa[0]
	# ctr+=1
	# if ctr==2:
	# 	break
	try:
		# words=word_tokenize(qa[0])

		words=qa[0].split()
	except Exception:
		raise  

	for word in words:
		if word not in wordsToCountDict:
			wordsToCountDict[word]=1
		else:
			wordsToCountDict[word]+=1

	try:
		# words=word_tokenize(qa[1])
		words=qa[1].split()
	except Exception:
		raise 
	# words=word_tokenize(qa[1])
	# words=qa[1].split()
	for word in words:
		if word not in wordsToCountDict:
			wordsToCountDict[word]=1
		else:
			wordsToCountDict[word]+=1

# print len(wordsToCountDict)

wordsToIDDict={}
UNK_threshold=0
ID_assigner=0

print "Words Counted."

for word in wordsToCountDict:
	if wordsToCountDict[word]>UNK_threshold:

		wordsToIDDict[word]=ID_assigner
		ID_assigner+=1

# print wordsToIDDict


print "Words(exceeding frequency threshold="+str(UNK_threshold)+") mapped to ID"

# Add the unique tokens to the vocabulary dictionaries.
specialSymbols = ['<PAD>','<EOS>','<UNK>','<GO>']
ctr=0
for symbol in specialSymbols:
	ctr+=1
	wordsToIDDict[symbol] = -ctr

IDToWordsDict = {ID: word for word, ID in wordsToIDDict.items()}

# print(len(wordsToIDDict))
# print(len(IDToWordsDict))

ques = []
ans = []
for qa in QAPairs_good:
	ques.append(qa[0].split())
	ans.append(qa[1].split())

# print(ques)

bitext = []
for i in range(len(ques)):
	bitext.append(AllignedSent(ques[i], ans[i]))



