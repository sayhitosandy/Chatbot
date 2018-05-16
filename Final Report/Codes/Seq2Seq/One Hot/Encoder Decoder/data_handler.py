import numpy as np
import tensorflow as tf
import re
import time
import re
from random import shuffle
import pickle
import keras.preprocessing.text
import keras.preprocessing.sequence 	

numLines=666576

LENGTH_THRESHOLD=(2,20)		#actually its 21 ,since included 'EOS'
LENGTH_THRESHOLD_OFFSET=1	#for the 'EOS'

lines=open('cornell movie-dialogs corpus/movie_lines.txt').read().split('\n')
conversations=open('cornell movie-dialogs corpus/movie_conversations.txt').read().split('\n')


linesList=['' for x in range(numLines+10)]
for line in lines:
	if line=='':
		break		
	tokens=line.split('+++$+++')
	lineID=tokens[0]
	lineID=int(lineID.strip()[1:])
	lineText=tokens[-1].lower()
	linesList[lineID]=lineText


QAPairs=[] #Question-Answer pairs
for conversation in conversations:	
	bracketi=conversation.find('[')
	conversation=conversation[bracketi:]
	conversationIDs = re.sub('\W+','', conversation ).split('L')[1:]
	for i in range(len(conversationIDs)-1):
		curI=int(conversationIDs[i])
		nextI=int(conversationIDs[i+1])
		QAPairs.append((linesList[curI],linesList[nextI]))


QAPairs_good=[]
for qa in QAPairs:
	q=keras.preprocessing.text.text_to_word_sequence(qa[0],
                                               filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                               lower=True,
                                               split=" ")
	a=keras.preprocessing.text.text_to_word_sequence(qa[1],
                                           filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                           lower=True,
                                           split=" ")
	lq=len(q)
	la=len(a)

	if (la<=LENGTH_THRESHOLD[1] and la >= LENGTH_THRESHOLD[0]) and (lq<=LENGTH_THRESHOLD[1] and lq>= LENGTH_THRESHOLD[0]):
		QAPairs_good.append(qa)


wordsToCountDict={}

########################
# TO REDUCE MEMORY LIMIT .USE ONLY 100 SAMPLES OF QAP"
QAPairs_good=QAPairs_good[:10000]
########################

for qa in QAPairs_good:
	try:
		words=keras.preprocessing.text.text_to_word_sequence(qa[0],
                                               filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                               lower=True,
                                               split=" ")
	except Exception:
		raise  

	for word in words:
		if word not in wordsToCountDict:
			wordsToCountDict[word]=1
		else:
			wordsToCountDict[word]+=1

	try:
		words=keras.preprocessing.text.text_to_word_sequence(qa[1],
                                       filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                       lower=True,
                                       split=" ")
	except Exception:
		raise 

	for word in words:
		if word not in wordsToCountDict:
			wordsToCountDict[word]=1
		else:
			wordsToCountDict[word]+=1

print ("Words Counted.")


wordsToIDDict={}
UNK_threshold=40
ID_assigner=0

for word in wordsToCountDict:
	if wordsToCountDict[word]>UNK_threshold:
		wordsToIDDict[word]=ID_assigner
		ID_assigner+=1

print ("Words(exceeding frequency threshold=10) mapped to ID")


# Add the unique tokens to the vocabulary dictionaries.
specialSymbols = ['vspad','vseos','vsunk','vsgo']

ctr=0
for symbol in specialSymbols:
	ctr+=1
	wordsToIDDict[symbol] = -ctr

IDToWordsDict = {ID: word for word, ID in wordsToIDDict.items()}


for i in range(len(QAPairs_good)):
	a=QAPairs_good[i][1]
	a=a+ ' vseos'
	QAPairs_good[i]=(QAPairs_good[i][0],a)


QAPairs_good_int=[]

# make sure only int values are here
for qa in QAPairs_good:
	try:
		wa=keras.preprocessing.text.text_to_word_sequence(qa[1],
                                       filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                       lower=True,
                                       split=" ")
		wq=keras.preprocessing.text.text_to_word_sequence(qa[0],
                                       filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                       lower=True,
                                       split=" ")
	except Exception:
		continue  

	aConvToID=[]
	qConvToID=[]

	for word in wa:
		if word not in wordsToIDDict:
			aConvToID.append(wordsToIDDict['vsunk'])
		else:
			aConvToID.append(wordsToIDDict[word])			

	for word in wq:
		if word not in wordsToIDDict:
			qConvToID.append(wordsToIDDict['vsunk'])
		else:
			qConvToID.append(wordsToIDDict[word])			

	q_rev=qConvToID[::-1]
	QAPairs_good_int.append((q_rev,aConvToID))


print ("all data samples conv to their int representations.")
# Sort questions and answers by the length of questions.
# This will reduce the amount of padding during training
# Which should speed up training and help to reduce the loss

QAPairs_good_int.sort(key=lambda t: len(t[0]))

QAPairs_good_int_padded=[]
for qa in QAPairs_good_int:
	lPad=LENGTH_THRESHOLD[1]+LENGTH_THRESHOLD_OFFSET-len(qa[0])
	qa0= np.lib.pad(qa[0], pad_width=(0,lPad), mode='constant', constant_values=wordsToIDDict['vspad']) 
	lPad=LENGTH_THRESHOLD[1]+LENGTH_THRESHOLD_OFFSET-len(qa[1])
	qa1= np.lib.pad(qa[1], pad_width=(0,lPad), mode='constant', constant_values=wordsToIDDict['vspad']) 
	QAPairs_good_int_padded.append((qa0,qa1))


with open('data+lookups.pickle','wb') as f:
	pickle.dump((QAPairs_good_int_padded,wordsToIDDict,IDToWordsDict),f, protocol=pickle.HIGHEST_PROTOCOL)

print ("saved QA int padded,lookups and pickled successfully. File name is : 'data+lookups.pickle' ")