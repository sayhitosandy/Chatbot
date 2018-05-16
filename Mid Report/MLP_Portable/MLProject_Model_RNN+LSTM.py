# import pandas as pd
import numpy as np
import tensorflow as tf
import re
import time
import nltk
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
QAPairs_good=QAPairs_good[:1000]

########################
for qa in QAPairs_good:
	print qa[0]
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

for i in range(len(QAPairs_good)):
	a=QAPairs_good[i][1]
	a=a+ ' <EOS>'
	QAPairs_good[i]=(QAPairs_good[i][0],a)


QAPairs_good_int=[]

# make sure only int values are here

for qa in QAPairs_good:
	# q=qa[0]
	# a=qa[1]

	try:
		# wa=word_tokenize(qa[0])
		# wq=word_tokenize(qa[1])
		wa=qa[1].split()
		wq=qa[0].split()

	except Exception:
		continue  

	aConvToID=[]
	qConvToID=[]

	for word in wa:
		if word not in wordsToIDDict:
			aConvToID.append(wordsToIDDict['<UNK>'])
		else:
			aConvToID.append(wordsToIDDict[word])			

	for word in wq:
		if word not in wordsToIDDict:
			qConvToID.append(wordsToIDDict['<UNK>'])
		else:
			qConvToID.append(wordsToIDDict[word])			

	QAPairs_good_int.append((qConvToID,aConvToID))

# print QAPairs_good_int[:10]

print "all data samples conv to their int representations."
# Sort questions and answers by the length of questions.
# This will reduce the amount of padding during training
# Which should speed up training and help to reduce the loss

QAPairs_good_int.sort(key=lambda t: len(t[0]))

# print QAPairs_good_int[:10]

QAPairs_good_int_padded=[]

for qa in QAPairs_good_int:
	lPad=length_threshold[1]+length_threshold_offset-len(qa[0])
	qa0= np.lib.pad(qa[0], pad_width=(0,lPad), mode='constant', constant_values=wordsToIDDict['<PAD>']) 
	# print qa[1]
	# print len(qa[1])
	lPad=length_threshold[1]+length_threshold_offset-len(qa[1])
	qa1= np.lib.pad(qa[1], pad_width=(0,lPad), mode='constant', constant_values=wordsToIDDict['<PAD>']) 
	QAPairs_good_int_padded.append((qa0,qa1))
	# print qa[0]
	# print qa[1]


# for qa in QAPairs_good_int_padded:
# 	print qa[0]
# 	print qa[1]

# print wordsToIDDict


encoderInput_3d=np.zeros((len(QAPairs_good_int_padded),length_threshold[1]+length_threshold_offset,len(wordsToIDDict)))
decoderInput_3d=np.zeros((len(QAPairs_good_int_padded),length_threshold[1]+length_threshold_offset,len(wordsToIDDict)))
decoderTarget_3d=np.zeros((len(QAPairs_good_int_padded),length_threshold[1]+length_threshold_offset,len(wordsToIDDict)))




# print inputVect3D

for i,qa in enumerate(QAPairs_good_int_padded):

	for j,word in enumerate(qa[0]):
		encoderInput_3d[i,j,word]=1.

	for k,word in enumerate(qa[1]):
		decoderInput_3d[i,k,word]=1.
		if k>0:
			decoderTarget_3d[i,k-1,word]=1.


# print encoderInput_3d
			

latentDimensions=2

encoderInputs=Input(shape=(None,len(wordsToIDDict)))
encoder=LSTM(latentDimensions,return_state=True)
encoderOutputs,stateH,stateC=encoder(encoderInputs)

encoderStates=[stateH,stateC]

# print encoderInput_3d
decoderInputs=Input(shape=(None,len(wordsToIDDict)))
decoderLSTM=LSTM(latentDimensions,return_sequences=True,return_state=True)
decoderOutputs, _, _ = decoderLSTM(decoderInputs,
									 initial_state=encoderStates)
decoderDense = Dense(len(wordsToIDDict), activation='softmax')
decoderOutputs = decoderDense(decoderOutputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
rnnLstm = Model([encoderInputs, decoderInputs], decoderOutputs)

# ########################
# 
# TRAINING PARAMS
# #######################
# Run training
batchSize=64
epochs=100
# 
# 
# 
# 
# 
# 
# 
rnnLstm.compile(optimizer='rmsprop', loss='categorical_crossentropy')
rnnLstm.fit([encoderInput_3d, decoderInput_3d], decoderTarget_3d,
		  batch_size=batchSize,
		  epochs=epochs,
		  validation_split=0.2)
# Save model
print "Saving..."
rnnLstm.save('s2s_new_LSTM.h5')




encoderModel = Model(encoderInputs, encoderStates)

decoderStateInputH = Input(shape=(latentDimensions,))
decoderStateInputC = Input(shape=(latentDimensions,))
decoderStateInputs = [decoderStateInputH, decoderStateInputC]


decoderOutputs, stateH, stateC = decoderLSTM(
	decoderInputs, initial_state=decoderStateInputs)


decoderStates = [stateH, stateC]
decoderOutputs = decoderDense(decoderOutputs)
decoderModel = Model(
	[decoderInputs] + decoderStateInputs,
	[decoderOutputs] + decoderStates)



def decodeASequence(inputSequence):
	# Encode the input as state vectors.
	statesValues = encoderModel.predict(inputSequence)

	# Generate empty target sequence of length 1.
	targetSequence = np.zeros((1, 1, len(wordsToIDDict)))
	# Populate the first character of target sequence with the start character.
	targetSequence[0, 0, wordsToIDDict['<GO>']] = 1.

	# Sampling loop for a batch of sequences
	# (to simplify, here we assume a batch of size 1).
	decodedOutput = ''
	itera = 0
	while (len(decodedOutput.split()) < length_threshold_offset+length_threshold[1] and itera < 1000	):
		outputTokens, h, c = decoderModel.predict(
			[targetSequence] + statesValues)
		print "loop"
		# Sample a token
		sampledWordIndex = np.argmax(outputTokens[0, -1, :])
		if sampledWordIndex not in IDToWordsDict:
			sampledWord="LOL"
		else:        	
			sampledWord = IDToWordsDict[sampledWordIndex]
			decodedOutput += (" "+sampledWord)

		if sampledWord=='<EOS>':
			break

		# Exit condition: either hit max length
		# or find stop character.

		# Update the target sequence (of length 1).
		targetSequence = np.zeros((1, 1, len(wordsToIDDict)))
		targetSequence[0, 0, sampledWordIndex] = 1.

		# Update states
		statesValues = [h, c]
		# print h
		# print c
		# print statesValues
		itera += 1

	return decodedOutput


for seqIndex in range(10):
	# print "herer"
	# Take one sequence (part of the training test)
	# for trying out decoding.
	inputSequence = encoderInput_3d[seqIndex: seqIndex + 1]
	decodedOutput = decodeASequence(inputSequence)
	print('-')
	print('Input sentence:', QAPairs_good[seqIndex][0])
	print('Decoded sentence:', decodedOutput)

# Epoch 50/50
# 80/80 [==============================] - 0s - loss: 5.7445 - val_loss: 5.7461

# Epoch 200/200
# 80/80 [==============================] - 0s - loss: 4.5463 - val_loss: 4.5581
