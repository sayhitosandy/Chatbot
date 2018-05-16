# import pandas as pd
import numpy as np
import tensorflow as tf
import re
import time
# import nltk
import re
import pickle
# from nltk.stem import PorterStemmer, WordNetLemmatizer
# from nltk.tokenize import word_tokenize
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import LSTM,RepeatVector,TimeDistributed,Dense,Activation,Input
from keras.models import Model
from random import shuffle
from keras.losses import categorical_crossentropy,binary_crossentropy
import keras as ker
# constant!!!
NUM_LINES=666576

LENGTH_THRESHOLD=(2,20)		#actually its 21 ,since included 'EOS'
LENGTH_THRESHOLD_OFFSET=1	#for the 'EOS'


QAPAirs_good_int_padded = []
QAPairs_good = []
wordsToIDDict = {}
IDToWordsDict = {}

with open('data+lookups.pickle','rb') as f:
	QAPairs_good_int_padded,wordsToIDDict,IDToWordsDict = pickle.load(f)
	print (len(wordsToIDDict))

def perplexity(y_true, y_pred):
    cross_entropy = categorical_crossentropy(y_true, y_pred)
    perplexity = np.power(2.0, cross_entropy)
    return perplexity



def decodeASequence(inputSequence,encoderModel,decoderModel):
	# Encode the input as state vectors.
	statesValues = encoderModel.predict(inputSequence)

	# Generate empty target sequence of length 1.
	targetSequence = np.zeros((1, 1, len(wordsToIDDict)))
	# Populate the first character of target sequence with the start character.
	targetSequence[0, 0, wordsToIDDict['vsgo']] = 1.

	# Sampling loop for a batch of sequences
	# (to simplify, here we assume a batch of size 1).
	decodedOutput = ''
	max_decoder_seq_length=30
	stopCondition=False
	while not stopCondition:
		outputTokens, h, c = decoderModel.predict(
			[targetSequence] + statesValues)

		# Sample a token
		sampledWordIndex = np.argmax(outputTokens[0, -1, :])
		if sampledWordIndex in IDToWordsDict:

			sampledWord = IDToWordsDict[sampledWordIndex]
		else:
			sampledWord= 'vsunk'
		decodedOutput += (' '+sampledWord)

		# Exit condition: either hit max length
		# or find stop character.
		if (sampledWord == 'vseos' or
		   len(decodedOutput) > max_decoder_seq_length):
			stopCondition = True

		# Update the target sequence (of length 1).
		targetSequence = np.zeros((1, 1, len(wordsToIDDict)))
		targetSequence[0, 0, sampledWordIndex] = 1.

		# Update states
		statesValues = [h, c]

	return decodedOutput


def trainModel(QAPairs_good_int_padded,wordsToIDDict,IDToWordsDict,batch_size,epochs):
	QAPairs_good_int_padded=np.array(QAPairs_good_int_padded)


	latentDimensions=512

	encoderInputs=Input(shape=(None,len(wordsToIDDict)))
	# print (encoderInputs)
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
	rnnLstm.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=[perplexity])


	# Run training


	for i in range(epochs):

		batch_inds = np.random.randint(len(QAPairs_good_int_padded), size=batch_size)
		a_batch=QAPairs_good_int_padded[batch_inds,:]

		encoderInput_3d=np.zeros((len(a_batch),LENGTH_THRESHOLD[1]+LENGTH_THRESHOLD_OFFSET,len(wordsToIDDict)))
		decoderInput_3d=np.zeros((len(a_batch),LENGTH_THRESHOLD[1]+LENGTH_THRESHOLD_OFFSET,len(wordsToIDDict)))
		decoderTarget_3d=np.zeros((len(a_batch),LENGTH_THRESHOLD[1]+LENGTH_THRESHOLD_OFFSET,len(wordsToIDDict)))

		for i,qa in enumerate(a_batch):
			encoderInpLine=qa[0][::-1]

			for j,word in enumerate(encoderInpLine):
				encoderInput_3d[i,j,word]=1.

			decoderInpLine=qa[1]

			for k,word in enumerate(decoderInpLine):
				decoderInput_3d[i,k,word]=1.
				if k>0:
					decoderTarget_3d[i,k-1,word]=1.

		batchSize=64
		epochs=100

		rnnLstm.fit([encoderInput_3d, decoderInput_3d], decoderTarget_3d,
				  batch_size=batchSize,
				  epochs=epochs,
				  validation_split=0.2)
		if epochs%50==0:
			pass
			rnnLstm.save('rnnLstm'+'+'+str(epochs)+'+'+str(batch_size)+'+'+str(latentDimensions)+'+seq2seq.h5')


	# Save model
	# print ("Saving...")
	# rnnLstm.save('LSTM+1000+64+2+YesReversal.h5')
	# Type+numEpochs+batchSize+latentDimensions+Special.h5

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

	print ("Saving encoder and decoder ...")

	encoderModel.save('LSTM'+'+'+str(epochs)+'+'+str(batch_size)+'+'+str(latentDimensions)+'+NoReversal_encoder.h5')
	decoderModel.save('LSTM'+'+'+str(epochs)+'+'+str(batch_size)+'+'+str(latentDimensions)+'+NoReversal_decoder.h5')


epochs=1000
batch_size=100
trainModel(QAPairs_good_int_padded,wordsToIDDict,IDToWordsDict,batch_size,epochs)

