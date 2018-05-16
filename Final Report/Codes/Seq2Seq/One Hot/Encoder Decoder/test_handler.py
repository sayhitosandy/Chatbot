
from keras.models import load_model
import keras
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
from keras.losses import categorical_crossentropy
import keras as ker
import keras.preprocessing.text
import keras.preprocessing.sequence 	

LENGTH_THRESHOLD=(2,20)		#actually its 21 ,since included 'EOS'
LENGTH_THRESHOLD_OFFSET=1	#for the 'EOS'


with open('data+lookups.pickle','rb') as f:
	QAPairs_good_int_padded,wordsToIDDict,IDToWordsDict = pickle.load(f)


def decodeASequence(inputSequence):
	# Encode the input as state vectors.

	words=keras.preprocessing.text.text_to_word_sequence(inputSequence,
                                               filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                               lower=True,
                                               split=" ")
	words_int=[]
	for i in range(len(words)):
		if words[i] in wordsToIDDict:
			words_int.append(wordsToIDDict[words[i]])
		else :
			words_int.append(wordsToIDDict['vsunk'])


	words_int=np.array(words_int)

	input_1hot=np.zeros((1,LENGTH_THRESHOLD[1]+LENGTH_THRESHOLD_OFFSET,len(wordsToIDDict)))
	input_1hot[0][np.arange(words_int.shape[0]),words_int]=1.


# 	>>> a = np.array([1, 0, 3])
# >>> b = np.zeros((3, 4))
# >>> b[np.arange(3), a] = 1
# >>> b
# array([[ 0.,  1.,  0.,  0.],
#        [ 1.,  0.,  0.,  0.],
#        [ 0.,  0.,  0.,  1.]])
# >>>



	statesValues = encoderModel.predict(input_1hot)

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

encoderModel=load_model('LSTM+1000+100+512+NoReversal_encoder.h5')
decoderModel=load_model('LSTM+1000+100+512+NoReversal_decoder.h5')


inputSequence=''

while inputSequence!='exit':	
	# Take one sequence (part of the training test)
	# for trying out decoding.
	# inputSequence = encoderInput_3d[seqIndex: seqIndex + 1]
	inputSequence=input('Enter a string :')
	decodedOutput = decodeASequence(inputSequence)
	print('-')
	print('Input sentence:', inputSequence)
	print('Decoded sentence:', decodedOutput)