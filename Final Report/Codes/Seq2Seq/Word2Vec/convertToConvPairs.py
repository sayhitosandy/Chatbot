import numpy as np
import tensorflow as tf
import re
import time
# import nltk
import re
# from nltk.stem import PorterStemmer, WordNetLemmatizer
# from nltk.tokenize import word_tokenize
from random import shuffle
import pickle
import keras.preprocessing.text
import keras.preprocessing.sequence 	

# constant!!!
numLines=666576

LENGTH_THRESHOLD=(2,20)#actually its 21 ,since included 'EOS'
LENGTH_THRESHOLD_OFFSET=1#for the 'EOS'

lines=open('cornell movie-dialogs corpus/movie_lines.txt').read().split('\n')
conversations=open('cornell movie-dialogs corpus/movie_conversations.txt').read().split('\n')


linesList=['' for x in range(numLines+10)]

# make array of lines

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

	# lineText=re.sub('[^A-Za-z.!?]+', ' ', lineText)
	# print lineID,
	# try:
	# 	lineText=" ".join([lemmatizerWordnet.lemmatize(i) for i in lineText.split()])
	# except Exception:
	# 	pass  
	linesList[lineID]=lineText
	# print linesList[lineID]
# print (linesList[:10])

QAPairs=[] #Question-Answer pairs
QAPairs_good=[]

# print ("lines loaded and formatted (numbers,punc removed except ?!.).")

# print ("lines loaded.")

for conversation in conversations:
	
	bracketi=conversation.find('[')
	conversation=conversation[bracketi:]
	conversationIDs = re.sub('\W+','', conversation ).split('L')[1:]
	# print conversationIDs
	for i in range(len(conversationIDs)-1):
		curI=int(conversationIDs[i])
		nextI=int(conversationIDs[i+1])
		QAPairs.append((linesList[curI],linesList[nextI]))

# print len(QAPairs)


for qa in QAPairs:
	# qq='<GO> '+qa[0]
	q=keras.preprocessing.text.text_to_word_sequence(qa[0],
                                               filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                               lower=True,
                                               split=" ")
	# aa=qa[1]+' <EOS>'#to mark end of sentence
	a=keras.preprocessing.text.text_to_word_sequence(qa[1],
                                           filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                           lower=True,
                                           split=" ")
	lq=len(q)
	la=len(a)

	if (la<=LENGTH_THRESHOLD[1] and la >= LENGTH_THRESHOLD[0]) and (lq<=LENGTH_THRESHOLD[1] and lq>= LENGTH_THRESHOLD[0]):
		
		# print([['<PAD>'] for i in range(LENGTH_THRESHOLD[1]-len(q))])
		# q=(['vrdStart']+q)+['vrdPad' for i in range(LENGTH_THRESHOLD[1]-len(q))]
		# a=(a+['vrdEnd'])+['vrdPad' for i in range(LENGTH_THRESHOLD[1]-len(a))]
		# print(len(q))
		# print(len(a))
		# exit()
		qStr='vrdStart '+qa[0]+''.join([' vrdPad' for i in range(LENGTH_THRESHOLD[1]-len(q))])
		# print (qStr)
		aStr=qa[1]+' vrdEnd'+''.join([' vrdPad' for i in range(LENGTH_THRESHOLD[1]-len(a))])
		# print (aStr)
		# exit()
		QAPairs_good.append((qStr,aStr))


print ("QA_pairs filtered, good pairs remain.")
# print ((QAPairs_good[10000:10002]))

QAPairs_good=QAPairs_good[:1000]

with open('QAPairs_good.pickle','wb') as f:
	pickle.dump(QAPairs_good,f, protocol=pickle.HIGHEST_PROTOCOL)

print ("saved QA Pairs and pickled successfully. File name is : 'QAPairs_good.pickle' ")
# finally obtain the question-response pairs...which are lesser than 20 words in size and greater than 2 words.
