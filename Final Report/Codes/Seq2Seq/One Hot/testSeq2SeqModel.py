from keras.models import load_model
import gensim
import nltk
import numpy as np
import keras.preprocessing.text
import keras.preprocessing.sequence 
LENGTH_THRESHOLD=(2,20)#actually its 21 ,since included 'EOS'
LENGTH_THRESHOLD_OFFSET=1#for the 'EOS'


myModel=load_model('LSTM50.h5')
# embeddingVectorSize=32
EMBEDDING_VECTOR_SIZE=300

embeddingModel=gensim.models.Word2Vec.load('embeddingModel_'+str(EMBEDDING_VECTOR_SIZE))

while True:
	message=input('Enter a message:')

	message_tokenized=keras.preprocessing.text.text_to_word_sequence(message,
                                               filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                               lower=True,
                                               split=" ")
	message_tokenized=['vrdstart']+message_tokenized+['vrdend' for i in range(LENGTH_THRESHOLD[1]-len(message_tokenized))]
	print (message_tokenized)

	message_tokenized_embedding=np.array([embeddingModel.wv[token] for token in message_tokenized])
	message_tokenized_embedding=np.array([message_tokenized_embedding])
	predictions=myModel.predict(message_tokenized_embedding)
	output=[embeddingModel.most_similar([predictions[0][i]])[0][0] for i in range(LENGTH_THRESHOLD[1])]
	outputStr=' '.join(output)
	print(outputStr)
	