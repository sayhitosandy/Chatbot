tagToWordsDict={}
tagAndNextDict={}
wordCountDict={}
tagCountDict={}
tags=set([])
			

def ReadFileGetParams(filename):
	words=set([])
	tags=set([])
	with open(filename) as data:

		for line in data:
			line=line[:-1]
			if line=="":
				continue
			(word,tag)=line.split('\t')
			words.add(word)
			tags.add(tag)

		# print len(words)
		# print len(tags)



def ReadDataAndFillDict(filename):	

	with open(filename) as data:
		ctr=0
		curWord="<s>"
		curTag="<t>"

		for line in data:
			line=line[:-1]
			# print line
			ctr+=1

			# if ctr==30:
			# 	break
			if line=="":
				nextWord='</s>'
				nextTag='</t>'
				# tagAndNextDict[curTag].append("</t>")
				# curWord="<s>"
				# curTag="<t>"
				# tagCountDict["<t>"]+=1
				# continue;
			else:

				(nextWord,nextTag)=line.split('\t')

			# print nextWord,nextTag

			# if nextTag==".":
			# 	continue
			# else:
			# 	tags.add(nextTag)

			if curWord not in wordCountDict:
				wordCountDict[curWord]=1
			else:
				wordCountDict[curWord]+=1


			if curTag not in tagCountDict:
				tagCountDict[curTag]=1
			else:
				tagCountDict[curTag]+=1
				

# 			dictionary with key as current tag, and values as word that has the tag
			
			if curTag not in tagToWordsDict:
				tagToWordsDict[curTag]=[curWord]
			else:
				tagToWordsDict[curTag].append(curWord)


# 			dictionary with key as current tag, and values as next adj. tags
			if curTag not in tagAndNextDict:
				tagAndNextDict[curTag]=[nextTag]
			else:
				tagAndNextDict[curTag].append(nextTag)

			if (nextWord=='</s>' and	nextTag=='</t>'):
				curTag="<t>"
				curWord="<s>"

			else:
				curTag=nextTag
				curWord=nextWord

		return tagAndNextDict,tagCountDict,tagToWordsDict,wordCountDict


def MakeTransitionMatrix(tagAndNextDict,tagCountDict):

	# make 2d transition matrix with a[i][j]=p(j/i);
	#going row by row here, i.e take a row and go thorugh each column of row
	# row i=> prob given 'i' e.g row 'NNS'=>prob given 'NNS'
	transitionDict2D={}

	for rowTag in tagAndNextDict:
		countRowTag=tagCountDict[rowTag]
		listWithPrevAsRowTag=tagAndNextDict[rowTag]
		curDict={} #i
		for colTag in tagCountDict:#columns
			
			numMatches=0
			for nextTag in listWithPrevAsRowTag:

				if nextTag==colTag:
					numMatches+=1
			# probaTag=numMatches*1.0/countRowTag
			probaTag=(numMatches*1.0+1.0)/(countRowTag+len(wordCountDict))#laplacian smoothing

			curDict[colTag]=probaTag
		transitionDict2D[rowTag]=curDict

	# for key in transitionDict2D:
	# 	print key,transitionDict2D[key]

	return transitionDict2D
					

def MakeObservationLikelihoodMatrix(tagToWordsDict,wordCountDict):

	obsLikelihoodDict2D={}

	for aTag in tagCountDict:
		curTagCount=tagCountDict[aTag]
		curDict={}
		curTagToWords=tagToWordsDict[aTag]

		for aWord in wordCountDict:
			numMatches=0
			for element in curTagToWords:
				if element==aWord:
					numMatches+=1
			# probaWord=numMatches*1.0/curTagCount
			probaWord=(numMatches*1.0+1.0)/(curTagCount+len(tagCountDict))#laplacian smoothing

			curDict[aWord]=probaWord
		obsLikelihoodDict2D[aTag]=curDict

	# for key in obsLikelihoodDict2D:
	# 	print key,obsLikelihoodDict2D[key]

	return obsLikelihoodDict2D



def Viterbi_Predict(transitionDict2D,obsLikelihoodDict2D,inputSequence):	
	totalProba=1.0
	Viterbi2D={}
	Parents2D={}

	splitInput=inputSequence.split(" ")

	curTransistioner=transitionDict2D['<t>']
	# print "\n\n",curTransistioner
	curDict={}
	curParentDict={}
	for key in curTransistioner:
		if key!='<t>':
			priori=curTransistioner[key]
			if splitInput[0] in wordCountDict:						
						posteriori=obsLikelihoodDict2D[key][splitInput[0]]
			else:
						posteriori=1e-3

			
			# print key,splitInput[0],priori*posteriori
			curDict[key]=priori*posteriori
			curParentDict[key]='O'


	# print "viterbi[0] is ",curDict
	Viterbi2D[0]=curDict
	Parents2D[0]=curParentDict

	# print Parents2D

	 #iterate column-wise

	for i in range(1,len(splitInput)):
		prevColumnDict=Viterbi2D[i-1]
		curColumnDict={}
		curParentDict={}

		# if i==2:
		# 	break
		for curKey in prevColumnDict:
			maxproba=-1
			maxPrev='NAN'
			for prevKey in prevColumnDict:

				if prevKey!='<t>':

					priori=transitionDict2D[prevKey][curKey]
					if splitInput[i] in wordCountDict:						
						posteriori=obsLikelihoodDict2D[curKey][splitInput[i]]
					else:
						posteriori=1e-3

					# print prevKey,curKey,splitInput[i],priori*posteriori,prevColumnDict[prevKey]
					probVal=prevColumnDict[prevKey]*priori*posteriori
					if probVal>maxproba:
						maxproba=probVal
						maxPrev=prevKey

					# print probVal
			curColumnDict[curKey]=maxproba
			curParent=maxPrev
			curParentDict[curKey]=maxPrev

		# print curColumnDict
		# print curParentDict
		Viterbi2D[i]=curColumnDict
		Parents2D[i]=curParentDict

	maxKey='NoKey'
	maxproba=-1
	last=len(splitInput)-1
	# print Viterbi2D[last]
	for key in Viterbi2D[last]:
		if Viterbi2D[last][key]>maxproba:
			maxKey=key
			maxproba=Viterbi2D[last][key]


	# backtrack
	ctr=2
	ansStr=""
	curCol=last
	curKey=maxKey
	while curCol!=-1:
		# print curKey
		ansStr=curKey+" "+ansStr
		curKey=Parents2D[curCol][curKey]
		# print (curKey,maxKey)
		try:
			totalProba*=transitionDict2D[maxKey][curKey]
			ctr+=1
		except Exception as e:
			pass

		# print(totalProba)
		curCol-=1

	print (ansStr)
	x= (pow(totalProba,1.0/(ctr)))	
	return (1.0/x)

def Viterbi_Train(filename):
	tagAndNextDict,tagCountDict,tagToWordsDict,wordCountDict=ReadDataAndFillDict(filename)
	transitionDict2D=MakeTransitionMatrix(tagAndNextDict,tagCountDict)
	obsLikelihoodDict2D=MakeObservationLikelihoodMatrix(tagToWordsDict,wordCountDict)
	return transitionDict2D,obsLikelihoodDict2D


	




filename="TrainingSet_Hmm.txt";


# inputSequence="i am a girl ."

transitionDict2D,obsLikelihoodDict2D=Viterbi_Train(filename)

queryStr=""
ctr=0

# with open('TestingSet_Hmm.txt') as f:
# 	for line in f:
# 		if line=='\n':
# 			# print (queryStr)
# 			# break

# 			x=Viterbi_Predict(transitionDict2D,obsLikelihoodDict2D,queryStr.lower())
# 			ctr+=1
# 			if(ctr==10):
# 				print("perplexity is :",x*1.0/ctr)
# 				break
# 			queryStr=""
# 		else:
# 			queryStr=queryStr+' '+line[:-1]


queryStr="hi how are you ."

queryStr="are you a computer?"
Viterbi_Predict(transitionDict2D,obsLikelihoodDict2D,queryStr)
queryStr="get lost!"
Viterbi_Predict(transitionDict2D,obsLikelihoodDict2D,queryStr)
queryStr="are you a human?"
Viterbi_Predict(transitionDict2D,obsLikelihoodDict2D,queryStr)
queryStr="who is the president?"
Viterbi_Predict(transitionDict2D,obsLikelihoodDict2D,queryStr)
queryStr="you are not making sense "
Viterbi_Predict(transitionDict2D,obsLikelihoodDict2D,queryStr)
queryStr="are you drunk?"
Viterbi_Predict(transitionDict2D,obsLikelihoodDict2D,queryStr)
queryStr="Hi!"
Viterbi_Predict(transitionDict2D,obsLikelihoodDict2D,queryStr)
queryStr="Am I a doctor?"
Viterbi_Predict(transitionDict2D,obsLikelihoodDict2D,queryStr)
queryStr="when will the world end?"
Viterbi_Predict(transitionDict2D,obsLikelihoodDict2D,queryStr)
queryStr="can you teach me something?  "
Viterbi_Predict(transitionDict2D,obsLikelihoodDict2D,queryStr)

# with open("Test set.txt") as testData:
# 	# print "ko"

# 	queryStr=""
# 	for line in testData:
# 		# print line
# 		line=line[:-1]

		
# 		if line==".":
# 			# print queryStr
# 			queryStr=queryStr.strip(" ")
# 			print (queryStr)
# 			Viterbi_Predict(transitionDict2D,obsLikelihoodDict2D,queryStr)
# 			queryStr=""

# 		else:
# 			queryStr=queryStr+line+" "


		
		



# Viterbi_Predict(transitionDict2D,obsLikelihoodDict2D,inputSequence)
