import sklearn, re, os, nltk,string, csv, random, json, scipy, numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def remove_punctuation(txt):
	return txt.translate(None, string.punctuation)

def stars_to_sentiment(number_of_stars):
	if int(number_of_stars) > 3:
		return +1
	else:
		return -1
	
	
def csvupdater(row, column, new_colname, function_to_apply):
	result=function_to_apply(column)
	return result
	

	


#moveable parts
outputname="amazon_edited.csv"


#reading in 
# inputi=open("amazon_baby.csv", "r")
# csvinputi=csv.reader(inputi, dialect="excel")
# 
# #writing out
# outputi=open(outputname, "w")
# csvoutputi=csv.writer(outputi, dialect="excel")
# 
# newcsv=[]
# 
# #stuff
# x=0
# 
# row0=csvinputi.next()
# row0=row0+['cleantext', 'sentiment']
# print row0
# newcsv.append(row0)
# print newcsv
# 
# 
# for row in csvinputi:
# #0 is name, 1 is review, 2 is star number
# 		#cleantext=remove_punctuation(row[1])
# 		cleantext=csvupdater(row, row[1], 'cleantext', remove_punctuation)
# 		sentiment=csvupdater(row, row[2], 'sentiment', stars_to_sentiment)
# # 		print sentiment
# 		newrow=row+[cleantext, sentiment]
# 		# print newrow
# 		if newrow[2] != '3':
# 			newcsv.append(newrow)
# 		# print newcsv
# # 		x=x+1
# #  		if x > 5:
# #  			break
# 
# 
# # 
# 	
# csvoutputi.writerows(newcsv)
# outputi.close()
# 


##NOW TRAINING
inputi=open(outputname, "r")
csvinputi=csv.reader(inputi, dialect="excel")

fullset=[]

for row in csvinputi:
	fullset.append(row)
	
print "the dataset has {} rows".format(len(fullset))

with open("test-idx.json") as jsontestfile:
	jsontestdata=json.load(jsontestfile)
	print len(jsontestdata)
	
	
with open("train-idx.json") as jsontrainfile:
	jsontraindata=json.load(jsontrainfile)
	print len(jsontraindata)
	
	
test_data=[fullset[i+1] for i in jsontestdata]
train_data=[fullset[i+1] for i in jsontraindata]

print "----newsflash-----"
print "the trainingset has {} rows".format(len(train_data))
print "the testset has {} rows".format(len(test_data))
print "that adds up to {} rows".format(len(test_data)+len(train_data))


# Learn a vocabulary (set of all words) from the training data. 
# Only the words that show up in the training data will be considered for feature extraction

vocab={}
wordcount=0

for row in train_data:
	review=row[3].split()
	for word in review:
		wordcount=wordcount+1
		if word not in vocab:
 			vocab[word]='TRUE'
# 			print word.lower(), "success"
		# else:
# 			print word.lower()
# 			break
print "the vocab is {} words".format(len(vocab))
print "the wordcount is {}". format(wordcount)

vectorizer=CountVectorizer(token_pattern=r'\b\w+\b')

for row in test_data:
# 	#wordvector=[row[3].split().count(i) for i in vocablist]

# Build a sparse matrix where each row is the word count vector for the corresponding review. 
# Call this matrix train_matrix.
# for row in test_data:
# 	#wordvector=[row[3].split().count(i) for i in vocablist]
# 	wordmatrix.append(scipy.sparse.csr_matrix([row[3].split().count(i) for i in vocablist]))
# 	#coo_matrix is also a thing
# 	#train_matrix.append(wordmatrix)
# 
# 
# print "the {} is {} items long".format("train matrix", len(train_matrix))
# #train_matrix=[numpy.sparse.csr_matrix(i) for i in wordmatrix]
# # for row in test_data:
# 	#wordvector=[row[3].split().count(i) for i in vocablist]
# 	test_matrix=scipy.sparse.csr_matrix([row[3].split().count(i) for i in vocablist])

print "this is done"

	
	

 	