import sklearn, re, os, nltk,string, csv, random, json

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
inputi=open("amazon_baby.csv", "r")
csvinputi=csv.reader(inputi, dialect="excel")

#writing out
outputi=open(outputname, "a")
csvoutputi=csv.writer(outputi, dialect="excel")

newcsv=[]

#stuff
x=0

row0=csvinputi.next()
row0=row0+['cleantext', 'sentiment']
print row0
newcsv.append(row0)
print newcsv


for row in csvinputi:
#0 is name, 1 is review, 2 is star number
		#cleantext=remove_punctuation(row[1])
		cleantext=csvupdater(row, row[1], 'cleantext', remove_punctuation)
		sentiment=csvupdater(row, row[2], 'sentiment', stars_to_sentiment)
# 		print sentiment
		newrow=row+[cleantext, sentiment]
		# print newrow
		if newrow[2] != '3':
			newcsv.append(newrow)
		# print newcsv
# 		x=x+1
#  		if x > 5:
#  			break


# 
	
# csvoutputi.writerows(newcsv)
outputi.close()



##NOW TRAINING
inputi=open(outputname, "r")
csvinputi=csv.reader(inputi, dialect="excel")

fullset=[]

for row in csvinputi:
	fullset.append(row)
	
print "the dataset has {} rows".format(len(fullset))

with open("test-idx.json") as jsontestfile:
	jsontestdata=json.load(jsontestfile)
	
with open("train-idx.json") as jsontrainfile:
	jsontraindata=json.load(jsontrainfile)
	
testset=[fullset[i] for i in jsontestdata]
trainset=[fullset[i] for i in jsontraindata]
print "----newsflash-----"
print "the trainingset has {} rows".format(len(trainset))
print "the testset has {} rows".format(len(testset))
print "that adds up to {} rows".format(len(testset)+len(trainset)*2)




	


 	