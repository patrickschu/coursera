import sklearn, re, os, nltk,string, csv, random, json, scipy, numpy as np, time
from sklearn import linear_model


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


#for second part of asgmt: 
vocab = ['love', 'great', 'easy', 'old', 'little', 'perfect', 'loves', 
      'well', 'able', 'car', 'broke', 'less', 'even', 'waste', 'disappointed', 
      'work', 'product', 'money', 'would', 'return']

# for row in train_data:
# 	review=row[3].split()
# 	for word in review:
# 		wordcount=wordcount+1
# 		if word.lower() not in vocab:
#  			vocab[word.lower()]=1
#  		else:
#  			vocab[word.lower()]=vocab[word.lower()]+1
# # 			print word.lower(), "success"
# 		# else:
# # 			print word.lower()
# # 			break
# print "the vocab is {} words".format(len(vocab))
# print "the wordcount is {}". format(wordcount)
# 
# vocab={i for i in vocab if vocab[i] > 9}
# print "the final vocab is {} words".format(len(vocab))


#just for keeping an eye on this
progress=range(0, len(train_data), 1000)
starttime=time.time()
print "----iterating the day away-----"


## Build a sparse matrix where each row is the word count vector for the corresponding review. 
# Call this matrix train_matrix.
train_wordvector=[]
count=0

#how much training & test data do we want?
train_data=train_data
test_data=test_data
tt=[]
for i in test_data:
	tt.append(i[4])
	
print set(tt)
print tt.count('1')
print tt.count('-1')


# train:
#112164
# 21252

# test:
# 28095
# 5241


for row in train_data:
	train_wordvector.append([row[3].split().count(i) for i in vocab])
	count=count+1
	if count in progress:
		print "we're at row {}, it took us {}".format(count, time.time()-starttime)

train_matrix=scipy.sparse.coo_matrix(train_wordvector)
print "lenght of train wordvector list is {}".format(len(train_wordvector))
print "length of matrix is {}".format(train_matrix.shape)


test_wordvector=[]
count=0

for row in test_data:
	test_wordvector.append([row[3].split().count(i) for i in vocab])
	count=count+1
	if count in progress:
		print "we're at row {}, it took us {}".format(count, time.time()-starttime)

print "building a matrix"
test_matrix=scipy.sparse.coo_matrix(test_wordvector)
print "lenght of test wordvector list is {}".format(len(test_wordvector))


#model biulding
sentiment_model=linear_model.LogisticRegression()
print "let us build the model {}".format(str(sentiment_model))
sentiment_model.fit(train_matrix, [i[4] for i in train_data])
#print type (sentiment_model.coef_)
print len(np.ndarray.tolist(sentiment_model.coef_)[0])
#print [len(i) for i in np.ndarray.tolist(sentiment_model.coef_)[0]]
print "non zero coefficients", len([i for i in np.ndarray.tolist(sentiment_model.coef_)[0] if float(i) > 0])
#non zero coefficients 7461
#quiz asks for "greater than or equal"




#Take the 11th, 12th, and 13th data points in the test data and save them 
#to sample_test_data.
sample_test_data=test_data[10:13]
# for i in sample_test_data:
# 	print len(i), i
# 	print "-----"
# Quiz question: Of the three data points in sample_test_data, 
# which one (first, second, or third) has the lowest probability of being classified as a positive review?
# [ 4.92644954 -0.60896787 -8.66589186], thus the 3rd one
#note that we feed it the output of the iter above
decfunc=sentiment_model.decision_function(test_matrix)
prediction=sentiment_model.predict(train_matrix)
print "decision function: {}".format(decfunc)

print "prediction: {}".format(prediction)

#print "real: {}".format("/".join([",".join([i[2], i[4]]) for i in sample_test_data]))

# Using the sentiment_model, find the 20 reviews in the entire test_data with the highest probability of being 
# classified as a positive review. We refer to these as the "most positive reviews."
# 
# To calculate these top-20 reviews, use the following steps:
# 
# Make probability predictions on test_data using the sentiment_model.
# Sort the data according to those predictions and pick the top 20.
# Quiz Question: Which of the following products are represented in the 20 most positive reviews?
# Most neg: Safety 1st Exchangeable Tip 3 in 1 Thermometer, 
#The First Years True Choice P400 Premium Digital Monitor, 2 Parent Unit
# Fisher-Price Ocean Wonders Aquarium Bouncer' VTech Communications Safe &amp; Sounds Full Color Video and Audio Monitor
# Adiri BPA Free Natural Nurser Ultimate Bottle Stage 1 White, Slow Flow (0-3 months)
# Levana Safe N'See Digital Video Baby Monitor with Talk-to-Baby Intercom and Lullaby Control (LV-TW501)
# Cloth Diaper Sprayer--styles may vary'
# Philips AVENT Newborn Starter Set
# Peg-Perego Tatamia High Chair, White Latte
# Ellaroo Mei Tai Baby Carrier - Hershey
# 'Safety 1st High-Def Digital Monitor'
# Baby Jogger Summit XC Double Stroller, Red/Black'
# Motorola Digital Video Baby Monitor with Room Temperature Thermometer
# 'Nuby Natural Touch Silicone Travel Infa Feeder, Colors May Vary, 3 Ounce'
# VTech Communications Safe &amp; Sound Digital Audio Monitor with two Parent Units'
# 'Jolly Jumper Arctic Sneak A Peek Infant Car Seat Cover Black'
# Fisher-Price Discover 'n Grow Take-Along Play Blanket"
# Baby Trend Inertia Infant Car Seat - Horizon
# Evenflo Take Me Too Premiere Tandem Stroller - Castlebay
# ABYBJORN Baby Carrier Active, Black/Red
# Samsung SEW-3037W Wireless Pan Tilt Video Baby Monitor Infrared Night Vision and Zoom, 3.5 inch
# Keekaroo Height Right High Chair, Infant Insert and Tray Combo, Natural/Cherry'
# Ergobaby Performance Collection Charcoal Grey Carrier
# Today's Baby&reg; Sarasota Elite Convertible Crib
# 
# 
# most pos:
# nfantino Wrap and Tie Baby Carrier, Black Blueberries
# CTA Digital 2-in-1 iPotty with Activity Seat for iPad
# Graco FastAction Fold Jogger Click Connect Stroller, Grapeade'
# Britax Decathlon Convertible Car Seat, Tiffany'
# Graco Pack 'n Play Element Playard - Flint
# Roundabout Convertible Car Seat - Grey Wicker
# Diono RadianRXT Convertible Car Seat, Plum
# Roan Rocco Classic Pram Stroller 2-in-1 with Bassinet and Seat Unit - Coffee'
# Britax 2012 B-Agile Stroller, Red
# 'Mamas &amp; Papas 2014 Urbo2 Stroller - Black'
# venflo 6 Pack Classic Glass Bottle, 4-Ounce
# Freemie Hands-Free Concealable Breast Pump Collection System'
# Simple Wishes Hands-Free Breastpump Bra, Pink, XS-L'
# Phil &amp; Teds Navigator Buggy Golden Kiwi Free 2nd Seat NAVPR12200'
# Fisher-Price Cradle 'N Swing,  My Little Snugabunny"
# Baby Jogger City Mini GT Double Stroller, Shadow/Orange',
# Stokke Scoot Stroller - Light Green',
# ["P'Kolino Silly Soft Seating in Tias, Green
# bumGenius One-Size Cloth Diaper Twilight
# "Dr. Brown's Bottle Warme
# Stork Craft Beatrice Combo Tower Chest, White'
# hermag Glider Rocker Combo, Pecan with Oatmeal
# Dream On Me / Mia Moda  Atmosferra Stroller, Nero'
# isher-Price Zen Collection Cradle Swing



decfunc_all=list(sentiment_model.decision_function(test_matrix))
print vocab
print sentiment_model.coef_
#this might be wrong cause i did not fully understand index
#ok since these numbers are probably quite unique it should work 
# but to make it more legit, we should change it
decfunc_tosort=[(i, test_data[decfunc_all.index(i)]) for i in decfunc_all]
decfunc_sorted=sorted(decfunc_tosort, key=lambda x: x[0])
decfunc_sorted_pos=sorted(decfunc_tosort, key=lambda x: x[0], reverse=True)
# 
# print "Top 20 to be negative: "
# for i in decfunc_sorted[:25]:
# 	print i [:3]
# 	print "\n---\n"
# 
# print "Top 20 to be positive: "
# for i in decfunc_sorted_pos[:25]:
# 	print i [:3]
# 	print "\n---\n"
# 
# print "this is done"



#NOW TESTING AVCCURACY
# Step 1: Use the sentiment_model to compute class predictions.
# Step 2: Count the number of data points when the predicted class labels match the ground truth labels.
# Step 3: Divide the total number of correct predictions by the total number of data points in the dataset.


# number of correct predictions in test set (test_set [4] == predict [x]) divided by len(test_set)
# print len(i) for i in test_set
# the last one is the correct sentiment
prediction_enumerated=enumerate(list(prediction))
print "lenght of prediction is {}".format(len(prediction))
correct_preds=[x for x in prediction_enumerated if x[1]== train_data[x[0]][4]]
print "lenght of correct prediction is {}".format(len(correct_preds))
accuracy=float(len(correct_preds))/len(prediction)
print "accuracy is: ", accuracy


# Quiz Question: What is the accuracy of the sentiment_model on the test_data? Round your answer to 2 decimal
#  places (e.g. 0.76).
# lenght of prediction is 33336
# lenght of correct prediction is 30841
# accuracy is:  0.925155987521




# Quiz Question: Does a higher accuracy value on the training_data always imply that the classifier is better?
# No, we might be overfitting


# Learn another classifier with fewer words
# 
# 16. There were a lot of words in the model we trained above. 
#We will now train a simpler logistic regression model using only a subet of words that occur in the reviews. 
#For this assignment, we selected 20 words to work with. These are:

significant_words = ['love', 'great', 'easy', 'old', 'little', 'perfect', 'loves', 
      'well', 'able', 'car', 'broke', 'less', 'even', 'waste', 'disappointed', 
      'work', 'product', 'money', 'would', 'return']
      
      
# Compute a new set of word count vectors using only these words. The CountVectorizer class 
# has a parameter that lets you limit the choice of words when building word count vectors:
# 
# vectorizer_word_subset = CountVectorizer(vocabulary=significant_words) # limit to 20 words
# train_matrix_word_subset = vectorizer_word_subset.fit_transform(train_data['review_clean'])
# test_matrix_word_subset = vectorizer_word_subset.transform(test_data['review_clean'])
# Compute word count vectors for the training and test data and obtain the sparse matrices 
# train_matrix_word_subset and test_matrix_word_subset, respectively.
# 
# 17. Now build a logistic regression classifier with train_matrix_word_subset as features and sentiment as the target. Call this model simple_model.
# 
# 18. Let us inspect the weights (coefficients) of the simple_model. 
# First, build a table to store (word, coefficient) pairs. 
# If you are using SFrame with scikit-learn, you can combine words with coefficients by running
# 
# simple_model_coef_table = sframe.SFrame({'word':significant_words,
#                                          'coefficient':simple_model.coef_.flatten()})
# Sort the data frame by the coefficient value in descending order.
# 
# Note: Make sure that the intercept term is excluded from this table.
# 
# Quiz Question: Consider the coefficients of simple_model. How many of the 20 coefficients
#  (corresponding to the 20 significant_words) are positive for the simple_model?
#non zero coefficients 10
# Quiz Question: Are the positive words in the simple_model also positive words in the sentiment_model?
# ['love', 'great', 'easy', 'old', 'little', 'perfect', 'loves', 'well', 'able', 'car', 'broke', 'less', 'even',
# 'waste', 'disappointed', 'work', 'product', 'money', 'would', 'return']
# [[ 1.29743576  0.85273751  1.13232095  0.09919418  0.54216826  1.44127014
#    1.67366002  0.55692924  0.18628488  0.07450177 -1.64784022 -0.21726529
#   -0.51081505 -1.91901022 -2.31443406 -0.63558387 -0.29819197 -0.93284695
#   -0.38788878 -2.08867765]]




# Comparing models
# 
# 19. We will now compare the accuracy of the sentiment_model and the simple_model.
# 
# First, compute the classification accuracy of the sentiment_model on the train_data.
# 
# Now, compute the classification accuracy of the simple_model on the train_data.
#0.864454038496
# 
# Quiz Question: Which model (sentiment_model or simple_model) has higher accuracy on the TRAINING set?
# 
# 20. Now, we will repeat this exercise on the test_data. Start by computing the classification accuracy of the sentiment_model on the test_data.
# 
# Next, compute the classification accuracy of the simple_model on the test_data.
#accuracy is:  0.867350611951
# 
# Quiz Question: Which model (sentiment_model or simple_model) has higher accuracy on the TEST set?
# sentiment_model
# Baseline: Majority class prediction
# 
# 21. It is quite common to use the majority class classifier as the a baseline (or reference)
 # model for comparison with your classifier model. 
#  The majority classifier model predicts the majority class for all data points. 
#  At the very least, you should healthily beat the majority class classifier, 
#  otherwise, the model is (usually) pointless.
# 
# Quiz Question: Enter the accuracy of the majority class classifier model on the test_data. 
#Round your answer to two decimal places (e.g. 0.76).
## train:
#112164
# 21252

# test:
# 28095
# 5241
# 0.842782577394
# 
# Quiz Question: Is the sentiment_model definitely better than the majority class classifier (the baseline)?
# 



os.system('say "you passed the test"')
	
	

 	