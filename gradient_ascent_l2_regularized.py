#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas
import json
import codecs
import string
import time
import numpy as np
from scipy.special import expit


# 1. For this assignment, we will use a subset of the Amazon product review dataset. The subset was chosen to contain similar numbers of positive and negative reviews, as the original dataset consisted primarily of positive reviews.
# 
# Load the dataset into a data frame named products. One column of this dataset is sentiment, corresponding to the class label with +1 indicating a review with positive sentiment and -1 for negative sentiment.

with codecs.open("/Users/ps22344/Downloads/amazon_baby_subset.csv", "r", "utf-8") as csvin:
	products=pandas.read_csv(csvin)


# 
# 2. Let us quickly explore more of this dataset. The name column indicates the name of the product. Try listing the name of the first 10 products in the dataset.
# 
print products['name'][:10]


# After that, try counting the number of positive and negative reviews.
# 
# Note: For this assignment, we eliminated class imbalance by choosing a subset of the data with a similar number of positive and negative reviews.
print products.columns
#Index([u'name', u'review', u'rating', u'sentiment'], dtype='object')

good=len(products[products['sentiment']==1])
bad=len(products[products['sentiment']==-1])
print "good", good
print "bad", bad
print "total", len(products.index), "versus", good+bad


# 
# Apply text cleaning on the review data
# 
# 
with codecs.open( '/Users/ps22344/Downloads/important_words.json',"r", "utf-8") as jsonin:
	important_words=json.load(jsonin)


products = products.fillna({'review':''})


# Write a function remove_punctuation that takes a line of text and removes all punctuation from that text. The function should be analogous to the following Python code:

def remove_punctuation(text):
	#this is not equivalent as it replaces with white space, not None
	#table=string.maketrans(string.punctuation, "{}".format(" " * len(string.punctuation)))
	return text.translate(None, string.punctuation)


# Apply the remove_punctuation function on every element of the review column and assign the result to the new column review_clean. Note. Many data frame packages support apply operation for this type of task. Consult appropriate manuals.
products['review_clean']= products['review'].apply(remove_punctuation)


start=time.time()
for word in important_words:
	print "Adding", word
	products.insert(0, word, products['review_clean'].apply(lambda s : s.split().count(word)))
	
end=time.time()
print "This took us {} minutes".format((end-start)/60)


with codecs.open('/Users/ps22344/Downloads/module-4-assignment-train-idx.json', "r", "utf-8") as trainin:
	trainindexes=json.load(trainin)

with codecs.open('/Users/ps22344/Downloads/module-4-assignment-validation-idx.json', "r", "utf-8") as valin:
	valindexes=json.load(valin)

train_data= products.iloc[trainindexes,:]
validation_data= products.iloc[valindexes,:]


# 
def get_numpy_data(data_frame, features_list, labels):
	"""
	Parameters
	----
	dataframe: a data frame to be converted
	features: a list of string, containing the names of the columns that are used as features.
	label: a string, containing the name of the single column that is used as class labels.
	Returns
	---
	The function should return two values:
	one 2D array for features 
	one 1D array for class labels
	"""
	data_frame.insert(0, 'constant', 1)
	matrix= data_frame[features_list].as_matrix()
	label_column= data_frame[labels].as_matrix()
	return  matrix, label_column


	
#feature_matrix, sentiment=get_numpy_data(products, ['baby', 'one'], 'sentiment')

feature_matrix_train, sentiment_train = get_numpy_data(train_data, important_words, 'sentiment')
feature_matrix_valid, sentiment_valid = get_numpy_data(validation_data, important_words, 'sentiment') 


print "matrix making done"

def predict_probability(feature_matrix, coefficients):
	"""
	Parameters: 
	---
	feature_matrix 
	coefficients.
	
	Returns:
	---
	predictions given by the link function.
	"""
	#First compute the dot product of feature_matrix and coefficients.
	#t=np.dot(feature_matrix, coefficients)
	score= np.sum(feature_matrix*coefficients, axis=1)
	#Then compute the link function P(y=+1|x,w).
	#this is the sigmoid function
	logit=np.apply_along_axis(lambda x: 1/(1+np.exp(-x)), axis=0, arr=score)
	#Return the predictions given by the link function.
	return logit
	
#predictions=predict_probability(feature_matrix_train, [1.,2000])




def feature_derivative(errors, feature_column):
	"""
	Compute derivative of log likelihood with respect to a single coefficient
	Parameters:
	---
	errors: vector whose i-th value contains
	1[yi=+1]−P(yi=+1|xi,w)
	feature: vector whose i-th value contains
	hj(xi)
	This corresponds to the j-th column of feature_matrix.
	Returns:
	---
	the dot product
	"""
	#print "errors", errors
	#print "column", feature_column
	# t=np.dot(feature_column, errors)
	# Compute the dot product of errors and feature.
	derivative= np.sum(feature_column*errors)
	# Return the dot product. This is the derivative with respect to a single coefficient w_j.
	return derivative

#feature_derivative(1-predictions, products['baby'].as_matrix())

def feature_derivative_with_L2(errors, feature_column, coefficient, l2_penalty, feature_is_constant):
	"""
	Compute derivative of log likelihood with respect to a single coefficient
	Parameters:
	---
	errors: vector whose i-th value contains
	1[yi=+1]−P(yi=+1|xi,w)
	feature: vector whose i-th value contains
	hj(xi)
	This corresponds to the j-th column of feature_matrix.
	Returns:
	---
	the dot product
	"""
	#print "errors", errors
	#print "column", feature_column
	# Compute the dot product of errors and feature
	derivative= np.sum(feature_column*errors)
	# add L2 penalty term for any feature that isn't the intercept.
	if not feature_is_constant: 
		derivative= derivative - 2*l2_penalty*np.sum(coefficient**2)
	return derivative

#r=feature_derivative_with_L2(1-predictions, products.as_matrix(), 1, 0.002, False)
#print "derri", r


def compute_log_likelihood(feature_matrix, sentiment, coefficients):
	"""
	The log-likelihood is computed using the following formula 
	 ℓℓ(w)=∑i=1N((1[yi=+1]−1)w⊺h(wi)−ln(1+exp(−w⊺h(xi))))
	Write a function compute_log_likelihood that implements the equation. 
	The function would be analogous to the following Python function.
	"""
	indicator= (sentiment==+1)
	#print "indicator", indicator-1
	#this gives us array([False, False,  True, False, False]
	#apparently this translates to [0,0,1,0,0]
	scores= np.sum(feature_matrix*coefficients, axis=1)
	#print scores
	#print np.dot(feature_matrix,coefficients)
	lp= np.sum((indicator-1)*scores - np.log(1.+np.exp(-scores)))
	return lp

#compute_log_likelihood(products[['baby','one']].as_matrix(), products['sentiment'].as_matrix(), [1,2])


def compute_log_likelihood_with_L2(feature_matrix, sentiment, coefficients, l2_penalty):
	"""
	The log-likelihood is computed using the following formula 
	 ℓℓ(w)=∑i=1N((1[yi=+1]−1)w⊺h(wi)−ln(1+exp(−w⊺h(xi))))
	Write a function compute_log_likelihood that implements the equation. 
	The function would be analogous to the following Python function.
	"""
	indicator= (sentiment==+1)
	#print "indicator", indicator-1
	#this gives us array([False, False,  True, False, False]
	#apparently this translates to [0,0,1,0,0]
	scores= np.sum(feature_matrix*coefficients, axis=1)
	#print scores
	#the 1: is to exclude the intercept
	lp= np.sum((indicator-1)*scores - np.log(1.+np.exp(-scores))) - (l2_penalty*np.sum(coefficients[1:]**2))
	return lp

# 
# 13. Now we are ready to implement our own logistic regression. All we have to do is to write a 
#gradient ascent function that takes gradient steps towards the optimum.
# 
# Write a function logistic_regression to fit a logistic regression model using gradient ascent.
# 



def logistic_regression_with_L2(feature_matrix, sentiment, initial_coefficients, step_size, l2_penalty, max_iter):
	"""
	Initialize vector coefficients to initial_coefficients.
	Predict the class probability P(yi=+1|xi,w) using your predict_probability function and save it to variable predictions.
	Compute indicator value for (yi=+1) by comparing sentiment against +1. Save it to variable indicator.
	Compute the errors as difference between indicator and predictions. Save the errors to variable errors.
	For each j-th coefficient, compute the per-coefficient derivative by calling feature_derivative with the j-th column of feature_matrix. Then increment the j-th coefficient by (step_size*derivative).
	Once in a while, insert code to print out the log likelihood.
	Repeat steps 2-6 for max_iter times.
	
	Parameters:
	---
	feature_matrix: 2D array of features
	sentiment: 1D array of class labels
	initial_coefficients: 1D array containing initial values of coefficients
	step_size: a parameter controlling the size of the gradient steps
	max_iter: number of iterations to run gradient ascent
	
	Return:
	---
	coefficients
	"""
	# Initialize vector coefficients to initial_coefficients.
	coefs=np.array(initial_coefficients)
	for iter in xrange(max_iter):
		print "---\n\nthis is iter", iter
		# Predict the class probability P(yi=+1|xi,w) using your predict_probability function and save it to variable predictions.
		predictions= predict_probability(feature_matrix, coefs)
		# Compute indicator value for (yi=+1) by comparing sentiment against +1. Save it to variable indicator.
		indicator= (sentiment==+1)
		# Compute the errors as difference between indicator and predictions. Save the errors to variable errors.
		errors= indicator-predictions
		#e.g. 1-.7, 1-.2, 0-.2
		# For each j-th coefficient, compute the per-coefficient derivative by calling 
		#feature_derivative with the j-th column of feature_matrix. 
		#Then increment the j-th coefficient by (step_size*derivative).
		for coef in xrange(len(coefs)):
			print "old coef", coef
			if coef == 0:
				print "intercept", coef
				derivative= feature_derivative_with_L2(errors, feature_matrix[:,coef], coef, l2_penalty, True)
				coefs[coef]=coefs[coef]+(step_size*derivative)
			else:
				derivative= feature_derivative_with_L2(errors, feature_matrix[:,coef], coef, l2_penalty, False)
				print "derivative", derivative
				coefs[coef]=coefs[coef]+(step_size*derivative)
				print "new coef", coef
		
		likelihood= compute_log_likelihood(feature_matrix, sentiment, coefs)
		print "likelihood is", likelihood
		#probs= predict_probability(feature_matrix, coefs)
		probs= np.sum(feature_matrix*coefs, axis=1)
		print "positive preds", np.sum(probs>0.0)
		print "negative preds", np.sum(probs<0.0)
		print "zero preds", np.sum(probs==0)
		merge= np.column_stack([probs,sentiment])
		correctpos= merge[(merge[:,0]>0) & (merge[:,1]==1)]
		correctneg= merge[(merge[:,0]<0) & (merge[:,1]==-1)]
		print "correct positives", len(correctpos)
		print "correct negatives", len(correctneg)
		print "ratio", (float(len(correctpos))+len(correctneg))/len(sentiment)
		tuplettes= [(w,c) for w,c in zip(important_words, coefs)]
		sortedtuplettes= sorted(tuplettes, key=lambda x:x[1], reverse=True)
		print "good", sortedtuplettes[:11]
		print "bad", sortedtuplettes[-11:]
		print "final model paras", coefs
		return {'coefs':coefs, 'good':sortedtuplettes[:11], 'bad':sortedtuplettes[-11:]}



features=train_data[important_words.append('constant')].as_matrix()
print "len features", features.shape
	
logistic_regression_with_L2(features, train_data['sentiment'].as_matrix(), np.zeros(features.shape[1]+1), 5e-6, 0, 501)	


coefficients_0_penalty=logistic_regression_with_L2(features, train_data['sentiment'].as_matrix(), np.zeros(features.shape[1]+1), 5e-6,  0, 501)	
with codecs.open("0.json", "w") as outi:
	json.dump(coefficients_0_penalty, outi)

coefficients_4_penalty= logistic_regression_with_L2(features, train_data['sentiment'].as_matrix(), np.zeros(features.shape[1]+1), 5e-6,  4, 501)	
with codecs.open("4.json", "w") as outi:
	json.dump(coefficients_4_penalty, outi)

coefficients_10_penalty= logistic_regression_with_L2(features, train_data['sentiment'].as_matrix(), np.zeros(features.shape[1]+1), 5e-6,  10, 501)	
with codecs.open("10.json", "w") as outi:
	json.dump(coefficients_10_penalty, outi)

coefficients_1e2_penalty= logistic_regression_with_L2(features, train_data['sentiment'].as_matrix(), np.zeros(features.shape[1]+1), 5e-6,  1e2, 501)	
with codecs.open("1e2.json", "w") as outi:
	json.dump(coefficients_1e2_penalty, outi)

coefficients_1e3_penalty= logistic_regression_with_L2(features, train_data['sentiment'].as_matrix(), np.zeros(features.shape[1]+1), 5e-6,  1e3, 501)	
with codecs.open("1e3.json", "w") as outi:
	json.dump(coefficients_1e3_penalty, outi)

coefficients_1e5_penalty=logistic_regression_with_L2(features, train_data['sentiment'].as_matrix(), np.zeros(features.shape[1]+1), 5e-6,  1e5, 501)	
with codecs.open("1e5.json", "w") as outi:
	json.dump(coefficients_1e5_penalty, outi)


