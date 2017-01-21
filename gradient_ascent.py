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
# 3. In this section, we will perform some simple feature cleaning using data frames. The last assignment used all words in building bag-of-words features, but here we limit ourselves to 193 words (for simplicity). We compiled a list of 193 most frequent words into the JSON file named important_words.json. Load the words into a list important_words.
# 
with codecs.open( '/Users/ps22344/Downloads/important_words.json',"r", "utf-8") as jsonin:
	important_words=json.load(jsonin)


# 4. Let us perform 2 simple data transformations:
# 
# Remove punctuation
# Compute word counts (only for important_words)
# We start with the first item as follows:
# 
# If your tool supports it, fill n/a values in the review column with empty strings. The n/a values indicate empty reviews. For instance, Pandas's the fillna() method lets you replace all N/A's in the review columns as follows:
# 
# 
# 1
# products = products.fillna({'review':''})  # fill in N/A's in the review column
products = products.fillna({'review':''})


# Write a function remove_punctuation that takes a line of text and removes all punctuation from that text. The function should be analogous to the following Python code:

def remove_punctuation(text):
	#this is not equivalent as it replaces with white space, not None
	table=string.maketrans(string.punctuation, "{}".format(" " * len(string.punctuation)))
	return text.translate(table)


# Apply the remove_punctuation function on every element of the review column and assign the result to the new column review_clean. Note. Many data frame packages support apply operation for this type of task. Consult appropriate manuals.
products['review_clean']= products['review'].apply(remove_punctuation)


#print products['review'][:10]
#print products['review_clean'][:10]



# 5. Now we proceed with the second item. For each word in important_words, we compute a count for the number of times the word occurs in the review. We will store this count in a separate column (one for each word). The result of this feature processing is a single column for each word in important_words which keeps a count of the number of times the respective word occurs in the review text.
# 
# Note: There are several ways of doing this. One way is to create an anonymous function that counts the occurrence of a particular word and apply it to every element in the review_clean column. Repeat this step for every word in important_words. Your code should be analogous to the following:
# 
# 
# 
# 1
# 2
# 3
# for word in important_words:
#     products[word] = products['review_clean'].apply(lambda s : s.split().count
#       (word))
#maybe this would be faster if it were its own dataframe and then merge
start=time.time()
for word in important_words:
	print "Adding", word
	products.insert(0, word, products['review_clean'].apply(lambda s : s.split().count(word)))
	
end=time.time()
print "This took us {} minutes".format((end-start)/60)

#print products['great']
# 6. After #4 and #5, the data frame products should contain one column for each of the 193 important_words. As an example, the column perfect contains a count of the number of times the word perfect occurs in each of the reviews.
# 
# 7. Now, write some code to compute the number of product reviews that contain the word perfect.
# 
# Hint:
# 
# First create a column called contains_perfect which is set to 1 if the count of the word perfect (stored in column perfect is >= 1.
#print len(products[products['perfect']!=0])

# Sum the number of 1s in the column contains_perfect.
# Quiz Question. How many reviews contain the word perfect?
# A: 3001
# Convert data frame to multi-dimensional array
# 
# 8. It is now time to convert our data frame to a multi-dimensional array. Look for a package that provides a highly optimized matrix operations. In the case of Python, NumPy is a good choice.
# 
# Write a function that extracts columns from a data frame and converts them into a multi-dimensional array. We plan to use them throughout the course, so make sure to get this function right.
# 
# The function should accept three parameters:
# 
# dataframe: a data frame to be converted
# features: a list of string, containing the names of the columns that are used as features.
# label: a string, containing the name of the single column that is used as class labels.
# The function should return two values:
# 
# one 2D array for features
# one 1D array for class labels
# The function should do the following:
# 
# Prepend a new column constant to dataframe and fill it with 1's. This column takes account of the intercept term. Make sure that the constant column appears first in the data frame.
# Prepend a string 'constant' to the list features. Make sure the string 'constant' appears first in the list.
# Extract columns in dataframe whose names appear in the list features.
# Convert the extracted columns into a 2D array using a function in the data frame library. If you are using Pandas, you would use as_matrix() function.
# Extract the single column in dataframe whose name corresponds to the string label.
# Convert the column into a 1D array.
# Return the 2D array and the 1D array.
# Users of SFrame or Pandas would execute these steps as follows:
# 
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






# 9. Using the function written in #8, extract two arrays feature_matrix and sentiment. The 2D array feature_matrix would contain the content of the columns given by the list important_words. The 1D array sentiment would contain the content of the column sentiment.
# 
# Quiz Question: How many features are there in the feature_matrix?
# 192 or however long important_words is


# Quiz Question: Assuming that the intercept is present, how does the number of features in feature_matrix relate to the number of features in the logistic regression model?
# they are features less 1, if you count the intercept
# 
# Estimating conditional probability with link function
# 
# 10. Recall from lecture that the link function is given by
# 
# P(yi=+1|xi,w)=11+exp(−w⊺h(xi)),
# where the feature vector h(xi) represents the word counts of important_words in the review xi. Write a function named predict_probability that implements the link function.
# 

# Take two parameters: feature_matrix and coefficients.
# First compute the dot product of feature_matrix and coefficients.
# Then compute the link function P(y=+1|x,w).
# Return the predictions given by the link function.
# Your code should be analogous to the following Python function:

# Aside. How the link function works with matrix algebra
# 
# Since the word counts are stored as columns in feature_matrix, each i-th row of the matrix corresponds to the feature vector h(xi):
# 
# [feature_matrix]=⎡⎣⎢⎢⎢⎢h(x1)⊺h(x2)⊺⋮h(xN)⊺⎤⎦⎥⎥⎥⎥=⎡⎣⎢⎢⎢⎢h0(x1)h0(x2)⋮h0(xN)h1(x1)h1(x2)⋮h1(xN)⋯⋯⋱⋯hD(x1)hD(x2)⋮hD(xN)⎤⎦⎥⎥⎥⎥
# By the rules of matrix multiplication, the score vector containing elements w⊺h(xi) is obtained by multiplying feature_matrix and the coefficient vector w:
# 
# [score]=[feature_matrix]w=⎡⎣⎢⎢⎢⎢h(x1)⊺h(x2)⊺⋮h(xN)⊺⎤⎦⎥⎥⎥⎥w=⎡⎣⎢⎢⎢⎢h(x1)⊺wh(x2)⊺w⋮h(xN)⊺w⎤⎦⎥⎥⎥⎥=⎡⎣⎢⎢⎢⎢w⊺h(x1)w⊺h(x2)⋮w⊺h(xN)⎤⎦⎥⎥⎥⎥


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
	logit=np.apply_along_axis(lambda x: 1/(1+np.exp(-x)), axis=0, arr=score)
	#Return the predictions given by the link function.
	return logit
	
#predictions=predict_probability(feature_matrix, [1.,2000])


# Compute derivative of log likelihood with respect to a single coefficient
# 
# 11. Recall from lecture:
# 
# ∂ℓ∂wj=∑i=1Nhj(xi)(1[yi=+1]−P(yi=+1|xi,w))
# We will now write a function feature_derivative that computes the derivative of log likelihood with respect to a single coefficient w_j. The function accepts two arguments:
# 
# errors: vector whose i-th value contains
# 1[yi=+1]−P(yi=+1|xi,w)
# feature: vector whose i-th value contains
# hj(xi)
# This corresponds to the j-th column of feature_matrix.
# 
# The function should do the following:
# 
# Take two parameters errors and feature.
# Compute the dot product of errors and feature.
# Return the dot product. This is the derivative with respect to a single coefficient w_j.
# Your code should be analogous to the following Python function:
# 
# 
# 
# 1
# 2
# 3
# 4
# 5
# def feature_derivative(errors, feature):     
#     # Compute the dot product of errors and feature
#     derivative = ...
#         # Return the derivative
#     return derivative

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




# 12. In the main lecture, our focus was on the likelihood. In the advanced optional video, however, we introduced a transformation of this likelihood---called the log-likelihood---that simplifies the derivation of the gradient and is more numerically stable. Due to its numerical stability, we will use the log-likelihood instead of the likelihood to assess the algorithm.
# 
# The log-likelihood is computed using the following formula (see the advanced optional video if you are curious about the derivation of this equation):
# 
# ℓℓ(w)=∑i=1N((1[yi=+1]−1)w⊺h(wi)−ln(1+exp(−w⊺h(xi))))
# Write a function compute_log_likelihood that implements the equation. The function would be analogous to the following Python function:
# 

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


# 
# 13. Now we are ready to implement our own logistic regression. All we have to do is to write a 
#gradient ascent function that takes gradient steps towards the optimum.
# 
# Write a function logistic_regression to fit a logistic regression model using gradient ascent.
# 

def logistic_regression(feature_matrix, sentiment, initial_coefficients, step_size, max_iter):
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
			#print "coef", coef
			#print "old coefs", coefs
			derivative= feature_derivative(errors, feature_matrix[:,coef])
			#print "derivative", derivative
			coefs[coef]=coefs[coef]+(step_size*derivative)
			#print "new coefs", coefs
		
		likelihood= compute_log_likelihood(feature_matrix, sentiment, coefs)
		print "likelihood is", likelihood
		
	# Once in a while, insert code to print out the log likelihood.
	
	# Repeat steps 2-6 for max_iter times.
	
	# At the end of day, your code should be analogous to the following Python function (with blanks filled in):
	# 
features=products[important_words].as_matrix()
	
logistic_regression(features, products['sentiment'].as_matrix(), np.zeros(features.shape[1]), 1e-7, 301)	

# The function accepts the following parameters:
# 
# feature_matrix: 2D array of features
# sentiment: 1D array of class labels
# initial_coefficients: 1D array containing initial values of coefficients
# step_size: a parameter controlling the size of the gradient steps
# max_iter: number of iterations to run gradient ascent
# The function returns the last set of coefficients after performing gradient ascent.
# 
# The function carries out the following steps:
# 
# Initialize vector coefficients to initial_coefficients.
# Predict the class probability P(yi=+1|xi,w) using your predict_probability function and save it to variable predictions.
# Compute indicator value for (yi=+1) by comparing sentiment against +1. Save it to variable indicator.
# Compute the errors as difference between indicator and predictions. Save the errors to variable errors.
# For each j-th coefficient, compute the per-coefficient derivative by calling feature_derivative with the j-th column of feature_matrix. Then increment the j-th coefficient by (step_size*derivative).
# Once in a while, insert code to print out the log likelihood.
# Repeat steps 2-6 for max_iter times.
# At the end of day, your code should be analogous to the following Python function (with blanks filled in):
# 
# 
# 
# 1
# 2
# 3
# 4
# 5
# 6
# 7
# 8
# 9
# 10
# 11
# 12
# 13
# 14
# 15
# 16
# 17
# 18
# 19
# 20
# 21
# 22
# 23
# 24
# 25
# 26
# 27
# 28
# 29
# 30
# 31
# from math import sqrt
# def logistic_regression(feature_matrix, sentiment, initial_coefficients, 
#   step_size, max_iter):
#     coefficients = np.array(initial_coefficients) # make sure it's a numpy array
#     for itr in xrange(max_iter):
#         # Predict P(y_i = +1|x_1,w) using your predict_probability() function
#         # YOUR CODE HERE
#         predictions = ...
#         # Compute indicator value for (y_i = +1)
#         indicator = (sentiment==+1)
#         # Compute the errors as indicator - predictions
#         errors = indicator - predictions
#         for j in xrange(len(coefficients)): # loop over each coefficient
#             # Recall that feature_matrix[:,j] is the feature column associated 
#               with coefficients[j]
#             # compute the derivative for coefficients[j]. Save it in a variable 
#               called derivative
#             # YOUR CODE HERE
#             derivative = ...
#             # add the step size times the derivative to the current coefficient
#             # YOUR CODE HERE
#             ...
#         # Checking whether log likelihood is increasing
#         if itr <= 15 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and itr % 
#           100 == 0) \
#         or (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:
#             lp = compute_log_likelihood(feature_matrix, sentiment, coefficients)
#             print 'iteration %*d: log likelihood of observed labels = %.8f' % \
#                 (int(np.ceil(np.log10(max_iter))), itr, lp)
#     return coefficients
# 14. Now, let us run the logistic regression solver with the parameters below:
# 
# feature_matrix = feature_matrix extracted in #9
# sentiment = sentiment extracted in #9
# initial_coefficients = a 194-dimensional vector filled with zeros
# step_size = 1e-7
# max_iter = 301
# Save the returned coefficients to variable coefficients.
# 
# Quiz question: As each iteration of gradient ascent passes, does the log likelihood increase or decrease?
# 
