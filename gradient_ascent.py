#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas
import json
import codecs
import string


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

for word in important_words:
	products[word]= products['review_clean'].apply(lambda s : s.split().count(word))

print products.columns

# 6. After #4 and #5, the data frame products should contain one column for each of the 193 important_words. As an example, the column perfect contains a count of the number of times the word perfect occurs in each of the reviews.
# 
# 7. Now, write some code to compute the number of product reviews that contain the word perfect.
# 
# Hint:
# 
# First create a column called contains_perfect which is set to 1 if the count of the word perfect (stored in column perfect is >= 1.
# Sum the number of 1s in the column contains_perfect.
# Quiz Question. How many reviews contain the word perfect?
# 
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
# 
# 1
# 2
# 3
# 4
# 5
# 6
# 7
# 8
# def get_numpy_data(dataframe, features, label):
#     dataframe['constant'] = 1
#     features = ['constant'] + features
#     features_frame = dataframe[features]
#     feature_matrix = features_frame.as_matrix()
#     label_sarray = dataframe[label]
#     label_array = label_sarray.as_matrix()
#     return(feature_matrix, label_array)
# 9. Using the function written in #8, extract two arrays feature_matrix and sentiment. The 2D array feature_matrix would contain the content of the columns given by the list important_words. The 1D array sentiment would contain the content of the column sentiment.
# 
# Quiz Question: How many features are there in the feature_matrix?
# 
# Quiz Question: Assuming that the intercept is present, how does the number of features in feature_matrix relate to the number of features in the logistic regression model?
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
# '''
# produces probablistic estimate for P(y_i = +1 | x_i, w).
# estimate ranges between 0 and 1.
# '''
# def predict_probability(feature_matrix, coefficients):
#     # Take dot product of feature_matrix and coefficients  
#     # YOUR CODE HERE
#     score = ...
#     
#     # Compute P(y_i = +1 | x_i, w) using the link function
#     # YOUR CODE HERE
#     predictions = ...
#     
#     # return predictions
#     return predictions
# Aside. How the link function works with matrix algebra
# 
# Since the word counts are stored as columns in feature_matrix, each i-th row of the matrix corresponds to the feature vector h(xi):
# 
# [feature_matrix]=⎡⎣⎢⎢⎢⎢h(x1)⊺h(x2)⊺⋮h(xN)⊺⎤⎦⎥⎥⎥⎥=⎡⎣⎢⎢⎢⎢h0(x1)h0(x2)⋮h0(xN)h1(x1)h1(x2)⋮h1(xN)⋯⋯⋱⋯hD(x1)hD(x2)⋮hD(xN)⎤⎦⎥⎥⎥⎥
# By the rules of matrix multiplication, the score vector containing elements w⊺h(xi) is obtained by multiplying feature_matrix and the coefficient vector w:
# 
# [score]=[feature_matrix]w=⎡⎣⎢⎢⎢⎢h(x1)⊺h(x2)⊺⋮h(xN)⊺⎤⎦⎥⎥⎥⎥w=⎡⎣⎢⎢⎢⎢h(x1)⊺wh(x2)⊺w⋮h(xN)⊺w⎤⎦⎥⎥⎥⎥=⎡⎣⎢⎢⎢⎢w⊺h(x1)w⊺h(x2)⋮w⊺h(xN)⎤⎦⎥⎥⎥⎥
