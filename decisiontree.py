#!/usr/bin/env python -W ignore::DeprecationWarning
 
 
import pandas
import numpy as np
import json
import codecs
import sklearn
from sklearn import preprocessing
from sklearn import model_selection
import sklearn.tree
header= "\n\n---\n\n"
# Identifying safe loans with decision trees
# 
# The LendingClub is a peer-to-peer leading company that directly connects borrowers and potential lenders/investors. In this notebook, you will build a classification model to predict whether or not a loan provided by LendingClub is likely to default.
# 
# In this notebook you will use data from the LendingClub to predict whether a loan will be paid off in full or the loan will be charged off and possibly go into default. In this assignment you will:
# 
# Use SFrames to do some feature engineering.
# Train a decision-tree on the LendingClub dataset.
# Visualize the tree.
# Predict whether a loan will default along with prediction probabilities (on a validation set).
# Train a complex tree model and compare it to simple tree model.
# If you are doing the assignment with IPython Notebook
# 

# 
# What you need to download
# 
# If you are using GraphLab Create:
# 
# Download the Lending club data in SFrame format:
# lending-club-data.gl.zip
# Download the companion IPython Notebook:
# module-5-decision-tree-assignment-1-blank.ipynb.zip
# Save both of these files in the same directory (where you are calling IPython notebook from) and unzip the data file.
# Follow the instructions contained in the IPython notebook.
# If you are not using GraphLab Create:
# 
# If you are using SFrame, download the LendingClub dataset in SFrame format:
# lending-club-data.gl.zip
# If you are using a different package, download the LendingClub dataset in CSV format:
# lending-club-data.csv.zip
# If you are using GraphLab Create and the companion IPython Notebook
# 
# 
# If you are using other tools
# 
# This section is designed for people using tools other than GraphLab Create. Even though some instructions are specific to scikit-learn, most part of the assignment should be applicable to other tools as well. However, we highly suggest you use SFrame since it is open source. In this part of the assignment, we describe general instructions, however we will tailor the instructions for SFrame and scikit-learn.
# 
# If you choose to use SFrame and scikit-learn, you should be able to follow the instructions here and complete the assessment. All code samples given here will be applicable to SFrame and scikit-learn.
# You are free to experiment with any tool of your choice, but some many not produce correct numbers for the quiz questions.
# Load the Lending Club dataset
# 
# We will be using a dataset from the LendingClub.
# 


loans= pandas.read_csv("/Users/ps22344/Downloads/coursera/lending-club-data.csv", low_memory=False)

print loans.describe()

# 2. Let's quickly explore what the dataset looks like. First, print out the column names to see what features we have in this dataset. On SFrame, you can run this code:
# 
print loans.columns

# Exploring the target column
# 
# The target column (label column) of the dataset that we are interested in is called bad_loans. In this column 1 means a risky (bad) loan 0 means a safe loan.
# 
# In order to make this more intuitive and consistent with the lectures, we reassign the target to be:
# 
# +1 as a safe loan
# -1 as a risky (bad) loan
# 3. We put this in a new column called safe_loans.
# 
loans['safe_loans']= loans['bad_loans'].apply(lambda x: -1 if x > 0 else +1 )


# 4. Now, let us explore the distribution of the column safe_loans. This gives us a sense of how many safe and risky loans are present in the dataset. Print out the percentage of safe loans and risky loans in the data frame.

print "good", float(sum(loans['safe_loans'] == 1))/loans.shape[0]*100
print "bad", float(sum(loans['safe_loans'] == -1))/loans.shape[0]*100


# 

# 5. In this assignment, we will be using a subset of features (categorical and numeric). The features we will be using are described in the code comments below. If you are a finance geek, the LendingClub website has a lot more details about these features. Extract these feature columns and target column from the dataset. We will only use these features.

features = ['grade',                     # grade of the loan
            'sub_grade',                 # sub-grade of the loan
            'short_emp',                 # one year or less of employment
            'emp_length_num',            # number of years of employment
            'home_ownership',            # home_ownership status: own, mortgage 
            'dti',                       # debt to income ratio
            'purpose',                   # the purpose of the loan
            'term',                      # the term of the loan
            'last_delinq_none',          # has borrower had a delinquincy
            'last_major_derog_none',     # has borrower had 90 day or worse 
            'revol_util',                # percent of available credit being 
            'total_rec_late_fee',        # total late fees received to day
           ]


target = 'safe_loans'                    # prediction target (y) (+1 means safe, 
#   -1 is risky)
# # Extract the feature columns and target column
loans = loans[features + [target]]
print "testi", loans.columns
# If you are NOT using SFrame, download the list of indices for the training and validation sets:
# 
# module-5-assignment-1-train-idx.json.zip
# module-5-assignment-1-validation-idx.json.zip
# Then follow the following steps:
# 
print "Apply one-hot encoding to loans. "
#Your tool may have a function for one-hot encoding. Alternatively, see #7 for implementation hints.


categorical_variables = []

for feat_name, feat_type in [(i,loans[i].dtype) for i in loans.columns]:
     if feat_type == 'O':
         categorical_variables.append(feat_name)
         
print "found these categoricals", categorical_variables


for feature in categorical_variables:
	loans_data_one_hot_encoded= pandas.get_dummies(loans[feature], prefix=feature)
	#merging in pandas sucks so much
	loans= pandas.concat([loans, loans_data_one_hot_encoded])
	loans = loans.drop(feature, 1)

for i in loans.columns:
	loans[i]= loans[i].fillna(0)
	
	
print "teti", loans.columns


# Load the JSON files into the lists train_idx and validation_idx.

with codecs.open('/Users/ps22344/Downloads/coursera/module-5-assignment-1-train-idx.json', 'r') as inputjson: 
	train_idx=json.load(inputjson)

with codecs.open('/Users/ps22344/Downloads/coursera/module-5-assignment-1-validation-idx.json', 'r') as inputjson: 
	validation_idx=json.load(inputjson)

train_data = loans.iloc[train_idx]
validation_data = loans.iloc[validation_idx]


#BUILD MODELS

# Build a decision tree classifier
# 
# 9. Now, let's use the built-in scikit learn decision tree learner (sklearn.tree.DecisionTreeClassifier) to create a loan prediction model on the training data. To do this, you will need to import sklearn, sklearn.tree, and numpy.
# 
# Note: You will have to first convert the SFrame into a numpy data matrix, and extract the target labels as a numpy array (Hint: you can use the .to_numpy() method call on SFrame to turn SFrames into numpy arrays). See the API for more information. Make sure to set max_depth=6.
# 
# Call this model decision_tree_model.

decision_tree_model= sklearn.tree.DecisionTreeClassifier(max_depth=6).fit(train_data.drop('safe_loans', axis=1), train_data['safe_loans'])

# 10. Also train a tree using with max_depth=2. Call this model small_model.

small_model= sklearn.tree.DecisionTreeClassifier(max_depth=2).fit(train_data.drop('safe_loans', axis=1), train_data['safe_loans'])

# Let's consider two positive and two negative examples from the validation set and see what the model predicts. We will do the following:

# Predict whether or not a loan is safe.
# Predict the probability that a loan is safe.
# 11. First, let's grab 2 positive examples and 2 negative examples.

validation_safe_loans = validation_data[validation_data['safe_loans'] == 1]
validation_risky_loans = validation_data[validation_data['safe_loans'] == -1]
sample_validation_data_risky = validation_risky_loans[0:2]
sample_validation_data_safe = validation_safe_loans[0:2]



#SCORER AND DATASET

def scorer(fitted_model, spread_sheet, gold_labels, message):
	"""
	compute accuracy
	"""
	print "Message: {}".format(message)
	data=spread_sheet.copy()
	data['predi']=fitted_model.predict(data.drop(gold_labels, axis=1))
	data['accu']=(data['predi']==data[gold_labels])
	accuracy= float(sum(data['accu']))/data.shape[0]
	print accuracy
	return accuracy
	





sample_validation_data= pandas.concat([sample_validation_data_risky, sample_validation_data_safe])

for i in sample_validation_data.iterrows():
	print i[1]['safe_loans']

count=0 


# 12. Now, we will use our model to predict whether or not a loan is likely to default. For each row in the sample_validation_data, use the decision_tree_model to predict whether or not the loan is classified as a safe loan. (Hint: if you are using scikit-learn, you can use the .predict() method)
# 

for i in sample_validation_data.drop('safe_loans', axis=1).iterrows():
	count=count+1
	#print (count, "risky loans and others", 		i[1].shape, 	i[1].values.reshape(1, -1).shape, 	i[1].as_matrix().shape)
	x=decision_tree_model.predict(i[1].values.reshape(1, -1))
	print "result", x


# Quiz Question: What percentage of the predictions on sample_validation_data did decision_tree_model get correct? 50%
#



# Explore probability predictions
# 
# 13. For each row in the sample_validation_data, what is the probability (according decision_tree_model) of a loan being classified as safe? (Hint: if you are using scikit-learn, you can use the .predict_proba() method)

for i in sample_validation_data.drop('safe_loans', axis=1).iterrows():
	count=count+1
	print count, "regular model probs"
	x=decision_tree_model.predict_proba(i[1].values.reshape(1, -1))
	print "result", x, decision_tree_model.predict(i[1].values.reshape(1, -1))

# 
# Quiz Question: Which loan has the highest probability of being classified as a safe loan? The 4th one
# 
# Checkpoint: Can you verify that for all the predictions with probability >= 0.5, the model predicted the label +1?
# Yes that is correct
# Tricky predictions!
# 
# 14. Now, we will explore something pretty interesting. For each row in the sample_validation_data, what is the probability (according to small_model) of a loan being classified as safe?
print header

for i in sample_validation_data.drop('safe_loans', axis=1).iterrows():
	count=count+1
	print count, "small model probs"
	x=small_model.predict_proba(i[1].values.reshape(1,-1))
	print "result", x, small_model.predict(i[1].values.reshape(1,-1))
# 
# Quiz Question: Notice that the probability preditions are the exact same for the 2nd and 3rd loans. Why would this happen?
# Cause they're in the same leaf
# Visualize the prediction on a tree
# 
# 14a. Note that you should be able to look at the small tree (of depth 2), traverse it yourself, and visualize the prediction being made. 
print "Consider the following point in the sample_validation_data[1]"


# 
# Quiz Question: Based on the visualized tree, what prediction would you make for this data point (according to small_model)? (If you don't have Graphviz, you can answer this quiz question by executing the next part.)
# 
print "Now, verify your prediction by examining the prediction made using small_model."

testpoint= sample_validation_data.drop('safe_loans', axis=1).iloc[1,:]
print small_model.predict_proba(testpoint.values.reshape(1,-1))
print small_model.predict(testpoint.values.reshape(1,-1))



# Evaluating accuracy of the decision tree model
# 
# Recall that the accuracy is defined as follows:
# 
# accuracy=# correctly classified data points# total data points
print "16. Evaluate the accuracy of small_model and decision_tree_model on the training data. (Hint: if you are using scikit-learn, you can use the .score() method)"


scorer(small_model, train_data, 'safe_loans', "small model on train")
scorer(decision_tree_model, train_data, 'safe_loans', "dec tree model on train")

#print small_model.score(train_data.drop('safe_loans', axis=1), train_data['safe_loans'])
#print decision_tree_model.score(train_data.drop('safe_loans', axis=1), train_data['safe_loans'])
# Checkpoint: You should see that the small_model performs worse than the decision_tree_model on the training data.
# 
print "17. Now, evaluate the accuracy of the small_model and decision_tree_model on the entire validation_data, not just the subsample considered above."
# 
scorer(small_model, validation_data, 'safe_loans', "small model on validation")
scorer(decision_tree_model, validation_data, 'safe_loans', "dec tree model on validation")



# Quiz Question: What is the accuracy of decision_tree_model on the validation set, rounded to the nearest .01?
# 0.851357173632
# Evaluating accuracy of a complex decision tree model
# 
# Here, we will train a large decision tree with max_depth=10. This will allow the learned tree to become very deep, and result in a very complex model. Recall that in lecture, we prefer simpler models with similar predictive power. This will be an example of a more complicated model which has similar predictive power, i.e. something we don't want.

big_model= sklearn.tree.DecisionTreeClassifier(max_depth=10).fit(train_data.drop('safe_loans', axis=1), train_data['safe_loans'])


# 18. Using sklearn.tree.DecisionTreeClassifier, train a decision tree with maximum depth = 10. Call this model big_model.
#

scorer(big_model, train_data, 'safe_loans', "giant model on train")
scorer(big_model, validation_data, 'safe_loans', "giant model on validation")

# 19. Evaluate the accuracy of big_model on the training set and validation set.
# 
# Checkpoint: We should see that big_model has even better performance on the training set than decision_tree_model did on the training set.
# 





# Quiz Question: How does the performance of big_model on the validation set compare to decision_tree_model on the validation set? Is this a sign of overfitting?
# YES
# Quantifying the cost of mistakes
# 
# Every mistake the model makes costs money. In this section, we will try and quantify the cost each mistake made by the model. Assume the following:
# 
# False negatives: Loans that were actually safe but were predicted to be risky. This results in an oppurtunity cost of loosing a loan that would have otherwise been accepted.
# False positives: Loans that were actually risky but were predicted to be safe. These are much more expensive because it results in a risky loan being given.
# Correct predictions: All correct predictions don't typically incur any cost.
# Let's write code that can compute the cost of mistakes made by the model. Complete the following 4 steps:
# 
# First, let us compute the predictions made by the model.
# Second, compute the number of false positives.
# Third, compute the number of false negatives.
# Finally, compute the cost of mistakes made by the model by adding up the costs of true positives and false positves.
# Quiz Question: Let's assume that each mistake costs us money: a false negative costs $10,000, while a false positive positive costs $20,000.
print " What is the total cost of mistakes made by decision_tree_model on validation_data?"

validation_data['predi']=decision_tree_model.predict(validation_data.drop('safe_loans', axis=1))
validation_data['false_pos']=((validation_data['predi']==1) & (validation_data['safe_loans']==-1))
#print validation_data[['safe_loans', 'predi', 'false_pos']]

validation_data['false_neg']=((validation_data['predi']==-1) & (validation_data['safe_loans']==1))
#print validation_data[['safe_loans', 'predi', 'false_pos', 'false_neg']]


print "Number of false positives is {} (cost: {}), number of false negatives is {} (cost: {}) out of a total {} rows".format(sum(validation_data['false_pos']), sum(validation_data['false_pos'])*20000, sum(validation_data['false_neg']), sum(validation_data['false_neg'])*10000, unicode(validation_data.shape[0]))

#Number of false positives is 671 (cost: 13420000), number of false negatives is 711 (cost: 7110000) out of a total 9284 rows