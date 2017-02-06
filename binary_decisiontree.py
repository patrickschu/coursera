 
 
import pandas
import numpy as np
import json
import codecs
import sklearn
from sklearn import preprocessing
from sklearn import model_selection
import sklearn.tree
header= "\n\n---\n\n"

loans= pandas.read_csv("/Users/ps22344/Downloads/coursera/lending-club-data.csv", low_memory=False)


loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
loans = loans.drop('bad_loans', axis=1)

features = ['grade',              # grade of the loan
             'term',               # the term of the loan
             'home_ownership',     # home_ownership status: own, mortgage or rent
             'emp_length',         # number of years of employment
            ]
target = 'safe_loans'





print "shape loans", loans.shape

# Extract these feature columns from the dataset, and discard the rest of the feature columns.
loans=loans[features+['safe_loans']]




# module-5-assignment-2-train-idx.json.zip
# module-5-assignment-2-test-idx.json.zip




# Then follow the following steps:
# 
# Apply one-hot encoding to loans. Your tool may have a function for one-hot encoding.
features= []
for f in [i for i in loans.columns if i not in ['safe_loans']]:
	print f, features.append(f), type(f)
	
encoded=pandas.DataFrame()

	
for f in features:
	hottie= pandas.get_dummies(loans[f], prefix=f)
	print "hottie", hottie.shape
	encoded=pandas.concat([encoded, hottie], axis=1)
	print "encoded", encoded.shape
	loans= loans.drop(f, axis=1)

	
loans= pandas.concat([loans,encoded])
print "shape loans", loans.shape
print loans.columns

for i in loans.columns:
	loans[i]= loans[i].fillna(0)

#print loans.describe 

# Load the JSON files into the lists train_idx and test_idx.
with codecs.open("/Users/ps22344/Downloads/coursera/module-5-assignment-2-train-idx.json", "r") as inputjson:
	train_idx= json.load(inputjson)

with codecs.open("/Users/ps22344/Downloads/coursera/module-5-assignment-2-test-idx.json", "r") as inputjson:
	test_idx= json.load(inputjson) 
	
	
	
# 
# Now proceed to the section "Decision tree implementation", skipping three sections below.

# 

# 

# 
# Decision tree implementation
# 
# In this section, we will implement binary decision trees from scratch. There are several steps involved in building a decision tree. For that reason, we have split the entire assignment into several sections.
# 
# Function to count number of mistakes while predicting majority class
# 
# Recall from the lecture that prediction at an intermediate node works by predicting the majority class for all data points that belong to this node. Now, we will write a function that calculates the number of misclassified examples when predicting the majority class. This will be used to help determine which feature is the best to split on at a given node of the tree.
# 
# Note: Keep in mind that in order to compute the number of mistakes for a majority classifier, we only need the label (y values) of the data points in the node.
# 
# Steps to follow:
# 
# Step 1: Calculate the number of safe loans and risky loans.
# Step 2: Since we are assuming majority class prediction, all the data points that are not in the majority class are considered mistakes.
# Step 3: Return the number of mistakes.
# 7. Now, let us write the function intermediate_node_num_mistakes which computes the number of misclassified examples of an intermediate node given the set of labels (y values) of the data points contained in the node. Your code should be analogous to
def intermediate_node_num_mistakes(labels):
	if len(labels) == 0:
		print "empty node"
		return 0
	#count 1s
	labels= np.array(labels)
	pos=sum((labels ==1))
	neg=sum((labels == -1))
	return labels.shape[0] - np.amax(np.array([pos, neg]))
	
	
intermediate_node_num_mistakes([-1, -1, 1, 1, 1])



# The function best_splitting_feature takes 3 arguments:
# 
# The data
# The features to consider for splits (a list of strings of column names to consider for splits)
# The name of the target/label column (string)
# The function will loop through the list of possible features, and consider splitting on each of them. It will calculate the classification error of each split and return the feature that had the smallest classification error when split on.
# 
# Recall that the classification error is defined as follows:
# 
# classification error=# mistakes# total examples
# 9. Follow these steps to implement best_splitting_feature:
# 
# Step 1: Loop over each feature in the feature list
# Step 2: Within the loop, split the data into two groups: one group where all of the data has feature value 0 or False (we will call this the left split), and one group where all of the data has feature value 1 or True (we will call this the right split). Make sure the left split corresponds with 0 and the right split corresponds with 1 to ensure your implementation fits with our implementation of the tree building process.
# Step 3: Calculate the number of misclassified examples in both groups of data and use the above formula to compute theclassification error.
# Step 4: If the computed error is smaller than the best error found so far, store this feature and its error.
# Note: Remember that since we are only dealing with binary features, we do not have to consider thresholds for real-valued features. This makes the implementation of this function much easier.
# 
# Your code should be analogous to
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
# 32
# 33
# 34
# 35
# 36
# 37
# 38
# 39
# 40
# def best_splitting_feature(data, features, target):
#     
#     target_values = data[target]
#     best_feature = None # Keep track of the best feature 
#     best_error = 10     # Keep track of the best error so far 
#     # Note: Since error is always <= 1, we should intialize it with something 
#       larger than 1.
#     # Convert to float to make sure error gets computed correctly.
#     num_data_points = float(len(data))  
#     
#     # Loop through each feature to consider splitting on that feature
#     for feature in features:
#         
#         # The left split will have all data points where the feature value is 0
#         left_split = data[data[feature] == 0]
#         
#         # The right split will have all data points where the feature value is 1
#         ## YOUR CODE HERE
#         right_split =  
#             
#         # Calculate the number of misclassified examples in the left split.
#         # Remember that we implemented a function for this! (It was called 
#           intermediate_node_num_mistakes)
#         # YOUR CODE HERE
#         left_mistakes =             
#         # Calculate the number of misclassified examples in the right split.
#         ## YOUR CODE HERE
#         right_mistakes = 
#             
#         # Compute the classification error of this split.
#         # Error = (# of mistakes (left) + # of mistakes (right)) / (# of data 
#           points)
#         ## YOUR CODE HERE
#         error = 
#         # If this is the best error we have found so far, store the feature as 
#           best_feature and the error as best_error
#         ## YOUR CODE HERE
#         if error < best_error:
#         
#     
#     return best_feature # Return the best feature we found
# Building the tree
# 
# With the above functions implemented correctly, we are now ready to build our decision tree. Each node in the decision tree is represented as a dictionary which contains the following keys and possible values:
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
# { 
#    'is_leaf'            : True/False.
#    'prediction'         : Prediction at the leaf node.
#    'left'               : (dictionary corresponding to the left tree).
#    'right'              : (dictionary corresponding to the right tree).
#    'splitting_feature'  : The feature that this node splits on
# }
# 10. First, we will write a function that creates a leaf node given a set of target values. Your code should be analogous to
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
# def create_leaf(target_values):    
#     # Create a leaf node
#     leaf = {'splitting_feature' : None,
#             'left' : None,
#             'right' : None,
#             'is_leaf':     }   ## YOUR CODE HERE 
#    
#     # Count the number of data points that are +1 and -1 in this node.
#     num_ones = len(target_values[target_values == +1])
#     num_minus_ones = len(target_values[target_values == -1])    
#     # For the leaf node, set the prediction to be the majority class.
#     # Store the predicted class (1 or -1) in leaf['prediction']
#     if num_ones > num_minus_ones:
#         leaf['prediction'] =          ## YOUR CODE HERE
#     else:
#         leaf['prediction'] =          ## YOUR CODE HERE        
#     # Return the leaf node
#     return leaf 
# We have provided a function that learns the decision tree recursively and implements 3 stopping conditions:
# 
# Stopping condition 1: All data points in a node are from the same class.
# Stopping condition 2: No more features to split on.
# Additional stopping condition: In addition to the above two stopping conditions covered in lecture, in this assignment we will also consider a stopping condition based on the max_depth of the tree. By not letting the tree grow too deep, we will save computational effort in the learning process.
# 11. Now, we will provide a Python skeleton of the learning algorithm. Note that this code is not complete; it needs to be completed by you if you are using Python. Otherwise, your code should be analogous to
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
# 32
# 33
# 34
# 35
# 36
# 37
# 38
# 39
# 40
# 41
# 42
# 43
# 44
# 45
# 46
# 47
# 48
# 49
# 50
# 51
# 52
# 53
# 54
# 55
# 56
# 57
# 58
# def decision_tree_create(data, features, target, current_depth = 0, max_depth = 
#   10):
#     remaining_features = features[:] # Make a copy of the features.
#     
#     target_values = data[target]
#     print "--------------------------------------------------------------------"
#     print "Subtree, depth = %s (%s data points)." % (current_depth, len
#       (target_values))
#     
#     # Stopping condition 1
#     # (Check if there are mistakes at current node.
#     # Recall you wrote a function intermediate_node_num_mistakes to compute this
#       .)
#     if  == 0:  ## YOUR CODE HERE
#         print "Stopping condition 1 reached."     
#         # If not mistakes at current node, make current node a leaf node
#         return create_leaf(target_values)
#     
#     # Stopping condition 2 (check if there are remaining features to consider 
#       splitting on)
#     if remaining_features == :   ## YOUR CODE HERE
#         print "Stopping condition 2 reached."    
#         # If there are no remaining features to consider, make current node a 
#           leaf node
#         return create_leaf(target_values)    
#     
#     # Additional stopping condition (limit tree depth)
#     if current_depth >= :  ## YOUR CODE HERE
#         print "Reached maximum depth. Stopping for now."
#         # If the max tree depth has been reached, make current node a leaf node
#         return create_leaf(target_values)
#     # Find the best splitting feature (recall the function 
#       best_splitting_feature implemented above)
#     ## YOUR CODE HERE
#     
#     # Split on the best feature that we found. 
#     left_split = data[data[splitting_feature] == 0]
#     right_split =       ## YOUR CODE HERE
#     remaining_features.remove(splitting_feature)
#     print "Split on feature %s. (%s, %s)" % (\
#                       splitting_feature, len(left_split), len(right_split))
#     
#     # Create a leaf node if the split is "perfect"
#     if len(left_split) == len(data):
#         print "Creating leaf node."
#         return create_leaf(left_split[target])
#     if len(right_split) == len(data):
#         print "Creating leaf node."
#         ## YOUR CODE HERE
#         
#     # Repeat (recurse) on left and right subtrees
#     left_tree = decision_tree_create(left_split, remaining_features, target, 
#       current_depth + 1, max_depth)        
#     ## YOUR CODE HERE
#     right_tree = 
#     return {'is_leaf'          : False, 
#             'prediction'       : None,
#             'splitting_feature': splitting_feature,
#             'left'             : left_tree, 
#             'right'            : right_tree}
# Build the tree!
# 
# 12. Train a tree model on the train_data. Limit the depth to 6 (max_depth = 6) to make sure the algorithm doesn't run for too long. Call this tree my_decision_tree. Warning: The tree may take 1-2 minutes to learn.
# 
# Making predictions with a decision tree
# 
# 13. As discussed in the lecture, we can make predictions from the decision tree with a simple recursive function. Write a function called classify, which takes in a learned tree and a test point x to classify. Include an option annotate that describes the prediction path when set to True. Your code should be analogous to
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
# def classify(tree, x, annotate = False):
#        # if the node is a leaf node.
#     if tree['is_leaf']:
#         if annotate:
#              print "At leaf, predicting %s" % tree['prediction']
#         return tree['prediction']
#      else:
#         # split on feature.
#         split_feature_value = x[tree['splitting_feature']]
#         if annotate:
#              print "Split on %s = %s" % (tree['splitting_feature'], 
#                split_feature_value)
#         if split_feature_value == 0:
#             return classify(tree['left'], x, annotate)
#         else:
#                ### YOUR CODE HERE
# 14. Now, let's consider the first example of the test set and see what my_decision_tree model predicts for this data point.
# 
# 
# 
# 1
# 2
# 3
# print test_data[0]
# print 'Predicted class: %s ' % classify(my_decision_tree, test_data[0])
# 15. Let's add some annotations to our prediction to see what the prediction path was that lead to this predicted class:
# 
# 
# 
# 1
# classify(my_decision_tree, test_data[0], annotate=True)
# Quiz question: What was the feature that my_decision_tree first split on while making the prediction for test_data[0]?
# 
# Quiz question: What was the first feature that lead to a right split of test_data[0]?
# 
# Quiz question: What was the last feature split on before reaching a leaf node for test_data[0]?
# 
# Evaluating your decision tree
# 
# 16. Now, we will write a function to evaluate a decision tree by computing the classification error of the tree on the given dataset. Write a function called evaluate_classification_error that takes in as input:
# 
# tree (as described above)
# data (a data frame of data points)
# This function should return a prediction (class label) for each row in data using the decision tree. Your code should be analogous to
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
# def evaluate_classification_error(tree, data):
#     # Apply the classify(tree, x) to each row in your data
#     prediction = data.apply(lambda x: classify(tree, x))
#     
#     # Once you've made the predictions, calculate the classification error and 
#       return it
#     ## YOUR CODE HERE
#     
# 17. Now, use this function to evaluate the classification error on the test set.
# 
# 
# 
# 1
# evaluate_classification_error(my_decision_tree, test_data)
# Quiz Question: Rounded to 2nd decimal point, what is the classification error of my_decision_tree on the test_data?
# 
# Printing out a decision stump
# 
# 18. As discussed in the lecture, we can print out a single decision stump (printing out the entire tree is left as an exercise to the curious reader). Here we provide Python code to visualize a decision stump. If you are using different software, make sure your code is analogous to:
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
# def print_stump(tree, name = 'root'):
#     split_name = tree['splitting_feature'] # split_name is something like 'term. 
#       36 months'
#     if split_name is None:
#         print "(leaf, label: %s)" % tree['prediction']
#         return None
#     split_feature, split_value = split_name.split('.')
#     print '                       %s' % name
#     print '         |---------------|----------------|'
#     print '         |                                |'
#     print '         |                                |'
#     print '         |                                |'
#     print '  [{0} == 0]               [{0} == 1]    '.format(split_name)
#     print '         |                                |'
#     print '         |                                |'
#     print '         |                                |'
#     print '    (%s)                         (%s)' \
#         % (('leaf, label: ' + str(tree['left']['prediction']) if 
#           tree['left']['is_leaf'] else 'subtree'),
#            ('leaf, label: ' + str(tree['right']['prediction']) if 
#              tree['right']['is_leaf'] else 'subtree'))
# 19. Using this function, we can print out the root of our decision tree:
# 
# 
# 
# 1
# print_stump(my_decision_tree)
# Quiz Question: What is the feature that is used for the split at the root node?
# 
# Exploring the intermediate left subtree
# 
# The tree is a recursive dictionary, so we do have access to all the nodes! We can use
# 
# my_decision_tree['left'] to go left
# my_decision_tree['right'] to go right
# 20. We can print out the left subtree by running the code
# 
# 
# 
# 1
# print_stump(my_decision_tree['left'], my_decision_tree['splitting_feature'])
# We can similarly print out the left subtree of the left subtree of the root by running the code
# 
# 
# 
# 1
# print_stump(my_decision_tree['left']['left'], 
#   my_decision_tree['left']['splitting_feature'])
# Quiz question: What is the path of the first 3 feature splits considered along the left-most branch of my_decision_tree?
# 
# Quiz question: What is the path of the first 3 feature splits considered along the right-most branch of my_decision_tree?
# 










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

# 
# loans= pandas.read_csv("/Users/ps22344/Downloads/coursera/lending-club-data.csv", low_memory=False)
# 
# print loans.describe()
# 
# # 2. Let's quickly explore what the dataset looks like. First, print out the column names to see what features we have in this dataset. On SFrame, you can run this code:
# # 
# print loans.columns
# 
# # Exploring the target column
# # 
# # The target column (label column) of the dataset that we are interested in is called bad_loans. In this column 1 means a risky (bad) loan 0 means a safe loan.
# # 
# # In order to make this more intuitive and consistent with the lectures, we reassign the target to be:
# # 
# # +1 as a safe loan
# # -1 as a risky (bad) loan
# # 3. We put this in a new column called safe_loans.
# # 
# loans['safe_loans']= loans['bad_loans'].apply(lambda x: -1 if x > 0 else +1 )
# 
# 
# # 4. Now, let us explore the distribution of the column safe_loans. This gives us a sense of how many safe and risky loans are present in the dataset. Print out the percentage of safe loans and risky loans in the data frame.
# 
# print "good", float(sum(loans['safe_loans'] == 1))/loans.shape[0]*100
# print "bad", float(sum(loans['safe_loans'] == -1))/loans.shape[0]*100
# 
# 
# # 
# 
# # 5. In this assignment, we will be using a subset of features (categorical and numeric). The features we will be using are described in the code comments below. If you are a finance geek, the LendingClub website has a lot more details about these features. Extract these feature columns and target column from the dataset. We will only use these features.
# 
# features = ['grade',                     # grade of the loan
#             'sub_grade',                 # sub-grade of the loan
#             'short_emp',                 # one year or less of employment
#             'emp_length_num',            # number of years of employment
#             'home_ownership',            # home_ownership status: own, mortgage 
#             'dti',                       # debt to income ratio
#             'purpose',                   # the purpose of the loan
#             'term',                      # the term of the loan
#             'last_delinq_none',          # has borrower had a delinquincy
#             'last_major_derog_none',     # has borrower had 90 day or worse 
#             'revol_util',                # percent of available credit being 
#             'total_rec_late_fee',        # total late fees received to day
#            ]
# 
# 
# target = 'safe_loans'                    # prediction target (y) (+1 means safe, 
# #   -1 is risky)
# # # Extract the feature columns and target column
# loans = loans[features + [target]]
# print "testi", loans.columns
# # If you are NOT using SFrame, download the list of indices for the training and validation sets:
# # 
# # module-5-assignment-1-train-idx.json.zip
# # module-5-assignment-1-validation-idx.json.zip
# # Then follow the following steps:
# # 
# print "Apply one-hot encoding to loans. "
# #Your tool may have a function for one-hot encoding. Alternatively, see #7 for implementation hints.
# 
# 
# categorical_variables = []
# 
# for feat_name, feat_type in [(i,loans[i].dtype) for i in loans.columns]:
#      if feat_type == 'O':
#          categorical_variables.append(feat_name)
#          
# print "found these categoricals", categorical_variables
# 
# 
# for feature in categorical_variables:
# 	loans_data_one_hot_encoded= pandas.get_dummies(loans[feature], prefix=feature)
# 	#merging in pandas sucks so much
# 	loans= pandas.concat([loans, loans_data_one_hot_encoded])
# 	loans = loans.drop(feature, 1)
# 
# for i in loans.columns:
# 	loans[i]= loans[i].fillna(0)
# 	
# 	
# print "teti", loans.columns
# 
# 
# # Load the JSON files into the lists train_idx and validation_idx.
# 
# with codecs.open('/Users/ps22344/Downloads/coursera/module-5-assignment-1-train-idx.json', 'r') as inputjson: 
# 	train_idx=json.load(inputjson)
# 
# with codecs.open('/Users/ps22344/Downloads/coursera/module-5-assignment-1-validation-idx.json', 'r') as inputjson: 
# 	validation_idx=json.load(inputjson)
# 
# train_data = loans.iloc[train_idx]
# validation_data = loans.iloc[validation_idx]
# 
# 
# #BUILD MODELS
# 
# # Build a decision tree classifier
# # 
# # 9. Now, let's use the built-in scikit learn decision tree learner (sklearn.tree.DecisionTreeClassifier) to create a loan prediction model on the training data. To do this, you will need to import sklearn, sklearn.tree, and numpy.
# # 
# # Note: You will have to first convert the SFrame into a numpy data matrix, and extract the target labels as a numpy array (Hint: you can use the .to_numpy() method call on SFrame to turn SFrames into numpy arrays). See the API for more information. Make sure to set max_depth=6.
# # 
# # Call this model decision_tree_model.
# 
# decision_tree_model= sklearn.tree.DecisionTreeClassifier(max_depth=6).fit(train_data.drop('safe_loans', axis=1), train_data['safe_loans'])
# 
# # 10. Also train a tree using with max_depth=2. Call this model small_model.
# 
# small_model= sklearn.tree.DecisionTreeClassifier(max_depth=2).fit(train_data.drop('safe_loans', axis=1), train_data['safe_loans'])
# 
# # Let's consider two positive and two negative examples from the validation set and see what the model predicts. We will do the following:
# 
# # Predict whether or not a loan is safe.
# # Predict the probability that a loan is safe.
# # 11. First, let's grab 2 positive examples and 2 negative examples.
# 
# validation_safe_loans = validation_data[validation_data['safe_loans'] == 1]
# validation_risky_loans = validation_data[validation_data['safe_loans'] == -1]
# sample_validation_data_risky = validation_risky_loans[0:2]
# sample_validation_data_safe = validation_safe_loans[0:2]
# 
# 
# 
# #SCORER AND DATASET
# 
# def scorer(fitted_model, spread_sheet, gold_labels, message):
# 	"""
# 	compute accuracy
# 	"""
# 	print "Message: {}".format(message)
# 	data=spread_sheet.copy()
# 	data['predi']=fitted_model.predict(data.drop(gold_labels, axis=1))
# 	data['accu']=(data['predi']==data[gold_labels])
# 	accuracy= float(sum(data['accu']))/data.shape[0]
# 	print accuracy
# 	return accuracy
# 	
# 
# 
# 
# 
# 
# sample_validation_data= pandas.concat([sample_validation_data_risky, sample_validation_data_safe])
# 
# for i in sample_validation_data.iterrows():
# 	print i[1]['safe_loans']
# 
# count=0 
# 
# 
# # 12. Now, we will use our model to predict whether or not a loan is likely to default. For each row in the sample_validation_data, use the decision_tree_model to predict whether or not the loan is classified as a safe loan. (Hint: if you are using scikit-learn, you can use the .predict() method)
# # 
# 
# for i in sample_validation_data.drop('safe_loans', axis=1).iterrows():
# 	count=count+1
# 	#print (count, "risky loans and others", 		i[1].shape, 	i[1].values.reshape(1, -1).shape, 	i[1].as_matrix().shape)
# 	x=decision_tree_model.predict(i[1].values.reshape(1, -1))
# 	print "result", x
# 
# 
# # Quiz Question: What percentage of the predictions on sample_validation_data did decision_tree_model get correct? 50%
# #
# 
# 
# 
# # Explore probability predictions
# # 
# # 13. For each row in the sample_validation_data, what is the probability (according decision_tree_model) of a loan being classified as safe? (Hint: if you are using scikit-learn, you can use the .predict_proba() method)
# 
# for i in sample_validation_data.drop('safe_loans', axis=1).iterrows():
# 	count=count+1
# 	print count, "regular model probs"
# 	x=decision_tree_model.predict_proba(i[1].values.reshape(1, -1))
# 	print "result", x, decision_tree_model.predict(i[1].values.reshape(1, -1))
# 
# # 
# # Quiz Question: Which loan has the highest probability of being classified as a safe loan? The 4th one
# # 
# # Checkpoint: Can you verify that for all the predictions with probability >= 0.5, the model predicted the label +1?
# # Yes that is correct
# # Tricky predictions!
# # 
# # 14. Now, we will explore something pretty interesting. For each row in the sample_validation_data, what is the probability (according to small_model) of a loan being classified as safe?
# print header
# 
# for i in sample_validation_data.drop('safe_loans', axis=1).iterrows():
# 	count=count+1
# 	print count, "small model probs"
# 	x=small_model.predict_proba(i[1].values.reshape(1,-1))
# 	print "result", x, small_model.predict(i[1].values.reshape(1,-1))
# # 
# # Quiz Question: Notice that the probability preditions are the exact same for the 2nd and 3rd loans. Why would this happen?
# # Cause they're in the same leaf
# # Visualize the prediction on a tree
# # 
# # 14a. Note that you should be able to look at the small tree (of depth 2), traverse it yourself, and visualize the prediction being made. 
# print "Consider the following point in the sample_validation_data[1]"
# 
# 
# # 
# # Quiz Question: Based on the visualized tree, what prediction would you make for this data point (according to small_model)? (If you don't have Graphviz, you can answer this quiz question by executing the next part.)
# # 
# print "Now, verify your prediction by examining the prediction made using small_model."
# 
# testpoint= sample_validation_data.drop('safe_loans', axis=1).iloc[1,:]
# print small_model.predict_proba(testpoint.values.reshape(1,-1))
# print small_model.predict(testpoint.values.reshape(1,-1))
# 
# 
# 
# # Evaluating accuracy of the decision tree model
# # 
# # Recall that the accuracy is defined as follows:
# # 
# # accuracy=# correctly classified data points# total data points
# print "16. Evaluate the accuracy of small_model and decision_tree_model on the training data. (Hint: if you are using scikit-learn, you can use the .score() method)"
# 
# 
# scorer(small_model, train_data, 'safe_loans', "small model on train")
# scorer(decision_tree_model, train_data, 'safe_loans', "dec tree model on train")
# 
# #print small_model.score(train_data.drop('safe_loans', axis=1), train_data['safe_loans'])
# #print decision_tree_model.score(train_data.drop('safe_loans', axis=1), train_data['safe_loans'])
# # Checkpoint: You should see that the small_model performs worse than the decision_tree_model on the training data.
# # 
# print "17. Now, evaluate the accuracy of the small_model and decision_tree_model on the entire validation_data, not just the subsample considered above."
# # 
# scorer(small_model, validation_data, 'safe_loans', "small model on validation")
# scorer(decision_tree_model, validation_data, 'safe_loans', "dec tree model on validation")
# 
# 
# 
# # Quiz Question: What is the accuracy of decision_tree_model on the validation set, rounded to the nearest .01?
# # 0.851357173632
# # Evaluating accuracy of a complex decision tree model
# # 
# # Here, we will train a large decision tree with max_depth=10. This will allow the learned tree to become very deep, and result in a very complex model. Recall that in lecture, we prefer simpler models with similar predictive power. This will be an example of a more complicated model which has similar predictive power, i.e. something we don't want.
# 
# big_model= sklearn.tree.DecisionTreeClassifier(max_depth=10).fit(train_data.drop('safe_loans', axis=1), train_data['safe_loans'])
# 
# 
# # 18. Using sklearn.tree.DecisionTreeClassifier, train a decision tree with maximum depth = 10. Call this model big_model.
# #
# 
# scorer(big_model, train_data, 'safe_loans', "giant model on train")
# scorer(big_model, validation_data, 'safe_loans', "giant model on validation")
# 
# # 19. Evaluate the accuracy of big_model on the training set and validation set.
# # 
# # Checkpoint: We should see that big_model has even better performance on the training set than decision_tree_model did on the training set.
# # 
# 
# 
# 
# 
# 
# # Quiz Question: How does the performance of big_model on the validation set compare to decision_tree_model on the validation set? Is this a sign of overfitting?
# # YES
# # Quantifying the cost of mistakes
# # 
# # Every mistake the model makes costs money. In this section, we will try and quantify the cost each mistake made by the model. Assume the following:
# # 
# # False negatives: Loans that were actually safe but were predicted to be risky. This results in an oppurtunity cost of loosing a loan that would have otherwise been accepted.
# # False positives: Loans that were actually risky but were predicted to be safe. These are much more expensive because it results in a risky loan being given.
# # Correct predictions: All correct predictions don't typically incur any cost.
# # Let's write code that can compute the cost of mistakes made by the model. Complete the following 4 steps:
# # 
# # First, let us compute the predictions made by the model.
# # Second, compute the number of false positives.
# # Third, compute the number of false negatives.
# # Finally, compute the cost of mistakes made by the model by adding up the costs of true positives and false positves.
# # Quiz Question: Let's assume that each mistake costs us money: a false negative costs $10,000, while a false positive positive costs $20,000.
# print " What is the total cost of mistakes made by decision_tree_model on validation_data?"
# 
# validation_data['predi']=decision_tree_model.predict(validation_data.drop('safe_loans', axis=1))
# validation_data['false_pos']=((validation_data['predi']==1) & (validation_data['safe_loans']==-1))
# #print validation_data[['safe_loans', 'predi', 'false_pos']]
# 
# validation_data['false_neg']=((validation_data['predi']==-1) & (validation_data['safe_loans']==1))
# #print validation_data[['safe_loans', 'predi', 'false_pos', 'false_neg']]
# 
# 
# print "Number of false positives is {} (cost: {}), number of false negatives is {} (cost: {}) out of a total {} rows".format(sum(validation_data['false_pos']), sum(validation_data['false_pos'])*20000, sum(validation_data['false_neg']), sum(validation_data['false_neg'])*10000, unicode(validation_data.shape[0]))
# 
# #Number of false positives is 671 (cost: 13420000), number of false negatives is 711 (cost: 7110000) out of a total 9284 rows