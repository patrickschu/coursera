from scipy import special
import scipy
import numpy as np


"""
these all assume we have been given a count for an x and the respective true label. 
"""


def listmultiplier(input_list):
	"""
	multiplies all items in a list. 
	"""
	return reduce(lambda x, y: x*y, input_list)

def likelihoodmachine(list_of_positive_x, list_of_negative_x):
	"""
	input list of items for plus and minus outputs respectively.
	"""
	poshoods=listmultiplier([special.expit(i) for i in list_of_positive_x])
	neghoods=listmultiplier([1-special.expit(i) for i in list_of_negative_x])
	print listmultiplier([poshoods,neghoods])
	return listmultiplier([poshoods,neghoods])

likelihoodmachine([2.5,2.8,0.5], [0.3])

def derivativemachine(weight_of_x, list_of_positive_x, list_of_negative_x):
	"""
	compute derivative as illustrated here:
	https://www.coursera.org/learn/ml-classification/lecture/UEmJg/example-of-computing-derivative-for-logistic-regression
	"""
	#derivative for each; sum.
	poshoods=[((weight_of_x*i) * (1 - special.expit(i))) for i in list_of_positive_x]
	neghoods=[((weight_of_x*i) * (0 - special.expit(i))) for i in list_of_negative_x]
	print poshoods
	print neghoods
	print sum(poshoods+neghoods)
	return sum(poshoods+neghoods)
	

x=derivativemachine(1, [2.5,2.8,0.5], [0.3])
print scipy.exp(x)
print np.log(x)
