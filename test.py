import scipy.sparse, time, sklearn, numpy as np
from sklearn import linear_model

# l=[[0,1.0,2,4,100,0,3,22], [0,2,33333]]
# 
# 
# 
a = scipy.sparse.coo_matrix([[1, 2, 0,6,7], [0, 0, 3,7], [4, 0, 5,7]])
print a.shape[1]
# #t=scipy.sparse.csr_matrix([[0,1.0,2,4,100,0,3,22], [0,2,33333]])
# print A
# t=time.time()
# print t
# dic={"an":1, "assi":3, "ff":1000}
# 
# for i in dic:
# 	print i
# 	s=time.time()-t
# 	print s
# 	
# classifier=linear_model.LogisticRegression()
# print classifier
# t=classifier.fit([[1,3,1], [0,0,0], [1,2,1]], [1,0,1])
# 
# #print t.coef_
# print len(np.ndarray.tolist(t.coef_)[0])
	
