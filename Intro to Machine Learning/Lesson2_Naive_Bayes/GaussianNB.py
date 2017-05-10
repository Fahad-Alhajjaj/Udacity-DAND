import numpy as np
from sklearn.naive_bayes import GaussianNB #importing needed libraries.

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]) # features
Y = np.array([1, 1, 1, 2, 2, 2]) #labels

clf = GaussianNB() #creating a classifier
clf.fit(X, Y) #fitting the data

print clf.predict([[-0.8, -1]]) #calling predict to make prediction.