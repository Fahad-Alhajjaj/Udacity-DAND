#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
import numpy
data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation
from sklearn.metrics import accuracy_score

features_train,features_test,labels_train,labels_test = cross_validation.train_test_split(features,
																							labels,
																							test_size=0.3,
																							random_state=42)
clf = DecisionTreeClassifier()
clf.fit(features_train,labels_train)

pred = clf.predict(features_test)
acc = accuracy_score(pred, labels_test)

print numpy.count_nonzero(pred)
print acc

counter = 0
for elem, label in zip(pred, labels_test):
	if elem == label and elem == 1:
		counter+=1
print counter

from sklearn.metrics import precision_score, recall_score

print precision_score(labels_test, pred)
print recall_score(labels_test, pred)