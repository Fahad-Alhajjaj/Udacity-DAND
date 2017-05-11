#!/usr/bin/python

""" lecture and example code for decision tree unit """

import sys
sys.path.append('../Lesson2_Naive_Bayes')
from class_vis import prettyPicture, output_image
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from classifyDT import classify

features_train, labels_train, features_test, labels_test = makeTerrainData()



### the classify() function in classifyDT is where the magic
### happens--fill in this function in the file 'classifyDT.py'!
from sklearn import tree
from sklearn.metrics import accuracy_score
print '***********GINI***********'
for i in range(2,100):
	clf = tree.DecisionTreeClassifier(min_samples_split = i, criterion='gini')
	clf.fit(features_train, labels_train)
	pred = clf.predict(features_test)
	acc = accuracy_score(pred, labels_test)

	print i, ' | ', acc

print '***********ENTROPY***********'
for i in range(2,100):
	clf = tree.DecisionTreeClassifier(min_samples_split = i, criterion='entropy')
	clf.fit(features_train, labels_train)
	pred = clf.predict(features_test)
	acc = accuracy_score(pred, labels_test)

	print i, ' | ', acc
'''
#### grader code, do not modify below this line
prettyPicture(clf, features_test, labels_test)
output_image("test.png", "png", open("test.png", "rb").read())
'''