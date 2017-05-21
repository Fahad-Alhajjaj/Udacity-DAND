#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import numpy as np

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
				'bonus',
				'salary',
				'total_payments',
				'total_stock_value',
				'long_term_incentive',
				'shared_receipt_with_poi',
				'from_this_person_to_poi',
				'from_poi_to_this_person'
				] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
'''
data_dict.pop('TOTAL', 0)
import pprint
for key in data_dict.keys():
	for k, v in data_dict[key].items():
		if k not in features_list:
			del data_dict[key][k]
'''
### Task 2: Remove outliers
counter = 0
for val in features_list[1:6]:
	for i in range(0,2):
		a = max(data_dict, key=lambda x: data_dict[x][val] if data_dict[x][val] != 'NaN' else 0)
		data_dict[a][val] = 0
		counter +=1
print 'Removed: ', counter, ' outliers.'
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
#np.random.seed(1410)
from sklearn import cross_validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, labels, test_size=0.3)

### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import AdaBoostClassifier as ABC
clf = ABC(n_estimators= 3)#RFC(n_estimators= 3, criterion= 'entropy', max_features= None, min_samples_split= 20)
clf.fit(X_train, y_train) #fitting the data
pred = clf.predict(X_test)
acc = accuracy_score(pred, y_test)
recall = recall_score(y_test, pred)
precision = precision_score(y_test, pred)

print 'Accuracy:  ', acc
print 'Recall:    ', recall
print 'Precision: ', precision

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

abc = ABC(n_estimators= 3)
abc.fit(features_train, labels_train) #fitting the data
pred = abc.predict(features_test)
acc = accuracy_score(pred, labels_test)
recall = recall_score(labels_test, pred)
precision = precision_score(labels_test, pred)

print 'Accuracy:  ', acc
print 'Recall:    ', recall
print 'Precision: ', precision
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)