#!/usr/bin/python

import sys
import pickle
from poi_email_addresses import poiEmails
from text_fix import find_signature, count_emails
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import numpy as np
import re
import pprint
from sklearn.grid_search import GridSearchCV

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
				
features_list = ['poi',
				'salary',
				#'deferral_payments',
				#'total_payments',
				#'loan_advances',
				#'bonus',
				#'restricted_stock_deferred',
				'deferred_income',
				'total_stock_value',
				'expenses',
				'exercised_stock_options',
				#'other',
				#'long_term_incentive',
				#'restricted_stock',
				#'director_fees',
				'to_messages',
				'from_poi_to_this_person',
				'from_messages',
				'from_this_person_to_poi',
				'shared_receipt_with_poi',
				] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

data_dict.pop('TOTAL', 0)
'''
emails = []
i=0
for key in data_dict.keys():
	if i == 0:
		pprint.pprint(data_dict[key])
		i+=1
	emails.append(data_dict[key]['email_address'])

emails_dict = count_emails(emails)

#pprint.pprint(emails_dict)
### Task 2: Remove outliers
data_list = {}
for key in data_dict.keys():
	for k, v in data_dict[key].items():
		if k not in features_list[1:] or v == 'NaN':
			continue
		if k in data_list:
			data_list[k].append(v)
		else:
			data_list[k] = []
			data_list[k].append(v)

counter = 0
for val in features_list[1:6]:
	for i in range(0,2):
		a = max(data_dict, key=lambda x: data_dict[x][val] if data_dict[x][val] != 'NaN' else 0)
		data_dict[a][val] = 'NaN'
		counter +=1
print 'Removed: ', counter, ' outliers.'

### Task 3: Create new feature(s)
for key in data_dict.keys():
	email = data_dict[key]['email_address']
	if email is not 'NaN':
		if email in emails_dict:
			data_dict[key]['from_count'] = emails_dict[email]['from']
			data_dict[key]['to_count'] = emails_dict[email]['to']
		else:
			data_dict[key]['from_count'] = 'NaN'
			data_dict[key]['to_count'] = 'NaN'
'''
#poi_emails = poiEmails()
#print 'GOT emails'
#find_signature(emails, poi_emails)
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
clf = ABC(n_estimators= 5)#RFC(n_estimators= 3, criterion= 'entropy', max_features= None, min_samples_split= 20)
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
    train_test_split(features, labels, test_size=0.1, random_state=42)

param_grid = {
         'n_estimators' : [3, 5, 7, 9, 11, 13, 15],
         'criterion' : ['gini', 'entropy'],
         'max_features': ['sqrt', 'log2', None]
          }
# for sklearn version 0.16 or prior, the class_weight parameter value is 'auto'
from sklearn.ensemble import RandomForestClassifier as RFC
#clf = GridSearchCV(RFC(), param_grid)

clf = RFC(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='log2', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=11, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
clf = clf.fit(features_train, labels_train)
#print "Best estimator found by grid search:"
#print clf.best_estimator_

#abc = ABC(n_estimators= 5)
#abc.fit(features_train, labels_train) #fitting the data
pred = clf.predict(features_test)
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