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
				'salary', #~
				'deferral_payments',
				'total_payments',
				#'loan_advances', ####
				#'bonus',
				#'restricted_stock_deferred',
				'deferred_income', #~
				'total_stock_value', #~
				'expenses',
				'exercised_stock_options',
				'other', ####
				#'long_term_incentive', ####
				'restricted_stock',
				'director_fees',
				'to_messages',
				'from_poi_to_this_person', #~
				'from_messages',
				'from_this_person_to_poi',
				'shared_receipt_with_poi',
				'bonus/salary',
				'director_fees/deferral_payments',
				'to_messages/deferral_payments', #~
				#'from_messages/deferral_payments',
				'bonus/total_payments',
				#'shared_receipt_with_poi/total_payments', #~
				'exercised_stock_options/total_stock_value',
				'restricted_stock/total_stock_value',
				'from_this_person_to_poi/to_messages', #~
				'shared_receipt_with_poi/to_messages',
				] # You will need to use more features

'''
~salary
deferral_payments
total_payments
bonus
restricted_stock_deferred
~deferred_income
~total_stock_value
expenses
exercised_stock_options
restricted_stock
director_fees
to_messages
~from_poi_to_this_person
from_messages
from_this_person_to_poi
shared_receipt_with_poi
'''
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

data_dict.pop('TOTAL', 0)

### Task 2: Remove outliers
keys_outliers = [#'SKILLING JEFFREY K',
				#'LAY KENNETH L',
				'FREVERT MARK A',
				'BHATNAGAR SANJAY',
				'LAVORATO JOHN J',
				#'DELAINEY DAVID W',
				#'BELDEN TIMOTHY N',
				'MARTIN AMANDA K',
				'DERRICK JR. JAMES V'
				#'PICKERING MARK R'
				]
for val in keys_outliers:
	del data_dict[val]
'''
for val in features_list[1:]:
	print 'features: ', val
	for i in range(0,4):
		a = max(data_dict, key=lambda x: data_dict[x][val] if data_dict[x][val] != 'NaN' else 0)
		print '\t', a, ' = ', data_dict[a][val]
		data_dict[a][val] = 0
'''
### Task 3: Create new feature(s)
for key in data_dict.keys():
	data_dict[key]['bonus/salary'] = \
		float(data_dict[key]['bonus'])/float(data_dict[key]['salary']) \
		if (data_dict[key]['salary'] != 0 and \
			data_dict[key]['bonus'] != 'NaN' and \
			data_dict[key]['salary'] != 'NaN') else 0

	data_dict[key]['director_fees/deferral_payments'] = \
		float(data_dict[key]['director_fees'])/float(data_dict[key]['deferral_payments']) \
		if (data_dict[key]['deferral_payments'] != 0 and \
			data_dict[key]['director_fees'] != 'NaN' and
			data_dict[key]['deferral_payments'] != 'NaN') else 0

	data_dict[key]['to_messages/deferral_payments'] = \
		float(data_dict[key]['to_messages'])/float(data_dict[key]['deferral_payments']) \
		if (data_dict[key]['deferral_payments'] != 0 and \
			data_dict[key]['to_messages'] != 'NaN' and \
			data_dict[key]['deferral_payments'] != 'NaN') else 0

	data_dict[key]['from_messages/deferral_payments'] = \
		float(data_dict[key]['from_messages'])/float(data_dict[key]['deferral_payments']) \
		if (data_dict[key]['deferral_payments'] != 0 and \
			data_dict[key]['from_messages'] != 'NaN' and \
			data_dict[key]['deferral_payments'] != 'NaN') else 0
	data_dict[key]['bonus/total_payments'] = \
		float(data_dict[key]['bonus'])/float(data_dict[key]['total_payments']) \
		if (data_dict[key]['total_payments'] != 0 and \
			data_dict[key]['bonus'] != 'NaN' and \
			data_dict[key]['total_payments'] != 'NaN') else 0

	data_dict[key]['shared_receipt_with_poi/total_payments'] = \
		float(data_dict[key]['shared_receipt_with_poi'])/float(data_dict[key]['total_payments']) \
		if (data_dict[key]['total_payments'] != 0 and \
			data_dict[key]['shared_receipt_with_poi'] != 'NaN' and \
			data_dict[key]['total_payments'] != 'NaN') else 0
	
	data_dict[key]['exercised_stock_options/total_stock_value'] = \
		float(data_dict[key]['exercised_stock_options'])/float(data_dict[key]['total_stock_value']) \
		if (data_dict[key]['total_stock_value'] != 0 and \
			data_dict[key]['exercised_stock_options'] != 'NaN' and \
			data_dict[key]['total_stock_value'] != 'NaN') else 0
	
	data_dict[key]['restricted_stock/total_stock_value'] = \
		float(data_dict[key]['restricted_stock'])/float(data_dict[key]['total_stock_value']) \
		if (data_dict[key]['total_stock_value'] != 0 and \
			data_dict[key]['restricted_stock'] != 'NaN' and \
			data_dict[key]['total_stock_value'] != 'NaN') else 0
	
	data_dict[key]['from_this_person_to_poi/to_messages'] = \
		float(data_dict[key]['from_this_person_to_poi'])/float(data_dict[key]['to_messages']) \
		if (data_dict[key]['to_messages'] != 0 and \
			data_dict[key]['from_this_person_to_poi'] != 'NaN' and \
			data_dict[key]['to_messages'] != 'NaN') else 0
	
	data_dict[key]['shared_receipt_with_poi/to_messages'] = \
		float(data_dict[key]['shared_receipt_with_poi'])/float(data_dict[key]['to_messages']) \
		if (data_dict[key]['to_messages'] != 0 and \
			data_dict[key]['shared_receipt_with_poi'] != 'NaN' and \
			data_dict[key]['to_messages'] != 'NaN') else 0

'''
bonus/salary

director_fees/deferral_payments
to_messages/deferral_payments
from_messages/deferral_payments

bonus/total_payments
shared_receipt_with_poi/total_payments

exercised_stock_options/total_stock_value
restricted_stock/total_stock_value

from_this_person_to_poi/to_messages
shared_receipt_with_poi/to_messages
'''
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
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
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
'''
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.1, random_state=42)

param_grid = {
         'n_estimators' : [3, 5, 7, 9, 11, 13, 15],
         'criterion' : ['gini', 'entropy'],
         'max_features': ['sqrt', 'log2', None]
          }
# for sklearn version 0.16 or prior, the class_weight parameter value is 'auto'
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
'''
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)