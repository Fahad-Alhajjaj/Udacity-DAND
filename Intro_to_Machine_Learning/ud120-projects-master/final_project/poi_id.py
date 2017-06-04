#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.naive_bayes import GaussianNB

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
				
features_list = ['poi',
				'salary',
				'deferral_payments',
				'total_payments',
				#'loan_advances', ####
				#'bonus', ####
				#'restricted_stock_deferred', ####
				'deferred_income',
				'total_stock_value',
				'expenses',
				'exercised_stock_options',
				'other',
				#'long_term_incentive', ####
				'restricted_stock',
				'director_fees',
				'to_messages',
				'from_poi_to_this_person',
				'from_messages',
				'from_this_person_to_poi',
				'shared_receipt_with_poi',
				'bonus/salary',
				'bonus/total_payments',
				'shared_receipt_with_poi/total_payments',
				'exercised_stock_options/total_stock_value',
				'restricted_stock/total_stock_value',
				'from_this_person_to_poi/to_messages',
				'shared_receipt_with_poi/to_messages',
				] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

data_dict.pop('TOTAL', 0)
temp = {'bonus': 0,
		'deferral_payments': 0,
		'deferred_income': 0,
		'director_fees': 0,
		'email_address': 0,
		'exercised_stock_options': 0,
		'expenses': 0,
		'from_messages': 0,
		'from_poi_to_this_person': 0,
		'from_this_person_to_poi': 0,
		'loan_advances': 0,
		'long_term_incentive': 0,
		'other': 0,
		'poi': 0,
		'restricted_stock': 0,
		'restricted_stock_deferred': 0,
		'salary': 0,
		'shared_receipt_with_poi': 0,
		'to_messages': 0,
		'total_payments': 0,
		'total_stock_value': 0
		}
for key in data_dict.keys():
	for val in temp.keys():
		if data_dict[key][val] == 'NaN' and data_dict[key]['poi'] == False:
			temp[val] += 1
import pprint
pprint.pprint(temp)
'''
### Task 2: Remove outliers
keys_outliers = ['FREVERT MARK A',
				'BHATNAGAR SANJAY',
				'LAVORATO JOHN J',
				'MARTIN AMANDA K',
				'KAMINSKI WINCENTY J'
				]
for val in keys_outliers:
	del data_dict[val]
'''
### Task 3: Create new feature(s)
def new_features(numerator, denumerator, key, data_dict):
	new_key = numerator + '/' + denumerator
	data_dict[key][new_key] = \
		float(data_dict[key][numerator])/float(data_dict[key][denumerator]) \
		if (data_dict[key][denumerator] != 0 and \
			data_dict[key][numerator] != 'NaN' and \
			data_dict[key][denumerator] != 'NaN') else 0
	return data_dict

for key in data_dict.keys():
	data_dict = new_features('bonus', 'salary', key, data_dict)
	data_dict = new_features('bonus', 'total_payments', key, data_dict)
	data_dict = new_features('shared_receipt_with_poi', 'total_payments', key, data_dict)
	data_dict = new_features('exercised_stock_options', 'total_stock_value', key, data_dict)
	data_dict = new_features('restricted_stock', 'total_stock_value', key, data_dict)
	data_dict = new_features('from_this_person_to_poi', 'to_messages', key, data_dict)
	data_dict = new_features('shared_receipt_with_poi', 'to_messages', key, data_dict)

'''
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
from sklearn import cross_validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, labels, test_size=0.1, random_state=42)

### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

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
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

features_train, features_test, labels_train, labels_test = \
	cross_validation.train_test_split(features, labels, test_size=0.1, random_state=42)

clf = ABC(algorithm='SAMME.R', base_estimator=None,
          learning_rate=1.0, n_estimators=27, random_state=None)

clf.fit(features_train, labels_train) #fitting the data

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
'''