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
				'deferred_income',
				'total_stock_value',
				'expenses',
				'exercised_stock_options',
				'other',
				'restricted_stock',
				'director_fees',
				'to_messages',
				'from_poi_to_this_person',
				'from_messages',
				'from_this_person_to_poi',
				'shared_receipt_with_poi',
				'shared_receipt_with_poi/to_messages'
				] # You will need to use more features
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

data_dict.pop('TOTAL', 0) #Remove the total record before doing any thing

### Task 2: Remove outliers
keys_outliers = ['FREVERT MARK A', #deferral_payments outlier
				'BHATNAGAR SANJAY', #restricted_stock_deferred outlier
				'LAVORATO JOHN J', #bonus and from_poi_to_this_person outlier
				'MARTIN AMANDA K', #long_term_incentive outlier
				'KAMINSKI WINCENTY J', #from_messages outlier
				'LOCKHART EUGENE E', #all of its values missing, 'NaN'
				'THE TRAVEL AGENCY IN THE PARK' # Not a real person
				]
for val in keys_outliers:
	del data_dict[val]

### Task 3: Create new feature(s)
#creating new feature by dividing one value by another.
def new_features(numerator, denumerator, key, data_dict):
	new_key = numerator + '/' + denumerator
	data_dict[key][new_key] = \
		float(data_dict[key][numerator])/float(data_dict[key][denumerator]) \
		if (data_dict[key][denumerator] != 0 and \
			data_dict[key][numerator] != 'NaN' and \
			data_dict[key][denumerator] != 'NaN') else 0
	return data_dict

for key in data_dict.keys():
	data_dict = new_features('shared_receipt_with_poi', 'to_messages', key, data_dict)

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
from sklearn.cross_validation import StratifiedShuffleSplit

cv_sss = StratifiedShuffleSplit(labels, n_iter= 1000, random_state = 42)

for train_index, test_index in cv_sss: 
    X_train = []
    X_test  = []
    y_train = []
    y_test  = []
    for i in train_index:
        X_train.append(features[i])
        y_train.append(labels[i])
    for j in test_index:
        X_test.append(features[j])
        y_test.append(labels[j])

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

'''
from sklearn.grid_search import GridSearchCV

Using param_grid and GridSearchCV to tune the algorithm's parameters
param_grid = {
			'n_estimators': [3, 7, 9, 11, 15, 21, 23, 27],
          }
clf = GridSearchCV(ABC(), param_grid)
'''
#The parameters are hardcoded because when leaving the GridSearchCV and testing with
#tester.py scores are low, but when hardcoded they are high as expected.
clf = ABC(algorithm='SAMME.R', base_estimator=None,
          learning_rate=1.0, n_estimators=23, random_state=None)

clf.fit(X_train, y_train) #fitting the data
'''
print "Best estimator found by grid search:"
print clf.best_estimator_
'''
pred = clf.predict(X_test)
acc = accuracy_score(pred, y_test)
recall = recall_score(y_test, pred)
precision = precision_score(y_test, pred)

print 'Accuracy:  ', acc
print 'Recall:    ', recall
print 'Precision: ', precision
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
