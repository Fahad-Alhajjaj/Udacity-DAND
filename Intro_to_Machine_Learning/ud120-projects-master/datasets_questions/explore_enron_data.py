import pickle
from math import isnan

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

import sys
sys.path.append('../tools')
from feature_format import *

a = 0
b = 0
c = 0
features = {'salary' : 0,
			'total_payments' : 0,
			'exercised_stock_options' : 0,
			'bonus' : 0,
			'restricted_stock' : 0,
			'shared_receipt_with_poi' : 0,
			'total_stock_value' : 0,
			'expenses' : 0,
			'other' : 0,
			'from_this_person_to_poi' : 0,
			'deferred_income' : 0,
			'long_term_incentive' : 0,
			'from_poi_to_this_person' : 0
			}

import pprint
for item in enron_data:
	for k, v in enron_data[item].items():
		if k == 'poi':
			if v == 1:
				c+=1
				for k in features.keys():
					if enron_data[item][k] != 'NaN':
						features[k] += 1
pprint.pprint(features)
print c

'''
salary
total_payments
exercised_stock_options
bonus
restricted_stock
shared_receipt_with_poi
total_stock_value
expenses
other
from_this_person_to_poi
deferred_income
long_term_incentive
from_poi_to_this_person
'''