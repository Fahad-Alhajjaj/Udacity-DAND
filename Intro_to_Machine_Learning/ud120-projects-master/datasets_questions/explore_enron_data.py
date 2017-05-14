import pickle
from math import isnan

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

import sys
sys.path.append('../tools')
from feature_format import *

a = 0
b = 0
for item in enron_data:
	for k, v in enron_data[item].items():
		if k == 'total_payments':
			if v == 'NaN':
				a+=1
			else:
				b+=1

print a, ' | ', b
'''
salary
to_messages
deferral_payments
total_payments
exercised_stock_options
bonus
restricted_stock
shared_receipt_with_poi
restricted_stock_deferred
total_stock_value
expenses
loan_advances
from_messages
other
from_this_person_to_poi
poi
director_fees
deferred_income
long_term_incentive
email_address
from_poi_to_this_person
'''