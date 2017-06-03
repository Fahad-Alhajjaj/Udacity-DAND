#!/usr/bin/python
import pickle
import sys
import matplotlib.pyplot as plt
import numpy

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ['salary',
			'deferral_payments',
			'total_payments',
			'loan_advances',
			'bonus',
			'restricted_stock_deferred',
			'deferred_income',
			'total_stock_value',
			'expenses',
			'exercised_stock_options',
			'other',
			'long_term_incentive',
			'restricted_stock',
			'director_fees',
			'to_messages',
			'from_poi_to_this_person',
			'from_messages',
			'from_this_person_to_poi',
			'shared_receipt_with_poi',
			'poi'
			]
print 'Done reading'
data_dict.pop('TOTAL', 0 )

data = featureFormat(data_dict, features)

### your code below
for i in range(0, 19):
	plt.figure(figsize=(30,20))
	for point in data:
		x = point[i]
		y = point[i]
		plt.scatter( x, y , color= ('blue' if point[19] == 1 else 'red'))
	plt.xlabel(features[i])
	plt.ylabel(features[i])
	plt.savefig(str(0) + features[i] + ".png")
	#plt.show()