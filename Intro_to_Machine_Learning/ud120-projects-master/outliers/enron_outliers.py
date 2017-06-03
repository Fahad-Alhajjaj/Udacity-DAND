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
			'bonus/salary',
			'director_fees/deferral_payments',
			'to_messages/deferral_payments',
			'from_messages/deferral_payments',
			'bonus/total_payments',
			'shared_receipt_with_poi/total_payments',
			'exercised_stock_options/total_stock_value',
			'restricted_stock/total_stock_value',
			'from_this_person_to_poi/to_messages',
			'shared_receipt_with_poi/to_messages',
			'poi'
			]
print 'Done reading'
data_dict.pop('TOTAL', 0 )

### Task 2: Remove outliers
keys_outliers = ['SKILLING JEFFREY K',
				'LAY KENNETH L',
				'FREVERT MARK A',
				'BHATNAGAR SANJAY',
				'LAVORATO JOHN J',
				'DELAINEY DAVID W',
				'BELDEN TIMOTHY N',
				'MARTIN AMANDA K',
				'DERRICK JR. JAMES V',
				'PICKERING MARK R'
				]
for val in keys_outliers:
	del data_dict[val]
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

data = featureFormat(data_dict, features)

### your code below
for i in range(19, 29):
	plt.figure(figsize=(30,20))
	for point in data:
		x = point[i]
		y = point[i]
		plt.scatter( x, y , color= ('blue' if point[19] == 1 else 'red'))
	plt.xlabel(features[i])
	plt.ylabel(features[i])
	name = str(2222) + features[i]
	name = name.replace('/', '--')
	plt.savefig(name + ".png")
	#plt.show()