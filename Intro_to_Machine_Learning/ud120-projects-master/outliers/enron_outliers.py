#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
import numpy

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]
data_dict.pop('TOTAL', 0 )
data = featureFormat(data_dict, features)
from outlier_cleaner import outlierCleaner

sals = data[0:,[0]]
outlier1 = sals.max()
print outlier1
ind = numpy.argmax(sals)
temp = numpy.delete(sals, ind)
outlier2 = numpy.max(temp)

for item in data_dict:
	for k, v in data_dict[item].items():
		if v == outlier1 or v == outlier2:
			print 'OUTLIER: \t', item
			print k, v

### your code below

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()