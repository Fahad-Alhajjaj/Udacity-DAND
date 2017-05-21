import sys
from time import time
sys.path.append('/Users/yousef/Udacity/Udacity-DAND/Intro_to_Machine_Learning/ud120-projects-master/tools')
from email_preprocess import preprocess

import numpy as np
import pandas as pd
import pylab as pl
from IPython.display import display
from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.ensemble import AdaBoostClassifier as ABC

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


features_train = features_train[:len(features_train)/100] 
labels_train = labels_train[:len(labels_train)/100]

# Code source: Gael Varoquaux
#              Andreas Muller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause
names = [
	"k nearest neighbors", "Linear SVM", "RBF SVM", "Naive Bayes",
	"Decision Tree", "Random Forest", "AdaBoost"
	]

classifiers = [
    KNC(3),
    SVC(kernel= 'linear', C=10000),
    SVC(kernel= 'rbf', gamma=2, C=1),
    GaussianNB(),
    DTC(max_depth=5),
    RFC(max_depth=5, n_estimators=10, max_features=1),
    ABC()
    ]

Classifiers_df = pd.DataFrame({
	'Classifier Name' : pd.Series([	"k nearest neighbors", "Linear SVM", "RBF SVM", "Naive Bayes",
									"Decision Tree", "Random Forest", "AdaBoost"]),
	'Training Time' : pd.Series([0.0,0.0,0.0,0.0,0.0,0.0,0.0]),
	'Prediction Time' : pd.Series([0.0,0.0,0.0,0.0,0.0,0.0,0.0]),
	'Total Time' : pd.Series([0.0,0.0,0.0,0.0,0.0,0.0,0.0]),
	'Accuracy' : pd.Series([0.0,0.0,0.0,0.0,0.0,0.0,0.0])})

Classifiers_df.set_index('Classifier Name', inplace=True)

# iterate over classifiers
for name, clf in zip(names, classifiers):
	t0 = time()
	clf.fit(features_train, labels_train)
	train_time = round(time()-t0, 3)
	t0 = time()
	pred = clf.predict(features_test)
	pred_time = round(time()-t0, 3)
	acc = accuracy_score(pred, labels_test)
	Classifiers_df.set_value(name, 'Training Time', train_time)
	Classifiers_df.set_value(name, 'Prediction Time', pred_time)
	Classifiers_df.set_value(name, 'Total Time', (train_time+pred_time))
	Classifiers_df.set_value(name, 'Accuracy', acc)

display(Classifiers_df.sort_values(by= ['Accuracy', 'Total Time'], ascending=[0, 1]))