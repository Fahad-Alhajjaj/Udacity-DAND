from sklearn.ensemble import AdaBoostClassifier as ABC
import features_and_lables as FandL

from time import time
import pandas as pd
from IPython.display import display
from sklearn.metrics import accuracy_score

Classifiers_df = pd.DataFrame(columns=('n_estimators', 'Train', 'Pred', 'Total', 'Accuracy'))
features_train, features_test, labels_train, labels_test = FandL.get_features_and_labels()

def run_clf(clf, n, counter):
	t0 = time()
	clf.fit(features_train, labels_train)
	train_time = round(time()-t0, 3)
	t0 = time()
	pred = clf.predict(features_test)
	pred_time = round(time()-t0, 3)
	acc = accuracy_score(pred, labels_test)
	Classifiers_df.loc[counter] = [n, train_time, pred_time, (train_time+pred_time), acc]

def aBoost_opt():
	counter = 1
	for n in range(1, 100, 10):
		clf = ABC(n_estimators= n)
		run_clf(clf, n, counter)
		counter+=1
	print 'Best Total Time'
	display((Classifiers_df.sort_values(by= 'Total', ascending=[1])).head(5))
	print 'Best Accuracy'
	display((Classifiers_df.sort_values(by= 'Accuracy', ascending=[0])).head(5))