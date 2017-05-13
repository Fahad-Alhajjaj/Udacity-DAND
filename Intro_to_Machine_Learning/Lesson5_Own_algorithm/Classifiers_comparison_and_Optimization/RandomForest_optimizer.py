from sklearn.ensemble import RandomForestClassifier as RFC
import features_and_lables as FandL

from time import time
import pandas as pd
from IPython.display import display
from sklearn.metrics import accuracy_score

Classifiers_df = pd.DataFrame(columns=('n_estimators', 'criterion', 'max_f', 'min_samples', 'Train', 'Pred', 'Total', 'Accuracy'))
features_train, features_test, labels_train, labels_test = FandL.get_features_and_labels()

def run_clf(clf, n, c, mf, mss, counter):
	t0 = time()
	clf.fit(features_train, labels_train)
	train_time = round(time()-t0, 3)
	t0 = time()
	pred = clf.predict(features_test)
	pred_time = round(time()-t0, 3)
	acc = accuracy_score(pred, labels_test)
	Classifiers_df.loc[counter] = [n, c, mf, mss, train_time, pred_time, (train_time+pred_time), acc]

def rf_opt():
	counter = 1
	for n in range(10, 100, 10):
		for c in ['gini', 'entropy']:
			for mf in ['sqrt', 'log2', None]:
				for mss in range(40, 100, 10):
						clf = RFC(n_estimators= n, criterion= c, max_features= mf, min_samples_split= mss)
						run_clf(clf, n, c, mf, mss, counter)
						counter+=1
	clf = RFC(n_estimators= 15, criterion= 'gini', max_features= None, min_samples_split= 35)
	run_clf(clf, 15, 'gini', None, 35, counter)
	print 'Best Total Time'
	display((Classifiers_df.sort_values(by= 'Total', ascending=[1])).head(5))
	print 'Best Accuracy'
	display((Classifiers_df.sort_values(by= 'Accuracy', ascending=[0])).head(5))
#n_estimators criterion max_f  min_samples  Train   Pred  Total  Accuracy
#        40.0   entropy  log2         40.0  0.112  0.084  0.196  0.849829