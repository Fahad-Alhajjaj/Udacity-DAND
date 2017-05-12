from sklearn.svm import SVC
import features_and_lables as FandL

from time import time
import pandas as pd
from IPython.display import display
from sklearn.metrics import accuracy_score

Classifiers_df = pd.DataFrame(columns=('kernel', 'C', 'gamma', 'degree', 'Training', 'Pred', 'Total', 'Accuracy'))
features_train, features_test, labels_train, labels_test = FandL.get_features_and_labels()

def run_clf(clf, k, c, g, d, counter):
	t0 = time()
	clf.fit(features_train, labels_train)
	train_time = round(time()-t0, 3)
	t0 = time()
	pred = clf.predict(features_test)
	pred_time = round(time()-t0, 3)
	acc = accuracy_score(pred, labels_test)
	Classifiers_df.loc[counter] = [k, c, g, d, train_time, pred_time, (train_time+pred_time), acc]

def svm_opt():
	# 'sigmoid', 'precomputed' 'callable'
	counter = 1
	for k in ['linear', 'poly', 'rbf']:
		for c in [10000, 100000]:
			for g in range(1, 100, 25):
				if k == 'poly':
					for d in range(1, 30, 10):
						clf = SVC(kernel= k, C= c , gamma= g, degree= d)
						run_clf(clf, k, c, g, d, counter)
						counter+=1
				else:
					d=0
					clf = SVC(kernel= k, C= c , gamma= g)
					run_clf(clf, k, c, g, d, counter)
					counter+=1
	print 'Best Total Time'
	display((Classifiers_df.sort_values(by= 'Total', ascending=[1])).head(5))
	print 'Best Accuracy'
	display((Classifiers_df.sort_values(by= 'Accuracy', ascending=[0])).head(5))