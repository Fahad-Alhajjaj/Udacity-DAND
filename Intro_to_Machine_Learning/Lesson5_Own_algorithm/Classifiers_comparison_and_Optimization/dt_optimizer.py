from sklearn.tree import DecisionTreeClassifier as DTC
import features_and_lables as FandL

from time import time
import pandas as pd
from IPython.display import display
from sklearn.metrics import accuracy_score

Classifiers_df = pd.DataFrame(columns=('criterion', 'splitter', 'max_f', 'min_samples', 'Train', 'Pred', 'Total', 'Accuracy'))
features_train, features_test, labels_train, labels_test = FandL.get_features_and_labels()

def run_clf(clf, c, s, mf, mss, counter):
	t0 = time()
	clf.fit(features_train, labels_train)
	train_time = round(time()-t0, 3)
	t0 = time()
	pred = clf.predict(features_test)
	pred_time = round(time()-t0, 3)
	acc = accuracy_score(pred, labels_test)
	Classifiers_df.loc[counter] = [c, s, mf, mss, train_time, pred_time, (train_time+pred_time), acc]

def dt_opt():
	counter = 1
	for c in ['gini', 'entropy']:
		for s in ['best', 'random']:
			for mf in ['sqrt', 'log2', None]:
				for mss in range(2,100):
						clf = DTC(criterion= c, splitter= s , max_features= mf, min_samples_split= mss)
						run_clf(clf, c, s, mf, mss, counter)
						counter+=1
	print 'Best Total Time'
	display((Classifiers_df.sort_values(by= 'Total', ascending=[1])).head(5))
	print 'Best Accuracy'
	display((Classifiers_df.sort_values(by= 'Accuracy', ascending=[0])).head(5))