from sklearn.neighbors import KNeighborsClassifier as KNC
import features_and_lables as FandL

from time import time
import pandas as pd
from IPython.display import display
from sklearn.metrics import accuracy_score

Classifiers_df = pd.DataFrame(columns=('n_neighbors', 'weights', 'algorithm', 'leaf_size', 'Train', 'Pred', 'Total', 'Accuracy'))
features_train, features_test, labels_train, labels_test = FandL.get_features_and_labels()

def run_clf(clf, nb, w, a, ls, counter):
	t0 = time()
	clf.fit(features_train, labels_train)
	train_time = round(time()-t0, 3)
	t0 = time()
	pred = clf.predict(features_test)
	pred_time = round(time()-t0, 3)
	acc = accuracy_score(pred, labels_test)
	Classifiers_df.loc[counter] = [nb, w, a, ls, train_time, pred_time, (train_time+pred_time), acc]

def kn_opt():
	counter = 1
	'''
	for nb in range(1,100):
		for w in ['uniform', 'distance']:
			for a in ['auto', 'ball_tree', 'kd_tree', 'brute']:
				for ls in range(1,100):
						clf = KNC(n_neighbors= nb, weights= w , algorithm= a, leaf_size= ls)
						run_clf(clf, nb, w, a, ls, counter)
						counter+=1
	'''
	clf = KNC(n_neighbors= 10, weights= 'uniform' , algorithm= 'ball_tree', leaf_size= 50)
	run_clf(clf, 10, 'uniform', 'ball_tree', 50, counter)
	print 'Best Total Time'
	display((Classifiers_df.sort_values(by= 'Total', ascending=[1])).head(5))
	print 'Best Accuracy'
	display((Classifiers_df.sort_values(by= 'Accuracy', ascending=[0])).head(5))