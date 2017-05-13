#get features and labels-----------------------------------
import sys
sys.path.append('/Users/yousef/Udacity/Udacity-DAND/Intro_to_Machine_Learning/ud120-projects-master/tools')
from time import time
from sklearn.metrics import accuracy_score

from email_preprocess import preprocess
features_train, features_test, labels_train, labels_test = preprocess()

features_train = features_train[:len(features_train)/100] 
labels_train = labels_train[:len(labels_train)/100]
#----------------------------------------------------------
#Run classifier--------------------------------------------
def run_clf(clf):
	t0 = time()
	clf.fit(features_train, labels_train)
	train_time = round(time()-t0, 3)
	t0 = time()
	pred = clf.predict(features_test)
	pred_time = round(time()-t0, 3)
	acc = accuracy_score(pred, labels_test)
	return (train_time, pred_time, (train_time+pred_time), acc)
#----------------------------------------------------------
def print_in():
	print 'Please choose from Available Classifiers'
	print '1. SVM\n2. Decision Tree\n3. K Nearest Neighbors\n4. Random Forest\n5. AdaBoost'

def get_input(printable):
	user_input = raw_input(printable)
	try:
		user_input = int(user_input)
		return user_input
	except ValueError:
		print 'ERROR: Not valid input - input is not an int'
		user_input = get_input()
		return user_input

print_in()
user_input = get_input('Please input a number:\n')
counter = 1
while(user_input = -1):
	while (user_input not in [1,2,3,4,5]):
		print 'ERROR Not valid input - input not in [1,2,3,4,5]'
		print_in()
		user_input = get_input('Please input a number:\n')

	Classifiers_df = pd.DataFrame(columns=('Class', 'parameters', 'Training', 'Pred', 'Total', 'Accuracy'))

	if user_input == 1:
		print 'You chose SVM Classifier'
		k= raw_input('Please input value for kernel: linear, poly, rbf, sigmoid, precomputed, callable:\n')
		c= get_input('Please input value for C:\t')
		g= get_input('Please input value for gamma:\t')
		d= get_input('Please input value for degree:\t')
		from sklearn.svm import SVC
		clf = SVC(kernel= k, C= c , gamma= g, degree= d)
		a, b, c, d = run_clf(clf)
		parameters = ('k= '+str(k)) + (', c= '+str(c)) + (', g= '+str(g)) + (', d= '+str(d))
		Classifiers_df.loc[counter] = ['SVM', parameters, a, b, c, d]
		display(Classifiers_df.loc[counter])
		counter += 1
	elif user_input == 2:
		print 'You chose Decision Tree Classifier'
		c= raw_input('Please input value for criterion: gini, entropy:\n')
		s= raw_input('Please input value for splitter: best, random:\n')
		mf= raw_input('Please input value for max_features: sqrt, log2, None:\n')
		if mf == 'None':
			mf = None
		mss= get_input('Please input value for min_samples_split:\t')
		from sklearn.tree import DecisionTreeClassifier as DTC
		clf = DTC(criterion= c, splitter= s , max_features= mf, min_samples_split= mss)
		a, b, c, d = run_clf(clf)
		parameters = ('c= '+c) + (', s= '+s) + (', mf= '+mf) + (', mss= '+str(mss))
		Classifiers_df.loc[counter] = ['DT', parameters, a, b, c, d]
		display(Classifiers_df.loc[counter])
		counter += 1
	elif user_input == 3:
		print 'You chose K Nearest Neighbors Classifier'
		nb= get_input('Please input value for n_neighbors:\t')
		w= raw_input('Please input value for weights: uniform, distance:\n')
		a= raw_input('Please input value for algorithm: auto, ball_tree, kd_tree, brute:\n')
		ls= get_input('Please input value for leaf_size:\t')
		from sklearn.neighbors import KNeighborsClassifier as KNC
		clf = KNC(n_neighbors= nb, weights= w , algorithm= a, leaf_size= ls)
		a, b, c, d = run_clf(clf)
		parameters = ('nb= '+str(nb)) + (', w= '+w) + (', a= '+a) + (', ls= '+str(ls))
		Classifiers_df.loc[counter] = ['KNN', parameters, a, b, c, d]
		display(Classifiers_df.loc[counter])
		counter += 1
	elif chosen_classifier == 4:
		print 'You chose Random Forest Classifier'
		n= get_input('Please input value for n_neighbors:\t')
		c= raw_input('Please input value for criterion: gini, entropy:\n')
		mf= raw_input('Please input value for max_features: sqrt, log2, None:\n')
		if mf == 'None':
			mf = None
		mss= get_input('Please input value for min_samples_split:\t')
		from sklearn.neighbors import KNeighborsClassifier as KNC
		clf = RFC(n_estimators= n, criterion= c, max_features= mf, min_samples_split= mss)
		a, b, c, d = run_clf(clf)
		parameters = ('n= '+str(n)) + (', c= '+c) + (', mf= '+mf) + (', mss= '+str(mss))
		Classifiers_df.loc[counter] = ['RF', parameters, a, b, c, d]
		display(Classifiers_df.loc[counter])
		counter += 1
	elif chosen_classifier == 5:
		print 'You chose AdaBoost Classifier'
		n = get_input('Please input value for n_estimators:\t')
		from sklearn.ensemble import AdaBoostClassifier as ABC
		clf = ABC(n_estimators= n)
		a, b, c, d = run_clf(clf)
		parameters = ('n= ' + str(n))
		Classifiers_df.loc[counter] = ['AB', parameters, a, b, c, d]
		display(Classifiers_df.loc[counter])
		counter += 1
	print_in()
	user_input = get_input('Please input a number:\n')
from IPython.display import display
