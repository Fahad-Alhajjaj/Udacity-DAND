import sys
sys.path.append('/Users/yousef/Udacity/Udacity-DAND/Intro_to_Machine_Learning/ud120-projects-master/tools')

from email_preprocess import preprocess
features_train, features_test, labels_train, labels_test = preprocess()

features_train = features_train[:len(features_train)/100] 
labels_train = labels_train[:len(labels_train)/100]

def get_features_and_labels():
	return (features_train, features_test, labels_train, labels_test)