def get_input():
	print 'Please choose from Available Classifiers'
	print '1. SVM\n2. Decision Tree\n3. K Nearest Neighbors\n4. Random Forest\n5. AdaBoost'
	chosen_classifier = raw_input('Please input a number (1-5):\n')
	try:
		chosen_classifier = int(chosen_classifier)
		return chosen_classifier
	except ValueError:
		print 'ERROR: Not valid input - input is not an int'
		chosen_classifier = get_input()

chosen_classifier = get_input()

while (chosen_classifier not in [1,2,3,4,5]):
	print 'ERROR Not valid input - input not in [1,2,3,4,5]'
	chosen_classifier = get_input()

if chosen_classifier == 1:
	print 'You chose SVM Classifier'
	from svm_optimizer import svm_opt
	svm_opt()
elif chosen_classifier == 2:
	print 'You chose Decision Tree Classifier'
	from dt_optimizer import dt_opt
	dt_opt()
elif chosen_classifier == 3:
	print 'You chose K Nearest Neighbors Classifier'
	from KNeighbors_optimizer import kn_opt
	kn_opt()
elif chosen_classifier == 4:
	print 'You chose Random Forest Classifier'
	from RandomForest_optimizer import rf_opt
	rf_opt()
elif chosen_classifier == 5:
	print 'You chose AdaBoost Classifier'
	from AdaBoost_optimizer import aBoost_opt
	aBoost_opt()

print 'DONE'