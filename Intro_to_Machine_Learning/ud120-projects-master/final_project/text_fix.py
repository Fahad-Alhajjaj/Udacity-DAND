import pickle
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import tree
from sklearn.metrics import accuracy_score
import os
import sys
import re
from nltk.stem.snowball import SnowballStemmer
import string
import numpy
from time import time
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.feature_selection import SelectPercentile, f_classif
import pprint

def find_signature(email_list, poi_emails):
	print 'IN find_signature'
	t0 = time()
	#vectorize_text(email_list, poi_emails)
	print 'Done reading and dumping files - Time taking= ', round(time()-t0, 3)
	words_file = "your_word_data.pkl" 
	authors_file = "your_email_authors.pkl"
	t0 = time()
	word_data = pickle.load( open(words_file, "r"))
	authors = pickle.load( open(authors_file, "r") )
	print 'Done loading files - Time taking= ', round(time()-t0, 3)

	t0 = time()
	features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data,
																								authors,
																								test_size=0.1,
																								random_state=42)
	print 'Done cross_validation - Time taking= ', round(time()-t0, 3)
	t0 = time()
	vectorizer = TfidfVectorizer(sublinear_tf=True,
								max_df=0.5,
								stop_words='english')

	features_train = vectorizer.fit_transform(features_train).toarray()
	print 'Done fit_transform - Time taking= ', round(time()-t0, 3)
	t0 = time()
	features_test  = vectorizer.transform(features_test).toarray()
	print 'Done transform - Time taking= ', round(time()-t0, 3)
	'''
	print pprint.pprint(features_train)
	print features_train.shape
	print len(labels_train)
	for i in range(0, len(features_train)):
		print numpy.count_nonzero(features_train[i])
	'''
	feature_names = vectorizer.get_feature_names()
	print 'feature_names length = ', len(feature_names)
	
	
	clf = tree.DecisionTreeClassifier(min_samples_split = 40,
									criterion='gini')
	t0 = time()
	clf.fit(features_train, labels_train)
	print 'Done fitting Classifier - Time taking= ', round(time()-t0, 3)
	t0 = time()
	pred = clf.predict(features_test)
	print 'Done predicting - Time taking= ', round(time()-t0, 3)
	t0 = time()
	acc = accuracy_score(pred, labels_test)
	recall = recall_score(labels_test, pred)
	precision = precision_score(labels_test, pred)
	print 'Done scoring - Time taking= ', round(time()-t0, 3)

	print 'Accuracy:  ', acc
	print 'Recall:    ', recall
	print 'Precision: ', precision

	importance = clf.feature_importances_
	for i in range(0, len(importance)):
		if importance[i] >= 0.04:
			print feature_names[i], ' = ', importance[i]

def vectorize_text(email_list, poi_emails):
	word_data = []
	from_data = []
	print 'IN vectorize_text'
	from_email = re.compile('from_(.*).txt')
	#to_email = re.compile('to_(.*).txt')
	list_of_emails  = open("emails_by_address\\fileNames.txt", "r")
	count_files = 0
	count_emails = 0
	total_emails = 0
	for file_name in list_of_emails:
		file_name = file_name[:-1]
		m1 = from_email.search(file_name)
		#m2 = to_email.search(file_name)
		if m1:
			search_email = m1.group(1)
			#elif m2:
			#	search_email = m2.group(1)
		else:
			break
		if search_email not in email_list:
			continue
		file_name = os.path.join('emails_by_address\\', file_name)
		emails_paths = open(file_name, "r")
		count_files+=1
		count_emails = 0
		print count_files, (1 if search_email in poi_emails else 0)
		all_emails = ''
		print file_name
		for path in emails_paths:
			'''
			if count_emails == 10:
				break
			'''
			path = path.replace("/", "\\")
			path = path.replace(".", "_")
			path = '..\..\..\..\\' + path[:-1]
			try:
				email = open(path, "r")

				stemmed_text = parseOutText(email)

				all_emails = ' '.join([all_emails, stemmed_text])
				
				email.close()
				count_emails += 1
			except IOError:
				pass
		emails_paths.close()
		word_data.append(all_emails)
		from_data.append(1 if search_email in poi_emails else 0)
		total_emails += count_emails
		print 'emails:\t', count_emails
	print 'Total emails = ', total_emails
	print "emails processed"
	list_of_emails.close()

	pickle.dump( word_data, open("your_word_data.pkl", "w") )
	pickle.dump( from_data, open("your_email_authors.pkl", "w") )

def parseOutText(f):
	f.seek(0)
	all_text = f.read()
	content = all_text.split("X-FileName:")
	words = ""
	if len(content) > 1:
		text_string = content[1].translate(string.maketrans("", ""), string.punctuation)
	stemmer = SnowballStemmer("english")
	words = ' '.join(stemmer.stem(word) for word in text_string.split())

	return words

def count_emails(emails_list):
	email = re.compile('.*_(.*).txt')
	email_dict = {}
	list_of_emails  = open("emails_by_address\\fileNames.txt", "r")
	for file_name in list_of_emails:
		is_from = 'from' in file_name
		file_name = file_name[:-1]
		m1 = email.search(file_name)
		if m1:
			search_email = m1.group(1)
		else:
			search_email = None
		if search_email not in emails_list:
			continue
		file_name = os.path.join('emails_by_address\\', file_name)
		emails_paths = open(file_name, "r")
		emails_count = len(emails_paths.readlines())
		if is_from:
			email_dict[search_email] = {}
			email_dict[search_email]['from'] = emails_count
		else:
			email_dict[search_email]['to'] = emails_count
	return email_dict