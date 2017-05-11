import sys

sys.path.append('../Lesson2_Naive_Bayes')
from class_vis import prettyPicture, output_image
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl


features_train, labels_train, features_test, labels_test = makeTerrainData()


########################## SVM #################################
### we handle the import statement and SVC creation for you here
from sklearn.svm import SVC
clf = SVC(kernel="rbf", C=1000000)


#### now your job is to fit the classifier
#### using the training features/labels, and to
#### make a set of predictions on the test data
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

#### store your predictions in a list named pred
t0 = time()
pred = clf.predict(features_test)
print "pred time:", round(time()-t0, 3), "s"

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)

print acc
def submitAccuracy():
    return acc

prettyPicture(clf, features_test, labels_test)
output_image("test.png", "png", open("test.png", "rb").read())