#!/opt/local/bin/python2.7

import input_data_LIDC

import sys
import os
import numpy as np
from time import strftime
import pickle

# from sklearn.datasets import load_iris
import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier


def esprint(msg):
  print("[%s] %s" % (strftime("%Y%m%d-%H%M%S"), msg))


# Consts.
inputDataFilePath = ""
inputTargetFilePath = ""
inputImageSOPToFileNameMapping = "LIDC_Complete_20141106/Extracts/DICOM_metadata_extracts/"


if len(sys.argv) <= 2:
  sys.exit("Specify the patient to process, e.g. 'LIDC-IDRI-0001'.")
patId = sys.argv[1]
imageType = sys.argv[2] # bin or orig

# If pickled dataset joining X & y values doesn't exist, then create it; else, use it.
pickle_file_name = "Evans-MacBook-Pro.local-8x8_edge_patches-"+imageType+"-"+patId+".pickle"
if os.path.isfile(pickle_file_name):
  input_data_LIDC.esprint("Unpickling: " + pickle_file_name)
  with open(pickle_file_name, "rb") as pickle_file:
    dataset_input = pickle.load(pickle_file)
else:
  dataset_input = input_data_LIDC.read_data_sets(
    '../../../LIDC_Complete_20141106/LIDC-IDRI-edge_patches/'+patId,
    (os.sep + imageType + os.sep),
    '*.tiff',
    '../../../LIDC_Complete_20141106/Extracts/master_join4.csv',
    '../../../LIDC_Complete_20141106/Extracts/DICOM_metadata_extracts/',
    '*.csv')
  input_data_LIDC.esprint("Pickling: " + pickle_file_name)
  with open(pickle_file_name, "wb") as pickle_file:
    pickle.dump(dataset_input, pickle_file)
esprint("Done pickling.")



# Randomize the image & label set in-place, taking care to maintain array correspondance.
# First, re-merge the training, validation, and test sets into a single set.
train_images, train_labels = dataset_input[0]
# validation_images, validation_labels = dataset_input[1]
test_images, test_labels = dataset_input[1]




# RUN DEC TREE CLF.
esprint("Done.")



clf = DecisionTreeClassifier(random_state=0)
train_cv_rslt = cross_val_score(clf, train_images, train_labels, cv=10)
esprint("* Training CV result: " + str(train_cv_rslt))
test_cv_rslt = cross_val_score(clf, test_images, test_labels, cv=10)
esprint("* Test CV result: " + str(test_cv_rslt))

fit_train = clf.fit(train_images, train_labels)
esprint("* Training fit result: " + str(fit_train))
fit_test = clf.fit(test_images, test_labels)
esprint("* Test fit result: " + str(fit_test))

pred_train = clf.predict_proba(train_images)
esprint("* Training prediction result: " + str(pred_train))
pred_test = clf.predict_proba(test_images)
esprint("* Test prediction result: " + str(pred_test))

score_train = clf.score(train_images, train_labels)
esprint("* Training score: " + str(score_train))
score_test = clf.score(test_images, test_labels)
esprint("* Test score: " + str(score_test))




# ## Main. ##
# print "----- IRIS DATA -----"
# print data
# print "----- IRIS TARGET -----"
# print iris.target
# print "----- IRIS CV SCORE -----"
# print cross_val_score(clf, iris.data, iris.target, cv=10)