#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                      'restricted_stock_deferred', 'deferred_income', 'total_stock_value',
                      'expenses', 'exercised_stock_options', 'other', 'long_term_incentive',
                      'restricted_stock', 'director_fees']

email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages',
                  'from_this_person_to_poi', 'shared_receipt_with_poi']

features_list = ["poi"] + financial_features + email_features


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
#There are two elements that are not employees/persons and can therefore be
#classified as outliers and should be removed:
data_dict.pop("TOTAL", 0) #discovered in the Outlier Lesson
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0) #from the accompanying PDF


### Task 3: Create new feature(s)
### FROM/TO POI
for k,v in data_dict.items():
    if v["from_poi_to_this_person"] != "NaN" and v["from_this_person_to_poi"] != "NaN":
        v["from_POI_rate"] = v["from_poi_to_this_person"] / float(v["to_messages"])
        v["to_POI_rate"] = v["from_this_person_to_poi"] / float(v["from_messages"])
    else:
        v["from_POI_rate"] = 0
        v["to_POI_rate"] = 0

features_list.append("from_POI_rate")
features_list.append("to_POI_rate")

### TOTAL VALUE
for k,v in data_dict.items():
    if not all([v[f] == "NaN" for f in financial_features]):
        v["total_value"] = sum([v[f] for f in financial_features if (v[f] != "NaN")])
    else:
        v["total_value"] = "NaN"

features_list.append("total_value")

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import cross_validation
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler


#Chosen and tuned classifier based on explorations in notebook
pipe_clf = DecisionTreeClassifier(min_samples_split=15, splitter='best', max_depth=15, min_samples_leaf=3)

steps = [('scaler', StandardScaler()),
        ('select_features', SelectKBest(k=15)),
        ('clf', pipe_clf)]

clf = Pipeline(steps)

clf.fit(features, labels)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

print "Done."
