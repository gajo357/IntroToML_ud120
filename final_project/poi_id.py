#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

def evaluate_classifier(clf, features_test, labels_test):
    from sklearn.metrics import confusion_matrix, precision_score, recall_score
    predictions = clf.predict(features_test)
    print clf.score(features_test, labels_test)

    ### evaluate the classifier
    print confusion_matrix(labels_test, predictions)
    print precision_score(labels_test, predictions)
    print recall_score(labels_test, predictions)

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                 'other',
                 #'salary', 
                 #'total_payments', 
                 #'exercised_stock_options', 
                 'total_stock_value', 
                 'expenses', 
                 #'deferred_income',
                 #'bonus', 
                 #'long_term_incentive',
                 #'shared_receipt_with_poi'
                 ] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

#for feature in data_dict['SKILLING JEFFREY K']:
#    if feature not in features_list \
#    and feature != 'email_address' \
#    and feature != 'shared_receipt_with_poi':
#        features_list.append(feature)
        
### Task 2: Remove outliers
data_dict.pop('TOTAL', 0)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
to_poi_percent = 'to_poi_percent'
from_poi_percent = 'from_poi_percent'
for person, features in my_dataset.iteritems():
    features[to_poi_percent] = 1.0 * float(features['from_this_person_to_poi'])/float(features['from_messages'])
    features[from_poi_percent] = 1.0 * float(features['from_poi_to_this_person'])/float(features['to_messages'])

# use new features
#features_list.append(to_poi_percent)
#features_list.append(from_poi_percent)

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html.
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(min_samples_split=50, class_weight='balanced')


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

test_classifier(clf, my_dataset, features_list)
fi = clf.feature_importances_
for i in xrange(0, len(fi)):
    print features_list[i + 1] + " " + str(fi[i])

dump_classifier_and_data(clf, my_dataset, features_list)