'''
Created on Jun 5, 2017

@author: Gajo
'''

#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def point_color(is_poi):
    if is_poi:
        return 'r'
    return 'b'

def plot_cross_features(features, poi, features_list, ignore_zero = True):
    for i in xrange(1, len(features_list)):
        for j in xrange(i + 1, len(features_list)):
            plt.clf()
            plt.xlabel(features_list[i])
            plt.ylabel(features_list[j])
            for data_index in xrange(0, len(features)):
                data = features[data_index]
                x = data[i - 1]
                y = data[j - 1]
                if ignore_zero and (x == 0 or y == 0):
                    continue
                plt.scatter(x, y, c = point_color(poi[data_index]))
            plt.savefig(features_list[i] + "_VS_" + features_list[j] + ".png")
            #plt.show()
        

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

data_dict.pop('TOTAL', 0)
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi'] # You will need to use more features
income_features_list = ['poi']
email_features_list = ['poi']
for feature in data_dict['SKILLING JEFFREY K']:
    if feature != 'poi' and feature != 'email_address':
        features_list.append(feature)
        if feature.startswith(('to_', 'from_', 'shared_')):
            email_features_list.append(feature)
        else:
            income_features_list.append(feature)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

to_poi_percent = 'to_poi_percent'
from_poi_percent = 'from_poi_percent'
plt.xlabel(to_poi_percent)
plt.ylabel(from_poi_percent)

for person, features in my_dataset.iteritems():
    features[to_poi_percent] = 1.0 * float(features['from_this_person_to_poi'])/float(features['from_messages'])
    features[from_poi_percent] = 1.0 * float(features['from_poi_to_this_person'])/float(features['to_messages'])
    plt.scatter(features[to_poi_percent], features[from_poi_percent], c = point_color(features['poi']))
#plt.show()
plt.clf()

features_list.append(to_poi_percent)
features_list.append(from_poi_percent)
    
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#income_data = featureFormat(my_dataset, income_features_list, sort_keys = True)
#income_labels, income_features = targetFeatureSplit(income_data)

#email_data = featureFormat(my_dataset, email_features_list, sort_keys = True)
#email_labels, email_features = targetFeatureSplit(email_data)

#plot_cross_features(income_features, income_labels, income_features_list, ignore_zero=True)
#plot_cross_features(email_features, email_labels, email_features_list, ignore_zero=False)


# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)


### try PCA
pca = PCA(n_components=5)
pca.fit(features_train)
#features_train_pca = pca.transform(features_train)
#features_test_pca = pca.transform(features_test)

#print pca.explained_variance_ratio_


from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
clf = Pipeline(
    [#('pca', PCA(n_components=5)),
     ('clf', DecisionTreeClassifier(#criterion='entropy',
                                    min_samples_split=50,
                                    class_weight='balanced'))])


