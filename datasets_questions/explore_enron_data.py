#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

#pois = list()
#with open("../final_project/poi_names.txt", 'r') as f:
#    for line in f:
#        if line[0] == '(':
#            name = line[4:].upper().replace(',', '')
#            pois.append(name)

import math
total_payments = 0
poi = 0
for person, features in enron_data.iteritems():
    if features['poi']:
        poi += 1
        if math.isnan(float(features['total_payments'])):
            total_payments += 1

print total_payments
print 1.0 * total_payments/poi

jeff = enron_data['SKILLING JEFFREY K']
lay = enron_data['LAY KENNETH L']
andrew = enron_data['FASTOW ANDREW S']

#for feature, value in features.iteritems():
#    print feature + ': ' + str(value)

