#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []
    
    from math import fabs
    ### your code goes here
    for i in xrange(0, len(predictions)):
        age = ages[i]
        net_worth = net_worths[i]
        pred = predictions[i]
        error = fabs(net_worth - pred)
        cleaned_data.append((age, net_worth, error))
    
    cleaned_data.sort(key = lambda x: x[2])
    return cleaned_data[:int(0.9*len(cleaned_data))]

