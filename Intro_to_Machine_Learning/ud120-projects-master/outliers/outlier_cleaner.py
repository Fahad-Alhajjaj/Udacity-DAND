#!/usr/bin/python

import numpy

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """

    cleaned_data = []
    ### your code goes here
    error = abs(predictions - net_worths)
    size = len(error) * 0.9
    while len(error) > size:
        index = numpy.argmax(error)
        error = numpy.delete(error, index)
        ages = numpy.delete(ages, index)
        net_worths = numpy.delete(net_worths, index)
    for i in range(0, len(error)):
        cleaned_data.append((ages.item(i), net_worths.item(i), error.item(i)))

    return cleaned_data

