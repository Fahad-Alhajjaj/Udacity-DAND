""" quiz materials for feature scaling clustering """

### FYI, the most straightforward implementation might 
### throw a divide-by-zero error, if the min and max
### values are the same
### but think about this for a second--that means that every
### data point has the same value for that feature!  
### why would you rescale it?  Or even use it at all?
def featureScaling(arr):
    mini = float(min(arr))
    maxi = float(max(arr))
    if min(arr) == max(arr):
        return arr
    for i in range(0, len(arr)):
    	arr[i] = (arr[i]-mini)/(maxi-mini)
    return arr

# tests of your feature scaler--line below is input data
data = [115, 140, 175]
print featureScaling(data)

