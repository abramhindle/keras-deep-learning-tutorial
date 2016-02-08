import numpy as np
import collections

def joint_shuffle(arr1,arr2):
    assert len(arr1) == len(arr2)
    indices = np.arange(len(arr1))
    np.random.shuffle(indices)
    arr1[0:len(arr1)] = arr1[indices]
    arr2[0:len(arr2)] = arr2[indices]

def mkcol(x):
    return x.reshape((x.shape[0]*x.shape[1],1))

def split_validation(percent, data, labels):
    ''' 
    split_validation splits a dataset of data and labels into
    2 partitions at the percent mark
    percent should be an int between 1 and 99
    '''
    s = int(percent * len(data) / 100)
    tdata = data[0:s]
    vdata = data[s:]
    tlabels = labels[0:s]
    vlabels = labels[s:]
    return ((tdata,tlabels),(vdata,vlabels))

def classifications(classification, truth):
    tp = sum((classification == True) & (truth == True))
    tn = sum((classification == False) & (truth == False))
    fn = sum((classification == False) & (truth == True))
    fp = sum((classification == True) & (truth == False))
    return [("tp",tp),("tn",tn),("fp",fp),("fn",fn)]
