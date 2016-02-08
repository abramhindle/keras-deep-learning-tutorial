# first off we load up some modules we want to use
import theanets
import scipy
import math
import numpy as np
import numpy.random as rnd
import logging
import sys
from numpy.random import power, normal, lognormal, uniform

# maximum number of iterations before we bail
mupdates = 1000

# setup logging
logging.basicConfig(stream = sys.stderr, level=logging.INFO)

# how we pose our problem to the deep belief network matters.

# lets make the task easier by scaling all values between 0 and 1
def min_max_scale(data):
    '''scales data by minimum and maximum values between 0 and 1'''
    dmin = np.min(data)
    return (data - dmin)/(np.max(data) - dmin)

bsize    = 100 # how many samples per each

# poor man's enum
LOGNORMAL=0
POWER=1
NORM=2
UNIFORM=3

def make_dataset1():
    # now lets make some samples 
    lns      = min_max_scale(lognormal(size=bsize)) #log normal
    powers   = min_max_scale(power(0.1,size=bsize)) #power law
    norms    = min_max_scale(normal(size=bsize))    #normal
    uniforms = min_max_scale(uniform(size=bsize))    #uniform
    

    # add our data together
    data = np.concatenate((lns,powers,norms,uniforms))

    # concatenate our labels
    labels = np.concatenate((
        (np.repeat(LOGNORMAL,bsize)),
        (np.repeat(POWER,bsize)),
        (np.repeat(NORM,bsize)),
        (np.repeat(UNIFORM,bsize))))
    tsize = len(labels)

    # make sure dimensionality and types are right
    data = data.reshape((len(data),1))
    data = data.astype(np.float32)
    labels = labels.astype(np.int32)
    labels = labels.reshape((len(data),))

    return data, labels, tsize

data, labels, tsize = make_dataset1()

test_data, test_labels, _ = make_dataset1()

# now lets shuffle
def joint_shuffle(arr1,arr2):
    assert len(arr1) == len(arr2)
    indices = np.arange(len(arr1))
    np.random.shuffle(indices)
    arr1[0:len(arr1)] = arr1[indices]
    arr2[0:len(arr2)] = arr2[indices]

# our data and labels are shuffled together
joint_shuffle(data,labels)

def split_validation(percent, data, labels):
    ''' percent should be an int '''
    s = percent * len(data) / 100
    tdata = data[0:s]
    vdata = data[s:]
    tlabels = labels[0:s]
    vlabels = labels[s:]
    return ((tdata,tlabels),(vdata,vlabels))

train, valid = split_validation(90, data, labels)
cnet = theanets.Classifier([1,4,4])
cnet.train(train,valid, algo='layerwise', patience=1, max_updates=mupdates)
cnet.train(train,valid, algo='rprop', patience=10, max_updates=mupdates)

print "%s / %s " % (sum(cnet.classify(data) == labels),tsize)
print "%s / %s " % (sum(cnet.classify(test_data) == test_labels),tsize)


# now that's kind of interesting, an accuracy of .3 to .5 max
# still pretty innaccurate, but 1 sample might never be enough.

width=40

def make_widedataset(width=width):
    # we're going to make rows of 40 features unsorted
    wlns      = min_max_scale(lognormal(size=(bsize,width))) #log normal
    wpowers   = min_max_scale(power(0.1,size=(bsize,width))) #power law
    wnorms    = min_max_scale(normal(size=(bsize,width)))    #normal
    wuniforms = min_max_scale(uniform(size=(bsize,width)))    #uniform

    wdata = np.concatenate((wlns,wpowers,wnorms,wuniforms))

    # concatenate our labels
    wlabels = np.concatenate((
        (np.repeat(LOGNORMAL,bsize)),
        (np.repeat(POWER,bsize)),
        (np.repeat(NORM,bsize)),
        (np.repeat(UNIFORM,bsize))))

    joint_shuffle(wdata,wlabels)
    wdata = wdata.astype(np.float32)
    wlabels = wlabels.astype(np.int32)
    wlabels = wlabels.reshape((len(data),))
    return wdata, wlabels

wdata, wlabels = make_widedataset()
test_wdata, test_wlabels = make_widedataset()


wtrain, wvalid = split_validation(90, wdata, wlabels)
wcnet = theanets.Classifier([width,width/2,4])
res = wcnet.train(wtrain,wvalid, algo='layerwise', patience=1, max_updates=mupdates)
print res
res = wcnet.train(wtrain,wvalid, algo='rprop',max_updates=mupdates, patience=1)
print res

print "%s / %s " % (sum(wcnet.classify(wdata) == wlabels),tsize)
import collections
print collections.Counter(wcnet.classify(wdata))

print "%s / %s " % (sum(wcnet.classify(test_wdata) == test_wlabels),tsize)
print collections.Counter(wcnet.classify(test_wdata))



# now lets input sorted values

wdata.sort(axis=1)
test_wdata.sort(axis=1)

swcnet = theanets.Classifier([width,width/2,4])
res = swcnet.train(wtrain,wvalid, algo='layerwise', patience=1, max_updates=mupdates)
print res
res = swcnet.train(wtrain,wvalid, algo='rprop', patience=1, max_updates=mupdates)
print res
print "%s / %s " % (sum(swcnet.classify(wdata) == wlabels),tsize)
print "%s / %s " % (sum(wcnet.classify(test_wdata) == test_wlabels),tsize)
print collections.Counter(wcnet.classify(test_wdata))



# a little better

# let's try actual binning

def bin(row):
    return np.histogram(row,bins=len(row),range=(0.0,1.0))[0]/float(len(row))

bdata = np.apply_along_axis(bin,1,wdata)
blabels = wlabels

test_bdata = np.apply_along_axis(bin,1,test_wdata)
test_blabels = test_wlabels


enum_funcs = [
    (LOGNORMAL,"log normal",lambda size: lognormal(size=size)),
    (POWER,"power",lambda size: power(0.1,size=size)),
    (NORM,"normal",lambda size: normal(size=size)),
    (UNIFORM,"uniforms",lambda size: uniform(size=size)),
]

def classify_test(bnet,ntests=1000):
    for tup in enum_funcs:
        enum, name, func = tup
        lns = min_max_scale(func(size=(ntests,width))) #log normal
        blns = np.apply_along_axis(bin,1,lns)
        blns_labels = np.repeat(enum,ntests)
        classification = bnet.classify(blns)
        print "%s %s / %s ::: %s " % (name,sum(classification == blns_labels),ntests, collections.Counter(classification))



btrain, bvalid = split_validation(90, bdata, blabels)
bnet = theanets.Classifier([width,width/2,4])
res = bnet.train(btrain,bvalid, algo='layerwise', patience=1, max_updates=mupdates)
print res
classify_test(bnet)
res = bnet.train(btrain,bvalid, algo='rprop', patience=1, max_updates=mupdates)
print res
print "%s / %s " % (sum(bnet.classify(bdata) == blabels),tsize)
print "%s / %s " % (sum(bnet.classify(test_bdata) == test_blabels),tsize)
classify_test(bnet)
# somtimes lognormal doesn't show up so well
