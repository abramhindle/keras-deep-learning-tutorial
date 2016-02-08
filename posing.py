import theanets
import scipy
import math
import numpy as np
import numpy.random as rnd
import logging
import sys
from numpy.random import power, normal, lognormal, uniform

logging.basicConfig(stream = sys.stderr, level=logging.INFO)

# how we pose our problem to the deep belief network matters.

# lets make the task easier by scaling all values between 0 and 1
def min_max_scale(data):
    dmin = np.min(data)
    return (data - dmin)/(np.max(data) - dmin)

# now lets make some samples 
bsize    = 100 # how many samples per each
lns      = min_max_scale(lognormal(size=bsize)) #log normal
powers   = min_max_scale(power(0.1,size=bsize)) #power law
norms    = min_max_scale(normal(size=bsize))    #normal
uniforms = min_max_scale(uniform(size=bsize))    #uniform

# poor man's enum
LOGNORMAL=0
POWER=1
NORM=2
UNIFORM=3

# add our data together
data = np.concatenate((lns,powers,norms,uniforms))

# concatenate our labels
labels = np.concatenate((
    (np.repeat(LOGNORMAL,bsize)),
    (np.repeat(POWER,bsize)),
    (np.repeat(NORM,bsize)),
    (np.repeat(UNIFORM,bsize))))
tsize = len(labels)

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

data = data.reshape((len(data),1))
data = data.astype(np.float32)
labels = labels.astype(np.int32)
labels = labels.reshape((len(data),))
test, valid = split_validation(90, data, labels)
cnet = theanets.Classifier([1,4,4])
cnet.train(test,valid, algo='layerwise', patience=1)
cnet.train(test,valid, algo='rprop', patience=10)

print "%s / %s " % (sum(cnet.classify(data) == labels),tsize)

# now that's kind of interesting, an accuracy of .3 to .5 max
# still pretty innaccurate, but 1 sample might never be enough.

# we're going to make rows of 40 features unsorted
width=40
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
wtest, wvalid = split_validation(90, wdata, wlabels)

wcnet = theanets.Classifier([width,width/2,4])
res = wcnet.train(wtest,wvalid, algo='layerwise', patience=1)
print res
res = wcnet.train(wtest,wvalid, algo='rprop', patience=10)
print res

print "%s / %s " % (sum(wcnet.classify(wdata) == labels),tsize)
import collections
print collections.Counter(wcnet.classify(wdata))

# now lets input sorted values

wdata.sort(axis=1)
swcnet = theanets.Classifier([width,width/2,4])
res = swcnet.train(wtest,wvalid, algo='layerwise', patience=1)
print res
res = swcnet.train(wtest,wvalid, algo='rprop', patience=1)
print res
print "%s / %s " % (sum(swcnet.classify(wdata) == labels),tsize)
