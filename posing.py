# Demonstration of how to pose the problem and how different formulations
# lead to different results!
#
# The MIT License (MIT)
# 
# Copyright (c) 2016 Abram Hindle <hindle1@ualberta.ca>, Leif Johnson <leif@lmjohns3.com>
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# first off we load up some modules we want to use
import keras
import scipy
import math
import numpy as np
import numpy.random as rnd
import logging
import sys
from numpy.random import power, normal, lognormal, uniform
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder
import theautil

# What are we going to do?
# - we're going to generate data derived from 4 different distributions
# - we're going to scale that data
# - we're going to create a RBM (1 hidden layer neural network)
# - we're going to train it to classify data as belonging to one of these distributions

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

# how many samples per each distribution
bsize    = 100 

# poor man's enum
LOGNORMAL=0
POWER=1
NORM=2
UNIFORM=3

print('''
########################################################################
# Experiment 1: can we classify single samples?
#
#
#########################################################################
''')

def make_dataset1():
    '''Make a dataset of single samples with labels from which distribution they come from'''
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

# this will be the training data and validation data
data, labels, tsize = make_dataset1()

# this is the test data, this is kept separate to prove we can
# actually work on the data we claim we can.
#
# Without test data, you might just have great performance on the
# train set.
test_data, test_labels, _ = make_dataset1()


# now lets shuffle
# If we're going to select a validation set we probably want to shuffle
def joint_shuffle(arr1,arr2):
    assert len(arr1) == len(arr2)
    indices = np.arange(len(arr1))
    np.random.shuffle(indices)
    arr1[0:len(arr1)] = arr1[indices]
    arr2[0:len(arr2)] = arr2[indices]

# our data and labels are shuffled together
joint_shuffle(data,labels)


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

# make a validation set from the train set
train1, valid1 = split_validation(90, data, labels)


print(train1[0].shape)
print(train1[1].shape)

enc1 = OneHotEncoder(handle_unknown='ignore')
enc1.fit(train1[1].reshape(len(train1[1]),1))
train1_y = enc1.transform(train1[1].reshape(len(train1[1]),1))
print(train1_y.shape)
valid1_y = enc1.transform(valid1[1].reshape(len(valid1[1]),1))
print(valid1_y.shape)
test1_y = enc1.transform(test_labels.reshape(len(test_labels),1))
print(test1_y.shape)

# build our classifier

print("We're building a MLP of 1 input layer node, 4 hidden layer nodes, and an output layer of 4 nodes. The output layer has 4 nodes because we have 4 classes that the neural network will output.")
cnet = Sequential()
cnet.add(Dense(4,input_shape=(1,),activation="sigmoid"))
cnet.add(Dense(4,activation="softmax"))
copt = SGD(lr=0.1)
# opt = Adam(lr=0.1)
cnet.compile(loss="categorical_crossentropy", optimizer=copt, metrics=["accuracy"])
history = cnet.fit(train1[0], train1_y, validation_data=(valid1[0], valid1_y),
	            epochs=100, batch_size=16)

#score = cnet.evaluate(test_data, test_labels)
#print("Scores: %s" % score)
classify = cnet.predict_classes(test_data)
print(theautil.classifications(classify,test_labels))
score = cnet.evaluate(test_data, test1_y)
print("Scores: %s" % score)

# now that's kind of interesting, an accuracy of .3 to .5 max
# still pretty inaccurate, but 1 sample might never be enough.

print("We could train longer and we might get better results, but there's ambiguity in each. As a human we might have a hard time determining them.")

print('''
########################################################################
# Experiment 2: can we classify a sample of data?
#
#
#########################################################################
''')
print("In this example we're going to input 40 values from a single distribution, and we'll see if we can classify the distribution.")

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

# make our train sets
wdata, wlabels = make_widedataset()
# make our test sets
test_wdata, test_wlabels = make_widedataset()

# split out our validation set
wtrain, wvalid = split_validation(90, wdata, wlabels)
print("At this point we have a weird decision to make, how many neurons in the hidden layer?")

encwc = OneHotEncoder(handle_unknown='ignore')
encwc.fit(wtrain[1].reshape(len(wtrain[1]),1))
wtrain_y = encwc.transform(wtrain[1].reshape(len(wtrain[1]),1))
wvalid_y = encwc.transform(wvalid[1].reshape(len(wvalid[1]),1))
wtest_y  = encwc.transform(test_wlabels.reshape(len(test_wlabels),1))

# wcnet = theanets.Classifier([width,width/4,4]) #267
wcnet = Sequential()
wcnet.add(Dense(width,input_shape=(width,),activation="sigmoid"))
wcnet.add(Dense(int(width/4),activation="sigmoid"))
wcnet.add(Dense(4,activation="softmax"))
wcnet.compile(loss="categorical_crossentropy", optimizer=SGD(lr=0.1), metrics=["accuracy"])
history = wcnet.fit(wtrain[0], wtrain_y, validation_data=(wvalid[0], wvalid_y),
	            epochs=100, batch_size=16)


classify = wcnet.predict_classes(test_wdata)
print(theautil.classifications(classify,test_wlabels))
score = wcnet.evaluate(test_wdata, wtest_y)
print("Scores: %s" % score)

# # You could try some of these alternative setups
# 
# [width,4]) #248
# [width,width/2,4]) #271
# [width,width,4]) #289
# [width,width*2,4]) #292
# [width,width/2,width/4,4]) #270
# [width,width/2,width/4,width/8,width/16,4]) #232
# [width,width*8,4]) #304

print("Ok that was neat, it definitely worked better, it had more data though.")

print("But what if we help it out, and we sort the values so that the first and last bins are always the min and max values?")

# now lets input sorted values

print('''
########################################################################
# Experiment 3: can we classify a SORTED sample of data?
#
#
#########################################################################
''')


print("Sorting the data")
wdata.sort(axis=1)
test_wdata.sort(axis=1)


swcnet = Sequential()
swcnet.add(Dense(width,input_shape=(width,),activation="sigmoid"))
swcnet.add(Dense(int(width/4),activation="sigmoid"))
swcnet.add(Dense(4,activation="softmax"))
swcnet.compile(loss="categorical_crossentropy", optimizer=SGD(lr=0.1), metrics=["accuracy"])
history = swcnet.fit(wtrain[0], wtrain_y, validation_data=(wvalid[0], wvalid_y),
	            epochs=100, batch_size=16)


classify = swcnet.predict_classes(test_wdata)
print(theautil.classifications(classify,test_wlabels))
score = swcnet.evaluate(test_wdata, wtest_y)
print("Scores: %s" % score)


# 
# swcnet = theanets.Classifier([width,width/2,4])
# res = swcnet.train(wtrain,wvalid, algo='layerwise', patience=1, max_updates=mupdates)
# print(res)
# res = swcnet.train(wtrain,wvalid, algo='rprop', patience=1, max_updates=mupdates)
# print(res)
# print("%s / %s " % (sum(swcnet.classify(wdata) == wlabels),tsize))
# print("%s / %s " % (sum(swcnet.classify(test_wdata) == test_wlabels),tsize))
# print(collections.Counter(swcnet.classify(test_wdata)))
#

print("That was an improvement!")

print("What if we add binning, where by we classify the histogram?")

# a little better

print('''
########################################################################
# Experiment 4: can we classify a discretized histogram of sample data?
#
#
#########################################################################
'''
)
# let's try actual binning


def bin(row):
    return np.histogram(row,bins=len(row),range=(0.0,1.0))[0]/float(len(row))

print("Apply the histogram to all the data rows")
bdata = np.apply_along_axis(bin,1,wdata).astype(np.float32)
blabels = wlabels

# ensure we have our test data
test_bdata = np.apply_along_axis(bin,1,test_wdata).astype(np.float32)
test_blabels = test_wlabels

# helper data 
enum_funcs = [
    (LOGNORMAL,"log normal",lambda size: lognormal(size=size)),
    (POWER,"power",lambda size: power(0.1,size=size)),
    (NORM,"normal",lambda size: normal(size=size)),
    (UNIFORM,"uniforms",lambda size: uniform(size=size)),
]

# uses enum_funcs to evaluate PER CLASS how well our classify operates
def classify_test(bnet,ntests=1000):
    for tup in enum_funcs:
        enum, name, func = tup
        lns = min_max_scale(func(size=(ntests,width))) #log normal
        blns = np.apply_along_axis(bin,1,lns).astype(np.float32)
        blns_labels = np.repeat(enum,ntests)
        blns_labels.astype(np.int32)
        classification = bnet.classify(blns)
        print("%s %s / %s ::: %s " % (name,sum(classification == blns_labels),ntests, collections.Counter(classification)))


# train & valid
btrain, bvalid = split_validation(90, bdata, blabels)
# similar network structure
bnet = theanets.Classifier([width,width/2,4])

# bnet = theanets.Classifier([width,32*width/2,4])

# layerwise training (RBM)
res = bnet.train(btrain,bvalid, algo='layerwise', patience=1, max_updates=mupdates)
print(res)
classify_test(bnet)
res = bnet.train(btrain,bvalid, algo='rprop', patience=1, max_updates=mupdates)
print(res)
print("%s / %s " % (sum(bnet.classify(bdata) == blabels),tsize))
print("%s / %s " % (sum(bnet.classify(test_bdata) == test_blabels),tsize))
classify_test(bnet)
# sometimes lognormal doesn't show up so well -- it can look like a powerlaw
# so after binning I have to say it is far more robust than before
