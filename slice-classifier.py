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
import theanets
import scipy
import math
import numpy as np
import numpy.random as rnd
import logging
import sys
import collections
import theautil

# setup logging
logging.basicConfig(stream = sys.stderr, level=logging.INFO)





mupdates = 1000
data = np.loadtxt("small-slice.csv", delimiter=",")
inputs  = data[0:,0:2].astype(np.float32)
outputs = data[0:,2:3].astype(np.int32)

theautil.joint_shuffle(inputs,outputs)

train_and_valid, test = theautil.split_validation(90, inputs, outputs)
train, valid = theautil.split_validation(90, train_and_valid[0], train_and_valid[1])

def linit(x):
    return x.reshape((len(x),))

train = (train[0],linit(train[1]))
valid = (valid[0],linit(valid[1]))
test  = (test[0] ,linit(test[1]))

# my solution
def in_circle(x,y,cx,cy,radius):
    return (x - float(cx)) ** 2 + (y - float(cy)) ** 2 < radius**2

def mysolution(pt,outer=0.3):
    return in_circle(pt[0],pt[1],0.5,0.5,outer) and not in_circle(pt[0],pt[1],0.5,0.5,0.1)

# apply my classifier
myclasses = np.apply_along_axis(mysolution,1,test[0])
print "My classifier!"
print "%s / %s " % (sum(myclasses == test[1]),len(test[1]))
print theautil.classifications(myclasses,test[1])




# try different combos here
net = theanets.Classifier([2,3,2])
net.train(train, valid, algo='layerwise', max_updates=mupdates, patience=1)
net.train(train, valid, algo='rprop',     max_updates=mupdates, patience=1)

print "Learner on the test set"
classify = net.classify(test[0])
print "%s / %s " % (sum(classify == test[1]),len(test[1]))
print collections.Counter(classify)
print theautil.classifications(classify,test[1])


print net.layers[2].params[0].get_value()
print net.layers[2].params[0].get_value()


def real_function(pt):
    rad = 0.1643167672515498
    in1 = in_circle(pt[0],pt[1],0.5,0.5,rad)
    in2 = in_circle(pt[0],pt[1],0.51,0.51,rad)
    return in1 ^ in2

print "And now on more unseen data that isn't 50/50"

bigtest = np.random.uniform(size=(3000,2)).astype(np.float32)
biglab = np.apply_along_axis(real_function,1,bigtest).astype(np.int32)
net.classify(bigtest)

classify = net.classify(bigtest)
print "%s / %s " % (sum(classify == biglab),len(biglab))
print collections.Counter(classify)
print theautil.classifications(classify,biglab)
