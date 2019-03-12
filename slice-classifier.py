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

mltrain = (train[0],linit(train[1]))
mlvalid = (valid[0],linit(valid[1]))
mltest  = (test[0] ,linit(test[1]))

# my solution
def in_circle(x,y,cx,cy,radius):
    return (x - float(cx)) ** 2 + (y - float(cy)) ** 2 < radius**2

def mysolution(pt,outer=0.3):
    return in_circle(pt[0],pt[1],0.5,0.5,outer) and not in_circle(pt[0],pt[1],0.5,0.5,0.1)

# apply my classifier
myclasses = np.apply_along_axis(mysolution,1,mltest[0])
print("My classifier!")
print("%s / %s " % (sum(myclasses == mltest[1]),len(mltest[1])))
print(theautil.classifications(myclasses,mltest[1]))

def euclid(pt1,pt2):
    return sum([ (pt1[i] - pt2[i])**2 for i in range(0,len(pt1)) ])

def oneNN(data,labels):
    def func(input):
        distance = None
        label = None
        for i in range(0,len(data)):
            d = euclid(input,data[i])
            if distance == None or d < distance:
                distance = d
                label = labels[i]
        return label
    return func

learner = oneNN(mltrain[0],mltrain[1])

oneclasses = np.apply_along_axis(learner,1,mltest[0])
print("1-NN classifier!")
print("%s / %s " % (sum(oneclasses == mltest[1]),len(mltest[1])))
print(theautil.classifications(oneclasses,mltest[1]))

print('''
########################################################################
# Part 3. Let's start using neural networks!
########################################################################
''')

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(train[1])
train_y = enc.transform(train[1])
valid_y = enc.transform(valid[1])
test_y = enc.transform(test[1])

print(train[0].shape)
print(train[1].shape)
print(train_y.shape)

# try different combos here
net = Sequential()
net.add(Dense(16,input_shape=(2,),activation="sigmoid"))
net.add(Dense(16,activation="relu"))
net.add(Dense(2,activation="softmax"))
opt = SGD(lr=0.1)
# opt = Adam(lr=0.1)
net.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
history = net.fit(train[0], train_y, validation_data=(valid[0], valid_y),
	            epochs=100, batch_size=16)

print("Learner on the test set")
score = net.evaluate(test[0], test_y)
print("Scores: %s" % score)
predictit = net.predict(test[0])
print(predictit.shape)
print(predictit[0:10,])
classify = net.predict_classes(test[0])

print("%s / %s " % (np.sum(classify == mltest[1]),len(mltest[1])))
print(collections.Counter(classify))
print(theautil.classifications(classify,mltest[1]))

def real_function(pt):
    rad = 0.1643167672515498
    in1 = in_circle(pt[0],pt[1],0.5,0.5,rad)
    in2 = in_circle(pt[0],pt[1],0.51,0.51,rad)
    return in1 ^ in2

print("And now on more unseen data that isn't 50/50")

bigtest = np.random.uniform(size=(3000,2)).astype(np.float32)
biglab = np.apply_along_axis(real_function,1,bigtest).astype(np.int32)

classify = net.predict_classes(bigtest)
print("%s / %s " % (sum(classify == biglab),len(biglab)))
print(collections.Counter(classify))
print(theautil.classifications(classify,biglab))
