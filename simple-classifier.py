import theanets
import math
import numpy as np
import numpy.random as rnd
import logging
import sys
logging.basicConfig(stream = sys.stderr, level=logging.INFO)

def f(x,y,z):
    return int(math.sqrt(x*x + y*y + z*z) <= 0.25)

def npf(vec):
    return f(vec[0],vec[1],vec[2])

def g(x,y,z):
    return round(x) or round(y) or round(z)

def npg(vec):
    return g(vec[0],vec[1],vec[2])


possible_points = rnd.uniform(0.0,0.2,(100,3))
unlikely_points = rnd.uniform(0.0,1,(100,3))
points = np.concatenate((possible_points, unlikely_points))
labels = np.apply_along_axis(npf,1,points)
glabels = np.apply_along_axis(npg,1,points)



points = points.astype(np.float32)
labels = labels.astype(np.float32)
ilabels = labels.astype(np.int32)
iglabels = glabels.astype(np.int32)

#res = list()
#for i in range(2,18):
cnet = theanets.Classifier([3,3,3, ('softmax',2)])
cnet.train([points, ilabels], algo='layerwise', patience=1)
cnet.train([points, ilabels], algo='rprop', patience=1)

cgnet = theanets.Classifier([3, ('tanh',3), ('softmax',2)])
cgnet.train([points, iglabels], algo='rprop', patience=1)

rnet = theanets.Regressor([3, 3, 1])
rnet.train([points,labels], algo='rprop', patience=10, batch_size=4)


