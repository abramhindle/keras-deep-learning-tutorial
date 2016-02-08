# Deep Learning Tutorial

#### Abram Hindle
#### <abram.hindle@ualberta.ca>
#### http://softwareprocess.ca/

Slides stolen gracefully from Ben Zittlau



## Start

First off lets get a useful Python environment!

Please install theanets and bpython.

`````
pip install theanets
pip install bpython
`````

otherwise consider provisioning a vagrant box defined by the
vagrantfile in the vagrant/ directory.



## Data
We're planning on using some data. TBD.




# Intro
### What is machine learning?

Building a function from data to classify, predict, group, or represent data.




# Intro
### Motivational Example

Imagine we have this data:

![2 crescent slices](images/slice.png "A function we want to learn
 f(x,y) -> z where z is red")



# Intro
### Machine Learning

There are a few kinds of tasks or functions that could help us here.

* Classification: given some input, predict the class that it belongs
  to. Given a point is it in the red or in the blue?
* Regression: Given a point what will its value be? In the case of a
  function with a continuous or numerous discrete outputs it might be
  appropriate.
* Representation: Learn a smaller representation of the input
  data. E.g. we have 300 features lets describe them in a 128bit hash.



# Intro 
### An example classifier

1-NN: 1 Nearest Neighbor.

Given the data, we produce a function that
outputs the CLASS of the nearest neighbour to the input data.

Whoever is closer, is the class. 3-NN is 3-nearest neighbors whereby
we use voting of the 3 neighbors instead.



# Intro
### An example classifier: 1-NN

``` python
def euclid(pt1,pt2):
    return sum([ (pt1[i] - pt2[i])**2 for i in range(0,len(pt1)) ])

def oneNN(data,labels):
    def func(input):
        distance = None
        label = None
        for i in range(0,len(data))
            d = euclid(input,data[i])
            if distance == None or d < distance:
                distance = d
                label = labels[i]
        return label
```







