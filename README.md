# Deep Learning Tutorial
Abram Hindle



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

![2 crescent slices](slice.png "A function we want to learn f(x,y) -> z where z is red")

There are a few kinds of tasks or functions that could help us here.

* Classification: given some input, predict the class that it belongs
  to. Given a point is it in the red or in the blue?
* Regression: Given a point what will its value be? In the case of a
  function with a continuous or numerous discrete outputs it might be
  appropriate.
* Representation: Learn a smaller representation of the input
  data. E.g. we have 300 features lets describe them in a 128bit hash.

### An example classifier

1-NN: 1 Nearest Neighbor. Given the data, we produce a function that
outputs the CLASS of the nearest neighbor to the input data.







