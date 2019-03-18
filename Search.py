# Copyright 2018 Abram Hindle <hindle1@ualberta.ca>
# 
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Search Module
  
   This module engages in search, grid search, or random search to
   allow for tuning. It is not clever, it does not pull values from
   distributions, it searches over discrete spaces. But given enough
   dimensions that doesn't matter does it.

   See example_search for an idea how to use the interface.

"""
import numpy
import pandas
import random
from time import time as now
import itertools

# here's what a distribution of parameters looks like

example_param_dist = {
    "alphas":    [0.0001,0.001,0.001,0.01,0.1,1.0,2.0,10.0],
    "topics":    [5,10,15,20,25,50,100,200,300,500],
    "thresholds":[0.001,0.01,0.1,0.2,0.5],
    "iterations":[50,100,200],
    "passes":    [1,2,8],
    "types":     ["knn","lr"],
}

def iterator_of_all_parameters(dist):
    rownames = dist.keys()
    product = itertools.product(*dist.values())
    return product, rownames
    

def random_parameters(dist):
    ''' choose random parameters from a distribution of discrete choices '''
    params = dict()
    for key in dist:
        params[key] = random.choice(dist[key])
    return params

def heuristic_function(state,params,reps=5,f=None):
    ''' example heuristic function
        you can pass your your own lambda(state,params) -> float as f
    '''
    # do 5 repetitions
    if f is None:
        raise "No Function to actually run! please provide an 'f' function!"
    else:
        evals = [f(state,params) for rep in range(reps)]
        return numpy.median(evals)

def random_search(state, distribution, heuristic_function=heuristic_function, iterations=100, time=None):
    ''' randomly search the space 
        if time is set in seconds, then it is which ever is shorter, iterations or time.
    '''
    results = list()
    start = now()
    for i in range(iterations):
        params = random_parameters(distribution)
        result = heuristic_function(state,params,reps=state.get("reps",5),f=state.get("f",None))
        params["Score"] = result
        # print(params)
        results.append(params)
        if time is not None:
            if now() > start + time:
                # print("Times Up for random search!")
                return results
    return results


def grid_search(state, distribution, heuristic_function):
    ''' not a generic grid search, just go through our parameters'''
    product, rownames = iterator_of_all_parameters(distribution)
    # this is all combos of all distribution values
    #   A X B X C
    results = [dict([("Score",
                     heuristic_function(state,dict(zip(rownames,params))
                                   ,reps=state.get("reps",5),f=state.get("f",None)))] + list(zip(rownames,params))) \
               for params in product]    
    return results


def example_search():
    import math
    state = {"reps":1}
    params = {"a":[1,2,4,5], "b":[2,5,9,0.1], "c":[0.01,0.02,0.3,0.5], "d":range(1,10) }
    f = lambda state,params: math.log(params["c"]/params["d"]) + params["a"] / params["b"] + math.sin(params["a"])
    state["f"] = f
    random_results = random_search(state,params,heuristic_function)
    random_results   = sorted(random_results, key=lambda x: x['Score'])
    print(random_results[-1])
    grid_results   = grid_search(state, params, heuristic_function)
    grid_results   = sorted(grid_results, key=lambda x: x['Score'])
    print(grid_results[-1])

if __name__ == "__main__":
    example_search()
