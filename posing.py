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



ln = lognormal(size=100)
