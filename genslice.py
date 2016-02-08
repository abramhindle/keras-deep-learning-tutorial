# purpose: make a slight difference of circles be the dataset to learn.
import numpy as np
gY, gX = np.meshgrid(np.arange(1000)/1000.0,np.arange(1000)/1000.0)

def intersect_circle(cx,cy,radius):
    equation = (gX - float(cx)) ** 2 + (gY - float(cy)) ** 2
    matches = equation < (radius**3) 
    return matches

# rad = 0.1643167672515498
rad = 0.3
x = intersect_circle(0.5,0.5,rad) ^ intersect_circle(0.51,0.51,rad)

def plotit(x):
    import matplotlib.pyplot as plt
    plt.imshow(x)
    plt.savefig('slice.png')
    plt.imshow(x)
    plt.savefig('slice.pdf')
    plt.show()

# plotit(x)

def mkcol(x):
    return x.reshape((x.shape[0]*x.shape[1],1))

# make the data set
big = np.concatenate((mkcol(gX),mkcol(gY),mkcol(1*x)),axis=1)
np.savetxt("big-slice.csv", big, delimiter=",")

# make a 50/50 data set
nots = big[big[0:,2]==0.0,]
np.random.shuffle(nots)
nots = nots[0:1000,]
trues = big[big[0:,2]==1.0,]
np.random.shuffle(trues)
trues = trues[0:1000,]
small = np.concatenate((trues,nots))
np.savetxt("small-slice.csv", small, delimiter=",")

