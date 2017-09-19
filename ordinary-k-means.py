import numpy as np
from math import sqrt

import re

from sklearn.metrics import adjusted_mutual_info_score
def eucl_dist(a, b):
    s = 0
    for i in range(len(a)):
        s += (a[i] - b[i])**2

    return sqrt(s)

def eucl_dist1(x, y):
    return (sum([np.math.fabs(x_i - y_i) ** k for x_i, y_i in zip(x, y)])) ** (1 / k)


def norm(a):
    n = 0
    for i in range(len(a)):
        n += a[i]*a[i]

    return sqrt(n)

def cos_dist(a, b):

    s = 0
    a1 = 0
    b1 = 0
    for i in range(len(a)):
        s += a[i]*b[i]
        a1 += a[i]*a[i]
        b1 += b[i]*b[i]

    a1 = sqrt(a1)
    b1 = sqrt(b1)

    return sqrt(1.0-round(s/(a1*b1), 5))


def distance(a, b):
    return sqrt(lambda_m**2*(cos_dist(a, b)**2)+ lambda_a**2*(eucl_dist(a, b)**2))

def distance1(a, b):
    return sqrt(lambda_m**2*(cos_dist(a, b)**2)+ lambda_a**2*(eucl_dist1(a, b)**2))


def calculate_start_lambda(data):

    n = data.shape[0]
    s = 0
    s1 = 0
    for i in range(n):
        print(i)
        for j in range(i,n):
            s += cos_dist(data[i,:], data[j,:])**2
            s1 += eucl_dist(data[i,:], data[j,:])**2


    return 1/sqrt((s*2)/(n**2)), 1/sqrt((s1*2)/(n**2))


def calculate_start_lambda_m(data):

    n = data.shape[0]
    s = 0
    for i in range(n):
        for j in range(n):
            s += eucl_dist(data[i,:], data[j,:])**2


    return 1/sqrt(s/(n**2))

class Cluster:

    def __init__(self, center):
        self.elements = []
        self.center = center
        self.changed = True

    def addElement(self,x):
        self.elements.append(x)

    def recalculate_center(self):
        if len(self.elements) == 0:
            #print("bu")
            self.changed = False
            return
        self.changed = True
        center = np.zeros(self.elements[0].shape[0])
        for i in range(len(self.elements)):
            center += self.elements[i]

        center /= len(self.elements)

        if np.equal(center, self.center).all():
            self.changed = False
        self.center = center


def kmeans(data, k, distance_function):

    cluters = []
    labels = np.zeros(data.shape[0])
    c = np.random.choice(data.shape[0], k, replace=False)
    for i in range(k):
        cluters.append(Cluster(data[c[i],:]))
        #print(cluters[i].center)

    while True:
        labels = np.zeros(data.shape[0])
        for i in range(k):
            cluters[i].elements = []

        for i in range(data.shape[0]):
            min_dist = []
            for j in range(k):
                min_dist.append(distance_function(data[i,:],cluters[j].center))

            index = min_dist.index(min(min_dist))
            cluters[index].elements.append(data[i,:])
            labels[i] = index

        for i in range(k):
            cluters[i].recalculate_center()

        e = False
        for i in range(k):
            if cluters[i].changed == True:
                e = True
                break

        if not e:
            break

    return labels


f = open("dim256.txt", 'r')
lines = f.readlines()
f.close()

p = re.compile(r"\d+")
n = len(lines)
m = len(p.findall(lines[0]))
print(n,m)
data = np.zeros((n,m), dtype=np.float64)

for i in range(n):
    x = p.findall(lines[i])
    for j in range(m):
        data[i,j] = int(x[j])

#lambda_a, lambda_m = calculate_start_lambda(data)
lambda_a = 2
lambda_m = 0.03
print(lambda_a, lambda_m)
print("Starting k means with new dist 1 1")
labels = kmeans(data, 16, distance)
#print(labels)
f = open("dim256.pa.txt",'r')
true_labels = [int(x) for x in f.readlines()]
#print(true_labels )
f.close()

print(adjusted_mutual_info_score(true_labels, labels))
k = 0.3
print("Starting k means with eucl_dist1 k = 0.3")
labels = kmeans(data, 16, eucl_dist1)
#print(labels)

print(adjusted_mutual_info_score(true_labels, labels))

print("Starting k means with eucl_dist1 k = 2")
labels = kmeans(data, 16, eucl_dist)
#print(labels)

print(adjusted_mutual_info_score(true_labels, labels))

print("Starting k means with cos")
labels = kmeans(data, 16, cos_dist)
#print(labels)

print(adjusted_mutual_info_score(true_labels, labels))

lambda_a = 2
lambda_m = 0.02
print(lambda_a, lambda_m)
print("Starting k means with new dist 1 1 k = 0.3")
labels = kmeans(data, 16, distance1)
#print(labels)
print(adjusted_mutual_info_score(true_labels, labels))