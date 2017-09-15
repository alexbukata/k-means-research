from math import sqrt
import numpy as np
import re


def eucl_dist(a, b):
    s = 0
    for i in range(len(a)):
        s += (a[i] - b[i])**2

    return sqrt(s)

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


def distance(a, b, lambda_a, lambda_m):
    return sqrt(lambda_m**2*(cos_dist(a, b)**2)+ lambda_a**2*(eucl_dist(a, b)**2))


def calculate_start_lambda_a(data):

    n = data.shape[0]
    s = 0
    for i in range(n):
        for j in range(n):
            s += cos_dist(data[i,:], data[j,:])**2


    return 1/sqrt(s/(n**2))


def calculate_start_lambda_m(data):

    n = data.shape[0]
    s = 0
    for i in range(n):
        for j in range(n):
            s += eucl_dist(data[i,:], data[j,:])**2


    return 1/sqrt(s/(n**2))




class Cluster:

    def __init__(self, x, n):
        self.elements = []
        self.elements.append(x)
        self.f_m = x
        self.x_m = norm(x)**2
        self.f_a = x/norm(x)
        self.x_a = 1
        self.s = 1
        self.n = n
        self.D = -1
        self.neighbors = []


    def update_parametrs(self, x):
        self.elements.append(x)
        self.f_m = self.s/(self.s+1)*self.f_m + 1/(self.s+1)*x
        self.x_m = self.s/(self.s+1)*self.x_m + 1/(self.s+1)*norm(x)**2
        self.f_a = self.s/(self.s+1)*self.f_a + 1/(self.s+1)*x/norm(x)
        self.s += 1

    def calculate_q(self, lambda_a, lambda_m):

        s = 0
        for i in range(self.s):
            for j in range(self.s):
                s += distance(self.elements[i], self.elements[j],lambda_a, lambda_m)**2

        q = sqrt(s/(self.s**2))
        print("Cluster N",self.n-1, "q =",q)
        return q
        #return 2*(self.x_m - norm(self.f_m)**2) + (1- norm(self.f_a)**2)

    def calculate_d(self, clusters, lambda_a, lambda_m):

        C = len(clusters)

        s = 0
        for i in range(C):
            for j in range(C):
                s += distance(clusters[i].f_m, clusters[j].f_m, lambda_a, lambda_m)**2

        z = 0

        for j in range(C):
            z += distance(self.f_m, clusters[j].f_m, lambda_a, lambda_m)**2

        self.D = self.s*s/(2*C*z)
        print("Cluster N", self.n-1, " D = ", self.D)



#new k means


def new_k_means(data):

    labels = []
    #stage 1
    #global
    k = 1
    C = 1
    mu_m = data[0,:]
    x_m = norm(data[0,:])**2
    mu_a = data[0,:]/norm(data[0,:])
    x_a = 1 #norm(mu_a)**2 = 1 ^_^

    lambda_a = 0
    lambda_m = 0

    #print(k,C,mu_m, x_m, mu_a, x_a)

    clusters = []
    n = data.shape[0]
    #n = 100
    for index in range(n):
        print("Index ",index,"/",n)
        x = data[index, :]
        if len(clusters) == 0:
            clusters.append(Cluster(x,1))
        else:
            #stage 2
            mu_m = k/(k+1)*mu_m + 1/(k+1)*x
            x_m =  k/(k+1)*x_m + 1/(k+1)*norm(x)**2
            mu_a =  k/(k+1)*mu_a + 1/(k+1)*x/norm(x)
            k = k + 1

            #MAIN

            lambda_a = 1/sqrt(1-norm(mu_a)**2)
            lambda_m = 1/sqrt(2*(x_m - norm(mu_m)**2))

            find_max = []
            find_min = []

            for i in range(C):
                find_max.append(distance(clusters[i].f_m, mu_m, lambda_a, lambda_m))
                find_min.append(distance(clusters[i].f_m, mu_m, lambda_a, lambda_m))

            d = distance(x, mu_m, lambda_a, lambda_m)
            if d > max(find_max) or d < min(find_min): # new cluster
                clusters.append(Cluster(x, C+1))
                C += 1

            else: #add it to nearest center

                nearest_center = 0

                for i in range(C):
                    if distance(x, clusters[i].f_m, lambda_a, lambda_m) < distance(x, clusters[nearest_center].f_m, lambda_a, lambda_m):
                        nearest_center = i


                clusters[nearest_center].update_parametrs(x)


    print("Number of clusters = ", C)
    lambda_a = 1/sqrt(1-norm(mu_a)**2)
    lambda_m = 1/sqrt(2*(x_m - norm(mu_m)**2))
    #stage 3

    constanta = 0
    count = 0

    for i in range(C):
        t = clusters[i].calculate_q(lambda_a, lambda_m)
        if t != 0:
            count += 1
        constanta += t

    constanta = constanta/count

    # constanta = 0
    # for i in range(C-1):
    #     for j in range(i+1, C):
    #         constanta += distance(clusters[i].f_m, clusters[j].f_m, lambda_a, lambda_m)**2
    #
    # constanta = sqrt(constanta)/C

    print("Constanta = ", constanta)

    for i in range(C):
        for j in range(C):
            if i != j:
                print("Distance between ", i, " and ", j, " = ", distance(clusters[i].f_m, clusters[j].f_m, lambda_a, lambda_m))
                if distance(clusters[i].f_m, clusters[j].f_m, lambda_a, lambda_m) > constanta:
                    clusters[i].neighbors.append(j)

        print("Number of neighbors for",i, "=",len(clusters[i].neighbors))


    local_max_f = []

    for i in range(C):
        clusters[i].calculate_d(clusters, lambda_a, lambda_m)

    print("#####")
    for i in range(C):
        max_d = 0

        for j in range(len(clusters[i].neighbors)):
            if clusters[clusters[i].neighbors[j]].D > max_d:
                max_d = clusters[clusters[i].neighbors[j]].D

        if clusters[i].D > max_d:
            print("i = ",i, "max_d = ", max_d)
            local_max_f.append(clusters[i].f_m)


    print("Number of clusters = ", len(local_max_f))
    for index in range(n):

        x = data[index, :]
        nearest_center = 0
        for i in range(len(local_max_f)):
            if distance(x, local_max_f[i], lambda_a, lambda_m) < distance(x, local_max_f[nearest_center], lambda_a, lambda_m):
                nearest_center = i

        labels.append(nearest_center)

    return labels

f = open("dim128.txt", 'r')
lines = f.readlines()
f.close()

p = re.compile(r"\d+")
n = len(lines)
m = len(p.findall(lines[0]))
data = np.zeros((n,m), dtype=np.int64)

for i in range(n):
    x = p.findall(lines[i])
    for j in range(m):
        data[i,j] = int(x[j])

#from sklearn import preprocessing
# normalize the data attributes
#data = preprocessing.normalize(data)
print(new_k_means(data))