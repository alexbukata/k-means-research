import random
from math import sqrt
import numpy as np
import re

from sklearn import metrics
from sklearn.metrics.cluster import  adjusted_mutual_info_score


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

    def write(self, f):
        pass
        #f.write(" ".join(("Cluster",str(self.n),"\n")))

        #for i in self.elements:
            #f.write(" ".join((str(i),"\n")))

        #f.write(" ".join(("Center -", str(self.f_m), "\n\n")))

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
        #q = sqrt(round(2*(self.x_m - norm(self.f_m)**2) + (1- norm(self.f_a)**2), 6))
        print("Cluster N",self.n-1, "q =",q)
        return q


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
    #f.write("START stage 1\n")
    k = 1
    C = 1
    mu_m = data[0,:]
    x_m = norm(data[0,:])**2
    mu_a = data[0,:]/norm(data[0,:])
    x_a = 1 #norm(mu_a)**2 = 1 ^_^

    lambda_a = 0
    lambda_m = 0

    #f.write(' '.join((str(k),str(C),str(mu_m), str(x_m), str(mu_a), str(x_a),"\n")))

    clusters = []
    n = data.shape[0]
    #n = 100
    #f.write("\n\nSTART stage 2\n")
    for index in range(n):

        #plt.scatter(data[1:,0], data[1:,1])
        #f.write("\nADD NEW POINT\n\n")
        print("Index ",index,"/",n)
        x = data[index, :]
        if len(clusters) == 0:
            clusters.append(Cluster(x,1))
            #f.write(' '.join(("Point",str(x),"added to new cluster 1","\n")))
        else:
            #stage 2
            mu_m = k/(k+1)*mu_m + 1/(k+1)*x
            x_m =  k/(k+1)*x_m + 1/(k+1)*norm(x)**2
            mu_a =  k/(k+1)*mu_a + 1/(k+1)*x/norm(x)
            k = k + 1

            #f.write(' '.join(("Global center is changed to", str(mu_m),"\n")))
            #MAIN

            lambda_a = 1/sqrt(1-norm(mu_a)**2)
            lambda_m = 1/sqrt(2*(x_m - norm(mu_m)**2))

            find_max = []
            find_min = []

            for i in range(C):
                find_max.append(distance(clusters[i].f_m, mu_m, lambda_a, lambda_m))
                find_min.append(distance(clusters[i].f_m, mu_m, lambda_a, lambda_m))

            d = distance(x, mu_m, lambda_a, lambda_m)
            #f.write("IF\n")
            #f.write(" ".join(("max between center of clusters and global center =",str(max(find_max)),"\n")))
            #f.write(" ".join(("min between center of clusters and global center =",str(min(find_min)),"\n")))
            #f.write(" ".join(("distance between point and global center =",str(d),"\n")))
            if d > max(find_max) or d < min(find_min): # new cluster
                clusters.append(Cluster(x, C+1))
                C += 1



                #f.write(' '.join(("Point",str(x),"added to new cluster", str(C),"\n")))

            else: #add it to nearest center

                nearest_center = 0

                for i in range(C):
                    if distance(x, clusters[i].f_m, lambda_a, lambda_m) < distance(x, clusters[nearest_center].f_m, lambda_a, lambda_m):
                        nearest_center = i


                clusters[nearest_center].update_parametrs(x)

                #f.write(' '.join(("Point",str(x),"added to cluster", str(nearest_center+1), "\n")))
                #f.write(' '.join(("Center of cluster",str(nearest_center+1), "is changed to", str(clusters[nearest_center].f_m),"\n")))



    print("Number of clusters = ", C)
    # w = open("clusters.txt",'w')
    # for i in range(C):
    #     clusters[i].write(w)
    # w.close()
    #f.write(" ".join(("Number of clusters =", str(C),"\n")))
    #f.write("\n\nSTART stage 3\n")
    lambda_a = 1/sqrt(1-norm(mu_a)**2)
    lambda_m = 1/sqrt(2*(x_m - norm(mu_m)**2))
    #stage 3

    constanta = 0
    count = 0

    # for i in range(C):
    #     t = clusters[i].calculate_q(lambda_a, lambda_m)    #статья
    #     constanta += t
    #
    # constanta = constanta/C

    constanta = 0
    for i in range(C-1):              #среднее квадр расстояние между центрами кластеров
        for j in range(i+1, C):
            constanta += distance(clusters[i].f_m, clusters[j].f_m, lambda_a, lambda_m)**2

    constanta = sqrt(constanta)/C

    print("Constanta = ", constanta)

    for i in range(C):
        for j in range(C):
            #print(distance(clusters[i].f_m, clusters[j].f_m, lambda_a, lambda_m))
            if i != j:
                print("Distance between ", i, " and ", j, " = ", distance(clusters[i].f_m, clusters[j].f_m, lambda_a, lambda_m))

                if distance(clusters[i].f_m, clusters[j].f_m, lambda_a, lambda_m) < constanta:
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
            print("i = ",i, "D = ", clusters[i].D)
            local_max_f.append(clusters[i].f_m)


    print("Number of clusters = ", len(local_max_f))
    w = open("final_clusters.txt", 'w')
    for index in range(n):

        x = data[index, :]
        nearest_center = 0
        for i in range(len(local_max_f)):
            if distance(x, local_max_f[i], lambda_a, lambda_m) < distance(x, local_max_f[nearest_center], lambda_a, lambda_m):
                nearest_center = i

        labels.append(nearest_center)
        w.write(" ".join((str(x),str(nearest_center), "\n")))

    w.close()


    return labels

f = open("datasets/dim256.txt", 'r')
lines = f.readlines()
f.close()

p = re.compile(r"\d+")
n = len(lines)
m = len(p.findall(lines[0]))
print(n,m)
data = np.zeros((n,m), dtype=np.int64)

for i in range(n):
    x = p.findall(lines[i])
    for j in range(m):
        data[i,j] = int(x[j])


# fig = plt.figure()
# plt.plot(data[:200,0], data[:200,1], 'o')
# plt.savefig('data.png', fmt='png')
#plt.show()
f = open("debug.log", 'w')
labels = new_k_means(data[:100,:])
print(labels)
f.close()

f = open("datasets/dim256.pa",'r')
true_labels = [int(x) for x in f.readlines()]
print(true_labels )
f.close()

print(adjusted_mutual_info_score(true_labels[:100], labels))





