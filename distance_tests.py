import numpy as np
from scipy.spatial.distance import euclidean, cityblock
from sklearn import metrics


def generateClusters(number, features):
    numberOfElements = number * features
    cluster1 = np.random.uniform(0, 5, numberOfElements).reshape((-1, features))
    cluster2 = np.random.uniform(5, 10, numberOfElements).reshape((-1, features))
    cluster3 = np.random.uniform(10, 15, numberOfElements).reshape((-1, features))
    cluster4 = np.random.uniform(15, 20, numberOfElements).reshape((-1, features))
    cluster5 = np.random.uniform(20, 25, numberOfElements).reshape((-1, features))
    cluster6 = np.random.uniform(25, 30, numberOfElements).reshape((-1, features))

    return np.concatenate((cluster1, cluster2, cluster3, cluster4, cluster5, cluster6))


def lk_norm(x, y, k):
    return (sum([np.math.fabs(x_i - y_i) ** k for x_i, y_i in zip(x, y)])) ** (1/k)


def kmeans(objs, clusters_number, k, centers=None):
    def split_clusters(objs, clusters_number, centers, k):
        clusters = [[] for i in range(clusters_number)]
        obj_distrib = [-1] * len(objs)
        for i, obj in enumerate(objs):
            distances = [lk_norm(center, obj, k) for center in centers]
            clust_number = np.argmin(distances)
            clusters[clust_number].append(obj)
            obj_distrib[i] = clust_number
        return clusters, obj_distrib

    dimension = len(objs[0])
    obj_distrib = None
    if centers is None:
        # avoid the situation when any cluster is empty
        while True:
            centers = np.random.uniform(0, 15, clusters_number * dimension).reshape((-1, dimension))
            clusters, obj_distrib = split_clusters(objs, clusters_number, centers, k)
            if all([len(cluster) > 0 for cluster in clusters]):
                break

    while True:
        # cluster and centers are already initialized
        new_centers = [np.mean(cluster, axis=0) for cluster in clusters]
        if np.equal(centers, new_centers).all():
            return centers, clusters, obj_distrib
        centers = new_centers
        clusters, obj_distrib = split_clusters(objs, clusters_number, centers, k)


def do_test(elems_number, features, k):
    def compute_error(clusters, centres):
        error = 0
        for index, result_center in enumerate(centres):
            # print(result_center)
            for obj in clusters[index]:
                error += euclidean(obj, result_center)

        return error

    objs = generateClusters(elems_number, features)
    result_centers, result_clusters, obj_distrib = kmeans(objs, 6, k)
    error = compute_error(result_clusters, result_centers)
    metric = metrics.calinski_harabaz_score(objs, obj_distrib)
    print("k-value: " + str(k) + ". Features: " + str(features) + ". Error: " + str(error)
          + ". Calinski&Harabaz score: " + str(metric))


if __name__ == '__main__':
    # for i in range(15, 20):
    do_test(700, 20, 0.3)
    do_test(700, 20, 2)
    print()

