import numpy as np
from scipy.spatial.distance import euclidean, cityblock


def generateClusters(number, features):
    numberOfElements = number * features
    cluster1 = np.random.uniform(0, 5, numberOfElements).reshape((-1, features))
    cluster2 = np.random.uniform(5, 10, numberOfElements).reshape((-1, features))
    cluster3 = np.random.uniform(10, 15, numberOfElements).reshape((-1, features))
    cluster4 = np.random.uniform(15, 20, numberOfElements).reshape((-1, features))
    cluster5 = np.random.uniform(20, 25, numberOfElements).reshape((-1, features))
    cluster6 = np.random.uniform(25, 30, numberOfElements).reshape((-1, features))

    return np.concatenate((cluster1, cluster2, cluster3, cluster4, cluster5, cluster6))


def kmeans(objs, clusters_number, centers=None, distance_fun=cityblock):
    dimension = len(objs[0])
    if centers is None:
        # avoid the situation when any cluster is empty
        while True:
            centers = np.random.uniform(0, 15, clusters_number * dimension).reshape((-1, dimension))
            clusters = split_clusters(objs, clusters_number, centers, distance_fun=distance_fun)
            if all([len(cluster) > 0 for cluster in clusters]):
                break

    while True:
        # cluster and centers are already initialized
        new_centers = [np.mean(cluster, axis=0) for cluster in clusters]
        if np.equal(centers, new_centers).all():
            return centers, clusters
        centers = new_centers
        clusters = split_clusters(objs, clusters_number, centers, distance_fun=distance_fun)


def split_clusters(objs, clusters_number, centers, distance_fun=cityblock):
    clusters = [[] for i in range(clusters_number)]
    for obj in objs:
        distances = [distance_fun(center, obj) for center in centers]
        clusters[np.argmin(distances)].append(obj)
    return clusters


def compute_error(clusters, centres):
    error = 0
    for index, result_center in enumerate(centres):
        # print(result_center)
        for obj in clusters[index]:
            error += euclidean(obj, result_center)

    return error


def do_test(elems_number, features, distance_func):
    objs = generateClusters(elems_number, features)
    result_centers, result_clusters = kmeans(objs, 6, None, euclidean)
    error = compute_error(result_clusters, result_centers)
    print("Distance: " + distance_func.__name__ + ". Features: " + str(features) + ". Error: " + str(error))


if __name__ == '__main__':
    # for i in range(15, 20):
    do_test(5000, 20, cityblock)
    do_test(5000, 20, euclidean)
    print()