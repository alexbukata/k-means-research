import random

import numpy as np
from scipy.spatial.distance import euclidean
from sklearn import metrics


def generateClusters(number, features):
    numberOfElements = number * features
    cluster1 = np.random.uniform(0, 5, numberOfElements).reshape((-1, features))
    cluster2 = np.random.uniform(50, 100, numberOfElements).reshape((-1, features))
    cluster3 = np.random.uniform(200, 250, numberOfElements).reshape((-1, features))
    cluster4 = np.random.uniform(400, 450, numberOfElements).reshape((-1, features))
    cluster5 = np.random.uniform(600, 650, numberOfElements).reshape((-1, features))
    cluster6 = np.random.uniform(1000, 1050, numberOfElements).reshape((-1, features))

    return np.concatenate((cluster1, cluster2, cluster3, cluster4, cluster5, cluster6))


def lk_norm(x, y, k):
    return (sum([np.math.fabs(x_i - y_i) ** k for x_i, y_i in zip(x, y)])) ** (1 / k)


def split_clusters(objs, clusters_number, centers, k):
    clusters = [[] for i in range(clusters_number)]
    obj_distrib = [-1] * len(objs)
    for i, obj in enumerate(objs):
        distances = [lk_norm(center, obj, k) for center in centers]
        clust_number = np.argmin(distances)
        clusters[clust_number].append(obj)
        obj_distrib[i] = clust_number
    return clusters, obj_distrib


def generate_centers(objs, clusters_number, k, method='kmeans++'):
    if method == 'random':
        return random.sample(objs, clusters_number)
    elif method == 'kmeans++':
        init_center = random.choice(objs)
        # init_distances = list(map(lambda obj: lk_norm(obj, init_center, k), objs))
        centers = [init_center]
        while len(centers) < clusters_number:
            distances = np.array([min([lk_norm(obj, center, k) for center in centers]) for obj in objs])
            sum_distances = np.sum(distances)
            new_center_index = np.random.choice(range(len(objs)), 1, p=distances / sum_distances)[0]
            centers.append(objs[new_center_index])
        return centers


def kmeans(objs, clusters_number, k, centers=None):
    if centers is None:
        # avoid the situation when any cluster is empty
        while True:
            centers = generate_centers(objs, clusters_number, k, method='kmeans++')
            # centers = np.random.uniform(np.min(objs), np.max(objs), clusters_number * dimension) \
            #     .reshape((-1, dimension))
            clusters, obj_distrib = split_clusters(objs, clusters_number, centers, k)
            print("Finding clusters")
            if all([len(cluster) > 0 for cluster in clusters]):
                break
    else:
        clusters, obj_distrib = split_clusters(objs, clusters_number, centers, k)

    while True:
        # cluster and centers are already initialized
        print("improving centers")
        new_centers = [np.mean(cluster, axis=0) for cluster in clusters]
        print([lk_norm(new_c, old_c, k) for new_c, old_c in zip(new_centers, centers)])
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

    # sk_kmeans = KMeans(n_clusters=6, init="random").fit(objs)
    # sk_result_clusters, sk_obj_distrib = split_clusters(objs, 6, sk_kmeans.cluster_centers_, k)
    # sk_error = compute_error(sk_result_clusters, sk_kmeans.cluster_centers_)
    # sk_metric = metrics.calinski_harabaz_score(objs, sk_obj_distrib)
    # print("sk learn: euclidian. Features: " + str(features) + ". n_iter=" + str(sk_kmeans.n_iter_) + ". Error: " +
    #       str(sk_error) + ". Calinski&Harabaz score: " + str(sk_metric))

    result_centers, result_clusters, obj_distrib = kmeans(objs, 6, k)
    error = compute_error(result_clusters, result_centers)
    metric = metrics.calinski_harabaz_score(objs, obj_distrib)
    print("k-value:" + str(k) + ". Features: " + str(features) + ". Error: " + str(error)
          + ". Calinski&Harabaz score: " + str(metric))


if __name__ == '__main__':
    # for i in range(15, 20):
    # do_test(700, 20, 0.3)
    do_test(500, 20, 0.3)
    print()
