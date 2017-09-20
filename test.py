from functools import partial

from kmeans_3 import *
from ordinary_kmeans import *


def lk_norm(x, y, k):
    return (sum([np.math.fabs(x_i - y_i) ** k for x_i, y_i in zip(x, y)])) ** (1 / k)


features = 20
numberOfElements = 1000

cluster1 = np.random.uniform(0, 1, numberOfElements * features).reshape((-1, features))
cluster2 = np.random.uniform(1, 2, numberOfElements * features).reshape((-1, features))
cluster3 = np.random.uniform(2, 3, numberOfElements * features).reshape((-1, features))
cluster4 = np.random.uniform(3, 4, numberOfElements * features).reshape((-1, features))
cluster5 = np.random.uniform(4, 5, numberOfElements * features).reshape((-1, features))
cluster6 = np.random.uniform(5, 6, numberOfElements * features).reshape((-1, features))
answers = [0] * numberOfElements + [1] * numberOfElements + [2] * numberOfElements + [3] * numberOfElements + [4] * numberOfElements + [5] * numberOfElements
objs = np.concatenate((cluster1, cluster2, cluster3, cluster4, cluster5, cluster6))

# objs = np.concatenate((cluster1, cluster2, cluster3))
for lk in [0.3, 0.5, 1, 2, 3]:
    partial_lk_norm = partial(lk_norm, k=lk)
    scores = []
    for i in range(10):
        labels, cluters = kmeans(objs, 6, partial_lk_norm, max_iter=50)
        scores.append(adjusted_mutual_info_score(answers, labels))
    print("k=" + str(lk) + ", score=" + str(np.median(scores)))

# cmap = plt.get_cmap('viridis')
# colors = cmap(np.linspace(0, 1, len(cluters)))
# for i, clust in enumerate(cluters):
#     plt.plot(np.array(clust.elements)[:, 0:1], np.array(clust.elements)[:, 1:2], c=colors[i], marker="+", linestyle="None")
#
# plt.show()
