from distance_tests import *

ds = [1, 2, 3, 4, 10, 15, 20, 100]
for d in ds:
    probs = 0
    for i in range(1000):
        objs = np.random.uniform(0, 1, 10 * d).reshape((-1, d))
        zero = np.zeros(d)
        distances_manh = [lk_norm(zero, obj, 1) for obj in objs]
        distances_eucl = [lk_norm(zero, obj, 2) for obj in objs]
        Dmax_manh = np.max(distances_manh)
        Dmin_manh = np.min(distances_manh)
        Dmax_eucl = np.max(distances_eucl)
        Dmin_eucl = np.min(distances_eucl)
        value_manh = (Dmax_manh - Dmin_manh) / Dmin_manh
        value_eucl = (Dmax_eucl - Dmin_eucl) / Dmin_eucl
        if value_eucl < value_manh:
            probs += 1
    print(probs / 1000.0)