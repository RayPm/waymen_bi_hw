import numpy as np
from operator import itemgetter


a = [0, 1, 2, 3]

b = np.random.rand(len(a))
c = zip(a, b)
print(list(zip(a, b)))
d = max(c, key=itemgetter(1))[0]
print(d)