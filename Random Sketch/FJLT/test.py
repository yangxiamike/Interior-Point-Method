from fjlt import Fast_JL_Transform as f
import numpy as np
from utils import Fast_JL_Transform


#cython implementation
a = np.random.random((100,100))
f(a,10)



#python implementation
y=np.random.random((100000,1000))
transform =Fast_JL_Transform(y,2000)

print(norm(transform))
print(norm(y))

from time import clock

"""
t1 = clock()
y.T.dot(y)
print(clock()-t1)
"""

t2 = clock()
transform.T.dot(transform)
print(clock()-t2)