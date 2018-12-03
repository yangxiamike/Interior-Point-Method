import numpy as np
cimport numpy as np
import numpy as np
cimport numpy as np
from sparsefht import fht
from numpy import random
from scipy import sparse
import math
from numpy.linalg import norm


def fast_sample(n,sample_size):
    if not (type(n) != np.int32 or type(sample_size) != np.int32):
        raise TypeError('Input n and sample_size must be int32 or int64!')
    return _fast_sample(n, sample_size)

cdef np.ndarray[np.int32_t, ndim = 1] _fast_sample(np.int32_t n, np.int32_t sample_size):
    swap_records = {}
    cdef np.ndarray[np.int32_t, ndim =1] sample_wor = np.empty(sample_size, dtype=np.int32)
    cdef np.int32_t i, rand_ix
    # 没有定义el1, el2
    for i in range(sample_size):
        rand_ix = np.random.randint(i, n)

        if i in swap_records:
            el1 = swap_records[i]
        else:
            el1 = i

        if rand_ix in swap_records:
            el2 = swap_records[rand_ix]
        else:
            el2 = rand_ix

        swap_records[rand_ix] = el1

        sample_wor[i] = el2
        if i in swap_records:
            del swap_records[i]
    return sample_wor

def nextPow(long d_act):
    return _nextPow(d_act)

cdef long _nextPow(long d_act):
    # return the nearest power 2 number
    d_act = d_act - 1
    d_act |= d_act >> 1
    d_act |= d_act >> 2
    d_act |= d_act >> 4
    d_act |= d_act >> 8
    d_act |= d_act >> 16
    d_act += 1
    return d_act