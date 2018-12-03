import numpy as np
cimport numpy as np
from sparsefht import fht
from numpy import random
from scipy import sparse
import math
from numpy.linalg import norm



def Fast_JL_Transform(A, k):
    return _Fast_JL_Transform(A,k)


cdef np.ndarray[np.float64_t, ndim = 2] _Fast_JL_Transform(np.ndarray[np.float64_t, ndim=2] A, np.int32_t k):
    cdef np.int32_t d,n,sample_size, d_act
    cdef np.float64_t q,sc_ft
    cdef np.ndarray[np.float64_t, ndim=1] p_data
    cdef np.ndarray[np.float64_t, ndim=2] DA, hda, result, D
    cdef np.ndarray[np.int32_t, ndim=1] indc,p_rows,p_cols
    # 没有解决scipy.csr 格式问题
    d = A.shape[0]
    n = A.shape[1]
    q = np.min([1.0, np.log(d) ** 2 / n])

    d_act = nextPow(d)
    sc_ft = np.sqrt(d_act / float(d * k))
    D = random.randint(0, 2, size=(d, 1)) * 2 * sc_ft - sc_ft
    DA = np.zeros((d_act, n))
    DA[0:d, :] = A * D
    hda = np.apply_along_axis(fht,0,DA)/np.sqrt(d_act)

    sample_size = random.binomial(k * d, q)
    indc = fast_sample(k * d, sample_size)

    #temp 数据格式没定
    temp = np.unravel_index(indc, (k, d))
    p_rows = np.array(temp[0],dtype=np.int32)
    p_cols = np.array(temp[1],dtype=np.int32)
    p_data = np.random.normal(loc=0, scale=math.sqrt(1 / q), size=len(p_rows))
    P = sparse.csr_matrix((p_data, (p_rows, p_cols)), shape=(k, d_act))
    result = P.dot(hda)

    return result



def fast_sample(n,sample_size):
    #if not (type(n) != np.int32 or type(sample_size) != np.int32):
    #    raise TypeError('Input n and sample_size must be int32 or int64!')
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

def nextPow(np.int32_t d_act):
    return _nextPow(d_act)


cdef np.int32_t _nextPow(np.int32_t d_act):
    # return the nearest power 2 number
    d_act = d_act - 1
    d_act |= d_act >> 1
    d_act |= d_act >> 2
    d_act |= d_act >> 4
    d_act |= d_act >> 8
    d_act |= d_act >> 16
    d_act += 1
    return d_act