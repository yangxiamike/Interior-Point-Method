from sparsefht import fht
import numpy as np
from numpy import random
from scipy import sparse
import math
from numpy.linalg import norm


def Fast_JL_Transform(A, k):
    """
    Implement fast JL transform on matrix along rows
    :param A: 2-D matrix or 1-D array
    :param k: target dimension of rows
    :return: 2-D matrix or 1-D array after sampling
    """
    (d, n) = A.shape
    q = np.min([1, np.log(d) ** 2 / n])

    d_act = nextPow(d)
    sc_ft = np.sqrt(d_act / float(d * k))
    D = random.randint(0, 2, size=(d, 1)) * 2 * sc_ft - sc_ft
    DA = np.zeros((d_act, n))
    DA[0:d, :] = A * D
    print(D.shape)
    print(A.shape)
    print((A*D).shape)
    hda = np.apply_along_axis(fht,0,DA)/np.sqrt(d_act)

    sample_size = random.binomial(k * d, q)
    indc = fast_sample(k * d, sample_size)
    temp = np.unravel_index(indc, (k, d))
    p_rows, p_cols = np.unravel_index(indc, (k, d))
    print(temp[0].ndim)
    p_data = np.random.normal(loc=0, scale=math.sqrt(1 / q), size=len(p_rows))
    P = sparse.csr_matrix((p_data, (p_rows, p_cols)), shape=(k, d_act))
    result = P.dot(hda)

    return result



def fast_sample(n, sample_size):
    swap_records = {}
    sample_wor = np.empty(sample_size, dtype=int)
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

def nextPow(d_act):
    # return the nearest power 2 number
    d_act = d_act - 1
    d_act |= d_act >> 1
    d_act |= d_act >> 2
    d_act |= d_act >> 4
    d_act |= d_act >> 8
    d_act |= d_act >> 16
    d_act += 1
    return d_act

if __name__ == '__main__':

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