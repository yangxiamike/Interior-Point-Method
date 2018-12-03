import numpy as np
from numpy.linalg import norm
from scipy._lib._util import check_random_state
def cwt_matrix(n_rows, n_columns, seed = None):
    S = np.zeros((n_rows,n_columns))
    nz_positions = np.random.randint(0,n_rows,n_columns)
    rng = check_random_state(seed)
    values = rng.choice([1,-1],n_columns)
    for i in range(n_columns):
        S[nz_positions[i]][i] =values[i]

    return S

def clarkson_woodruff_transform(input_matrix, sketch_size, seed=None):
    m,n = input_matrix.shape
    S = cwt_matrix(sketch_size,m,seed)
    return np.dot(S,input_matrix)



if __name__ == '__main__':
    k, n,d = 2000,10000,10
    X = np.random.random((n,d))
    F = clarkson_woodruff_transform(X,k)
    print(norm(F))
    print(norm(X))