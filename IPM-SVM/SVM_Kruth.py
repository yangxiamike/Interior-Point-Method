from IPM import IPM
from sklearn.utils import Memory
from sklearn.datasets import load_svmlight_file
import os
import numpy as np
from scipy.sparse import coo_matrix,block_diag,eye,dia_matrix,csc_matrix,hstack,vstack
from scipy.sparse.linalg import spsolve
from numpy.linalg import norm
mem = Memory('./cache')

from IPM import IPM
from sklearn.utils import Memory
from sklearn.datasets import load_svmlight_file
import os
import numpy as np
from scipy.sparse import coo_matrix, block_diag, eye, dia_matrix, csc_matrix, hstack, vstack
from scipy.sparse.linalg import spsolve
from numpy.linalg import norm

mem = Memory('./cache')


class SVM(IPM):
    def __init__(self, tao):
        os.chdir('./IPM-SVM')
        self.tao = tao
        pass

    def load_data(self, filename):

        data = load_svmlight_file(filename)
        self.X = data[0]
        self.Y = data[1]
        print('data loaded')
        return

    def data_preprocess(self):
        self.X = self.X.T
        self.n = self.X.shape[1]
        self.m = self.X.shape[0]

        for i in range(self.n):
            self.X[:, i] = self.X[:, i] * self.Y[i]
        self.XY = self.X
        del self.X
        print('XY computed')
        Q = np.append(np.ones(self.m), np.zeros(self.n))
        self.G = block_diag(Q)
        print('G prepared')
        assert self.G.shape == (self.m + self.n, self.m + self.n)
        del Q
        # 需要改成稀疏矩阵形式存储
        temp1 = hstack((block_diag(np.ones(self.m)), -self.XY))
        temp2 = hstack((csc_matrix((1, self.m)), self.Y.T))
        A = vstack((temp1, temp2))
        self.A = csc_matrix(A.T)
        assert self.A.shape == (self.m + self.n, self.m + 1)
        print('A prepared')
        del A

        temp = np.append(np.zeros(self.m), -np.ones(self.n))
        self.g = csc_matrix(temp)
        assert self.g.shape == (1, self.m + self.n)
        print('g prepared')

        temp1 = csc_matrix((self.n, self.m))
        temp2 = block_diag(np.ones(self.n))
        temp3 = block_diag(-np.ones(self.n))
        C1 = hstack((temp1, temp2))
        C2 = hstack((temp1, temp2))
        self.C = vstack((C1, C2)).T
        assert self.C.shape == (self.m + self.n, 2 * self.n)
        print('C prepared')
        del C1
        del C2
        temp1 = np.zeros(self.n)
        temp2 = -self.tao * np.ones(self.n)
        temp = np.append(temp1, temp2)
        self.d = csc_matrix(temp)
        assert self.d.shape == (1, 2 * self.n)
        print('d prepared')
        del temp1
        del temp2
        del temp
        return

    def sketch(self, data):
        pass

    # 初始解输入为np.array格式
    # x = [w,z]对应gondzio 论文
    def fit_general_QP(self, verbose=0, init=(), eta=0.95,
                       maxk=200, eps_L=1e-5, eps_A=1e-5,
                       eps_C=1e-5, eps_mu=1e-5):
        x, y, z, s = init
        # residuals
        mA, nA = self.A.shape
        mC, nC = self.C.shape
        rL = self.G.dot(x) + self.g - self.A.dot(y) - self.C.dot(z)
        rL = np.asarray(rL).T
        rA = -self.A.T.dot(x)
        rA = rA[:, np.newaxis]
        rC = -self.C.T.dot(x) + s + self.d
        rC = np.asarray(rC).T
        rsz = s * z
        mu = rsz.sum() / (nC)

        # iters,epsilon,tolerances
        k = 0
        while k <= maxk and (norm(rL) >= eps_L or norm(rA) >= eps_A or norm(rC) >= eps_C or abs(mu) >= eps_mu):
            print(k)
            print('rL', norm(rL))
            print('rA', norm(rA))
            print('rC', norm(rC))
            print('mu', abs(mu))
            # solve system with a Newton-like method/factorization
            lhs1 = hstack((self.G, -self.A, -self.C))
            lhs2 = hstack((-self.A.T, csc_matrix((nA, nA)), csc_matrix((nA, nC))))
            lhs3 = hstack((-self.C.T, csc_matrix((nC, nA)), -block_diag(s / z)))
            lhs = vstack((lhs1, lhs2, lhs3))
            del lhs1, lhs2, lhs3
            rhs = vstack((-rL, -rA, -rC + (rsz / z)[:, np.newaxis]))
            # 这里应该用cholesky
            dxyz_a = spsolve(lhs, rhs)
            # 传入的x,y,z,s都是rank 1 array
            dx_a = dxyz_a[:x.shape[0]]
            dy_a = dxyz_a[x.shape[0]:x.shape[0] + y.shape[0]]
            dz_a = dxyz_a[x.shape[0] + y.shape[0]:x.shape[0] + y.shape[0] + z.shape[0]]
            ds_a = -(rsz + s * dz_a) / z

            # compute alpha_aff
            alpha_a = 1
            idx_z = np.where(dz_a < 0)[0]
            idx_s = np.where(ds_a < 0)[0]
            if idx_z.size != 0:
                alpha_a = min(alpha_a, np.min(-z[idx_z] / dz_a[idx_z]))
            if idx_s.size != 0:
                alpha_a = min(alpha_a, np.min(-s[idx_s] / ds_a[idx_s]))

            # affine duality gap
            mu_a = (z + alpha_a * dz_a).dot(s + alpha_a * ds_a) / nC

            # compute centering paramter
            sigma = (mu_a / mu) ** 3

            # solve system
            e = np.ones(nC)
            rsz = rsz + ds_a * dz_a - sigma * mu * e
            rhs = vstack((-rL, -rA, -rC + (rsz / z)[:, np.newaxis]))
            dxyz = spsolve(lhs, rhs)
            dx = dxyz[:x.shape[0]]
            dy = dxyz[x.shape[0]:x.shape[0] + y.shape[0]]
            dz = dxyz[x.shape[0] + y.shape[0]:x.shape[0] + y.shape[0] + z.shape[0]]
            ds = -(rsz + s * dz) / z

            # compute alpha
            alpha = 1
            idx_z = np.where(dz < 0)[0]
            idx_s = np.where(ds < 0)[0]
            if idx_z.size != 0:
                alpha_a = min(alpha_a, np.min(-z[idx_z] / dz[idx_z]))
            if idx_s.size != 0:
                alpha_a = min(alpha_a, np.min(-s[idx_s] / ds[idx_s]))
            # print('dx',dx)
            # print('dy',dy)
            # print('dz',dz)
            # print('ds',ds)
            # update
            x = x + eta * alpha * dx
            y = y + eta * alpha * dy
            z = z + eta * alpha * dz
            s = s + eta * alpha * ds
            print('alpha', alpha)
            k += 1
            print(k)
            # update rhs
            rL = self.G.dot(x) + self.g - self.A.dot(y) - self.C.dot(z)
            rL = np.asarray(rL).T
            rA = -self.A.T.dot(x)
            rA = rA[:, np.newaxis]
            rC = -self.C.T.dot(x) + s + self.d
            rC = np.asarray(rC).T
            rsz = s * z
            mu = rsz.sum() / nC

        return (x, y, s, z)

    def fit(self, data):
        return
        '''
        while 1>2 :
            #Compute M for Normal equation

            temp1 = s/z
            temp2 = v/(self.tao*np.ones(self.n)-z)
            temp = temp1+temp2
            Theta_inv = dia_matrix((temp,[0]),shape = (self.n,self.n))
            M = (self.A.dot(self.Q+Theta_inv)).dot(self.A.T)

            #Compute Residuals
            r_b  = -self.A.dot(x)
'''











