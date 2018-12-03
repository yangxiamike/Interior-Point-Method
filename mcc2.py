#!/usr/bin/env python
# coding: utf-8

from sklearn.datasets import load_svmlight_file
from scipy.linalg import cholesky
import numpy as np
from numpy.linalg import solve
from scipy.sparse import coo_matrix, block_diag, eye, dia_matrix, csc_matrix, hstack, vstack
from scipy.linalg import ldl
from scipy.sparse.linalg import spsolve
from numpy.linalg import norm
from numpy.linalg import inv as dense_inv
from scipy.sparse.linalg import inv


class SVM():
    def __init__(self, tao):
        self.tao = tao
        pass

    def init_point(self, A_w, A_z, Q_w, Q_z, c_w, c_z, b, u):
        y_ = inv(A_z.dot(A_z.T)).dot(2 * b - A_z.dot(u))
        z = (A_z.T.dot(y_) + u) / 2
        y = inv(A_z.dot(A_z.T)).dot(A_z).dot(c_z)
        s = 1 / 2 * (A_z.T.dot(y) - c_z)
        w = np.zeros(A_w.shape[1])
        v = -s
        v = np.clip(v, 1e-5, None)
        s = np.clip(s, 1e-5, None)
        z = np.clip(z, 1e-5, u - 1e-5)
        return w, z, y, s, v

    def load_data(self, filename):

        data = load_svmlight_file(filename)
        self.X = data[0][:600, :]
        self.Y = data[1][:600]
        self.Y[self.Y == 0] = -1
        print('data loaded')
        return

    def data_preprocess(self):

        self.X = self.X.T
        self.n = self.X.shape[1]
        self.m = self.X.shape[0]
        assert self.m < self.n
        '''
        self.Q_w = block_diag(np.ones(self.m))#.toarray()
        self.A_w = csc_matrix(np.eye(self.m+1,self.m))#.toarray()
        self.A_z = csc_matrix(np.eye(self.m+1,self.n))#.toarray()
        #self.Q_z = block_diag(np.ones(self.n))
        self.Q_z = csc_matrix(np.zeros((self.n,self.n)))#.toarray()
        self.c_w = np.zeros(self.m)
        self.c_z = np.zeros(self.n)
        self.u = 10*np.ones(self.n)
        self.b = np.append(-5*np.ones(self.m),5)
        print(self.b)
        '''

        for i in range(self.n):
            self.X[:, i] = self.X[:, i] * self.Y[i]
        self.XY = self.X
        # del self.X
        print('XY computed')
        Q_w = np.ones(self.m)
        self.Q_w = block_diag(Q_w)
        print('Q_w prepared')
        assert self.Q_w.shape == (self.m, self.m)
        del Q_w
        self.Q_z = csc_matrix((self.n, self.n))
        print('Q_z prepared')
        # 需要改成稀疏矩阵形式存储
        self.A_w = vstack((block_diag(np.ones(self.m)), csc_matrix((1, self.m))))
        self.A_z = vstack((-self.XY, self.Y.T))
        assert self.A_w.shape == (self.m + 1, self.m)
        assert self.A_z.shape == (self.m + 1, self.n)
        print('A_w,A_z prepared')

        self.c_z = np.ones(self.n)
        self.c_w = np.zeros(self.m)
        print('c_w,c_z prepared')

        self.b = np.zeros(self.m + 1)
        print('b prepared')

        self.u = self.tao * np.ones(self.n)
        print('u prepared')

        return self.A_w, self.A_z, self.Q_w, self.Q_z, self.c_w, self.c_z, self.b, self.u

    def sketch(self, data):
        pass

    def fit(self, A_w, A_z, Q_w, Q_z, c_w, c_z, b, u,
            init=(),
            maxk=130, eta=0.95, eps_r_b=1e-5, K=3,
            eps_r_cw=1e-5, eps_r_cz=1e-5, eps_mu=1e-5, delta=0.1, beta_min=0.1, beta_max=10, gamma=0.3,
            ):
        # print(A_w.shape)
        w, z, y, s, v = init
        k = 0
        assert w.shape == (self.m,)
        assert z.shape == (self.n,)
        assert s.shape == (self.n,)
        assert v.shape == (self.n,)
        assert y.shape == (self.m + 1,)
        # print(A_w.shape,(self.m + 1, self.m))
        # print(A_w.shape == (self.m + 1, self.m))
        assert A_w.shape == (self.m + 1, self.m)
        assert A_z.shape == (self.m + 1, self.n)
        assert Q_w.shape == (self.m, self.m)
        assert Q_z.shape == (self.n, self.n)
        assert c_w.shape == (self.m,)
        assert c_z.shape == (self.n,)
        assert b.shape == (self.m + 1,)
        assert u.shape == (self.n,)
        # 输入为 rank 1 array
        Theta_inv = block_diag(s / z + v / (u - z))
        # Theta = block_diag(z/s+(u-z)/v)
        r_b = -(A_w.dot(w) + A_z.dot(z) - b)
        # print(A_z.shape)
        # print('r_b',r_b)
        r_cw = -(-Q_w.dot(w) + A_w.T.dot(y) - c_w)
        # print('r_cw',r_cw)
        r_sz = -z * s
        r_vz = -(u - z) * v
        r_cz = -A_z.T.dot(y) + c_z + Q_z.dot(z) - s + v
        r_cz_hat = r_cz - r_sz / z + r_vz / (u - z)  # Theta_inv.dot(z))
        # print('r_cz',r_cz)
        # 2n((z0)Ts0+(u−z0)Tv0)#.toarray()
        # self.Theta_inv = Theta_inv
        # print('Theta_inv',Theta_inv)
        mu = (s.dot(z) + (u - z).dot(v)) / (2 * self.n)
        # r_b_hat = r_b + A_z.dot(Q_z + Theta_inv).dot(r_cz)
        # print('r_b_hat',r_b_hat)
        # print(A_w.dot(inv(Q_w)).dot(r_cw).shape)
        r_b_hat = r_b + A_w.dot(inv(Q_w)).dot(r_cw) + A_z.dot(inv(Q_z + Theta_inv)).dot(r_cz_hat)
        while k <= maxk and (norm(r_b) / (1 + norm(self.b)) >= eps_r_b and norm(r_cw) / (1 + norm(c_w)) >= eps_r_cw \
                                     and norm(r_cz) / (1 + norm(c_z)) >= eps_r_cz and mu >= eps_mu):
            print(k)
            M_w = A_w.dot(inv(Q_w)).dot(A_w.T)
            # print(M_w)
            M_z = A_z.dot(inv((Q_z + Theta_inv))).dot(A_z.T)
            M = (M_w + M_z).toarray()
            # Factorization
            # print(M)

            L = cholesky(M)
            # print(L)
            # d_y_aff = dense_inv(L).dot(dense_inv(L.T).dot(r_b_hat_hat.T.ravel()))
            d_y_aff = dense_inv(L).dot(dense_inv(L.T).dot(r_b_hat.T.ravel()))
            d_w_aff = inv(Q_w).dot(A_w.T.dot(d_y_aff) - r_cw)
            d_z_aff = inv(Q_z + Theta_inv).dot(A_z.T.dot(d_y_aff) - r_cz_hat)
            d_s_aff = -(s + s / z * d_z_aff)
            d_v_aff = -v + v / (u - z) * d_z_aff
            print('test affine')
            print('r_b', norm(A_w.dot(w + d_w_aff) + A_z.dot(z + d_z_aff) - b))
            print('r_cw', norm(-Q_w.dot(d_w_aff) + A_w.T.dot(d_y_aff) - r_cw))
            print('r_cz', norm(-(Q_z + Theta_inv).dot(d_z_aff) + A_z.T.dot(d_y_aff) - r_cz_hat))
            print('end')
            print('\n')
            # print('d_w_aff',d_w_aff)
            # print('d_z_aff',d_z_aff)
            # print('d_s_aff',d_s_aff)
            # print('d_v_aff',d_v_aff)
            # print('d_y_aff',d_y_aff)
            alpha_aff_p = 1
            alpha_aff_d = 1
            idx_z = np.where(d_z_aff < 0)[0]
            idx_uz = np.where(d_z_aff > 0)[0]
            idx_s = np.where(d_s_aff < 0)[0]
            idx_v = np.where(d_v_aff < 0)[0]
            if idx_z.size != 0:
                alpha_aff_p = min(alpha_aff_p, np.min(-z[idx_z] / d_z_aff[idx_z]))
            if idx_uz.size != 0:
                alpha_aff_p = min(alpha_aff_p, np.min((u - z)[idx_uz] / d_z_aff[idx_uz]))
            if idx_s.size != 0:
                alpha_aff_d = min(alpha_aff_d, np.min(-s[idx_s] / d_s_aff[idx_s]))
            if idx_v.size != 0:
                alpha_aff_d = min(alpha_aff_d, np.min(-v[idx_v] / d_v_aff[idx_v]))
            print('affine step length')
            print('alpha_aff_p', alpha_aff_p)
            print('alpha_aff_d', alpha_aff_d)
            print('\n')
            z_aff = z + d_z_aff * alpha_aff_p
            s_aff = s + d_s_aff * alpha_aff_d
            v_aff = v + d_v_aff * alpha_aff_d
            y_aff = y + d_y_aff * alpha_aff_d
            w_aff = w + d_w_aff * alpha_aff_p
            assert (z_aff >= -1e-8).all(), '%d' % z_aff[np.where(z_aff <= 0)]
            assert (s_aff >= -1e-8).all(), '%d' % s_aff[np.where(s_aff <= 0)]
            assert (v_aff >= -1e8).all(), '%d' % v_aff[np.where(v_aff <= 0)]
            assert (u - z_aff >= -1e-6).all(), '%d' % (u - z_aff)[np.where((u - z_aff) <= 0)]
            # print('z_aff',z_aff)
            # print('s_aff',s_aff)
            # print('w_aff',w_aff)
            # print('v_aff',v_aff)
            # print('y_aff',y_aff)
            mu_aff = ((s_aff).dot(z_aff) + (u - z_aff).dot(v_aff)) / (2 * self.n)
            print('affine gap', mu_aff)
            print('\n')
            # Determine centering parameter
            sigma = (mu_aff / mu) ** 3
            mu_t = sigma * mu
            e = np.ones(z.shape[0])
            r_sz = -z * s - d_z_aff * d_s_aff + mu_t * e
            r_vz = -v * (u - z) + d_z_aff * d_v_aff + mu_t * np.ones(z.shape[0])
            r_cz = -A_z.T.dot(y) + c_z + Q_z.dot(z) - s + v
            r_cz_hat = r_cz - r_sz / z + r_vz / (u - z)  # Theta_inv.dot(z))
            r_b_hat = r_b + A_w.dot(inv(Q_w)).dot(r_cw) + A_z.dot(inv(Q_z + Theta_inv)).dot(r_cz_hat)
            # Solve system
            d_y_pre_corr = dense_inv(L).dot(dense_inv(L.T).dot(r_b_hat.ravel()))
            d_w_pre_corr = inv(Q_w).dot(A_w.T.dot(d_y_pre_corr) - r_cw)
            d_z_pre_corr = inv(Q_z + Theta_inv).dot(A_z.T.dot(d_y_pre_corr) - r_cz_hat)
            d_s_pre_corr = r_sz / z - s * d_z_pre_corr / z
            d_v_pre_corr = r_vz / (u - z) + v * d_z_pre_corr / (u - z)
            print('test correct')
            print('r_b', norm(A_w.dot(w + d_w_pre_corr) + A_z.dot(z + d_z_pre_corr) - b))
            print('r_cw', norm(-Q_w.dot(d_w_pre_corr) + A_w.T.dot(d_y_pre_corr) - r_cw))
            print('r_cz', norm(-(Q_z + Theta_inv).dot(d_z_pre_corr) + A_z.T.dot(d_y_pre_corr) - r_cz_hat))
            print('end')
            print('\n')
            # Compute predictor step alpha_p
            alpha_p = 1
            alpha_d = 1
            idx_z = np.where(d_z_pre_corr < 0)[0]
            idx_uz = np.where(d_z_pre_corr > 0)[0]
            idx_s = np.where(d_s_pre_corr < 0)[0]
            idx_v = np.where(d_v_pre_corr < 0)[0]
            if idx_z.size != 0:
                alpha_p = min(alpha_p, np.min(-z[idx_z] / d_z_pre_corr[idx_z]))
            if idx_uz.size != 0:
                alpha_p = min(alpha_p, np.min((u - z)[idx_uz] / d_z_pre_corr[idx_uz]))
            if idx_s.size != 0:
                alpha_d = min(alpha_d, np.min(-s[idx_s] / d_s_pre_corr[idx_s]))
            if idx_v.size != 0:
                alpha_d = min(alpha_d, np.min(-v[idx_v] / d_v_pre_corr[idx_v]))
            print()
            print('correct step length')
            print('alpha_p', alpha_p)
            print('alpha_d', alpha_d)
            print('\n')
            d_w = d_w_pre_corr
            d_z = d_z_pre_corr
            d_s = d_s_pre_corr
            d_v = d_v_pre_corr
            d_y = d_y_pre_corr

            w += d_w * eta * alpha_p
            # _hat
            z += d_z * eta * alpha_p
            # _hat
            s += d_s * eta * alpha_d
            # _hat
            v += d_v * eta * alpha_d
            # _hat
            y += d_y * eta * alpha_d
            # print('z',z)
            # print('s',s)
            # print('w',w)
            # print('v',v)
            # print('y',y)

            # print(A_z.shape)
            # print('r_b',r_b)

            # print('r_cw',r_cw)
            print('###############')
            print('result')
            r_b = -(A_w.dot(w) + A_z.dot(z) - b)
            print('r_b', norm(r_b) / (1 + norm(self.b)))
            # print('d_w',d_w)
            # print('r_b',r_b)
            r_cw = -(-Q_w.dot(w) + A_w.T.dot(y) - c_w)
            print('r_cw', norm(r_cw) / (1 + norm(c_w)))
            r_sz = -z * s
            r_vz = -(u - z) * v
            r_cz = -A_z.T.dot(y) + c_z + Q_z.dot(z) - s + v
            r_cz_hat = r_cz - r_sz / z + r_vz / (u - z)  # Theta_inv.dot(z))
            print('r_cz', norm(r_cz) / (1 + norm(r_cz)))
            # print('r_cz',r_cz)
            print('mu', mu)
            print('\n')
            # 2n((z0)Ts0+(u−z0)Tv0)
            Theta_inv = block_diag(s / z + v / (u - z))  # .toarray()
            # print('Theta_inv',Theta_inv)
            mu = (s.dot(z) + (u - z).dot(v)) / (2 * self.n)
            # print('r_b_hat',r_b_hat)
            r_b_hat = r_b + A_w.dot(inv(Q_w)).dot(r_cw) + A_z.dot(inv(Q_z + Theta_inv)).dot(r_cz_hat)
            # print('r_b_hat_hat',r_b_hat_hat)
            k += 1

        return y, w, z

# In[2]:

c = SVM(tao=0.01)
c.load_data('IPM-SVM/svm.txt')
Q_w,Q_z,A_w,A_z,c_w,c_z,b,u = c.data_preprocess()
"""
w = np.ones(c.m)
z = np.ones(c.n)
s = np.ones(c.n)
v = np.ones(c.n)
y = np.ones(c.m+1)
"""
w, z, y, s, v = c.init_point(Q_w,Q_z,A_w,A_z,c_w,c_z,b,u)

#w, z, y, s, v = c.init_point(Q_w,Q_z,A_w,A_z,c_w,c_z,b,u)


# In[31]:


y,w,z =c.fit(Q_w,Q_z,A_w,A_z,c_w,c_z,b,u,init= (w,z,y,s,v),maxk=10)


# In[8]:




# In[14]:




# In[18]:




# In[17]:





# In[ ]:





# In[ ]:




