from IPM import IPM
from sklearn.datasets import load_svmlight_file
import os
from scipy.linalg import cholesky
import numpy as np
from numpy.linalg import solve
from scipy.sparse import coo_matrix, block_diag, eye, dia_matrix, csc_matrix, hstack, vstack
from scipy.linalg import ldl
from scipy.sparse.linalg import spsolve
from numpy.linalg import norm
from numpy.linalg import inv as dense_inv
from scipy.sparse.linalg import inv


class SVM(IPM):
    def __init__(self, tao):
        self.tao = tao
        pass

    def load_data(self, filename):

        data = load_svmlight_file(filename)
        self.X = data[0]
        self.Y = data[1]
        self.Y[self.Y == 0] = -1
        print('data loaded')
        return

    def data_preprocess(self):
        self.X = self.X.T
        self.n = self.X.shape[1]
        self.m = self.X.shape[0]

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

    # 初始解输入为np.array格式
    # x = [w,z]对应gondzio 论文
    def fit(self, A_w, A_z, Q_w, Q_z, c_w, c_z, b, u,
            init=(),
            maxk=200, eta=0.99995, eps_r_b=1e-5, K=3,
            eps_r_cw=1e-5, eps_r_cz=1e-5, eps_mu=1e-5, delta=0.1, beta_min=0.1, beta_max=10, gamma=0.1,
            ):
        w, z, y, s, v = init
        assert w.shape == (self.m,)
        assert z.shape == (self.n,)
        assert s.shape == (self.n,)
        assert v.shape == (self.n,)
        assert y.shape == (self.m + 1,)
        assert A_w.shape == (self.m + 1, self.m)
        assert A_z.shape == (self.m + 1, self.n)
        assert Q_w.shape == (self.m, self.m)
        assert Q_z.shape == (self.n, self.n)
        assert c_w.shape == (self.m,)
        assert c_z.shape == (self.n,)
        assert b.shape == (self.m + 1,)
        assert u.shape == (self.n,)
        # 输入为 rank 1 array
        n = z.shape[0]
        k = 0
        # b = 0, cw =0 ,Qz =0
        # residual
        r_b = -(A_w.dot(w) + A_z.dot(z) - b)
        # print('r_b',r_b.shape)
        r_cw = -(-Q_w.dot(w) + A_w.T.dot(y) - c_w)
        # print('r_cw',r_cw.shape)
        r_cz = -(A_z.T.dot(y) - c_z + s - v - Q_z.dot(z))
        # print('r_cz',r_cz.shape)
        # 2n((z0)Ts0+(u−z0)Tv0)
        Theta_inv = block_diag(s / z + v / (u - z))
        self.Theta_inv = Theta_inv
        # print('Theta_inv',Theta_inv.shape)
        mu = (s.dot(z) + (u - z).dot(v)) / (2 * n)

        r_b_hat = r_b + A_z.dot(Q_z + Theta_inv).dot(r_cz)
        # print(r_b_hat.shape)
        # print(A_w.dot(inv(Q_w)).dot(r_cw).shape)
        r_b_hat_hat = r_b_hat + A_w.dot(inv(Q_w)).dot(r_cw)

        # stopping criterion page 15  mu may be wrong
        while k <= maxk or norm(r_b) / (1 + norm(self.b)) >= eps_r_b or norm(r_cw) / (1 + norm(c_w)) >= eps_r_cw \
                or norm(r_cz) / (1 + norm(c_z)) >= eps_r_cz or abs(mu) >= eps_mu:
            # Compute M for Normal equation
            print(k)
            print('mu', mu)
            M_w = A_w.dot(inv(Q_w)).dot(A_w.T)
            M_z = A_z.dot((Q_z + Theta_inv)).dot(A_z.T)
            self.M_z = M_z
            M = M_w + M_z
            self.s = s
            self.z = z
            self.v = v
            self.M = M
            # Factorization
            L = cholesky(M.toarray())
            self.L = L
            """
            affine step
            """
            # solve
            d_y_aff = dense_inv(L).dot(dense_inv(L.T).dot(r_b_hat_hat))
            d_w_aff = inv(Q_w).dot(A_w.T.dot(d_y_aff) - r_cw)
            d_z_aff = inv(Q_z + Theta_inv).dot(A_z.T.dot(d_y_aff) - r_cz)
            d_s_aff = -(s + s / z * d_z_aff)
            d_v_aff = -v + v / (u - z) * d_z_aff
            # decide step length alpha
            alpha_aff_p = 1
            alpha_aff_d = 1
            idx_z = np.where(d_z_aff < 1e-8)[0]
            idx_uz = np.where(d_z_aff > 1e-8)[0]
            idx_s = np.where(d_s_aff < 1e-8)[0]
            idx_v = np.where(d_v_aff < 1e-8)[0]
            if idx_z.size != 0:
                alpha_aff_p = max(0, min(alpha_aff_p, np.min(-z[idx_z] / d_z_aff[idx_z])))
                # if idx_uz.size != 0:
                #     alpha_p = max(0,min(alpha_p, np.min(-(u-z)[idx_uz] /(u-z-d_z_aff)[idx_uz])))
            if idx_s.size != 0:
                alpha_aff_d = min(alpha_aff_d, np.min(-s[idx_s] / d_s_aff[idx_s]))
            if idx_v.size != 0:
                alpha_aff_d = min(alpha_aff_d, np.min(-v[idx_v] / d_v_aff[idx_v]))
            print('alpha_aff_p', alpha_aff_p)
            print('alpha_aff_d', alpha_aff_d)
            z_aff = z + d_z_aff * alpha_aff_p
            s_aff = s + d_s_aff * alpha_aff_d
            v_aff = v + d_v_aff * alpha_aff_d
            print('aff_gap', (s_aff.dot(z_aff) + (u - z_aff).dot(v_aff)) / (2 * n))
            print('z:', z_aff[np.where(z_aff <= 0)])
            print('s:', s_aff[np.where(s_aff <= 0)])
            print('v:', v_aff[np.where(v_aff <= 0)])
            #  assert (z_aff>=0).all()
            #  assert (s_aff>=0).all()
            #  assert (v_aff>=0).all()
            #  assert (u-z_aff>=0).all()
            # Compute affine duality gap
            mu_aff = ((s_aff).dot(z_aff) + (u - z_aff).dot(v_aff)) / (2 * n)
            # Determine centering parameter
            sigma = (mu_aff / mu) ** 3
            mu_t = sigma * mu
            '''
            Corrector step
            '''
            r_b = np.zeros(self.m + 1)
            r_cw = np.zeros(self.m)
            r_cz = mu_t * (1 / (u - z) - 1 / z) + d_z_aff * d_s_aff / z + 1 / (u - z) * d_z_aff * d_v_aff
            r_b_hat = r_b + A_z.dot(Q_z + Theta_inv).dot(r_cz)
            r_b_hat_hat = r_b_hat + A_w.dot(inv(Q_w)).dot(r_cw)
            # Solve system
            d_y_corr = dense_inv(L).dot(dense_inv(L.T).dot(r_b_hat_hat))
            d_w_corr = inv(Q_w).dot(A_w.T.dot(d_y_aff) - r_cw)
            d_z_corr = inv(Q_z + Theta_inv).dot(A_z.T.dot(d_y_corr) - r_cz)
            d_s_corr = -(s + s / z * d_z_corr)
            d_v_corr = -v + v / (u - z) * d_z_corr

            # Get predictor direction
            d_y_p = d_y_aff + d_y_corr
            d_w_p = d_w_aff + d_w_corr
            d_z_p = d_z_aff + d_z_corr
            d_s_p = d_s_aff + d_s_corr
            d_v_p = d_v_aff + d_v_corr
            # Compute predictor step alpha_p
            alpha_p = 1
            alpha_d = 1
            idx_z = np.where(d_z_p < 1e-8)[0]
            # idx_uz = np.where(d_z_P > 1e-8)[0]
            idx_s = np.where(d_s_p < 1e-8)[0]
            idx_v = np.where(d_v_p < 1e-8)[0]
            if idx_z.size != 0:
                alpha_p = max(0, min(alpha_p, np.min(-z[idx_z] / d_z_p[idx_z])))
                # if idx_uz.size != 0:
                #     alpha_p = max(0,min(alpha_p, np.min(-(u-z)[idx_uz] /(u-z-d_z_aff)[idx_uz])))
            if idx_s.size != 0:
                alpha_d = min(alpha_d, np.min(-s[idx_s] / d_s_p[idx_s]))
            if idx_v.size != 0:
                alpha_d = min(alpha_d, np.min(-v[idx_v] / d_v_p[idx_v]))
            # modify centering direction
            alpha_tilde_p = np.min((alpha_p + delta, 1))
            alpha_tilde_d = np.min((alpha_d + delta, 1))

            # Mutiple correct step
            k_mcc = 0
            while k_mcc < K:
                # Compute trial point
                w_tilde = w + alpha_tilde_p * d_w_p
                z_tilde = z + alpha_tilde_p * d_z_p
                s_tilde = s + alpha_tilde_d * d_s_p
                v_tilde = v + alpha_tilde_d * d_v_p
                y_tilde = y + alpha_tilde_d * d_y_p
                vec_tilde = np.append(z_tilde * s_tilde, (u - z_tilde) * v_tilde)
                vec_t = np.clip(vec_tilde, beta_min * mu_t, beta_max * mu_t)
                vec_bar = np.clip(vec_t - vec_tilde, -beta_max * mu_t, None)
                # compute direction
                # print('vec_bar',vec_bar.shape)
                r_cw_mcc = np.zeros(self.m)
                r_cz_mcc = -(-vec_bar[:n] / z + vec_bar[n:] / (u - z))
                # print('r_cz_mcc',r_cz_mcc.shape)
                r_b_mcc = np.zeros(self.m + 1)
                r_b_hat_mcc = r_b_mcc + A_z.dot(Q_z + Theta_inv).dot(r_cz_mcc)
                # print(A_w.dot(inv(Q_w)).shape)
                # print('r_b_hat_mcc',r_b_hat_mcc.shape)
                r_b_hat_hat_mcc = r_b_hat_mcc + A_w.dot(inv(Q_w)).dot(r_cw_mcc)
                # print('r_b_hat_hat_mcc',r_b_hat_hat_mcc.shape)
                d_y_mcc = dense_inv(L).dot(dense_inv(L.T).dot(r_b_hat_hat_mcc))
                d_w_mcc = inv(Q_w).dot(A_w.T.dot(d_y_mcc) - r_cw_mcc)
                d_z_mcc = inv(Q_z + Theta_inv).dot(A_z.T.dot(d_y_mcc) - r_cz)
                d_s_mcc = -(s + s / z * d_z_mcc)
                d_v_mcc = -v + v / (u - z) * d_z_mcc

                d_z_t = d_z_p + d_z_mcc
                d_s_t = d_s_p + d_s_mcc
                d_v_t = d_v_p + d_v_mcc
                idx_z = np.where(d_z_t < 1e-8)[0]
                idx_uz = np.where(d_z_t > 1e-8)[0]
                idx_s = np.where(d_s_t < 1e-8)[0]
                idx_v = np.where(d_v_t < 1e-8)[0]
                alpha_p_hat = 1
                alpha_d_hat = 1
                if idx_z.size != 0:
                    alpha_p_hat = min(alpha_p_hat, np.min(-z[idx_z] / d_z_t[idx_z]))
                    #  if idx_uz.size != 0:
                    #      alpha_p_hat = max(0,min(alpha_p_hat, np.min(-(u-z)[idx_uz] /(u-z-d_z_t)[idx_uz])))
                if idx_s.size != 0:
                    alpha_d_hat = min(alpha_d_hat, np.min(-s[idx_s] / d_s_t[idx_s]))
                if idx_v.size != 0:
                    alpha_d_hat = min(alpha_d_hat, np.min(-v[idx_v] / d_v_t[idx_v]))

                print('alpha_p_hat', alpha_p_hat)
                print('alpha_d_hat', alpha_d_hat)
                print('alpha_p + gamma*delta', alpha_p + gamma * delta)
                print('alpha_d + gamma*delta', alpha_d + gamma * delta)
                if (alpha_p_hat >= alpha_p + gamma * delta and alpha_d_hat >= alpha_d + gamma * delta):
                    print('enter mcc', k_mcc)
                    k_mcc += 1
                    # update mcc
                    d_w_p += d_w_mcc  # + d_w_aff
                    d_y_p += d_y_mcc  # + d_y_aff
                    d_z_p += d_z_mcc  # + d_z_aff
                    d_s_p += d_s_mcc  # + d_s_aff
                    d_v_p += d_v_mcc  # + d_v_aff
                    # update alpha
                    alpha_p = alpha_p_hat
                    alpha_d = alpha_d_hat
                    alpha_tilde_p = np.min((alpha_p + delta, 1))
                    alpha_tilde_d = np.min((alpha_d + delta, 1))
                else:
                    print('exit mcc')
                    d_w = d_w_p
                    d_z = d_z_p
                    d_s = d_s_p
                    d_v = d_v_p
                    d_y = d_y_p
                    break

                    # alpha_p = min(1,alpha_p+delta)
                    # alpha_d = min(1,alpha_d+delta)
            '''
            idx_z = np.where(d_z < 0)[0]
            idx_s = np.where(d_s < 0)[0]
            idx_v = np.where(d_v < 0)[0]
            if idx_z.size != 0:
                alpha_p = min(alpha_p, np.min(-z[idx_z] / d_z[idx_z]))
            if idx_s.size != 0:
                alpha_d = min(alpha_d, np.min(-s[idx_s] / d_s[idx_s]))
            if idx_v.size != 0:
                alpha_d = min(alpha_d, np.min(-v[idx_v] / d_v[idx_v]))
            '''
            # update

            w += d_w * eta * alpha_p_hat
            z += d_z * eta * alpha_p_hat
            s += d_s * eta * alpha_d_hat
            v += d_v * eta * alpha_d_hat
            y += d_y * eta * alpha_d_hat
            print(z)
            assert (z > 0).all() == True
            assert (s > 0).all() == True
            assert (v > 0).all() == True
            assert (u - z >= 0).all() == True

            # Compute Residuals
            r_b = -(A_w.dot(w) + A_z.dot(z) - b)
            r_cw = -(-Q_w.dot(w) + self.A_w.T.dot(y) - c_w)
            r_cz = -(A_z.T.dot(y) - c_z + s - v - Q_z.dot(z))
            mu = (s.dot(z) + (u - z).dot(v)) / (2 * self.n)
            Theta_inv = block_diag(s / z + v / (u - z))
            k += 1
            r_b_hat = r_b + A_z.dot(Q_z + Theta_inv).dot(r_cz)
            r_b_hat_hat = r_b_hat + A_w.dot(inv(Q_w)).dot(r_cw)

        return w, z, y, s, v







