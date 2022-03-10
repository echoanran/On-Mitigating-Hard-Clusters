import numpy as np
import time
from multiprocessing import Process, Manager
import math


class Density():
    def __init__(self, knn, dist, density_type='naive', gt=None, **kwargs):
        super().__init__()

        self.knn = knn
        self.dist = dist
        self.density_type = density_type
        self.gt = gt
        self.density_args = kwargs

    def transport_worker(self, knn, P, sid, eid, process_i, res_dict):
        n, k = knn.shape
        pos = np.zeros(n).astype('int32') - 1
        y = np.zeros((k, k)).astype('float32')
        res_dist = []
        for i in range(sid, eid):
            pos[knn[i]] = np.arange(k)
            knn_tmp = knn[knn[i]]
            P_tmp = P[knn[i]]
            y.fill(0)
            x_ind, y_ind = np.where(pos[knn_tmp] >= 0)
            pos_index = pos[knn_tmp[x_ind, y_ind]]
            y[x_ind, pos_index] = P_tmp[x_ind, y_ind]
            pos[knn[i]] = -1
            tmp_dist = np.dot(P[i], y.T)
            res_dist.append(tmp_dist)
        result = np.array(res_dist)
        res_dict[process_i] = result

    def TPDi(self, sigma=0.05, processNum=16):
        n, _ = self.knn.shape
        P = np.exp(-self.dist / sigma)
        ss = P.sum(axis=1)
        for i in range(len(P)):
            P[i] = P[i] / (ss[i] + 1e-5)
        P = np.sqrt(P)
        step = math.ceil(n / processNum)
        pool = []
        res_dict = Manager().dict()
        for process_i in range(processNum):
            sid = process_i * step
            eid = min((process_i + 1) * step, n)
            t = Process(target=self.transport_worker,
                        args=(self.knn, P, sid, eid, process_i, res_dict))
            pool.append(t)
        for process_i in range(processNum):
            pool[process_i].start()
        for process_i in range(processNum):
            pool[process_i].join()

        dist = 1 - np.concatenate([res_dict[i] for i in range(processNum)], 0)
        return dist

    def approximate_NDDe(self, sigma=0.5, k=20, verbose=True, **kwargs):

        assert self.knn.shape[0] == self.dist.shape[0]
        n, _ = self.dist.shape
        density = np.zeros(n)

        knn, dist = self.knn[:, :k], self.dist[:, :k]
        P = np.exp(-dist / sigma)
        ss = P.sum(axis=1)

        for i in range(len(P)):
            density[self.knn[i][:k]] += P[i] / (ss[i] + 1e-5)
        return density

    def NDDe(self,
             sigma=0.5,
             d=0.1,
             k=20,
             epsilon=0.05,
             verbose=True,
             **kwargs):

        assert self.knn.shape[0] == self.dist.shape[0]
        n, _ = self.dist.shape
        knn, dist = self.knn[:, :k], self.dist[:, :k]
        density = np.ones((n, 1)) * (1 / n)

        P = np.exp(-dist / sigma)
        ss = P.sum(axis=1)

        link_p = np.zeros((n, n))
        for i in range(len(P)):  # normalize density for each node
            link_p[i][knn[i]] = P[i] / ss[i]
            link_p[i][i] = 0
        link_p = link_p.T

        while True:
            density_tmp = d * density + \
                (1-d) * np.matmul(link_p, density)
            if np.abs(density_tmp - density).sum() < epsilon:
                return density
            density = density_tmp

    def naive_density(self, rho=0.64, verbose=True, **kwargs):

        tmp = self.dist < rho
        return tmp.sum(axis=1)

    def compute_density(self, normalize=False, compute_time=False):

        total_time = 0

        if compute_time:
            self.density_args['verbose'] = False

        tic = time.time()

        if self.density_type == 'naive':
            density = self.naive_density(**self.density_args)
        elif self.density_type == 'NDDe':
            density = self.NDDe(**self.density_args)
        elif self.density_type == 'approximate_NDDe':
            density = self.approximate_NDDe(**self.density_args)

        toc = time.time()
        total_time = toc - tic

        if compute_time:
            print("Time for computing density: ", total_time)

        if normalize:
            minn = density.min()
            maxx = density.max()
            if minn == maxx:
                pass
            else:
                density = (density - minn) / (maxx - minn)

        return density, total_time