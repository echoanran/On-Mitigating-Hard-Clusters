import os
import numpy as np

from utils.compute_density import Density
from utils.dpc import DPC

np.set_printoptions(threshold=np.inf)

import warnings

warnings.filterwarnings("ignore")


class Clustering():
    def __init__(self,
                 knn,
                 dist,
                 gt,
                 density_args,
                 cluster_args,
                 work_dir,
                 prefix,
                 verbose=False,
                 ori_knn=None,
                 ori_dist=None,
                 sim=None,
                 **kwargs):
        super().__init__()

        self.knn = knn
        self.dist = dist
        self.gt = gt

        self.density_args = density_args
        self.cluster_args = cluster_args

        self.work_dir = work_dir
        self.prefix = prefix

        self.verbose = verbose

        self.ori_knn = ori_knn
        self.ori_dist = ori_dist

        self.sim = sim

    def run(self):

        if not os.path.exists(os.path.join(self.work_dir, 'results')):
            os.makedirs(os.path.join(self.work_dir, 'results'))

        if self.ori_dist is not None:
            dsty = Density(self.ori_knn,
                           self.ori_dist,
                           **self.density_args,
                           gt=self.gt)
        else:
            dsty = Density(self.knn,
                           self.dist,
                           **self.density_args,
                           gt=self.gt)

        density, total_time = dsty.compute_density(
            normalize=False, compute_time=self.cluster_args['compute_time'])

        np.save(
            os.path.join(self.work_dir, 'results', self.prefix + '_density'),
            density)

        np.save(os.path.join(self.work_dir, 'results', self.prefix + '_gt'),
                self.gt)

        if 'TPDi' in self.density_args and self.density_args['TPDi']:
            print("Computing TPDi...")
            new_dist = dsty.TPDi()
            new_dist = new_dist[:, :self.dist.shape[-1]]
        else:
            new_dist = self.dist

        best_pairwise_fscore, best_bcubed_fscore, best_ARI, best_NMI = 0.0, 0.0, 0.0, 0.0
        best_bcubed_pre, best_bcubed_rec = 0.0, 0.0
        best_pairwise_pre, best_pairwise_rec = 0.0, 0.0
        best_pred = []
        best_tau = 0.0

        dpc = DPC(self.knn,
                  new_dist,
                  self.gt,
                  sim=self.sim,
                  **self.cluster_args,
                  verbose=self.verbose)

        taus = np.linspace(self.cluster_args['tau_start'],
                           self.cluster_args['tau_end'],
                           self.cluster_args['tau_step']).tolist()

        # '''
        for tidx, tau in enumerate(taus):
            print("tidx: {}, tau: {}".format(tidx, tau))
            pred = dpc.run(confidence=density, tau=tau)

            pairwise_fscore, bcubed_fscore, ARI, NMI, bcubed_pre, bcubed_rec, pairwise_pre, pairwise_rec = dpc.evaluate(
            )

            print(
                'pairwise_fscore: {}, bcubed_fscore: {}, ARI: {}, NMI: {}, bcubed_pre: {}, bcubed_rec: {}, pairwise_pre: {}, pairwise_rec: {}'
                .format(pairwise_fscore, bcubed_fscore, ARI, NMI, bcubed_pre,
                        bcubed_rec, pairwise_pre, pairwise_rec))

            if self.cluster_args['select_by'] == 'pairwise':
                if pairwise_fscore > best_pairwise_fscore:
                    best_pairwise_fscore = pairwise_fscore
                    best_bcubed_fscore = bcubed_fscore
                    best_ARI = ARI
                    best_NMI = NMI
                    best_pred = pred
                    best_tau = tau
                    best_bcubed_pre = bcubed_pre
                    best_bcubed_rec = bcubed_rec
                    best_pairwise_pre = pairwise_pre
                    best_pairwise_rec = pairwise_rec

            elif self.cluster_args['select_by'] == 'bcubed':
                if bcubed_fscore > best_bcubed_fscore:
                    best_pairwise_fscore = pairwise_fscore
                    best_bcubed_fscore = bcubed_fscore
                    best_ARI = ARI
                    best_NMI = NMI
                    best_pred = pred
                    best_tau = tau
                    best_bcubed_pre = bcubed_pre
                    best_bcubed_rec = bcubed_rec
                    best_pairwise_pre = pairwise_pre
                    best_pairwise_rec = pairwise_rec

        print("Pairwise fscore: {}".format(best_pairwise_fscore))
        print("Bcubed fscore: {}".format(best_bcubed_fscore))
        print("Bcubed precision: {}".format(best_bcubed_pre))
        print("Bcubed recall: {}".format(best_bcubed_rec))
        print("Pairwise precision: {}".format(best_pairwise_pre))
        print("Pairwise recall: {}".format(best_pairwise_rec))
        print("ARI: {}".format(best_ARI))
        print("NMI: {}".format(best_NMI))
        print("tau: {}".format(best_tau))
        print("#clusters: {}".format(len(np.unique(best_pred))))

        print(
            'best_pairwise_fscore: {}, best_bcubed_fscore: {}, best_ARI: {}, best_NMI: {}, best_bcubed_pre: {}, best_bcubed_rec: {}, best_pairwise_pre: {}, best_pairwise_rec: {}\n'
            .format(best_pairwise_fscore, best_bcubed_fscore, best_ARI,
                    best_NMI, best_bcubed_pre, best_bcubed_rec,
                    best_pairwise_pre, best_pairwise_rec))

        np.savetxt(
            os.path.join(self.work_dir, 'results',
                         self.prefix + '_pairwise_f1.txt'),
            np.array([best_pairwise_fscore]))
        np.savetxt(
            os.path.join(self.work_dir, 'results',
                         self.prefix + '_bcubed_f1.txt'),
            np.array([best_bcubed_fscore]))
        np.savetxt(
            os.path.join(self.work_dir, 'results', self.prefix + '_ARI.txt'),
            np.array([best_ARI]))
        np.savetxt(
            os.path.join(self.work_dir, 'results', self.prefix + '_NMI.txt'),
            np.array([best_NMI]))
        np.savetxt(
            os.path.join(self.work_dir, 'results',
                         self.prefix + '_bcubed_pre.txt'),
            np.array([best_bcubed_pre]))
        np.savetxt(
            os.path.join(self.work_dir, 'results',
                         self.prefix + '_bcubed_rec.txt'),
            np.array([best_bcubed_rec]))
        np.savetxt(
            os.path.join(self.work_dir, 'results',
                         self.prefix + '_pairwise_pre.txt'),
            np.array([best_pairwise_pre]))
        np.savetxt(
            os.path.join(self.work_dir, 'results',
                         self.prefix + '_pairwise_rec.txt'),
            np.array([best_pairwise_rec]))
        np.savetxt(
            os.path.join(self.work_dir, 'results', self.prefix + '_tau.txt'),
            np.array([
                self.cluster_args['tau_start'], self.cluster_args['tau_end'],
                self.cluster_args['tau_step'], best_tau
            ]))

        np.savetxt(os.path.join(self.work_dir, 'results',
                                self.prefix + '_pred.txt'),
                   np.array(best_pred),
                   fmt='%d')
        # '''

        if 'compute_size' in self.cluster_args and self.cluster_args[
                'compute_size']:
            # '''
            # evaluate size bin
            pred = dpc.run(confidence=density, tau=taus[0])
            pairwise_fscore, bcubed_fscore, ARI, NMI, bcubed_pre, bcubed_rec, pairwise_pre, pairwise_rec = dpc.evaluate_bin_size(
            )

            print(
                'pairwise_fscore: {}\n, bcubed_fscore: {}\n, ARI: {}\n, NMI: {}\n, bcubed_pre: {}\n, bcubed_rec: {}\n, pairwise_pre: {}\n, pairwise_rec: {}\n'
                .format(pairwise_fscore, bcubed_fscore, ARI, NMI, bcubed_pre,
                        bcubed_rec, pairwise_pre, pairwise_rec))

            np.savetxt(
                os.path.join(self.work_dir, 'results',
                             self.prefix + '_size_bin_pairwise_f1.txt'),
                np.array([pairwise_fscore]))
            np.savetxt(
                os.path.join(self.work_dir, 'results',
                             self.prefix + '_size_bin_bcubed_f1.txt'),
                np.array([bcubed_fscore]))
            np.savetxt(
                os.path.join(self.work_dir, 'results',
                             self.prefix + '_size_bin_bcubed_pre.txt'),
                np.array([bcubed_pre]))
            np.savetxt(
                os.path.join(self.work_dir, 'results',
                             self.prefix + '_size_bin_bcubed_rec.txt'),
                np.array([bcubed_rec]))
            np.savetxt(
                os.path.join(self.work_dir, 'results',
                             self.prefix + '_size_bin_pairwise_pre.txt'),
                np.array([pairwise_pre]))
            np.savetxt(
                os.path.join(self.work_dir, 'results',
                             self.prefix + '_size_bin_pairwise_rec.txt'),
                np.array([pairwise_rec]))
            np.savetxt(
                os.path.join(self.work_dir,
                             'results', self.prefix + '_size_bin_ARI.txt'),
                np.array([ARI]))
            np.savetxt(
                os.path.join(self.work_dir,
                             'results', self.prefix + '_size_bin_NMI.txt'),
                np.array([NMI]))
            # '''

        if 'compute_sim' in self.cluster_args and self.cluster_args[
                'compute_sim']:
            # evaluate sim bin
            if self.sim is not None:
                pred = dpc.run(confidence=density, tau=taus[0])
                pairwise_fscore, bcubed_fscore, ARI, NMI, bcubed_pre, bcubed_rec, pairwise_pre, pairwise_rec = dpc.evaluate_bin_sim(
                )

                print(
                    'pairwise_fscore: {}\n, bcubed_fscore: {}\n, ARI: {}\n, NMI: {}\n, bcubed_pre: {}\n, bcubed_rec: {}\n, pairwise_pre: {}\n, pairwise_rec: {}\n'
                    .format(pairwise_fscore, bcubed_fscore, ARI, NMI,
                            bcubed_pre, bcubed_rec, pairwise_pre,
                            pairwise_rec))

                np.savetxt(
                    os.path.join(self.work_dir, 'results',
                                 self.prefix + '_sim_bin_pairwise_f1.txt'),
                    np.array([pairwise_fscore]))
                np.savetxt(
                    os.path.join(self.work_dir, 'results',
                                 self.prefix + '_sim_bin_bcubed_f1.txt'),
                    np.array([bcubed_fscore]))
                np.savetxt(
                    os.path.join(self.work_dir, 'results',
                                 self.prefix + '_sim_bin_bcubed_pre.txt'),
                    np.array([bcubed_pre]))
                np.savetxt(
                    os.path.join(self.work_dir, 'results',
                                 self.prefix + '_sim_bin_bcubed_rec.txt'),
                    np.array([bcubed_rec]))
                np.savetxt(
                    os.path.join(self.work_dir, 'results',
                                 self.prefix + '_sim_bin_pairwise_pre.txt'),
                    np.array([pairwise_pre]))
                np.savetxt(
                    os.path.join(self.work_dir, 'results',
                                 self.prefix + '_sim_bin_pairwise_rec.txt'),
                    np.array([pairwise_rec]))
                np.savetxt(
                    os.path.join(self.work_dir, 'results',
                                 self.prefix + '_sim_bin_ARI.txt'),
                    np.array([ARI]))
                np.savetxt(
                    os.path.join(self.work_dir, 'results',
                                 self.prefix + '_sim_bin_NMI.txt'),
                    np.array([NMI]))

        return best_pairwise_fscore, best_bcubed_fscore