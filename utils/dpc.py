import numpy as np
import json
import sys

from utils.eval_metrics import Evaluator


class DPC():
    def __init__(self, knn, dist, gt=None, sim=None, **kwargs):
        super().__init__()

        self.knn = knn
        self.dist = dist
        self.gt = gt
        self.sim = sim

        self.cluster_args = kwargs

    def _find_parent(self, parent, u):
        idx = []
        # parent is a fixed point
        while (u != parent[u]):
            idx.append(u)
            u = parent[u]
        for i in idx:
            parent[i] = u
        return u

    def _edge_to_connected_graph(self, edges):
        n_points = self.knn.shape[0]

        parent = list(range(n_points))
        for u, v in edges:
            p_u = self._find_parent(parent, u)
            p_v = self._find_parent(parent, v)
            parent[p_u] = p_v

        for i in range(n_points):
            parent[i] = self._find_parent(parent, i)
        remap = {}
        uf = np.unique(np.array(parent))
        for i, f in enumerate(uf):
            remap[f] = i
        cluster_id = np.array([remap[f] for f in parent])
        return cluster_id

    def _peaks_to_edges(self, peaks, dist2peak):
        edges = []
        for src in peaks:
            dsts = peaks[src]
            dists = dist2peak[src]
            for dst, dist in zip(dsts, dists):
                if src == dst or dist >= self.cluster_args['tau']:
                    continue
                edges.append([src, dst])
        return edges

    def _peaks_to_labels(self, peaks, dist2peak):
        edges = self._peaks_to_edges(peaks, dist2peak)
        pred_labels = self._edge_to_connected_graph(edges)
        return pred_labels

    def _confidence_to_peaks(self, confidence):
        assert len(confidence) > 0, "confidence is required."

        assert self.dist.shape[0] == confidence.shape[
            0], "dist shape not equal to confidence shape in dimension 0"
        assert self.dist.shape == self.knn.shape, "dist shape not equal to knn shape"

        n_points, _ = self.dist.shape
        dist2peak = {i: [] for i in range(n_points)}
        peaks = {i: [] for i in range(n_points)}

        for i, nbr in enumerate(self.knn):
            dist_tmp = self.dist[i]
            ind = dist_tmp.argsort()
            dist_tmp = dist_tmp[ind]
            nbr = nbr[ind]

            nbr_conf = confidence[nbr]

            for j, c in enumerate(nbr_conf):
                nbr_idx = nbr[j]
                if i == nbr_idx or c < confidence[i]:
                    continue
                dist2peak[i].append(dist_tmp[j])
                peaks[i].append(nbr_idx)
                if len(dist2peak[i]) >= self.cluster_args['max_conn']:
                    break
        return dist2peak, peaks

    def run(self, confidence, tau):
        self.cluster_args['tau'] = tau
        self.confidence = confidence

        dist2peak, peaks = self._confidence_to_peaks(confidence=confidence)
        self.pred = self._peaks_to_labels(peaks, dist2peak)

        return self.pred

    def evaluate_Fscore(self):
        assert self.gt is not None, 'ground truth (gt) is required for evaluation'

        Eval = Evaluator(self.gt, self.pred, remove_single=False)

        pairwise_pre, pairwise_rec, pairwise_fscore = Eval.pairwise()
        bcubed_pre, bcubed_rec, bcubed_fscore = Eval.bcubed()

        if 'verbose' in self.cluster_args and self.cluster_args[
                'verbose'] is True:
            print("Number of clusters: gt: {}, pred: {}".format(
                Eval.get_gt_clusters(), Eval.get_pred_clusters()))
            print("Pairwise precison: {}, recall: {}, fscore: {}".format(
                pairwise_pre, pairwise_rec, pairwise_fscore))
            print("Bcubed precison: {}, recall: {}, fscore: {}".format(
                bcubed_pre, bcubed_rec, bcubed_fscore))
            print('\n')

        return pairwise_fscore, bcubed_fscore

    def evaluate_benchmark(self):
        assert self.gt is not None, 'ground truth (gt) is required for evaluation'

        Eval = Evaluator(self.gt, self.pred, remove_single=False)

        ARI = Eval.ARI()
        NMI = Eval.NMI()

        if 'verbose' in self.cluster_args and self.cluster_args[
                'verbose'] is True:
            print("Number of clusters: gt: {}, pred: {}".format(
                Eval.get_gt_clusters(), Eval.get_pred_clusters()))
            print("ARI: {}, NMI: {}".format(ARI, NMI))
            print('\n')

        return ARI, NMI

    def evaluate(self):
        assert self.gt is not None, 'ground truth (gt) is required for evaluation'

        Eval = Evaluator(self.gt, self.pred, remove_single=False)

        pairwise_pre, pairwise_rec, pairwise_fscore = Eval.pairwise()
        bcubed_pre, bcubed_rec, bcubed_fscore = Eval.bcubed()
        ARI = Eval.ARI()
        NMI = Eval.NMI()

        if 'verbose' in self.cluster_args and self.cluster_args[
                'verbose'] is True:
            print("Number of clusters: gt: {}, pred: {}".format(
                Eval.get_gt_clusters(), Eval.get_pred_clusters()))
            print("Pairwise precison: {}, recall: {}, fscore: {}".format(
                pairwise_pre, pairwise_rec, pairwise_fscore))
            print("Bcubed precison: {}, recall: {}, fscore: {}".format(
                bcubed_pre, bcubed_rec, bcubed_fscore))
            print('\n')

        return pairwise_fscore, bcubed_fscore, ARI, NMI, bcubed_pre, bcubed_rec, pairwise_pre, pairwise_rec

    def evaluate_bin_size(self):
        assert self.gt is not None, 'ground truth (gt) is required for evaluation'

        # get bin
        class_pred = {}
        for label, pred in zip(self.gt, self.pred):
            label = int(label)
            if label not in class_pred:
                class_pred[label] = []
                class_pred[label].append(pred)
            else:
                class_pred[label].append(pred)

        class_cnt = {}
        max_cnt = 0
        min_cnt = 1e6
        for k, v in class_pred.items():
            class_cnt[k] = len(v)
            if len(v) > max_cnt:
                max_cnt = len(v)
            if len(v) < min_cnt:
                min_cnt = len(v)
        print("min cluster size: {}, max cluster size: {}".format(
            min_cnt, max_cnt))

        bin_pairwise_fscore = []
        bin_bcubed_fscore = []
        bin_ARI = []
        bin_NMI = []
        bin_bcubed_pre = []
        bin_bcubed_rec = []
        bin_pairwise_pre = []
        bin_pairwise_rec = []

        class_bin_pred = {}
        class_bin_gt = {}

        bin_list = []
        for i in range(5):
            bin_list.append(0 + i * (len(class_cnt) - 0) // 5)
        bin_list.append(len(class_cnt) + 1)

        class_sort = {}
        for (k, v) in sorted(class_cnt.items(), key=lambda d: d[1]):
            class_sort[k] = v

        bin_cnt = np.zeros(len(bin_list))
        for cidx, (k, v) in enumerate(class_sort.items()):
            for bin_idx, cnt in enumerate(bin_list):
                if cidx < cnt:
                    bin = bin_idx
                    bin_cnt[bin_idx] += 1
                    break

            if bin not in class_bin_pred:
                class_bin_pred[bin] = []
            if bin not in class_bin_gt:
                class_bin_gt[bin] = []
            c_pred = class_pred[k]
            c_gt = [k] * len(c_pred)
            class_bin_pred[bin].extend(c_pred)
            class_bin_gt[bin].extend(c_gt)

        print("clusters in each bin: {}".format(bin_cnt[1:]))

        for k, v in class_bin_pred.items():
            bin_pred = np.array(v)
            bin_gt = np.array(class_bin_gt[k])

            Eval = Evaluator(bin_gt, bin_pred, remove_single=False)

            pairwise_pre, pairwise_rec, pairwise_fscore = Eval.pairwise()
            bcubed_pre, bcubed_rec, bcubed_fscore = Eval.bcubed()
            ARI = Eval.ARI()
            NMI = Eval.NMI()

            if 'verbose' in self.cluster_args and self.cluster_args[
                    'verbose'] is True:
                print("Number of clusters: gt: {}, pred: {}".format(
                    Eval.get_gt_clusters(), Eval.get_pred_clusters()))
                print("Pairwise precison: {}, recall: {}, fscore: {}".format(
                    pairwise_pre, pairwise_rec, pairwise_fscore))
                print("Bcubed precison: {}, recall: {}, fscore: {}".format(
                    bcubed_pre, bcubed_rec, bcubed_fscore))
                print('\n')

            bin_pairwise_fscore.append(pairwise_fscore)
            bin_bcubed_fscore.append(bcubed_fscore)
            bin_ARI.append(ARI)
            bin_NMI.append(NMI)
            bin_bcubed_pre.append(bcubed_pre)
            bin_bcubed_rec.append(bcubed_rec)
            bin_pairwise_pre.append(pairwise_pre)
            bin_pairwise_rec.append(pairwise_rec)

        return bin_pairwise_fscore, bin_bcubed_fscore, bin_ARI, bin_NMI, bin_bcubed_pre, bin_bcubed_rec, bin_pairwise_pre, bin_pairwise_rec

    def evaluate_bin_sim(self):
        assert self.gt is not None, 'ground truth (gt) is required for evaluation'

        # get bin
        class_pred = {}
        for label, pred in zip(self.gt, self.pred):
            label = int(label)
            if label not in class_pred:
                class_pred[label] = []
                class_pred[label].append(pred)
            else:
                class_pred[label].append(pred)

        class_sim = {}
        max_sim = 0
        min_sim = 1e6

        assert len(list(class_pred.keys())) == len(list(
            self.sim.keys())), print("Error, class sim size != gt size")

        for k, v in self.sim.items():
            class_sim[k] = v
            if v > max_sim:
                max_sim = v
            if v < min_sim:
                min_sim = v
        print("min cluster sparsity: {}, max cluster sparsity: {}".format(
            1 - max_sim, 1 - min_sim))

        bin_pairwise_fscore = []
        bin_bcubed_fscore = []
        bin_ARI = []
        bin_NMI = []
        bin_bcubed_pre = []
        bin_bcubed_rec = []
        bin_pairwise_pre = []
        bin_pairwise_rec = []

        class_bin_pred = {}
        class_bin_gt = {}

        bin_list = []
        for i in range(5):
            bin_list.append(0 + i * (len(class_sim) - 0) // 5)
        bin_list.append(len(class_sim) + 1)

        class_sort = {}
        for (k, v) in sorted(class_sim.items(), key=lambda d: d[1]):
            class_sort[k] = v

        bin_cnt = np.zeros(len(bin_list))
        for cidx, (k, v) in enumerate(class_sort.items()):
            for bin_idx, cnt in enumerate(bin_list):
                if cidx < cnt:
                    bin = bin_idx
                    bin_cnt[bin_idx] += 1
                    break

            if bin not in class_bin_pred:
                class_bin_pred[bin] = []
            if bin not in class_bin_gt:
                class_bin_gt[bin] = []
            c_pred = class_pred[k]
            c_gt = [k] * len(c_pred)
            class_bin_pred[bin].extend(c_pred)
            class_bin_gt[bin].extend(c_gt)

        print("clusters in each bin: {}".format(bin_cnt[1:]))

        for bin_idx, (k, v) in enumerate(class_bin_pred.items()):
            bin_pred = np.array(v)
            bin_gt = np.array(class_bin_gt[k])

            Eval = Evaluator(bin_gt, bin_pred, remove_single=False)

            pairwise_pre, pairwise_rec, pairwise_fscore = Eval.pairwise()
            bcubed_pre, bcubed_rec, bcubed_fscore = Eval.bcubed()
            ARI = Eval.ARI()
            NMI = Eval.NMI()

            if 'verbose' in self.cluster_args and self.cluster_args[
                    'verbose'] is True:
                print("Number of clusters: gt: {}, pred: {}".format(
                    Eval.get_gt_clusters(), Eval.get_pred_clusters()))
                print("Pairwise precison: {}, recall: {}, fscore: {}".format(
                    pairwise_pre, pairwise_rec, pairwise_fscore))
                print("Bcubed precison: {}, recall: {}, fscore: {}".format(
                    bcubed_pre, bcubed_rec, bcubed_fscore))
                print('\n')

            bin_pairwise_fscore.append(pairwise_fscore)
            bin_bcubed_fscore.append(bcubed_fscore)
            bin_ARI.append(ARI)
            bin_NMI.append(NMI)
            bin_bcubed_pre.append(bcubed_pre)
            bin_bcubed_rec.append(bcubed_rec)
            bin_pairwise_pre.append(pairwise_pre)
            bin_pairwise_rec.append(pairwise_rec)

        return bin_pairwise_fscore, bin_bcubed_fscore, bin_ARI, bin_NMI, bin_bcubed_pre, bin_bcubed_rec, bin_pairwise_pre, bin_pairwise_rec
