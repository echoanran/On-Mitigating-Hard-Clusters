from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn import metrics
from sklearn.metrics.cluster import contingency_matrix
import pandas as pd


class Evaluator():
    def __init__(self, gt, pred, confidence=None, remove_single=True):
        super().__init__()
        assert len(gt) == len(pred)

        self.gt_labels = gt
        self.pred_labels = pred
        self.confidence = confidence
        self.n_total = len(self.pred_labels)
        # remove single node clusters
        if remove_single:
            pred_label_dic = {}
            for lab in self.pred_labels:
                if lab in pred_label_dic:
                    pred_label_dic[lab] += 1
                else:
                    pred_label_dic[lab] = 1
            ind = []
            for i in range(len(self.pred_labels)):
                if pred_label_dic[self.pred_labels[i]] > 1:
                    ind.append(i)
            self.gt_labels = self.gt_labels[ind]
            self.pred_labels = self.pred_labels[ind]
            self.confidence = self.confidence[ind]
        self.n_arxiv = len(self.pred_labels)

    # Cover anchor based accuracy
    def CoverAnchorAcc(self):
        assert self.confidence != None, 'confidence is required for CoverAnchorAcc'

        cover_dict = {}
        pred_labels_dic = {}
        # Find cover id for each cluster
        for i in range(len(self.pred_labels)):
            if self.pred_labels[i] not in pred_labels_dic:
                pred_labels_dic[self.pred_labels[i]] = [i]
                cover_dict[self.pred_labels[i]] = (i, self.confidence[i])
            else:
                pred_labels_dic[self.pred_labels[i]].append(i)
                if cover_dict[self.pred_labels[i]][1] < self.confidence[i]:
                    cover_dict[self.pred_labels[i]] = (i, self.confidence[i])
        # If share same label with cover in groud-truth, view as a success
        acc_count, total_count = 0, 0
        for key in pred_labels_dic.keys():
            cover_id = cover_dict[key][0]
            for item in pred_labels_dic[key]:
                total_count += 1
                if self.gt_labels[item] == self.gt_labels[cover_id]:
                    acc_count += 1
        return acc_count / (total_count * 1.0)

    # Normalized mutual information
    def NMI(self):
        return metrics.normalized_mutual_info_score(self.pred_labels,
                                                    self.gt_labels)

    # Adjusted rand index
    def ARI(self):
        return metrics.adjusted_rand_score(self.pred_labels, self.gt_labels)

    # pairwised fscore
    def pairwise(self):
        n_samples = self.gt_labels.shape[0]
        c = contingency_matrix(self.gt_labels, self.pred_labels, sparse=True)
        tk = np.dot(c.data, c.data) - n_samples
        pk = np.sum(np.asarray(c.sum(axis=0)).ravel()**2) - n_samples
        qk = np.sum(np.asarray(c.sum(axis=1)).ravel()**2) - n_samples
        avg_pre = tk / (pk + 1e-4)
        avg_rec = tk / (qk + 1e-4)
        fscore = 2. * avg_rec * avg_pre / (avg_pre + avg_rec + 1e-4)
        return avg_pre, avg_rec, fscore

    # Bcubed fscore
    def bcubed(self):
        gt_lb2idxs = {}
        pred_lb2idxs = {}
        for idx, lb in enumerate(self.gt_labels):
            if lb not in gt_lb2idxs:
                gt_lb2idxs[lb] = []
            gt_lb2idxs[lb].append(idx)

        for idx, lb in enumerate(self.pred_labels):
            if lb not in pred_lb2idxs:
                pred_lb2idxs[lb] = []
            pred_lb2idxs[lb].append(idx)

        num_lbs = len(gt_lb2idxs)
        pre = np.zeros(num_lbs)
        rec = np.zeros(num_lbs)
        gt_num = np.zeros(num_lbs)

        for i, gt_idxs in enumerate(gt_lb2idxs.values()):
            all_pred_lbs = np.unique(self.pred_labels[gt_idxs])
            gt_num[i] = len(gt_idxs)
            for pred_lb in all_pred_lbs:
                pred_idxs = pred_lb2idxs[pred_lb]
                n = 1. * np.intersect1d(gt_idxs, pred_idxs).size
                pre[i] += n**2 / len(pred_idxs)
                rec[i] += n**2 / gt_num[i]

        gt_num = gt_num.sum()
        avg_pre = pre.sum() / gt_num
        avg_rec = rec.sum() / gt_num
        fscore = 2. * avg_rec * avg_pre / (avg_pre + avg_rec + 1e-4)
        return avg_pre, avg_rec, fscore

    def SplitDegree(self):
        nclus_pred = len(np.unique(self.pred_labels))
        nclus_gt = len(np.unique(self.gt_labels))
        return nclus_pred, nclus_gt

    def get_gt_clusters(self):

        return len(np.unique(self.gt_labels))

    def get_pred_clusters(self):

        return len(np.unique(self.pred_labels))