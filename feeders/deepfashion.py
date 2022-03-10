from argparse import ArgumentError
import os
import numpy as np

from utils import (read_meta, read_probs, l2norm, build_knns,
                   knns2ordered_nbrs, intdict2ndarray, Timer,
                   get_weighted_feature)


class DeepFashion(object):
    def __init__(
        self,
        feature_path,
        label_path,
        feature_dim,
        topk=80,
        knn_path=None,
        is_norm_feat=False,
        prefix='../save',
        knn_method='faiss',
        is_reload=False,
        is_train=True,
    ):

        self.feature_path = feature_path
        self.label_path = label_path
        self.knn_path = knn_path
        self.topk = topk
        self.feature_dim = feature_dim
        self.is_norm_feat = is_norm_feat
        self.prefix = prefix
        self.knn_method = knn_method
        self.is_reload = is_reload
        self.is_train = is_train

        with Timer('read meta and feature'):
            if label_path is not None:
                self.lb2idxs, self.idx2lb = read_meta(label_path)
                self.inst_num = len(self.idx2lb)
                self.gt_labels = intdict2ndarray(self.idx2lb)
                self.ignore_label = False
            else:
                self.inst_num = -1
                self.ignore_label = True

            self.features = read_probs(self.feature_path, self.inst_num,
                                       self.feature_dim)
            if self.is_norm_feat:
                self.features = l2norm(self.features)
            if self.inst_num == -1:
                self.inst_num = self.features.shape[0]
            self.size = self.features.shape[0]

        with Timer('read knn'):
            if not self.is_reload and self.knn_path is not None and os.path.isfile(
                    self.knn_path):
                knns = np.load(self.knn_path)['data']
            else:
                if self.knn_path is not None:
                    print('knn_path does not exist: {}'.format(self.knn_path))
                knn_prefix = os.path.join(
                    self.prefix, 'knns', 'deepfashion',
                    feature_path.split('/')[-1].split('.')[0])
                knns = build_knns(knn_prefix, self.features, self.knn_method,
                                  self.topk)

            # convert knns to (dists, nbrs)
            self.dists, self.nbrs = knns2ordered_nbrs(knns)

        print('feature shape: {}, norm_feat: {}'.format(
            self.features.shape, self.is_norm_feat))
        print('Num of gt classes: ', len(np.unique(self.gt_labels)))

        cnt = 0
        for cur_idx, nbr_idxs in enumerate(self.nbrs):
            cnt = cnt + (sum([
                1 for nbr in nbr_idxs
                if self.gt_labels[nbr] == self.gt_labels[nbr_idxs[0]]
            ]))
        print("same label rate: ", cnt / (self.topk * self.nbrs.shape[0]))

        aggre_path = os.path.join(self.prefix, 'aggre_feats', 'deepfashion',
                                  feature_path.split('/')[-1].split('.')[0],
                                  'faiss_k_' + str(self.topk) + '.npy')
        if not self.is_reload and os.path.exists(aggre_path):
            print('load aggregated feature from {}'.format(aggre_path))
            self.aggregated_features = np.load(os.path.join(aggre_path))
        else:
            print('recompute aggregated feature and save in {}'.format(
                aggre_path))
            self.aggregated_features = []

            for cur_idx, nbr_idxs in enumerate(self.nbrs):
                features = self.features[nbr_idxs]
                nbr_dists = self.dists[cur_idx]
                aggregated_feature = get_weighted_feature(features, nbr_dists)
                self.aggregated_features.append(aggregated_feature)
            self.aggregated_features = np.array(self.aggregated_features)

            if not os.path.exists(os.path.dirname(aggre_path)):
                os.makedirs(os.path.dirname(aggre_path))
            np.save(aggre_path, self.aggregated_features)
        print('aggregated_features shape: {}'.format(
            self.aggregated_features.shape))
        
        self.aggre_labels = []
        for cur_idx, nbr_idxs in enumerate(self.nbrs):
            self.aggre_labels.append(self.gt_labels[self.nbrs[cur_idx]])
        self.aggre_labels = np.array(self.aggre_labels)

    def __getitem__(self, index):

        ori_feature = self.features[self.nbrs[index]]
        feature = self.aggregated_features[self.nbrs[index]]
        label = self.gt_labels[self.nbrs[index]]

        return (feature, label, ori_feature)

    def __len__(self):
        return self.size


if __name__ == '__main__':
    pass
