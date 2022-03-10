from argparse import ArgumentError
import os
from pyexpat import features
import numpy as np

from utils import (read_meta, read_probs, l2norm, build_knns,
                   knns2ordered_nbrs, intdict2ndarray, Timer,
                   get_weighted_feature)


class MS1M(object):
    def __init__(
        self,
        feature_path,
        label_path,
        feature_dim,
        topk=80,
        knn_path=None,
        is_norm_feat=False,
        prefix='../save',
        knn_method='faiss_gpu',
        is_reload=False,
        is_train=True,
        is_sim=False,
        is_size=False,
        is_similarity=False,
        model_topk=80,
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
        self.is_sim = is_sim
        self.is_size = is_size
        self.is_similarity = is_similarity
        self.model_topk = model_topk

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

        with Timer('compute class size'):
            if self.is_size:
                size_path = os.path.join(
                    self.prefix, 'class_size', 'ms1m',
                    feature_path.split('/')[-1].split('.')[0], 'size.npy')

                if not os.path.exists(os.path.dirname(size_path)):
                    os.makedirs(os.path.dirname(size_path))

                if not os.path.exists(size_path):
                    self._compute_size()
                    np.save(size_path, self.class_size)

        with Timer('compute class similarity'):
            if self.is_similarity:
                similarity_path = os.path.join(
                    self.prefix, 'class_sim', 'ms1m',
                    feature_path.split('/')[-1].split('.')[0], 'sim.npy')

                if not os.path.exists(os.path.dirname(similarity_path)):
                    os.makedirs(os.path.dirname(similarity_path))

                if not os.path.exists(similarity_path):
                    self._compute_similarity()
                    np.save(similarity_path, self.class_sim)

        with Timer('read knn'):
            if not self.is_reload and self.knn_path is not None and os.path.isfile(
                    self.knn_path):
                knns = np.load(self.knn_path)['data']
            else:
                if not self.is_reload and self.knn_path is not None:
                    print('knn_path does not exist: {}'.format(self.knn_path))
                knn_prefix = os.path.join(
                    self.prefix, 'knns', 'ms1m',
                    feature_path.split('/')[-1].split('.')[0])
                knns = build_knns(knn_prefix, self.features, self.knn_method,
                                  max(self.topk, self.model_topk))

            # convert knns to (dists, nbrs)
            self.dists, self.nbrs = knns2ordered_nbrs(knns)

        print('feature shape: {}, norm_feat: {}'.format(
            self.features.shape, self.is_norm_feat))

        aggre_path = os.path.join(self.prefix, 'aggre_feats', 'ms1m',
                                  feature_path.split('/')[-1].split('.')[0],
                                  'faiss_k_' + str(self.topk) + '.npy')
        if not self.is_reload and os.path.exists(aggre_path):
            print('load aggregated feature from {}'.format(aggre_path))
            self.aggregated_features = np.load(os.path.join(aggre_path))
        else:
            print('recompute aggregated feature and save in {}'.format(
                aggre_path))
            self.aggregated_features = []

            for cur_idx, nbr_idxs in enumerate(self.nbrs[:, :self.topk]):
                features = self.features[nbr_idxs]
                nbr_dists = self.dists[cur_idx, :self.topk]
                aggregated_feature = get_weighted_feature(features, nbr_dists)
                self.aggregated_features.append(aggregated_feature)
            self.aggregated_features = np.array(self.aggregated_features)

            if not os.path.exists(os.path.dirname(aggre_path)):
                os.makedirs(os.path.dirname(aggre_path))
            np.save(aggre_path, self.aggregated_features)
        print('aggregated_features shape: {}'.format(
            self.aggregated_features.shape))

        self.aggre_labels = []
        for cur_idx, _ in enumerate(self.nbrs):
            self.aggre_labels.append(
                self.gt_labels[self.nbrs[cur_idx, :self.topk]])
        self.aggre_labels = np.array(self.aggre_labels)

    def _compute_size(self):

        self.gt_labels
        self.features

        unique_labels = np.unique(self.gt_labels)
        self.class_size = {}
        for label in unique_labels:
            features = self.features[self.gt_labels == label]
            self.class_size[int(label)] = len(features)

        return self.class_size

    def _compute_similarity(self):

        self.gt_labels
        self.features

        unique_labels = np.unique(self.gt_labels)
        self.class_sim = {}
        for label in unique_labels:
            features = self.features[self.gt_labels == label]
            similiarity = 0
            cnt = 0
            for i in range(len(features)):
                for j in range(i + 1, len(features)):
                    similiarity += np.dot(features[i], features[j].T) / (
                        np.linalg.norm(features[i]) *
                        np.linalg.norm(features[j]))
                    cnt += 1

            if cnt == 0:
                self.class_sim[int(label)] = 1
            else:
                self.class_sim[int(label)] = similiarity / cnt

        return self.class_sim

    def __getitem__(self, index):

        ori_feature = self.features[self.nbrs[index][:self.model_topk]]
        feature = self.aggregated_features[self.nbrs[index][:self.model_topk]]
        label = self.gt_labels[self.nbrs[index][:self.model_topk]]

        return (feature, label, ori_feature)

    def __len__(self):
        return self.size


if __name__ == '__main__':
    pass
