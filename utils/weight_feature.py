import torch
import numpy as np

__all__ = ['get_weighted_feature']


def get_weighted_feature(features, nbr_dists, sigma=0.5):

    anchor_feature = features[0]
    nbr_features = features[1:]

    weights = np.exp(-nbr_dists[1:]) / sigma
    weights = weights / weights.sum()
    
    weighted_feature = np.sum(np.repeat(
        np.expand_dims(weights, axis=1), features.shape[-1], axis=1) *
                              nbr_features,
                              axis=0)

    aggregated_feature = np.concatenate([anchor_feature, weighted_feature],
                                        axis=0)
    
    return aggregated_feature