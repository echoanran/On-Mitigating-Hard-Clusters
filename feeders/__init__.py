from re import I
from .ms1m import MS1M


__factory__ = {
    'ms1m': MS1M
}


def build_dataset(dataset_type, cfg):
    if dataset_type not in __factory__:
        raise KeyError("Unknown dataset type:", dataset_type)

    return __factory__[dataset_type](**cfg)
