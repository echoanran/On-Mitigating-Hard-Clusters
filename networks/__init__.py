from .transformer import *
from .losses import * 
from .models import TransformerBased

__factory__ = {
    'transformer': TransformerBased
}


def build_model(model_type, cfg):
    if model_type not in __factory__:
        raise KeyError("Unknown dataset type:", model_type)
    return __factory__[model_type](**cfg)
