import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math

from networks import TransformerEncoderLayer, TransformerEncoder


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        self.d_model = d_model
        self.max_len = max_len

    def forward(self, x):
        pe = torch.zeros(self.max_len, self.d_model, device=x.device)
        position = torch.arange(0, self.max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0.0, self.d_model, 2) *
            -(math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return Variable(pe[:, :x.size(-1)], requires_grad=False)


class TransformerBased(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=True,
        num_encoder_layers=2,
        num_of_feats=80,
        **kwargs,
    ):
        super().__init__()
        self.position_encoding = PositionalEncoding(d_model=d_model,
                                                    max_len=num_of_feats)

        encoder_layer = TransformerEncoderLayer(d_model, nhead,
                                                dim_feedforward, dropout,
                                                activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers,
                                          encoder_norm)

        self.clf = nn.Sequential(nn.Linear(d_model * 2, d_model),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(d_model, int(d_model / 2)),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(int(d_model / 2), 1))

    def forward(self, x):
        memory = self.encoder(x,
                              src_key_padding_mask=None,
                              pos=self.position_encoding(x))

        output = memory

        batch_size, topk, dim = output.shape
        anchor = output[:, 0, :].unsqueeze(1)
        anchor = anchor.repeat(1, topk, 1)
        pair_feat = torch.cat([anchor, output], 2)

        pair_res = self.clf(pair_feat)

        return pair_res