import os
import torch

__all__ = ['save_model']


def save_model(save_dir, model, optimizer, current_epoch):
    out = os.path.join(save_dir, "checkpoint_{}.tar".format(current_epoch))
    state = {
        'net': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': current_epoch
    }
    torch.save(state, out)
