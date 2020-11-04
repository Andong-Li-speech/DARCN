import torch
import torch.nn as nn
import numpy as np
EPSILON = 1e-10


def mse_loss(esti_list, label, mask_for_loss):
    masked_esti= esti_list * mask_for_loss
    masked_label = label * mask_for_loss
    loss = ((masked_esti - masked_label) ** 2).sum()  / mask_for_loss.sum() + EPSILON
    return loss


def mse_loss_stage(esti_list, label, nframes):
    with torch.no_grad():
        mask_for_loss_list = []
        for frame_num in nframes:
            mask_for_loss_list.append(torch.ones((frame_num, label.size()[-1]), dtype=torch.float32))
        mask_for_loss = nn.utils.rnn.pad_sequence(mask_for_loss_list, batch_first=True).cuda()

    stage_number = len(esti_list)
    loss = 0
    for i in range(stage_number):
        loss += (((esti_list[i] - label) ** 2) * mask_for_loss).sum() / mask_for_loss.sum()
    return loss / stage_number + EPSILON


def numParams(net):
    num = 0
    for param in net.parameters():
        if param.requires_grad:
            num += int(np.prod(param.size()))
    return num