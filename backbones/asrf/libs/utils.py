import torch
import torch.nn as nn
from typing import Optional, Any, Tuple
import numpy as np
import torch.nn as nn
from torch.autograd import Function
import torch
import torch.nn.functional as F


def entropy(predictions, reduction='none'):
    epsilon = 1e-5
    H = -predictions * torch.log(predictions + epsilon)
    H = H.sum(dim=1)
    if reduction == 'mean':
        return H.mean()
    else:
        return H


def get_segment_info(action_idx, batch_input, weight, batch_target=None):

    segment_idx = [0]
    prev_seg = action_idx[0]
    for ii, idx in enumerate(action_idx):
        if idx != prev_seg:
            segment_idx.append(ii)
        prev_seg = idx
    segment_idx.append(len(action_idx))
    ### get segment lable and feat
    GTlabel_list = list()
    segment_feat = list()
    label_probility = list()
    Predseg_list = list()
    for s_i in range(len(segment_idx) - 1):
        prev_idx = segment_idx[s_i]
        curr_idx = segment_idx[s_i + 1]
        curr_seg = batch_input[:, :, prev_idx:curr_idx]
        ### groundtruth segment label info
        if batch_target is not None:
            GTseg = batch_target[:, prev_idx:curr_idx]
            seg_label = torch.where(torch.bincount(GTseg[0])>0)[0]
            # seg_label = torch.argmax(torch.bincount(GTseg[0]))
            label_prob = torch.bincount(GTseg[0])[seg_label]/torch.bincount(GTseg[0])[seg_label].sum()
            label_probility.append(label_prob)
            GTlabel_list.append(seg_label)
        ## predict info
        # Predseg_info = torch.mean(action_idx[prev_idx:curr_idx].float()).long()
        Predseg_info = torch.argmax(torch.bincount(action_idx[prev_idx:curr_idx]))
        Predseg_list.append(Predseg_info)

        ## segment feat info
        curr_seg_weight = weight[:, prev_idx:curr_idx]
        curr_seg_weight = curr_seg_weight / torch.sum(curr_seg_weight)
        curr_feat =(curr_seg_weight.unsqueeze(0)*curr_seg).sum(dim=2, keepdim=True)
        segment_feat.append(curr_feat.permute(0, 2, 1))
    segment_feat = torch.cat(segment_feat, dim=1)
    return segment_idx, segment_feat, label_probility, GTlabel_list, Predseg_list


def seg_encode(action_idx, batch_input, weight, batch_target=None):

    segment_idx, segment_feats, label_probility, GTlabel_list, Predseg_list = \
        get_segment_info(action_idx, batch_input, weight, batch_target)
    return segment_feats, segment_idx, label_probility, GTlabel_list, Predseg_list


def rollout(segment_idx, refine_pred): ## segment label to frame label
    refine_rollout = []
    for s_i in range(len(segment_idx)-1):
        prev_idx = segment_idx[s_i]
        curr_idx = segment_idx[s_i+1]
        curr_refine = refine_pred[0, s_i, :].view(1, -1).repeat(curr_idx-prev_idx, 1)
        refine_rollout.append(curr_refine)
    refine_rollout = torch.cat(refine_rollout, dim=0).unsqueeze(0).transpose(2, 1) # B D L

    return refine_rollout


def get_segment_index(p_hard_index, segment_idx, segment_len):

    segment_index = list()
    for i in range(segment_len):
        prev_idx = segment_idx[i]
        curr_idx = segment_idx[i + 1]
        for value in p_hard_index[0]:
            if value >= prev_idx and value < curr_idx:
                segment_index.append(i)
            else:
                continue

    return segment_index













