import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from libs.utils import entropy, seg_encode, rollout
from libs.postprocess import PostProcessor
import configs.asrf_config as asrf_cfg


class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, device, reduction='mean'):

        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.reduction = reduction
        self.device = device

    def forward(self, inputs, targets, weights=1.0):

        log_probs = self.logsoftmax(inputs)
        if len(targets.size()) == 1:
            targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1).to(self.device)
        loss = (-targets * log_probs).sum(dim=1)*weights
        if self.reduction=='mean':
            return loss.mean()
        else:
            return loss


class AlignSeg(nn.Module):

    def __init__(self, num_classes, device):

        super(AlignSeg, self).__init__()
        self.num_classes = num_classes
        self.T = 1
        self.device = device

    def forward(self, segment_feat, f_feat):

        segment_feat = F.normalize(segment_feat, dim=1)
        f_feat = F.normalize(f_feat, dim=1)
        sim_matrix = 1.0 / (torch.cdist(segment_feat, f_feat) + 1.0)
        # sim_matrix = torch.einsum('nc,kc->nk', [segment_feat, f_feat])
        cost_matrix = 1 - sim_matrix      ## cost matrix
        pro = (torch.ones((len(cost_matrix), 1)) * (1 / len(cost_matrix))).to(self.device)
        t_dist = F.softmax(sim_matrix/self.T, dim=1) ##transport distance
        loss = ((cost_matrix * t_dist).sum(1)*pro.squeeze(1)).sum()
        return loss


def get_merge(labels, preds, tot):
    ans = []
    for label, pred in zip(labels, preds):
        cur = torch.zeros(tot)
        for l, p in zip(label, pred):
            cur[l.item()] = p
        ans.append(cur)
    return ans

def JS_Divergence_With_Temperature(p, q, temp_factor, get_softmax=True):
    KLDivLoss = nn.KLDivLoss(reduction='none')
    if get_softmax:
        p_softmax_output = F.softmax(p / temp_factor, dim=1)
        q_softmax_output = F.softmax(q / temp_factor, dim=1)
    log_mean_softmax_output = ((p_softmax_output + q_softmax_output) / 2).log()
    return (KLDivLoss(log_mean_softmax_output, p_softmax_output) + KLDivLoss(log_mean_softmax_output, q_softmax_output)) / 2


def frame_segment_adaptation_asrf(train_loader, model, num_classes, optimizer, optimizer_refine, refine_net,
                                  class_weight, criterion_bound, device):

    ce = nn.CrossEntropyLoss(weight=class_weight, ignore_index=-100)
    soft_ce = CrossEntropyLabelSmooth(num_classes, device)
    epoch_loss = 0.0
    total = 0
    correct = 0
    label_embedding = nn.Embedding(num_classes, 64).to(device)
    transport_loss = AlignSeg(num_classes, device)

    for idx, sample in enumerate(train_loader):
        x = sample['feature']
        t = sample['label']
        # b = sample["boundary"]
        # mask_b = sample["mask"]
        x, t = x.to(device), t.to(device)
        # b = b.to(device)
        # mask_b = mask_b.to(device)
        mask = torch.ones(1, num_classes, np.shape(t)[1]).to(device)
        action_pred, output_bound, frames_feat = model(x) ## backbone model's prediction
        optimizer.zero_grad()  ##
        optimizer_refine.zero_grad()

        ## training refiner: improving accuracy
        loss_ema = 0.0
        for j, p_ema in enumerate(action_pred):
            frame_pre_ema = refine_net(F.softmax(p_ema.detach(), dim=1))
            loss_ema += ce(frame_pre_ema.transpose(2, 1).contiguous().view(-1, num_classes), t.view(-1))
        loss_ema.backward()

        # train backbone model: reduce over-segmentation
        loss = 0.0
        for j, (p, f_feat) in enumerate(zip(action_pred, frames_feat)):

            if j < 2 :
                continue
            p_frame = refine_net(F.softmax(p, dim=1))

            ### Segment encoder: get segment feature and segment pseudo-label
            action_idx = torch.argmax(p, dim=1).squeeze()
            weight = 1.0 + torch.exp(-entropy(F.softmax(p, dim=1)))
            segment_feat, segment_idx, label_probility, GTlabel_list, Predseg_list = \
                seg_encode(action_idx.to(device), f_feat.to(device), weight, t)
            label_emb = label_embedding(torch.hstack(Predseg_list))
            segment_feats = segment_feat + label_emb.unsqueeze(0)
            segment_pred = model.asb[j-1].conv_out(segment_feats.permute(0, 2, 1))  ## segment predict
            segment_soft_label = torch.stack(get_merge(GTlabel_list, label_probility, num_classes))
            segment_predictions = segment_pred.transpose(2, 1).contiguous().view(-1, num_classes)
            seg_roll_predict = rollout(segment_idx, segment_pred.permute(0, 2, 1))
            confidences = F.softmax(segment_pred, dim=1).max(dim=1)[0].squeeze()  # confidence as weights

            ###training loss
            loss_kl = F.kl_div(F.log_softmax(seg_roll_predict.detach(), dim=1), F.softmax(p_frame, dim=1))
            loss_segment = soft_ce(segment_predictions, segment_soft_label.to(device), weights=confidences)  ## segment loss
            loss_align = transport_loss(segment_feat.squeeze(0), f_feat.squeeze(0).permute(1, 0))  ## transport loss
            loss += (loss_segment + loss_align - loss_kl)

        ## boundary loss
        # n = len(output_bound)
        # for out in output_bound:
        #     loss += 0.01*criterion_bound(out, b, mask_b) / n

        loss.backward()
        optimizer_refine.step()
        optimizer.step()
        # _, predicted = torch.max(action_pred[-1].data, 1)
        predicted_refine = refine_net(F.softmax(action_pred[-1], dim=1))  ##

        postprocessor = PostProcessor("refinement_with_boundary", asrf_cfg.boundary_th)
        predicted = postprocessor(predicted_refine.cpu().data.numpy(), boundaries=output_bound[-1].cpu().data.numpy(),
                                  masks=torch.ones(1, 1, x.shape[-1]).bool().data.numpy())
        predicted = torch.from_numpy(predicted).cuda()
        # _, predicted = torch.max(predicted_refine.data, 1)  ## source prediction

        correct += ((predicted == t).float()*mask[:, 0, :].squeeze(1)).sum().item()
        total += torch.sum(mask[:, 0, :]).item()
        epoch_loss += loss.item()

    acc = float(correct)/total
    return epoch_loss/len(train_loader), acc

def frame_segment_adaptation_tcn(train_loader, model, num_classes, optimizer, optimizer_refine, refine_net, device):

    ce = nn.CrossEntropyLoss(ignore_index=-100)
    soft_ce = CrossEntropyLabelSmooth(num_classes, device)
    epoch_loss = 0.0
    total = 0
    correct = 0
    label_embedding = nn.Embedding(num_classes, 64).to(device)
    transport_loss = AlignSeg(num_classes, device)

    for idx, sample in enumerate(train_loader):
        refine_net.train()
        model.train()
        x = sample['feature']
        t = sample['label']
        x, t = x.to(device), t.to(device)
        mask = torch.ones(1, num_classes, np.shape(t)[1]).to(device)
        action_pred, frames_feat = model(x, mask) ## backbone model's prediction
        optimizer.zero_grad()  ##
        optimizer_refine.zero_grad()

        ## training refiner: improving accuracy
        loss_ema = 0.0
        for j, p_ema in enumerate(action_pred):
            frame_pre_ema  = refine_net(F.softmax(p_ema.detach(), dim=1))
            loss_ema += ce(frame_pre_ema.transpose(2, 1).contiguous().view(-1, num_classes), t.view(-1))
        loss_ema.backward()

        # train backbone model: reduce over-segmentation
        loss = 0.0
        for j, (p, f_feat) in enumerate(zip(action_pred, frames_feat)):

            if j < 2:
                continue
            p_frame = refine_net(F.softmax(p, dim=1))

            ### Segment encoder: get segment feature and segment pseudo-label
            action_idx = torch.argmax(p, dim=1).squeeze()
            weight = 1.0 + torch.exp(-entropy(F.softmax(p, dim=1))) ##
            # weight = torch.ones(1, p.size(-1)).to(device)  ## no weights
            segment_feat, segment_idx, label_probility, GTlabel_list, Predseg_list = \
                seg_encode(action_idx.to(device), f_feat.to(device), weight, t)
            label_emb = label_embedding(torch.hstack(Predseg_list))
            segment_feats = segment_feat + label_emb.unsqueeze(0)
            segment_pred = model.stages[j - 1].conv_out(segment_feats.permute(0, 2, 1))  ## segment predict
            segment_soft_label = torch.stack(get_merge(GTlabel_list, label_probility, num_classes))
            segment_predictions = segment_pred.transpose(2, 1).contiguous().view(-1, num_classes)
            seg_roll_predict = rollout(segment_idx, segment_pred.permute(0, 2, 1))
            confidences = F.softmax(segment_pred, dim=1).max(dim=1)[0].squeeze()  # confidence as weights

            ###training loss
            loss_kl = F.kl_div(F.log_softmax(seg_roll_predict.detach(), dim=1), F.softmax(p_frame, dim=1))
            # loss_kl = torch.nn.L1Loss()(F.softmax(seg_roll_predict.detach(), dim=1), F.softmax(p_frame, dim=1)) ## l1
            # loss_kl = torch.mean(JS_Divergence_With_Temperature(seg_roll_predict.detach(), p_frame,1)) ## JS
            loss_segment = soft_ce(segment_predictions, segment_soft_label.to(device), weights=confidences)  ## segment loss
            loss_align = transport_loss(segment_feat.squeeze(0), f_feat.squeeze(0).permute(1, 0))  ## transport loss
            loss += (loss_segment + loss_align - loss_kl)

        loss.backward()
        optimizer_refine.step()
        optimizer.step()
        # _, predicted = torch.max(action_pred[-1].data, 1)
        predicted_refine = refine_net(F.softmax(action_pred[-1], dim=1))  ##
        _, predicted = torch.max(predicted_refine.data, 1)  ## source prediction

        correct += ((predicted == t).float()*mask[:, 0, :].squeeze(1)).sum().item()
        total += torch.sum(mask[:, 0, :]).item()
        epoch_loss += loss.item()

    acc = float(correct)/total
    return epoch_loss/len(train_loader), acc

def frame_segment_adaptation_tcn_plus(train_loader, model, num_classes, optimizer, optimizer_refine, refine_net, device):

    ce = nn.CrossEntropyLoss(ignore_index=-100)
    # ce = nn.CrossEntropyLoss(weight=class_weight, ignore_index=-100)
    soft_ce = CrossEntropyLabelSmooth(num_classes, device)
    epoch_loss = 0.0
    total = 0
    correct = 0
    label_embedding = nn.Embedding(num_classes, 64).to(device)
    transport_loss = AlignSeg(num_classes, device)

    for idx, sample in enumerate(train_loader):
        x = sample['feature']
        t = sample['label']
        x, t = x.to(device), t.to(device)
        mask = torch.ones(1, num_classes, np.shape(t)[1]).to(device)
        optimizer.zero_grad()  ##
        optimizer_refine.zero_grad()
        action_pred, frames_feat = model(x) ## backbone model's prediction
        ## training refiner: improving accuracy
        loss_ema = 0.0
        for j, p_ema in enumerate(action_pred):
            frame_pre_ema = refine_net(F.softmax(p_ema.detach(), dim=1))
            loss_ema += ce(frame_pre_ema.transpose(2, 1).contiguous().view(-1, num_classes), t.view(-1))
        loss_ema.backward()

        # train backbone model: reduce over-segmentation
        loss = 0.0
        for j, (p, f_feat) in enumerate(zip(action_pred, frames_feat)):
            if j < 2:
                continue
            p_frame = refine_net(F.softmax(p, dim=1))

            ### Segment encoder: get segment feature and segment pseudo-label
            action_idx = torch.argmax(p, dim=1).squeeze()
            weight = 1.0 + torch.exp(-entropy(F.softmax(p, dim=1)))
            segment_feat, segment_idx, label_probility, GTlabel_list, Predseg_list = \
                seg_encode(action_idx.to(device), f_feat.to(device), weight, t)
            label_emb = label_embedding(torch.hstack(Predseg_list))
            segment_feats = segment_feat + label_emb.unsqueeze(0)
            segment_pred = model.Rs[j - 1].conv_out(segment_feats.permute(0, 2, 1))  ## segment predict
            segment_soft_label = torch.stack(get_merge(GTlabel_list, label_probility, num_classes))
            segment_soft_label = segment_soft_label.to(device)
            segment_predictions = segment_pred.transpose(2, 1).contiguous().view(-1, num_classes)
            seg_roll_predict = rollout(segment_idx, segment_pred.permute(0, 2, 1))
            confidences = F.softmax(segment_pred, dim=1).max(dim=1)[0].squeeze()  # confidence as weights

            ###training loss
            loss_adv = F.kl_div(F.log_softmax(seg_roll_predict.detach(), dim=1), F.softmax(p_frame, dim=1))
            loss_segment = soft_ce(segment_predictions, segment_soft_label, weights=confidences)  ## segment loss
            loss_align = transport_loss(segment_feat.squeeze(0), f_feat.squeeze(0).permute(1, 0))  ## transport loss
            loss += (loss_segment + loss_align - loss_adv)

        loss.backward()
        optimizer_refine.step()
        optimizer.step()
        # _, predicted = torch.max(action_pred[-1].data, 1)
        predicted_refine = refine_net(F.softmax(action_pred[-1], dim=1))  ##
        _, predicted = torch.max(predicted_refine.data, 1)  ## source prediction

        correct += ((predicted == t).float()*mask[:, 0, :].squeeze(1)).sum().item()
        total += torch.sum(mask[:, 0, :]).item()
        epoch_loss += loss.item()

    acc = float(correct)/total
    return epoch_loss/len(train_loader), acc

def frame_segment_adaptation_ASF(train_loader, model, num_classes, optimizer, optimizer_refine, refine_net, device):

    ce = nn.CrossEntropyLoss(ignore_index=-100)
    soft_ce = CrossEntropyLabelSmooth(num_classes, device)
    epoch_loss = 0.0
    total = 0
    correct = 0
    label_embedding = nn.Embedding(num_classes, 64).to(device)
    transport_loss = AlignSeg(num_classes, device)

    for idx, sample in enumerate(train_loader):
        refine_net.train()
        model.train()
        x = sample['feature']
        t = sample['label']
        x, t = x.to(device), t.to(device)
        mask = torch.ones(1, num_classes, np.shape(t)[1]).to(device)
        action_pred, frames_feat = model(x, mask)
        optimizer.zero_grad()  ##
        optimizer_refine.zero_grad()

        ## training refiner: improving accuracy
        loss_ema = 0.0
        for j, p_ema in enumerate(action_pred):
            frame_pre_ema = refine_net(F.softmax(p_ema.detach(), dim=1))
            # frame_pre_ema = refine_net(F.softmax(p_ema.detach(), dim=1), frames_feat[j].detach(), mask) ## ASFormer
            loss_ema += ce(frame_pre_ema.transpose(2, 1).contiguous().view(-1, num_classes), t.view(-1))
        loss_ema.backward()

        # train segment model: reduce over-segmentation
        loss = 0.0
        for j, (p, f_feat) in enumerate(zip(action_pred, frames_feat)):

            if j < 2:
                continue
            p_frame = refine_net(F.softmax(p, dim=1))
            # p_emas = refine_net(F.softmax(p, dim=1), f_feat, mask) ### ASFormer
            ### Segment encoder: get segment feature and segment pseudo-label
            action_idx = torch.argmax(p, dim=1).squeeze()
            weight = 1.0 + torch.exp(-entropy(F.softmax(p, dim=1)))
            segment_feat, segment_idx, label_probility, GTlabel_list, Predseg_list = \
                seg_encode(action_idx.to(device), f_feat.to(device), weight, t)
            label_emb = label_embedding(torch.hstack(Predseg_list))
            segment_feat = segment_feat + label_emb.unsqueeze(0)
            segment_pred = model.decoders[j - 1].conv_out(segment_feat.permute(0, 2, 1))  ## segment predict
            segment_soft_label = torch.stack(get_merge(GTlabel_list, label_probility, num_classes))
            segment_predictions = segment_pred.transpose(2, 1).contiguous().view(-1, num_classes)
            seg_roll_predict = rollout(segment_idx, segment_pred.permute(0, 2, 1))
            confidences = F.softmax(segment_pred, dim=1).max(dim=1)[0].squeeze()  # confidence as weights

            ###training loss
            loss_kl = F.kl_div(F.log_softmax(seg_roll_predict.detach(), dim=1), F.softmax(p_frame, dim=1))  #
            loss_segment = soft_ce(segment_predictions, segment_soft_label.to(device), weights=confidences)  ## segment loss
            loss_align = transport_loss(segment_feat.squeeze(0), f_feat.squeeze(0).permute(1, 0))  ## transport loss
            loss += (loss_segment + loss_align - loss_kl)  #

        # loss.backward()
        loss.backward()
        optimizer_refine.step()
        optimizer.step()
        # _, predicted = torch.max(action_pred[-1].data, 1)
        predicted_refine = refine_net(F.softmax(action_pred[-1], dim=1))  ##
        _, predicted = torch.max(predicted_refine.data, 1)  ## source prediction

        correct += ((predicted == t).float()*mask[:, 0, :].squeeze(1)).sum().item()
        total += torch.sum(mask[:, 0, :]).item()
        epoch_loss += loss.item()


    acc = float(correct)/total
    return epoch_loss/len(train_loader), acc