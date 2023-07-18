import torch
import torch.nn as nn

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

def correct_p_hard(segment_idx, p_hard_index, segment_index, logits, p, p_hard_before):

    logits_hard = logits[segment_index, :]
    cor = []
    for i in range(len(p_hard_before)):
        seg = segment_index[i]
        prev_idx = segment_idx[seg]
        curr_idx = segment_idx[seg + 1]
        A1 = (p[prev_idx: curr_idx] * logits_hard[i, :][prev_idx: curr_idx].unsqueeze(1)).sum(dim=0)-logits_hard[i,:][p_hard_index[0][i]]*p_hard_before[i,:]
        # A2 = (p*logits_hard[i,:].unsqueeze(1)).sum(dim=0) - (p[prev_idx: curr_idx] * logits_hard[i, :][prev_idx: curr_idx].unsqueeze(1)).sum(dim=0)
        # A =  (A1 - A2)/2
        cor.append(A1)
    return p_hard_before+torch.vstack(cor)


def MI(outputs_target):
    batch_size = outputs_target.size(0)
    softmax_outs_t = nn.Softmax(dim=1)(outputs_target)
    avg_softmax_outs_t = torch.sum(softmax_outs_t, dim=0) / float(batch_size)
    log_avg_softmax_outs_t = torch.log(avg_softmax_outs_t)
    item1 = -torch.sum(avg_softmax_outs_t * log_avg_softmax_outs_t)
    item2 = -torch.sum(softmax_outs_t * torch.log(softmax_outs_t)) / float(batch_size)
    return item1 - item2

#### second way: ratio filter
# ratios_all = torch.hstack(ratios)
# ratio_ind, ind = ratios_all.sort(descending=True)
# ratio_ind_select, ind_select = ratio_ind[:math.ceil(len(ratio_ind)*0.85)], ind[:math.ceil(len(ind)*0.85)]
# for j in range(len(ind_select)):
#     pos_element = torch.exp(similarity)[ind_select][j,:][affinity_matrix[ind_select][j, :]==1]
#     neg_element = torch.exp(similarity)[ind_select][j,:][affinity_matrix[ind_select][j, :]==-1]
#     if pos_element.sum() ==0: ## for single frame
#         continue
#     loss += -torch.log(pos_element.sum()/(pos_element.sum()+neg_element.detach().sum()))/len(ind_select)


# for i in range(seg_len):
#     prev_idx = segment_idx[i]
#     curr_idx = segment_idx[i + 1]
#     ## In segment
#     mask_segment = (segment_predict_label[0][i] == frame_predict_label).squeeze()
#     index = mask_frame[prev_idx:curr_idx] ## predict is correct with the GT
#     index_correct = mask_segment[prev_idx:curr_idx] ## predict is the segment label
#     affinity_matrix[i][prev_idx:curr_idx][index*index_correct] = 1
#     # affinity_matrix[i, :][prev_idx:curr_idx][index] = 1
#
#     ## Out segment
#     index1 = mask_frame[:prev_idx]
#     index1_correct = mask_segment[:prev_idx]
#     # affinity_matrix[i][:prev_idx][index1] = -1  ## easy negative
#     affinity_matrix[i][:prev_idx][index1*index1_correct] = -1
#     index2 = mask_frame[curr_idx:]
#     index2_correct = mask_segment[curr_idx:]
#     # affinity_matrix[i][curr_idx:][index2] = -1  ## easy negative
#     affinity_matrix[i][curr_idx:][index2*index2_correct] = -1
#
#     # #### Calculate contrastive loss:
#     if len(similarity[i][affinity_matrix[i] == 1]) == 0:
#         continue
#     # ##
#     q_s = 0.1
#     lamb = 0.025
#     posive_mean = (similarity[i][affinity_matrix[i] == 1]).mean()
#     posive_mean_score = (torch.exp(posive_mean))**q_s/q_s
#     negative_sum = (torch.exp(similarity)[i][affinity_matrix[i]==0]).sum() ## hard negative
#     negative_score = (lamb*(posive_mean_score + negative_sum))**q_s/q_s
#     l_nce = -posive_mean_score + negative_score
#     loss += 0.1*l_nce
#
#     ###
# q_s = 0.01
# lamb = 0.025
# posive_ele = (similarity_seg[i][affinity_matrix[i] == 1])#.mean()
# if len(posive_ele) == 0:
#     continue
# posive_mean = posive_ele.sort(descending=True)[0][0]
# posive_mean_score = (torch.exp(posive_mean))**q_s/q_s
# negative_sum = (torch.exp(similarity_seg)[i][affinity_matrix[i]==-1]).sum()
# negative_score = (lamb*(posive_mean_score + negative_sum))**q_s/q_s
# l_nce = -posive_mean_score + negative_score
# loss += 0.1*l_nce

# pos = torch.exp((similarity[i][affinity_matrix[i] == 1]).mean())
# neg = (torch.exp(similarity)[i][affinity_matrix[i]==0]).sum()
# loss += -0.1*torch.log(pos/(pos+neg))

#### Calculate contrastive loss:
# posive_element = torch.exp(similarity)[i][affinity_matrix[i] == 1]
# negative_element = torch.exp(similarity)[i][affinity_matrix[i]==-1]
# if len(posive_element) == 0:
#     continue
# ratio = ((similarity[i][affinity_matrix[i] == 1]).sum() /
#          (similarity[i][affinity_matrix[i] == -1]).sum()).item()
# loss += -ratio*(torch.log(posive_element.sum()/(posive_element.sum()+negative_element.sum()))/seg_len)
#
#
# def frame_segment_adaptation1(train_loader, model, num_classes, optimizer, device):
#
#     # normal_ce = nn.CrossEntropyLoss()
#     soft_ce = CrossEntropyLabelSmooth(num_classes, device)
#     transfer_loss = AlignSeg(num_classes, device)
#     transfer_loss.train()
#     epoch_loss = 0.0
#     total = 0
#     correct = 0
#     for idx, sample in enumerate(train_loader):
#         model.train()
#         x = sample['feature']
#         t = sample['label']
#         x, t = x.to(device), t.to(device)
#         mask = torch.ones(x.size(), device=device)
#         action_pred, frames_feat = model(x, mask)
#         loss = 0.0
#         for i,(p, f_feat) in enumerate(zip(action_pred, frames_feat)):
#             action_idx = torch.argmax(p, dim=1).squeeze().detach()
#             weight = 1.0 + torch.exp(-entropy(F.softmax(p, dim=1).detach()))
#             segment_feat, segment_idx, label_probility, GTlabel_list = \
#                 seg_encode(action_idx.to(device), f_feat.to(device), weight, num_classes, t)
#             if i == 0:
#                 segment_pred = model.stage1.conv_out(segment_feat.permute(0, 2, 1))
#             else:
#                 segment_pred = model.stages[i - 1].conv_out(segment_feat.permute(0, 2, 1))
#             ### segment loss
#             segment_soft_label = torch.stack(get_merge(GTlabel_list, label_probility, num_classes))
#             segment_predictions = segment_pred.transpose(2, 1).contiguous().view(-1, num_classes)
#             loss += soft_ce(segment_predictions, segment_soft_label.to(device))
#
#             ### transfer loss
#             segment_len = segment_feat.size(1)
#             for ind in range(segment_len):
#                 query_segment_feat = segment_feat[:,ind,:] ## query
#                 positive_ind = list(range(segment_idx[ind], segment_idx[ind+1]))
#                 # negative_ind = [item for item in range(segment_idx[-1]) if item not in positive_ind]
#                 # random.shuffle(positive_ind)
#                 # positive_ind_select = random.choice(positive_ind)
#                 # positive_f_feat = f_feat[:, :, positive_ind_select]
#                 # negative_f_feat = f_feat[:, :, negative_ind]
#                 # single_f_feat = f_feat[:,:, segment_idx[ind]:segment_idx[ind+1]]
#                 # loss += transfer_loss(query_segment_feat.detach(), single_f_feat, device)
#                 # loss += 0.01*transfer_loss(query_segment_feat, positive_f_feat, negative_f_feat)
#                 loss += 0.1 * transfer_loss(query_segment_feat, positive_ind, f_feat)
#             # loss += transfer_loss(segment_feat.detach(), f_feat, device)
#             # print(loss)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         _, predicted = torch.max(action_pred[-1].data, 1)
#         correct += ((predicted == t).float()*mask[:, 0, :].squeeze(1)).sum().item()
#         total += torch.sum(mask[:, 0, :]).item()
#         epoch_loss += loss / len(train_loader)
#
#     acc = float(correct)/total
#
#     return epoch_loss.item(), acc
#
#
#
#
# def refiner_train_origin(cfg, dataset, train_loader, model, backbones, backbone_names, optimizer, epoch, split_dict, device):
#     normal_ce = nn.CrossEntropyLoss()
#     total_loss = 0.0
#     for idx, sample in enumerate(train_loader):
#         model.train()
#         x = sample['feature']
#         t = sample['label']
#
#         split_idx = 0
#         for i in range(eval('cfg.num_splits["{}"]'.format(dataset))):
#             if sample['feature_path'][0].split('/')[-1].split('.')[0] in split_dict[i + 1]:
#                 split_idx = i + 1
#                 break
#         bb_key = random.choice(backbone_names)
#         curr_backbone = backbones[bb_key][split_idx]
#         curr_backbone.load_state_dict(torch.load('{}/{}/{}/split_{}/epoch-{}.model'.format(cfg.model_root,
#                                                                                            bb_key,
#                                                                                            dataset,
#                                                                                            str(i + 1),
#                                                                                            np.random.randint(10, 51))))
#         curr_backbone.to(device)
#         curr_backbone.eval()
#         x, t = x.to(device), t.to(device)
#
#         if bb_key == 'mstcn':
#             mask = torch.ones(x.size(), device=device)
#             action_pred = curr_backbone(x, mask)
#             action_idx = torch.argmax(action_pred[-1], dim=1).squeeze().detach()
#
#         elif bb_key == 'mgru':
#             action_pred = curr_backbone(x)
#             action_idx = torch.argmax(action_pred, dim=1).squeeze().detach()
#
#         elif bb_key == 'sstda':
#             mask = torch.ones(x.size(), device=device)
#             action_pred, _, _, _, _, _, _, _, _, _, _, _, _, _ = curr_backbone(x,
#                                                                                x,
#                                                                                mask,
#                                                                                mask,
#                                                                                [0, 0],
#                                                                                reverse=False)
#             action_idx = torch.argmax(action_pred[:, -1, :, :], dim=1).squeeze().detach()
#
#         elif bb_key == 'asrf':
#             out_cls, out_bound = curr_backbone(x)
#             postprocessor = PostProcessor("refinement_with_boundary", cfg.boundary_th)
#             refined_output_cls = postprocessor(out_cls.cpu().data.numpy(), boundaries=out_bound.cpu().data.numpy(),
#                                                masks=torch.ones(1, 1, x.shape[-1]).bool().data.numpy())
#             action_idx = torch.Tensor(refined_output_cls).squeeze().detach()
#
#         refine_pred, refine_rollout, GTlabel_list = model(action_idx.to(device), x, t)
#         loss = 0.0
#         loss += normal_ce(refine_pred[0], GTlabel_list.view(-1))
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         total_loss += loss / len(train_loader)
#
#     return total_loss.item()