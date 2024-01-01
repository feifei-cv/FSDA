import csv
import sys
import os
import numpy as np
import torch
import random
import time
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from scipy.ndimage import median_filter
import torch.nn.functional as F

sys.path.append('./backbones/DiffAct')
from backbones.DiffAct.utils import load_config_file, func_eval, get_labels_start_end_time, mode_filter
from backbones.DiffAct.dataset import get_data_dict, VideoFeatureDataset, restore_full_sequence
from backbones.DiffAct.model import ASDiffusionModel

from src.utils import load_meta, Logger
import configs.DiffAct_config as cfg
from src.refiner_model import RefineAction
from src.refiner_train import frame_segment_adaptation_DiffAct

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class Trainer:
    def __init__(self, encoder_params, decoder_params, diffusion_params, event_list, sample_rate, temporal_aug,
                 set_sampling_seed, postprocess, device):

        self.device = device
        self.num_classes = len(event_list)
        self.encoder_params = encoder_params
        self.decoder_params = decoder_params
        self.event_list = event_list
        self.sample_rate = sample_rate
        self.temporal_aug = temporal_aug
        self.set_sampling_seed = set_sampling_seed
        self.postprocess = postprocess

        self.model = ASDiffusionModel(encoder_params, decoder_params, diffusion_params, self.num_classes, self.device)
        print('Model Size: ', sum(p.numel() for p in self.model.parameters()))

    def train(self, train_train_dataset, loss_weights, class_weighting, soft_label, num_epochs, batch_size,
              learning_rate, weight_decay, result_dir, save_dir, log_freq):

        device = self.device
        self.model.to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        optimizer.zero_grad()
        restore_epoch = -1
        step = 1
        if os.path.exists(result_dir):
            if 'latest.pt' in os.listdir(result_dir):
                if os.path.getsize(os.path.join(result_dir, 'latest.pt')) > 0:
                    saved_state = torch.load(os.path.join(result_dir, 'latest.pt'))
                    self.model.load_state_dict(saved_state['model'])
                    optimizer.load_state_dict(saved_state['optimizer'])
                    restore_epoch = saved_state['epoch']
                    step = saved_state['step']

        if class_weighting:
            class_weights = train_train_dataset.get_class_weights()
            class_weights = torch.from_numpy(class_weights).float().to(device)
            ce_criterion = nn.CrossEntropyLoss(ignore_index=-100, weight=class_weights, reduction='none')
        else:
            ce_criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

        bce_criterion = nn.BCELoss(reduction='none')
        mse_criterion = nn.MSELoss(reduction='none')

        train_train_loader = torch.utils.data.DataLoader(
            train_train_dataset, batch_size=1, shuffle=True, num_workers=4)

        if result_dir:
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            # logger = SummaryWriter(result_dir)

        for epoch in range(restore_epoch + 1, num_epochs):

            self.model.train()

            epoch_running_loss = 0

            for _, data in enumerate(train_train_loader):

                feature, label, boundary, video = data
                feature, label, boundary = feature.to(device), label.to(device), boundary.to(device)

                loss_dict = self.model.get_training_loss(
                     feature, event_gt=F.one_hot(label.long(), num_classes=self.num_classes).permute(0, 2, 1),
                     boundary_gt=boundary,
                     encoder_ce_criterion=ce_criterion,
                     encoder_mse_criterion=mse_criterion,
                     encoder_boundary_criterion=bce_criterion,
                     decoder_ce_criterion=ce_criterion,
                     decoder_mse_criterion=mse_criterion,
                     decoder_boundary_criterion=bce_criterion,
                     soft_label=soft_label)
                total_loss = 0
                for k, v in loss_dict.items():
                    total_loss += loss_weights[k] * v
                total_loss /= batch_size
                total_loss.backward()
                epoch_running_loss += total_loss.item()
                if step % batch_size == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                step += 1

            epoch_running_loss /= len(train_train_dataset)
            print(f'Epoch {epoch} - Running Loss {epoch_running_loss}')
            if epoch % log_freq == 0:
                torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch) + ".model")

    def test_single_video(self, video_idx, test_dataset, mode, device, model_path=None):

        assert (test_dataset.mode == 'test')
        assert (mode in ['encoder', 'decoder-noagg', 'decoder-agg'])
        assert (self.postprocess['type'] in ['median', 'mode', 'purge', None])

        self.model.eval()
        self.model.to(device)

        if model_path:
            self.model.load_state_dict(torch.load(model_path))

        if self.set_sampling_seed:
            seed = video_idx
        else:
            seed = None

        with torch.no_grad():

            feature, label, _, video = test_dataset[video_idx]

            # feature:   [torch.Size([1, F, Sampled T])]
            # label:     torch.Size([1, Original T])
            # output: [torch.Size([1, C, Sampled T])]

            if mode == 'encoder':
                output = [self.model.encoder(feature[i].to(device))
                          for i in range(len(feature))]  # output is a list of tuples
                output = [F.softmax(i, 1).cpu() for i in output]
                left_offset = self.sample_rate // 2
                right_offset = (self.sample_rate - 1) // 2

            if mode == 'decoder-agg':
                output = [self.model.ddim_sample(feature[i].to(device), seed)
                          for i in range(len(feature))]  # output is a list of tuples
                output = [i.cpu() for i in output]
                left_offset = self.sample_rate // 2
                right_offset = (self.sample_rate - 1) // 2

            if mode == 'decoder-noagg':  # temporal aug must be true
                output = [
                    self.model.ddim_sample(feature[len(feature) // 2].to(device), seed)]  # output is a list of tuples
                output = [i.cpu() for i in output]
                left_offset = self.sample_rate // 2
                right_offset = 0

            assert (output[0].shape[0] == 1)

            min_len = min([i.shape[2] for i in output])
            output = [i[:, :, :min_len] for i in output]
            output = torch.cat(output, 0)  # torch.Size([sample_rate, C, T])
            output = output.mean(0).numpy()

            if self.postprocess['type'] == 'median':  # before restoring full sequence
                smoothed_output = np.zeros_like(output)
                for c in range(output.shape[0]):
                    smoothed_output[c] = median_filter(output[c], size=self.postprocess['value'])
                output = smoothed_output / smoothed_output.sum(0, keepdims=True)

            output = np.argmax(output, 0)

            output = restore_full_sequence(output,
                                           full_len=label.shape[-1],
                                           left_offset=left_offset,
                                           right_offset=right_offset,
                                           sample_rate=self.sample_rate
                                           )

            if self.postprocess['type'] == 'mode':  # after restoring full sequence
                output = mode_filter(output, self.postprocess['value'])

            if self.postprocess['type'] == 'purge':

                trans, starts, ends = get_labels_start_end_time(output)

                for e in range(0, len(trans)):
                    duration = ends[e] - starts[e]
                    if duration <= self.postprocess['value']:

                        if e == 0:
                            output[starts[e]:ends[e]] = trans[e + 1]
                        elif e == len(trans) - 1:
                            output[starts[e]:ends[e]] = trans[e - 1]
                        else:
                            mid = starts[e] + duration // 2
                            output[starts[e]:mid] = trans[e - 1]
                            output[mid:ends[e]] = trans[e + 1]

            label = label.squeeze(0).cpu().numpy()

            assert (output.shape == label.shape)

            return video, output, label

    def test_single_video_f(self, video_idx, test_dataset, mode, refine_net, device, model_path=None):

        assert (test_dataset.mode == 'test')
        assert (mode in ['encoder', 'decoder-noagg', 'decoder-agg'])
        assert (self.postprocess['type'] in ['median', 'mode', 'purge', None])

        self.model.eval()
        self.model.to(device)

        if model_path:
            self.model.load_state_dict(torch.load(model_path))

        if self.set_sampling_seed:
            seed = video_idx
        else:
            seed = None

        with torch.no_grad():

            feature, label, _, video = test_dataset[video_idx]

            # feature:   [torch.Size([1, F, Sampled T])]
            # label:     torch.Size([1, Original T])
            # output: [torch.Size([1, C, Sampled T])]

            if mode == 'encoder':
                output = [self.model.encoder(feature[i].to(device))
                          for i in range(len(feature))]  # output is a list of tuples
                output = [F.softmax(i, 1).cpu() for i in output]
                left_offset = self.sample_rate // 2
                right_offset = (self.sample_rate - 1) // 2

            if mode == 'decoder-agg':
                output = [F.softmax(refine_net(self.model.ddim_sample(feature[i].to(device), seed)), dim=1)
                          for i in range(len(feature))]  # output is a list of tuples
                output = [i.cpu() for i in output]
                left_offset = self.sample_rate // 2
                right_offset = (self.sample_rate - 1) // 2

            if mode == 'decoder-noagg':  # temporal aug must be true
                output = [
                    self.model.ddim_sample(feature[len(feature) // 2].to(device), seed)]  # output is a list of tuples
                output = [i.cpu() for i in output]
                left_offset = self.sample_rate // 2
                right_offset = 0

            assert (output[0].shape[0] == 1)

            min_len = min([i.shape[2] for i in output])
            output = [i[:, :, :min_len] for i in output]
            output = torch.cat(output, 0)  # torch.Size([sample_rate, C, T])
            output = output.mean(0).numpy()

            if self.postprocess['type'] == 'median':  # before restoring full sequence
                smoothed_output = np.zeros_like(output)
                for c in range(output.shape[0]):
                    smoothed_output[c] = median_filter(output[c], size=self.postprocess['value'])
                output = smoothed_output / smoothed_output.sum(0, keepdims=True)

            output = np.argmax(output, 0)

            output = restore_full_sequence(output,
                                           full_len=label.shape[-1],
                                           left_offset=left_offset,
                                           right_offset=right_offset,
                                           sample_rate=self.sample_rate
                                           )

            if self.postprocess['type'] == 'mode':  # after restoring full sequence
                output = mode_filter(output, self.postprocess['value'])

            if self.postprocess['type'] == 'purge':

                trans, starts, ends = get_labels_start_end_time(output)

                for e in range(0, len(trans)):
                    duration = ends[e] - starts[e]
                    if duration <= self.postprocess['value']:

                        if e == 0:
                            output[starts[e]:ends[e]] = trans[e + 1]
                        elif e == len(trans) - 1:
                            output[starts[e]:ends[e]] = trans[e - 1]
                        else:
                            mid = starts[e] + duration // 2
                            output[starts[e]:mid] = trans[e - 1]
                            output[mid:ends[e]] = trans[e + 1]

            label = label.squeeze(0).cpu().numpy()

            assert (output.shape == label.shape)

            return video, output, label

    def test(self, test_dataset, mode, device, label_dir, result_dir=None, model_path=None):

        assert (test_dataset.mode == 'test')

        self.model.eval()
        self.model.to(device)

        if model_path:
            self.model.load_state_dict(torch.load(model_path))

        with torch.no_grad():

            for video_idx in tqdm(range(len(test_dataset))):

                video, pred, label = self.test_single_video(
                    video_idx, test_dataset, mode, device, model_path)

                pred = [self.event_list[int(i)] for i in pred]

                if not os.path.exists(os.path.join(result_dir, 'prediction')):
                    os.makedirs(os.path.join(result_dir, 'prediction'))

                file_name = os.path.join(result_dir, 'prediction', f'{video}.txt')
                file_ptr = open(file_name, 'w')
                file_ptr.write('### Frame level recognition: ###\n')
                file_ptr.write(' '.join(pred))
                file_ptr.close()

        acc, edit, f1s = func_eval(label_dir, os.path.join(result_dir, 'prediction'), test_dataset.video_list)

        print("Acc: %.4f" % (acc))
        print('Edit: %.4f' % (edit))
        print('F1@0.10: %.4f' % (f1s[0]))
        print('F1@0.25: %.4f' % (f1s[1]))
        print('F1@0.50: %.4f' % (f1s[2]))

        result_dict = {
            'accu': acc,
            'edit': edit,
            'F1@0.10': f1s[0],
            'F1@0.25': f1s[1],
            'F1@0.50': f1s[2]
        }

        return result_dict


    def refine_predict(self, test_dataset, mode, refine_net, device, label_dir, result_dir, model_path, refine_path):

        assert (test_dataset.mode == 'test')

        self.model.eval()
        self.model.to(device)
        refine_net.eval()

        self.model.load_state_dict(torch.load(model_path))
        refine_net.load_state_dict(torch.load(refine_path))

        with torch.no_grad():

            for video_idx in tqdm(range(len(test_dataset))):

                video, pred, label = self.test_single_video_f(
                    video_idx, test_dataset, mode, refine_net, device, model_path)

                pred = [self.event_list[int(i)] for i in pred]

                if not os.path.exists(os.path.join(result_dir, 'prediction')):
                    os.makedirs(os.path.join(result_dir, 'prediction'))

                file_name = os.path.join(result_dir, 'prediction', f'{video}.txt')
                file_ptr = open(file_name, 'w')
                file_ptr.write('### Frame level recognition: ###\n')
                file_ptr.write(' '.join(pred))
                file_ptr.close()

        acc, edit, f1s = func_eval(label_dir, os.path.join(result_dir, 'prediction'), test_dataset.video_list)

        print("Acc: %.4f" % (acc))
        print('Edit: %.4f' % (edit))
        print('F1@0.10: %.4f' % (f1s[0]))
        print('F1@0.25: %.4f' % (f1s[1]))
        print('F1@0.50: %.4f' % (f1s[2]))

        result_dict = {
            'accu': acc,
            'edit': edit,
            'F1@0.10': f1s[0],
            'F1@0.25': f1s[1],
            'F1@0.50': f1s[2]
        }

        return result_dict


if __name__ == '__main__':

    # init_seeds(seed=1)
    device = 'cuda'
    backbone_name = 'DiffAct'
    model_name = 'FSDA' + '-' + '-'.join([backbone_name])
    ### log record
    logs_dir = './logs'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    mapping_file = 'logs/train_FSDA_DiffAct_' + time.ctime() + '.txt'
    sys.stdout = Logger(mapping_file)
    with open(mapping_file, 'a') as f:
        f.write('Begin training FSDA DiffAct with 3090 GPU \n')

    for dataset in ['gtea', '50salads', 'breakfast']:
        for spl in ([1, 2, 3, 4, 5]):
            if spl == 5 and dataset != '50salads':
                continue
            print(dataset, spl)

            config_file = 'configs/' + dataset + '-S' + str(spl) + '.json'
            print('config_file:', config_file)
            all_params = load_config_file(config_file)
            locals().update(all_params)

            actions_dict, \
            num_actions, \
            gt_path, \
            features_path, \
            vid_list_file, \
            vid_list_file_tst, \
            rate, \
            model_dir, \
            result_dir, \
            record_dir = load_meta(cfg.dataset_root, cfg.model_root, cfg.result_root, cfg.record_root, dataset, spl,
                                   model_name)

            mapping_file = os.path.join(cfg.dataset_root, dataset, 'mapping.txt')
            event_list = np.loadtxt(mapping_file, dtype=str)
            event_list = [i[1] for i in event_list]
            num_classes = len(event_list)

            train_video_list = np.loadtxt(vid_list_file, dtype=str)
            test_video_list = np.loadtxt(vid_list_file_tst, dtype=str)

            train_video_list = [i.split('.')[0] for i in train_video_list]
            test_video_list = [i.split('.')[0] for i in test_video_list]

            train_data_dict = get_data_dict(
                feature_dir=features_path,
                label_dir=gt_path,
                video_list=train_video_list,
                event_list=event_list,
                sample_rate=sample_rate,
                temporal_aug=temporal_aug,
                boundary_smooth=boundary_smooth
            )

            test_data_dict = get_data_dict(
                feature_dir=features_path,
                label_dir=gt_path,
                video_list=test_video_list,
                event_list=event_list,
                sample_rate=sample_rate,
                temporal_aug=temporal_aug,
                boundary_smooth=boundary_smooth
            )

            train_train_dataset = VideoFeatureDataset(train_data_dict, num_classes, mode='train')
            test_test_dataset = VideoFeatureDataset(test_data_dict, num_classes, mode='test')

            trainer = Trainer(dict(encoder_params), dict(decoder_params), dict(diffusion_params), event_list,
                              sample_rate, temporal_aug, set_sampling_seed, postprocess, device)

            ### load backbone model
            curr_model = trainer.model
            model_pt = os.path.join(model_dir.replace(model_name, backbone_name)+'-1', 'release.model')
            print('model_pt:', model_pt)
            curr_model.load_state_dict(torch.load(model_pt))
            curr_model.to(device)  ### backbone
            refine_net = RefineAction(num_layers=encoder_params['num_layers'],
                                         num_f_maps=encoder_params['num_f_maps'],
                                         dim=num_actions,
                                         num_classes=num_actions)
            refine_net.to(device)  ### refine net

            learning_rate = 0.0001
            # learning_rate = 0.00001 ## for 'breakfast'
            weight_decay = 5e-6
            optimizer = torch.optim.Adam(curr_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            optimizer_refine = torch.optim.Adam(refine_net.parameters(), lr=learning_rate * 3, weight_decay=weight_decay)

            ###
            num_epochs = 100
            log_freq = 1
            for epoch in range(num_epochs):
                train_loss, acc = frame_segment_adaptation_DiffAct(train_train_dataset, curr_model, num_actions,
                                                                   optimizer, optimizer_refine, refine_net, device)
                print("[epoch %d]: lr = %f,  epoch loss = %f,   acc = %f" % (
                                                        epoch + 1, optimizer.param_groups[0]["lr"], train_loss, acc))
                if epoch % log_freq == 0:
                    torch.save(curr_model.state_dict(), os.path.join(model_dir, "epoch-" + str(epoch) + ".model"))
                    torch.save(refine_net.state_dict(), os.path.join(model_dir, "epoch-" + str(epoch) + ".opt"))

            #### saving result
            max_epoch = -1
            max_val = 0.0
            max_results = dict()
            f = open(os.path.join(record_dir, 'split_{}_all.csv'.format(spl)), 'w')
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['epoch', 'accu', 'edit',
                             'F1@{}'.format(cfg.iou_thresholds[0]),
                             'F1@{}'.format(cfg.iou_thresholds[1]),
                             'F1@{}'.format(cfg.iou_thresholds[2])])

            for epoch in range(num_epochs):
                if epoch % log_freq == 0:
                    model_path = os.path.join(model_dir, 'epoch-'+str(epoch)+'.model')
                    refine_path = os.path.join(model_dir, 'epoch-' + str(epoch) + '.opt')
                    print('======================EPOCH {}====================='.format(epoch))
                    for mode in ['decoder-agg']:  # Default: decoder-agg. The results of decoder-noagg are similar
                        results = trainer.refine_predict(test_test_dataset, mode, refine_net, device, gt_path,
                                                         result_dir, model_path, refine_path)
                    writer.writerow([epoch, '%.4f' % (results['accu']), '%.4f' % (results['edit']),
                                     '%.4f' % (results['F1@%0.2f' % (cfg.iou_thresholds[0])]),
                                     '%.4f' % (results['F1@%0.2f' % (cfg.iou_thresholds[1])]),
                                     '%.4f' % (results['F1@%0.2f' % (cfg.iou_thresholds[2])])])

                    curr_val = sum([results[k] for k in results.keys()])
                    max_val = max(max_val, curr_val)

                    if curr_val == max_val:
                        max_epoch = epoch
                        max_results = results

            # ##### test release model
            # for epoch in range(1):
            #     if epoch % log_freq == 0:
            #         model_path = os.path.join(model_dir + '/release.model')
            #         refine_path = os.path.join(model_dir + '/release.opt')
            #         print('======================EPOCH {}====================='.format(epoch))
            #         for mode in ['decoder-agg']:  # Default: decoder-agg. The results of decoder-noagg are similar
            #             results = trainer.refine_predict(test_test_dataset, mode, refine_net, device, gt_path,
            #                                              result_dir, model_path, refine_path)
            #         writer.writerow([epoch, '%.4f' % (results['accu']), '%.4f' % (results['edit']),
            #                          '%.4f' % (results['F1@%0.2f' % (cfg.iou_thresholds[0])]),
            #                          '%.4f' % (results['F1@%0.2f' % (cfg.iou_thresholds[1])]),
            #                          '%.4f' % (results['F1@%0.2f' % (cfg.iou_thresholds[2])])])
            #
            #         curr_val = sum([results[k] for k in results.keys()])
            #         max_val = max(max_val, curr_val)
            #
            #         if curr_val == max_val:
            #             max_epoch = epoch
            #             max_results = results


            print('EARNED MAXIMUM PERFORMANCE IN EPOCH {}'.format(max_epoch))
            print(max_results)
            f.close()

            ## saving best result
            f = open(os.path.join(record_dir, 'split_{}_best.csv'.format(spl)), 'w')
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['epoch', 'accu', 'edit',
                             'F1@{}'.format(cfg.iou_thresholds[0]),
                             'F1@{}'.format(cfg.iou_thresholds[1]),
                             'F1@{}'.format(cfg.iou_thresholds[2])])
            writer.writerow([max_epoch, '%.4f' % (max_results['accu']), '%.4f' % (max_results['edit']),
                             '%.4f' % (max_results['F1@%0.2f' % (cfg.iou_thresholds[0])]),
                             '%.4f' % (max_results['F1@%0.2f' % (cfg.iou_thresholds[1])]),
                             '%.4f' % (max_results['F1@%0.2f' % (cfg.iou_thresholds[2])])])
            f.close()

