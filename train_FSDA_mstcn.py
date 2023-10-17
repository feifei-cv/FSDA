import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import random
import csv
import sys
import os
import time

sys.path.append('./backbones')
sys.path.append('./backbones/ms-tcn')
from model import MultiStageModel, SingleStageModel

from libs.dataset import ActionSegmentationDataset, collate_fn
from libs.transformer import TempDownSamp, ToTensor
from src.utils import eval_txts, load_meta, Logger
from src.predict import predict_refiner
from src.refiner_train import frame_segment_adaptation_tcn
from src.refiner_model import RefineAction
import configs.refiner_config_mstcn as cfg



os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def init_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('seed:', seed)



if __name__ == '__main__':

    init_seeds(seed=1)
    device = 'cuda'
    backbone_name = 'mstcn'
    model_name = 'FSDA' + '-' + '-'.join([backbone_name])

    ### log record
    logs_dir = './logs'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    mapping_file = 'logs/train_FSDA_mstcn_' + time.ctime() + '.txt'
    with open(mapping_file, 'a') as f:
        f.write('Begin training FSDA with 3090 GPU \n')
    sys.stdout = Logger(mapping_file)

    for dataset in ['gtea','50salads']: ##
    # for dataset in ['breakfast']:  ##
        for split in ([1, 2, 3, 4, 5]):#
            if split == 5 and dataset != '50salads':
                continue
            print(dataset, split)

            actions_dict, \
            num_actions, \
            gt_path, \
            features_path, \
            vid_list_file, \
            vid_list_file_tst, \
            sample_rate, \
            model_dir, \
            result_dir, \
            record_dir = load_meta(cfg.dataset_root, cfg.model_root, cfg.result_root, cfg.record_root,
                                   dataset, split, model_name)
            train_data = ActionSegmentationDataset(
                dataset,
                transform=Compose([ToTensor(), TempDownSamp(sample_rate)]),
                mode="trainval",
                split=split,
                dataset_dir=cfg.dataset_root,
                csv_dir=cfg.csv_dir,
            )
            train_loader = DataLoader(
                train_data,
                batch_size=cfg.batch_size,
                shuffle=True,
                drop_last=True if cfg.batch_size > 1 else False,
                collate_fn=collate_fn,
                pin_memory=True
            )

            if dataset == 'breakfast':
                cfg.lr = 1e-5
                cfg.weight_decay = 1e-5
                cfg.max_epoch = 30

            curr_model = MultiStageModel(cfg.num_stages,
                                         num_layers=cfg.num_layers,
                                         num_f_maps=cfg.num_f_maps,
                                         dim=cfg.features_dim,
                                         num_classes=num_actions)
            model_pt = os.path.join(cfg.model_root, 'mstcn', dataset,'split_{}'.format(split),
                                    'epoch-{}.model'.format(cfg.best['mstcn'][dataset][split-1]))

            print(model_pt)
            curr_model.load_state_dict(torch.load(model_pt))
            curr_model.to(device)  ### backbone

            refine_net = RefineAction(num_layers=cfg.num_layers,
                                         num_f_maps=cfg.num_f_maps,
                                         dim=num_actions,
                                         num_classes=num_actions)
            refine_net.to(device)  ### refine net

            optimizer = torch.optim.Adam(curr_model.parameters(),  lr=cfg.lr, weight_decay=cfg.weight_decay)
            optimizer_refine = torch.optim.Adam(refine_net.parameters(),  lr=cfg.lr*3, weight_decay=cfg.weight_decay)

            for epoch in range(cfg.max_epoch):
                train_loss, acc = frame_segment_adaptation_tcn(train_loader, curr_model, num_actions,
                                                           optimizer, optimizer_refine, refine_net, device)
                torch.save(curr_model.state_dict(), os.path.join(model_dir, "epoch-" + str(epoch + 1) + ".model"))
                torch.save(refine_net.state_dict(), os.path.join(model_dir, "epoch-" + str(epoch + 1) + ".opt"))
                print("[epoch %d]: lr = %f,  epoch loss = %f,   acc = %f" % (epoch + 1, optimizer.param_groups[0]["lr"], train_loss, acc))

            ##saving
            max_epoch = -1
            max_val = 0.0
            max_results = dict()
            f = open(os.path.join(record_dir, 'split_{}_all.csv'.format(split)), 'w')
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['epoch', 'accu', 'edit',
                             'F1@{}'.format(cfg.iou_thresholds[0]),
                             'F1@{}'.format(cfg.iou_thresholds[1]),
                             'F1@{}'.format(cfg.iou_thresholds[2])])
            for epoch in range(1, cfg.max_epoch + 1):
                print('======================EPOCH {}====================='.format(epoch))
                predict_refiner(curr_model, refine_net, backbone_name, model_dir, result_dir,
                                features_path, vid_list_file_tst,
                                epoch, actions_dict, device, sample_rate)
                results = eval_txts(cfg.dataset_root, result_dir, dataset, split, model_name)
                writer.writerow([epoch, '%.4f' % (results['accu']), '%.4f' % (results['edit']),
                                 '%.4f' % (results['F1@%0.2f' % (cfg.iou_thresholds[0])]),
                                 '%.4f' % (results['F1@%0.2f' % (cfg.iou_thresholds[1])]),
                                 '%.4f' % (results['F1@%0.2f' % (cfg.iou_thresholds[2])])])
                curr_val = sum([results[k] for k in results.keys()])
                max_val = max(max_val, curr_val)

                if curr_val == max_val:
                    max_epoch = epoch
                    max_results = results
            print('EARNED MAXIMUM PERFORMANCE IN EPOCH {}'.format(max_epoch))
            print(max_results)
            f.close()

            ##saving best
            f = open(os.path.join(record_dir, 'split_{}_best.csv'.format(split)), 'w')
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