import csv
import sys
import os
import numpy as np
import torch
import random
import time

sys.path.append('./backbones/ms-tcn')
from batch_gen import BatchGenerator
from model import MultiStageModel, Trainer

from src.utils import load_meta, eval_txts, Logger
from src.predict import predict_backbone
import configs.mstcn_config as cfg

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

if __name__ == '__main__':

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    device = 'cuda'
    model_name = 'mstcn'

    ### log record
    logs_dir = './logs'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    mapping_file = 'logs/train_backbone_mstcn_' + time.ctime()+ '.txt'
    sys.stdout = Logger(mapping_file)
    with open(mapping_file, 'a') as f:
        f.write('Begin training backbone MS-TCN with 3090 GPU \n')

    for dataset in ['gtea', '50salads', 'breakfast']: #
        for split in ([1, 2, 3, 4, 5]):
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
            record_dir = load_meta(cfg.dataset_root, cfg.model_root, cfg.result_root, cfg.record_root, dataset, split,
                                   model_name)

            cfg.num_epochs = 100
            if dataset == "gtea":
                cfg.num_epochs = 60

            batch_gen = BatchGenerator(num_actions, actions_dict, gt_path, features_path, sample_rate)
            batch_gen.read_data(vid_list_file)

            trainer = Trainer(cfg.num_stages, cfg.num_layers, cfg.num_f_maps, cfg.features_dim, num_actions)

            trainer.train(save_dir=model_dir,
                          batch_gen=batch_gen,
                          num_epochs=cfg.num_epochs,
                          batch_size=cfg.batch_size,
                          learning_rate=cfg.lr,
                          device=device)

            ## saving result
            max_epoch = -1
            max_val = 0.0
            max_results = dict()
            f = open(os.path.join(record_dir, 'split_{}_all.csv'.format(split)), 'w')
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['epoch', 'accu', 'edit',
                             'F1@{}'.format(cfg.iou_thresholds[0]),
                             'F1@{}'.format(cfg.iou_thresholds[1]),
                             'F1@{}'.format(cfg.iou_thresholds[2])])
            for epoch in range(1, cfg.num_epochs + 1):
                print('======================EPOCH {}====================='.format(epoch))
                predict_backbone(model_name, trainer.model, model_dir, result_dir, features_path, vid_list_file_tst,
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

            ## saving best result
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