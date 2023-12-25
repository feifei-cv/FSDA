import os
import sys
import csv
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import random
import time

sys.path.append('./backbones/asrf')

from libs import models
from libs.optimizer import get_optimizer
from libs.loss_fn import ActionSegmentationLoss, BoundaryRegressionLoss
from libs.class_weight import get_class_weight, get_pos_weight
from libs.dataset import ActionSegmentationDataset, collate_fn
from libs.transformer import TempDownSamp, ToTensor
from libs.helper import train



from src.utils import load_meta, eval_txts, Logger
from src.predict import predict_backbone
import configs.asrf_config as cfg



os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def init_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    print('seed:', seed)


if __name__ == '__main__':


    init_seeds(seed=1)
    # cpu or cuda
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    model_name = 'asrf'

    ### log record
    logs_dir = './logs'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    mapping_file = 'logs/train_backbone_asrf_' + time.ctime()+ '.txt'
    sys.stdout = Logger(mapping_file)
    with open(mapping_file, 'a') as f:
        f.write('Begin training backbone asrf with 3090 GPU \n')

    for dataset in ['gtea', '50salads', 'breakfast']:
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

            cfg.max_epoch = 100

            train_data = ActionSegmentationDataset(
                dataset,
                transform=Compose([ToTensor(), TempDownSamp(sample_rate)]),
                mode="trainval" if not cfg.param_search else "training",
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
            )

            model = models.ActionSegmentRefinementFramework(
                in_channel=cfg.in_channel,
                n_features=cfg.n_features,
                n_classes=num_actions,
                n_stages=cfg.n_stages,
                n_layers=cfg.n_layers,
                n_stages_asb=cfg.n_stages_asb,
                n_stages_brb=cfg.n_stages_brb
            )
            model.to(device)

            # send the model to cuda/cpu
            model.to(device)

            optimizer = get_optimizer(
                'Adam',
                model,
                cfg.learning_rate,
                momentum=cfg.momentum,
                dampening=cfg.dampening,
                weight_decay=cfg.weight_decay,
                nesterov=cfg.nesterov,
            )

            if cfg.class_weight:
                class_weight = get_class_weight(
                    dataset=dataset,
                    split=split,
                    dataset_dir=cfg.dataset_root,
                    csv_dir=cfg.csv_dir,
                    mode="training" if cfg.param_search else "trainval",
                )
                class_weight = class_weight.to(device)
            else:
                class_weight = None
            print(class_weight)

            criterion_cls = ActionSegmentationLoss(
                ce=cfg.ce,
                focal=cfg.focal,
                tmse=cfg.tmse,
                gstmse=cfg.gstmse,
                weight=class_weight,
                ignore_index=255,
                ce_weight=cfg.ce_weight,
                focal_weight=cfg.focal_weight,
                tmse_weight=cfg.tmse_weight,
                gstmse_weight=cfg.gstmse,
            )

            pos_weight = get_pos_weight(
                dataset=dataset,
                split=split,
                csv_dir=cfg.csv_dir,
                mode="training" if cfg.param_search else "trainval",
            ).to(device)

            criterion_bound = BoundaryRegressionLoss(pos_weight=pos_weight)

            for epoch in range(0, cfg.max_epoch):
                # training
                train_loss = train(
                    train_loader,
                    model,
                    criterion_cls,
                    criterion_bound,
                    cfg.lambda_b,
                    optimizer,
                    epoch,
                    device,
                )
                torch.save(model.state_dict(), os.path.join(model_dir, "epoch-" + str(epoch + 1) + ".model"))
                print("epoch: {}\tlr: {:.4f}\ttrain loss: {:.4f}".format(epoch + 1, optimizer.param_groups[0]["lr"],
                                                                         train_loss))

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

            for epoch in range(1, cfg.max_epoch + 1):
                print('======================EPOCH {}====================='.format(epoch))
                predict_backbone(model_name, model, model_dir, result_dir, features_path, vid_list_file_tst,
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