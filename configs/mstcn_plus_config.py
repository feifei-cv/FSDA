import os
import csv

num_stages = 4
num_layers = 10
num_f_maps = 64
features_dim = 2048
batch_size = 1
lr = 0.0005
weight_decay = 0.00001
num_epochs=100
max_epoch = 50

dataset_root = './dataset'
model_root = './model'
best_root = './best_model'
result_root ='./result'
record_root = './record'
csv_dir='./backbones/asrf/csv'


iou_thresholds = [0.1, 0.25, 0.5]
num_layers_R = 10
num_layers_PG=11
num_R=3

num_splits = dict()
num_splits['gtea'] = 4
num_splits['50salads']=5
num_splits['breakfast']=4

dataset_names = ['gtea', '50salads', 'breakfast']
backbone_names = ['mstcn++']
best = {bn:{dn:[] for dn in dataset_names} for bn in backbone_names}

for bn in backbone_names:
    for k, v in num_splits.items():
        record_dir = os.path.join(record_root, bn, k)
        if os.path.exists(record_dir):
            define_flag = True
            for i in range(v):
                if 'split_{}_best.csv'.format(i+1) not in os.listdir(record_dir):
                    define_flag = False
            if define_flag:
                for i in range(v):
                    record_fp = os.path.join(record_dir, 'split_{}_best.csv'.format(i+1))
                    with open(record_fp, 'r') as f:
                        reader = csv.reader(f, delimiter='\t')
                        for ri, row in enumerate(reader):
                            if ri > 0:
                                best_epoch = row[0]
                        best[bn][k].append(int(best_epoch))