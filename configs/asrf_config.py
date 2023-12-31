import os
import csv

batch_size = 1
boundary_th = 0.5
ce = True
ce_weight = 1.0
class_weight = True
csv_dir='./backbones/asrf/csv'
dampening = 0.0
dataset_root = './dataset'
model_root = './model'
result_root ='./result'
record_root = './record'
focal = False
focal_weight = 1.0
gstmse = True
gstmse_index = 'feature'
gstmse_weight = 1.0
in_channel = 2048
iou_thresholds = [0.1, 0.25, 0.5]
lambda_b = 0.1
learning_rate = 0.0005
max_epoch = 50
momentum = 0.9
n_features = 64
n_layers = 10
n_stages = 4
n_stages_asb = 4
n_stages_brb = 4
nesterov = True
param_search = False
tmse = False
tmse_weight= 0.15
tolerance= 5
weight_decay= 0.00001


num_stages = n_stages
num_layers = n_layers
num_f_maps = n_features
features_dim = in_channel
lr = learning_rate
num_epochs = max_epoch

num_splits = dict()
num_splits['gtea'] = 4
num_splits['50salads']=5
num_splits['breakfast']=4
dataset_names = ['gtea', '50salads', 'breakfast']
backbone_names = ['asrf']
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