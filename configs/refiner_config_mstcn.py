import os
import csv

batch_size = 1
in_channel = 2048
iou_thresholds = [0.1, 0.25, 0.5]
learning_rate = 1e-4 ##
max_epoch = 50
n_features = 64
n_layers = 10 #10
n_stages = 4
weight_decay= 1e-5 ##

num_stages = n_stages
num_layers = n_layers
num_f_maps = n_features
features_dim = in_channel
lr = learning_rate
num_epochs = max_epoch

dataset_root = './dataset'
model_root = './model'
best_root = './best_model'
result_root ='./result'
record_root = './record'
csv_dir='./backbones/asrf/csv'

num_splits = dict()
num_splits['gtea'] = 4
num_splits['50salads']=5
num_splits['breakfast']=4

dataset_names = ['gtea', '50salads', 'breakfast']
backbone_names = ['mstcn', 'ASFormer']
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

