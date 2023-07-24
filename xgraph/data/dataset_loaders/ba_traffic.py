
import torch
from torch.utils.data import random_split
from xgraph import data_args
from xgraph.definitions import ROOT_DIR
from xgraph.kernel.utils import Metric
from xgraph.data.dataset_gen import BA_Traffic
import os

def ba_traffic(name):
    data_args.dataset_type = 'syn'
    data_args.model_level = 'graph'

    dataset = BA_Traffic(root=os.path.join(ROOT_DIR, 'xgraph', 'datasets'),
                         name=name,
                         num_per_class=1000)
    # --- dataset._data_list is the data buffer, you have to clear it after changing data ---
    dataset.data.x = dataset.data.x.to(torch.float32)
    # --- the code above write dataset[0] to dataset._data_list automatically.
    # Indeed it can be considered as a kind of bug ---
    data_args.dim_node = dataset.num_node_features
    data_args.dim_edge = dataset.num_edge_features
    data_args.num_targets = dataset.num_classes  # This so-called classes are actually targets.

    # Define models' output shape.
    if Metric.cur_task == 'bcs':
        data_args.num_classes = 2
    elif Metric.cur_task == 'reg':
        data_args.num_classes = 1

    assert data_args.target_idx != -1, 'Explaining on multi tasks is meaningless.'
    assert data_args.target_idx <= dataset.data.y.shape[1], 'No such target in the dataset.'

    dataset.data.y = dataset.data.y[:, data_args.target_idx]
    # --- clear buffer dataset._data_list ---
    dataset._data_list = None
    data_args.num_targets = 1

    dataset_len = len(dataset)
    dataset_split = [int(dataset_len * data_args.dataset_split[0]),
                     int(dataset_len * data_args.dataset_split[1]),
                     0]
    dataset_split[2] = dataset_len - dataset_split[0] - dataset_split[1]
    train_set, val_set, test_set = \
        random_split(dataset, dataset_split)

    return {'train': train_set, 'val': val_set, 'test': test_set}
