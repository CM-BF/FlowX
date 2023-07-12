"""
FileName: ba_shapes.py
Description: 
Time: 2021/11/12 16:20
Project: GNN_benchmark
Author: Shurui Gui
"""
import torch
from torch.utils.data import random_split
from benchmark import data_args
from definitions import ROOT_DIR
from benchmark.kernel.utils import Metric
from benchmark.data.dataset_gen import BA_Shapes
import os, sys
import copy


def ba_shapes(name):
    data_args.dataset_type = 'syn'
    data_args.model_level = 'node'

    dataset = BA_Shapes(root=os.path.join(ROOT_DIR, 'benchmark', 'datasets', name),
                        num_base_node=300, num_shape=80)
    dataset.data.x = dataset.data.x.to(torch.float32)
    data_args.dim_node = dataset.num_node_features
    data_args.dim_edge = dataset.num_edge_features
    data_args.num_targets = 1

    # Define models' output shape.
    if Metric.cur_task == 'bcs':
        data_args.num_classes = 2
    elif Metric.cur_task == 'reg':
        data_args.num_classes = 1
    else:
        data_args.num_classes = dataset.num_classes

    assert data_args.target_idx != -1, 'Explaining on multi tasks is meaningless.'
    if data_args.model_level != 'node':

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
    else:
        # --- clear buffer dataset._data_list ---
        dataset._data_list = None
        data_args.num_targets = 1
        train_set = dataset
        val_set = copy.deepcopy(dataset)
        test_set = copy.deepcopy(dataset)
        train_set.data.mask = train_set.data.train_mask
        train_set.slices['mask'] = train_set.slices['train_mask']
        val_set.data.mask = val_set.data.val_mask
        val_set.slices['mask'] = val_set.slices['val_mask']
        test_set.data.mask = test_set.data.test_mask
        test_set.slices['mask'] = test_set.slices['test_mask']
        return {'train': train_set, 'val': val_set, 'test': test_set}