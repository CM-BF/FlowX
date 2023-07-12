"""
FileName: molecule_datasets.py
Description: 
Time: 2021/11/12 16:18
Project: GNN_benchmark
Author: Shurui Gui
"""
from torch_geometric.datasets import MoleculeNet
import torch
from torch.utils.data import random_split
from benchmark import data_args
from definitions import ROOT_DIR
from benchmark.kernel.utils import Metric
import os, sys


def molecule_datasets(name):
    data_args.dataset_type = 'mol'
    data_args.model_level = 'graph'

    dataset = MoleculeNet(root=os.path.join(ROOT_DIR, 'benchmark', 'datasets'), name=name)
    dataset.data.x = dataset.data.x.to(torch.float32)
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