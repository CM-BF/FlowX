"""
FileName: train_utils.py
Description: Training tools we use to optimize models
Time: 2020/7/30 15:40
Project: GNN_benchmark
Author: Shurui Gui
"""


import torch
from benchmark import TrainArgs

class TrainUtils(object):
    optimizer: torch.optim.Adam = None
    scheduler: torch.optim.lr_scheduler._LRScheduler = None


def set_train_utils(model: torch.nn.Module, args: TrainArgs):
    TrainUtils.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    TrainUtils.scheduler = torch.optim.lr_scheduler.MultiStepLR(TrainUtils.optimizer, milestones=args.mile_stones, gamma=0.1)

