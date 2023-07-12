"""
FileName: initial.py
Description: initialization
Time: 2020/7/30 11:48
Project: GNN_benchmark
Author: Shurui Gui
"""
import numpy as np
import torch
import random
from benchmark import TrainArgs


def init(args: TrainArgs):

    # Fix Random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Default state is a training state
    torch.enable_grad()

