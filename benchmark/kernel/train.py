"""
FileName: train.py
Description: batch training functions
Time: 2020/7/30 11:18
Project: GNN_benchmark
Author: Shurui Gui
"""
from torch_geometric.data.batch import Batch
import torch
from benchmark import TrainArgs, data_args
import torch.nn.functional as F
from .utils import nan2zero_get_mask
from benchmark.kernel.train_utils import TrainUtils as tr_utils
from benchmark.kernel.utils import Metric


def train_batch(model: torch.nn.Module, data: Batch, args: TrainArgs):
    data = data.to(args.device)
    mask, targets = nan2zero_get_mask(data, args)
    tr_utils.optimizer.zero_grad()

    logits: torch.tensor = model(data=data)
    loss: torch.tensor = Metric.loss_func(logits, targets, reduction='none') * mask
    loss = loss.sum() / mask.sum()
    loss.backward()
    # logger.debug(f'Loss: {loss.item():.4f}')

    tr_utils.optimizer.step()




