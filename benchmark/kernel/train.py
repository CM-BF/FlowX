"""
FileName: train.py
Description: batch training functions
Time: 2020/7/30 11:18
Project: GNN_benchmark
Author: Shurui Gui
"""
import os

from torch_geometric.data.batch import Batch
import torch
from benchmark import TrainArgs, data_args
import torch.nn.functional as F

from definitions import ROOT_DIR
from .utils import nan2zero_get_mask
from benchmark.kernel.train_utils import TrainUtils as tr_utils
from benchmark.kernel.utils import Metric
from benchmark.models.explainers import PGExplainer, VGIB, RC_Explainer_Batch_star
from benchmark.models.explainers_backup import Gem
from pathlib import Path
from benchmark.args import gem_args


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


def dataset_method_train(explainer, args, loader, dataset, model):
    use_pred_label = args['explain'].explain_pred_label
    if isinstance(explainer, PGExplainer):
        if use_pred_label:
            train_ckpt = os.path.join(ROOT_DIR, 'pgxtmp',
                                      f'{args["explain"].dataset_name}_{args["explain"].model_name}_PL.pt')
        else:
            train_ckpt = os.path.join(ROOT_DIR, 'pgxtmp',
                                      f'{args["explain"].dataset_name}_{args["explain"].model_name}.pt')
        if not os.path.exists(train_ckpt) or gem_args.force_retrain:
            explainer.pg.train_explanation_network(loader['explain'].dataset, use_pred_label=use_pred_label)
            torch.save(explainer.state_dict(), train_ckpt)
        state_dict = torch.load(train_ckpt)
        explainer.load_state_dict(state_dict, strict=False)
    elif isinstance(explainer, Gem):
        top_k = gem_args.top_k
        threshold = None
        force_regen = gem_args.force_regen
        force_retrain = gem_args.force_retrain
        save_root = os.path.join(ROOT_DIR, 'Gem')
        dataset_name = dataset['train'].dataset.name
        save_dir = f"{dataset_name}_top_{top_k}_thres_{threshold}"
        gen_output = os.path.join(save_root, 'distillation', save_dir)
        train_output = os.path.join(save_root, 'explanation', save_dir)
        if force_regen or not os.path.exists(gen_output):
            explainer.gen_gt(dataset, model, device=data_args.device, top_k=top_k, threshold=threshold, output=gen_output)
            print('finish generation')
            force_retrain = True
        train_args = explainer.train_args[data_args.model_level]
        train_args.distillation = gen_output
        train_args.output = train_output
        train_args.dataset = dataset_name
        if force_retrain or not os.path.exists(train_output):
            explainer.train_vae(train_args)
            print('finish training explainer')
    elif isinstance(explainer, VGIB):
        explainer.fit_a_single_model(dataset['test'], use_pred_label=True)
    elif isinstance(explainer, RC_Explainer_Batch_star):
        train_ckpt_path = Path(ROOT_DIR) / 'checkpoints' / 'RC_explainer' / args["explain"].dataset_name / f'{args["explain"].model_name}_PL.pt'
        # whether the file's parent directory exists
        if not train_ckpt_path.parent.exists():
            train_ckpt_path.parent.mkdir(parents=True)
        if train_ckpt_path.exists() and not gem_args.force_retrain:
            explainer.load_policy_net(path=train_ckpt_path)
        else:
            explainer.train_policy(loader['train'], loader['explain'], save_model_path=train_ckpt_path)
