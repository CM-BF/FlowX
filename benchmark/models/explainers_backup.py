"""
FileName: explainers_backup.py
Description: Explainable methods' set
Time: 2020/8/4 8:56
Project: GNN_benchmark
Author: Shurui Gui
"""
import os
from typing import Any, Callable, List, Tuple, Union, Dict, Sequence

from math import sqrt

import torch
from torch import Tensor
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.nn import MessagePassing
from torch_geometric.utils.loop import add_self_loops, remove_self_loops
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_networkx
from benchmark.models.utils import subgraph, normalize
from torch.nn.functional import binary_cross_entropy as bceloss
from typing_extensions import Literal
from benchmark.kernel.utils import Metric
from benchmark import data_args
from benchmark.args import x_args, fs_args
from definitions import ROOT_DIR
from rdkit import Chem
from matplotlib.axes import Axes
from matplotlib.patches import Path, PathPatch


import captum
import captum.attr as ca
from captum.attr._utils.typing import (
    BaselineType,
    Literal,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)
from captum.attr._core.deep_lift import DeepLiftShap
from captum.attr._utils.attribution import GradientAttribution, LayerAttribution
from captum.attr._utils.common import (
    ExpansionTypes,
    _call_custom_attribution_func,
    _compute_conv_delta_and_format_attrs,
    _expand_additional_forward_args,
    _expand_target,
    _format_additional_forward_args,
    _format_attributions,
    _format_baseline,
    _format_callable_baseline,
    _format_input,
    _tensorize_baseline,
    _validate_input,
)
from captum.attr._utils.gradient import (
    apply_gradient_requirements,
    compute_layer_gradients_and_eval,
    undo_gradient_requirements,
)
import benchmark.models.gradient_utils as gu

from itertools import combinations
import numpy as np
from benchmark.models.models import GlobalMeanPool, GraphSequential, GNNPool
from benchmark.models.ext.deeplift.layer_deep_lift import LayerDeepLift, DeepLift
from benchmark.models.ext.lrp.lrp import LRP_gamma
from benchmark.models.ext.pgexplainer.pgexplainer import PGExplainer as pgex
from benchmark.models.utils import gumbel_softmax
import benchmark.models.ext.PGM_Graph.pgm_explainer_graph as pgmg
import benchmark.models.ext.PGM_Node.Explain_GNN.pgm_explainer as pgmn
import shap
import time

EPS = 1e-15


class ExplainerBase(nn.Module):

    def __init__(self, model: nn.Module, epochs=0, lr=0, explain_graph=False, molecule=False):
        super().__init__()
        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.explain_graph = explain_graph
        self.molecule = molecule
        self.mp_layers = [module for module in self.model.modules() if isinstance(module, MessagePassing)]
        self.num_layers = len(self.mp_layers)

        self.ori_pred = None
        self.ex_labels = None
        self.edge_mask = None
        self.hard_edge_mask = None

        self.num_edges = None
        self.num_nodes = None
        self.device = x_args.device
        self.table = Chem.GetPeriodicTable().GetElementSymbol

    def __set_masks__(self, x, edge_index, init="normal"):
        (N, F), E = x.size(), edge_index.size(1)

        std = 0.1
        self.node_feat_mask = torch.nn.Parameter(torch.randn(F, requires_grad=True, device=self.device) * 0.1)

        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        self.edge_mask = torch.nn.Parameter(torch.randn(E, requires_grad=True, device=self.device) * std)
        # self.edge_mask = torch.nn.Parameter(100 * torch.ones(E, requires_grad=True))

        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask

    def __clear_masks__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None
        self.node_feat_masks = None
        self.edge_mask = None

    @property
    def __num_hops__(self):
        if self.explain_graph:
            return -1
        else:
            return self.num_layers

    def __flow__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                return module.flow
        return 'source_to_target'

    def __subgraph__(self, node_idx, x, edge_index, **kwargs):
        num_nodes, num_edges = x.size(0), edge_index.size(1)

        subset, edge_index, mapping, edge_mask = subgraph(
            node_idx, self.__num_hops__, edge_index, relabel_nodes=True,
            num_nodes=num_nodes, flow=self.__flow__())

        x = x[subset]
        for key, item in kwargs.items():
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                item = item[subset]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                item = item[edge_mask]
            kwargs[key] = item

        return x, edge_index, mapping, edge_mask, kwargs


    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                **kwargs
                ):
        self.num_edges = edge_index.shape[1]
        self.num_nodes = x.shape[0]
        self.device = x.device


    def control_sparsity(self, mask, sparsity=None):
        r"""

        :param mask: mask that need to transform
        :param sparsity: sparsity we need to control i.e. 0.7, 0.5
        :return: transformed mask where top 1 - sparsity values are set to inf.
        """
        if sparsity is None:
            sparsity = 0.7

        if data_args.model_level == 'node':
            assert self.hard_edge_mask is not None
            mask_indices = torch.where(self.hard_edge_mask)[0]
            if mask.shape[0] == self.hard_edge_mask.sum():
                sub_mask = mask
            else:
                sub_mask = mask[self.hard_edge_mask]
            mask_len = sub_mask.shape[0]
            _, sub_indices = torch.sort(sub_mask, descending=True)
            split_point = int((1 - sparsity) * mask_len)
            important_sub_indices = sub_indices[: split_point]
            important_indices = mask_indices[important_sub_indices]
            unimportant_sub_indices = sub_indices[split_point:]
            unimportant_indices = mask_indices[unimportant_sub_indices]
            trans_mask = self.hard_edge_mask.clone().float()
            trans_mask[:] = - float('inf')
            trans_mask[important_indices] = float('inf')
        else:
            _, indices = torch.sort(mask, descending=True)
            mask_len = mask.shape[0]
            split_point = int((1 - sparsity) * mask_len)
            important_indices = indices[: split_point]
            unimportant_indices = indices[split_point:]
            trans_mask = mask.clone()
            trans_mask[important_indices] = float('inf')
            trans_mask[unimportant_indices] = - float('inf')

        return trans_mask


    def visualize_graph(self, node_idx, edge_index, edge_mask, y=None,
                           threshold=None, **kwargs) -> Tuple[Axes, nx.DiGraph]:
        r"""Visualizes the subgraph around :attr:`node_idx` given an edge mask
        :attr:`edge_mask`.

        Args:
            node_idx (int): The node id to explain.
            edge_index (LongTensor): The edge indices.
            edge_mask (Tensor): The edge mask.
            y (Tensor, optional): The ground-truth node-prediction labels used
                as node colorings. (default: :obj:`None`)
            threshold (float, optional): Sets a threshold for visualizing
                important edges. If set to :obj:`None`, will visualize all
                edges with transparancy indicating the importance of edges.
                (default: :obj:`None`)
            **kwargs (optional): Additional arguments passed to
                :func:`nx.draw`.

        :rtype: :class:`matplotlib.axes.Axes`, :class:`networkx.DiGraph`
        """
        edge_index, _ = add_self_loops(edge_index, num_nodes=kwargs.get('num_nodes'))
        assert edge_mask.size(0) == edge_index.size(1)

        if self.molecule:
            atomic_num = torch.clone(y)

        if data_args.dataset_type == 'nlp':
            sentence = kwargs.get('sentence')
            assert sentence is not None

        # Only operate on a k-hop subgraph around `node_idx`.
        subset, edge_index, _, hard_edge_mask = subgraph(
            node_idx, self.__num_hops__, edge_index, relabel_nodes=True,
            num_nodes=None, flow=self.__flow__())

        edge_mask = edge_mask[hard_edge_mask]

        # --- temp ---
        edge_mask[edge_mask == float('inf')] = 1
        edge_mask[edge_mask == - float('inf')] = 0
        # ---

        if threshold is not None:
            edge_mask = (edge_mask >= threshold).to(torch.float)

        if data_args.dataset_name == 'ba_lrp' or data_args.dataset_type =='nlp':
            y = torch.zeros(edge_index.max().item() + 1,
                            device=edge_index.device)
        if y is None:
            y = torch.zeros(edge_index.max().item() + 1,
                            device=edge_index.device)
        else:
            y = y[subset]

        if self.molecule:
            atom_colors = {6: '#8c69c5', 7: '#71bcf0', 8: '#aef5f1', 9: '#bdc499', 15: '#c22f72', 16: '#f3ea19',
                           17: '#bdc499', 35: '#cc7161'}
            node_colors = [None for _ in range(y.shape[0])]
            for y_idx in range(y.shape[0]):
                node_colors[y_idx] = atom_colors[y[y_idx].int().tolist()]
        else:
            atom_colors = {0: '#8c69c5', 1: '#c56973', 2: '#a1c569', 3: '#69c5ba'}
            node_colors = [None for _ in range(y.shape[0])]
            for y_idx in range(y.shape[0]):
                node_colors[y_idx] = atom_colors[y[y_idx].int().tolist()]

        if data_args.dataset_name == 'ba_infe':
            data = kwargs.get('data')
            atom_colors = {0: '#8c69c5', 1: '#c56973', 2: '#a1c569', 3: '#69c5ba'}
            node_colors = [None for _ in range(y.shape[0])]
            for y_idx in range(y.shape[0]):
                node_colors[y_idx] = atom_colors[torch.where(data.x[y_idx] == 1)[0].squeeze().item()]


        data = Data(edge_index=edge_index, att=edge_mask, y=y,
                    num_nodes=y.size(0)).to('cpu')
        G = to_networkx(data, node_attrs=['y'], edge_attrs=['att'])
        mapping = {k: i for k, i in enumerate(subset.tolist())}
        G = nx.relabel_nodes(G, mapping)

        kwargs['with_labels'] = kwargs.get('with_labels') or True
        kwargs['font_size'] = kwargs.get('font_size') or 10
        kwargs['node_size'] = kwargs.get('node_size') or 250
        kwargs['cmap'] = kwargs.get('cmap') or 'cool'

        # calculate Graph positions
        pos = nx.kamada_kawai_layout(G)
        ax = plt.gca()


        for source, target, data in G.edges(data=True):
            ax.annotate(
                '', xy=pos[target], xycoords='data', xytext=pos[source],
                textcoords='data', arrowprops=dict(
                    arrowstyle="->",
                    lw=max(data['att'], 0.5) * 2,
                    alpha=max(data['att'], 0.4),  # alpha control transparency
                    color='#e1442a',  # color control color
                    shrinkA=sqrt(kwargs['node_size']) / 2.0,
                    shrinkB=sqrt(kwargs['node_size']) / 2.0,
                    connectionstyle="arc3,rad=0.08",  # rad control angle
                ))

        node_att = [G.adj[i][i]['att'] for i in G.adj.keys()]
        nx.draw_networkx_nodes(G, pos, node_color='#e1442a', alpha=node_att, node_size=400)
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, **kwargs)
        # define node labels
        if self.molecule:
            if x_args.nolabel:
                node_labels = {n: f'{self.table(atomic_num[n].int().item())}'
                               for n in G.nodes()}
                nx.draw_networkx_labels(G, pos, labels=node_labels, **kwargs)
            else:
                node_labels = {n: f'{n}:{self.table(atomic_num[n].int().item())}'
                               for n in G.nodes()}
                nx.draw_networkx_labels(G, pos, labels=node_labels, **kwargs)
        elif data_args.dataset_type == 'nlp':
            if x_args.nolabel:
                node_labels = {n: f'{sentence[n]}'
                               for n in G.nodes()}
                nx.draw_networkx_labels(G, pos, labels=node_labels, **kwargs)
            else:
                node_labels = {n: f'{n}:{sentence[n]}'
                               for n in G.nodes()}
                nx.draw_networkx_labels(G, pos, labels=node_labels, **kwargs)
            pass
        else:
            if not x_args.nolabel:
                nx.draw_networkx_labels(G, pos, **kwargs)

        return ax, G

    def visualize_walks(self, node_idx, edge_index, walks, edge_mask, y=None,
                        threshold=None, **kwargs) -> Tuple[Axes, nx.DiGraph]:
        r"""Visualizes the subgraph around :attr:`node_idx` given an edge mask
        :attr:`edge_mask`.

        Args:
            node_idx (int): The node id to explain.
            edge_index (LongTensor): The edge indices.
            edge_mask (Tensor): The edge mask.
            y (Tensor, optional): The ground-truth node-prediction labels used
                as node colorings. (default: :obj:`None`)
            threshold (float, optional): Sets a threshold for visualizing
                important edges. If set to :obj:`None`, will visualize all
                edges with transparancy indicating the importance of edges.
                (default: :obj:`None`)
            **kwargs (optional): Additional arguments passed to
                :func:`nx.draw`.

        :rtype: :class:`matplotlib.axes.Axes`, :class:`networkx.DiGraph`
        """
        self_loop_edge_index, _ = add_self_loops(edge_index, num_nodes=kwargs.get('num_nodes'))
        assert edge_mask.size(0) == self_loop_edge_index.size(1)

        if self.molecule:
            atomic_num = torch.clone(y)

        # Only operate on a k-hop subgraph around `node_idx`.
        subset, edge_index, _, hard_edge_mask = subgraph(
            node_idx, self.__num_hops__, self_loop_edge_index, relabel_nodes=True,
            num_nodes=None, flow=self.__flow__())

        edge_mask = edge_mask[hard_edge_mask]

        # --- temp ---
        edge_mask[edge_mask == float('inf')] = 1
        edge_mask[edge_mask == - float('inf')] = 0
        # ---

        if threshold is not None:
            edge_mask = (edge_mask >= threshold).to(torch.float)

        if data_args.dataset_name == 'ba_lrp':
            y = torch.zeros(edge_index.max().item() + 1,
                            device=edge_index.device)
        if y is None:
            y = torch.zeros(edge_index.max().item() + 1,
                            device=edge_index.device)
        else:
            y = y[subset]

        if self.molecule:
            atom_colors = {6: '#8c69c5', 7: '#71bcf0', 8: '#aef5f1', 9: '#bdc499', 15: '#c22f72', 16: '#f3ea19',
                           17: '#bdc499', 35: '#cc7161'}
            node_colors = [None for _ in range(y.shape[0])]
            for y_idx in range(y.shape[0]):
                node_colors[y_idx] = atom_colors[y[y_idx].int().tolist()]
        else:
            atom_colors = {0: '#8c69c5', 1: '#c56973', 2: '#a1c569', 3: '#69c5ba'}
            node_colors = [None for _ in range(y.shape[0])]
            for y_idx in range(y.shape[0]):
                node_colors[y_idx] = atom_colors[y[y_idx].int().tolist()]

        data = Data(edge_index=edge_index, att=edge_mask, y=y,
                    num_nodes=y.size(0)).to('cpu')
        G = to_networkx(data, node_attrs=['y'], edge_attrs=['att'])
        mapping = {k: i for k, i in enumerate(subset.tolist())}
        G = nx.relabel_nodes(G, mapping)

        kwargs['with_labels'] = kwargs.get('with_labels') or True
        kwargs['font_size'] = kwargs.get('font_size') or 8
        kwargs['node_size'] = kwargs.get('node_size') or 200
        kwargs['cmap'] = kwargs.get('cmap') or 'cool'

        # calculate Graph positions
        pos = nx.kamada_kawai_layout(G)
        ax = plt.gca()

        for source, target, data in G.edges(data=True):
            ax.annotate(
                '', xy=pos[target], xycoords='data', xytext=pos[source],
                textcoords='data', arrowprops=dict(
                    arrowstyle="-",
                    lw=1.5,
                    alpha=0.5,  # alpha control transparency
                    color='grey',  # color control color
                    shrinkA=sqrt(kwargs['node_size']) / 2.0,
                    shrinkB=sqrt(kwargs['node_size']) / 2.0,
                    connectionstyle="arc3,rad=0",  # rad control angle
                ))


        # --- try to draw a walk ---
        walks_ids = walks['ids']
        walks_score = walks['score']
        walks_node_list = []
        for i in range(walks_ids.shape[1]):
            if i == 0:
                walks_node_list.append(self_loop_edge_index[:, walks_ids[:, i].view(-1)].view(2, -1))
            else:
                walks_node_list.append(self_loop_edge_index[1, walks_ids[:, i].view(-1)].view(1, -1))
        walks_node_ids = torch.cat(walks_node_list, dim=0).T

        walks_mask = torch.zeros(walks_node_ids.shape, dtype=bool, device=self.device)
        for n in G.nodes():
            walks_mask = walks_mask | (walks_node_ids == n)
        walks_mask = walks_mask.sum(1) == walks_node_ids.shape[1]

        sub_walks_node_ids = walks_node_ids[walks_mask]
        sub_walks_score = walks_score[walks_mask]

        for i, walk in enumerate(sub_walks_node_ids):
            verts = [pos[n.item()] for n in walk]
            if walk.shape[0] == 3:
                codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
            else:
                codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
            path = Path(verts, codes)
            if sub_walks_score[i] > 0:
                patch = PathPatch(path, facecolor='none', edgecolor='red', lw=1.5,#e1442a
                                  alpha=(sub_walks_score[i] / (sub_walks_score.max() * 2)).item())
            else:
                patch = PathPatch(path, facecolor='none', edgecolor='blue', lw=1.5,#18d66b
                                  alpha=(sub_walks_score[i] / (sub_walks_score.min() * 2)).item())
            ax.add_patch(patch)


        nx.draw_networkx_nodes(G, pos, node_color=node_colors, **kwargs)
        # define node labels
        if self.molecule:
            if x_args.nolabel:
                node_labels = {n: f'{self.table(atomic_num[n].int().item())}'
                               for n in G.nodes()}
                nx.draw_networkx_labels(G, pos, labels=node_labels, **kwargs)
            else:
                node_labels = {n: f'{n}:{self.table(atomic_num[n].int().item())}'
                               for n in G.nodes()}
                nx.draw_networkx_labels(G, pos, labels=node_labels, **kwargs)
        else:
            if not x_args.nolabel:
                nx.draw_networkx_labels(G, pos, **kwargs)

        return ax, G

    def eval_related_pred(self, x, edge_index, edge_masks, **kwargs):

        node_idx = kwargs.get('node_idx')
        node_idx = 0 if node_idx is None else node_idx  # graph level: 0, node level: node_idx
        related_preds = []

        for ex_label, edge_mask in enumerate(edge_masks):

            self.edge_mask.data = float('inf') * torch.ones(edge_mask.size(), device=data_args.device)
            ori_pred = self.model(x=x, edge_index=edge_index, **kwargs)

            self.edge_mask.data = edge_mask
            masked_pred = self.model(x=x, edge_index=edge_index, **kwargs)

            # mask out important elements for fidelity calculation
            self.edge_mask.data = - edge_mask  # keep Parameter's id
            maskout_pred = self.model(x=x, edge_index=edge_index, **kwargs)

            # zero_mask
            self.edge_mask.data = - float('inf') * torch.ones(edge_mask.size(), device=data_args.device)
            zero_mask_pred = self.model(x=x, edge_index=edge_index, **kwargs)

            related_preds.append({'zero': zero_mask_pred[node_idx],
                                  'masked': masked_pred[node_idx],
                                  'maskout': maskout_pred[node_idx],
                                  'origin': ori_pred[node_idx]})

            # Adding proper activation function to the models' outputs.
            if 'cs' in Metric.cur_task:
                related_preds[ex_label] = {key: pred.softmax(0)[ex_label].item()
                                        for key, pred in related_preds[ex_label].items()}

        return related_preds



class FlowBase(ExplainerBase):

    def __init__(self, model: nn.Module, epochs=0, lr=0, explain_graph=False, molecule=False):
        super().__init__(model, epochs, lr, explain_graph, molecule)

    def extract_step(self, x, edge_index, detach=True, split_fc=False):

        layer_extractor = []
        hooks = []

        def register_hook(module: nn.Module):
            # if hasattr(module, 'weight'):
            #     print(module) # for debug
            if not list(module.children()) or isinstance(module, MessagePassing):
                hooks.append(module.register_forward_hook(forward_hook))

        def forward_hook(module: nn.Module, input: Tuple[Tensor], output: Tensor):
            # input contains x and edge_index
            if detach:
                layer_extractor.append((module, input[0].clone().detach(), output.clone().detach()))
            else:
                layer_extractor.append((module, input[0], output))

        # --- register hooks ---
        self.model.apply(register_hook)

        pred = self.model(x, edge_index)

        for hook in hooks:
            hook.remove()

        # --- divide layer sets ---

        walk_steps = []
        fc_steps = []
        pool_flag = False
        step = {'input': None, 'module': [], 'output': None}
        for layer in layer_extractor:
            if isinstance(layer[0], MessagePassing) or isinstance(layer[0], GNNPool):
                if isinstance(layer[0], GNNPool):
                    pool_flag = True
                if step['module'] and step['input'] is not None:
                    walk_steps.append(step)
                step = {'input': layer[1], 'module': [], 'output': None}
            if pool_flag and split_fc and isinstance(layer[0], nn.Linear):
                if step['module']:
                    fc_steps.append(step)
                step = {'input': layer[1], 'module': [], 'output': None}
            step['module'].append(layer[0])
            step['output'] = layer[2]


        for walk_step in walk_steps:
            if hasattr(walk_step['module'][0], 'nn') and walk_step['module'][0].nn is not None:
                # We don't allow any outside nn during message flow process in GINs
                walk_step['module'] = [walk_step['module'][0]]


        if split_fc:
            if step['module']:
                fc_steps.append(step)
            return walk_steps, fc_steps
        else:
            fc_step = step


        return walk_steps, fc_step

    def walks_pick(self,
                   edge_index: Tensor,
                   pick_edge_indices: List,
                   walk_indices: List=[],
                   num_layers=0
                   ):
        walk_indices_list = []
        for edge_idx in pick_edge_indices:

            # Adding one edge
            walk_indices.append(edge_idx)
            _, new_src = src, tgt = edge_index[:, edge_idx]
            next_edge_indices = np.array((edge_index[0, :] == new_src).nonzero().view(-1))

            # Finding next edge
            if len(walk_indices) >= num_layers:
                # return one walk
                walk_indices_list.append(walk_indices.copy())
            else:
                walk_indices_list += self.walks_pick(edge_index, next_edge_indices, walk_indices, num_layers)

            # remove the last edge
            walk_indices.pop(-1)

        return walk_indices_list

    def eval_related_pred(self, x, edge_index, masks, **kwargs):

        node_idx = kwargs.get('node_idx')
        node_idx = 0 if node_idx is None else node_idx # graph level: 0, node level: node_idx

        related_preds = []

        for label, mask in enumerate(masks):
            # origin pred
            for edge_mask in self.edge_mask:
                edge_mask.data = float('inf') * torch.ones(mask.size(), device=data_args.device)
            ori_pred = self.model(x=x, edge_index=edge_index, **kwargs)

            for edge_mask in self.edge_mask:
                edge_mask.data = mask
            masked_pred = self.model(x=x, edge_index=edge_index, **kwargs)

            # mask out important elements for fidelity calculation
            for edge_mask in self.edge_mask:
                edge_mask.data = - mask
            maskout_pred = self.model(x=x, edge_index=edge_index, **kwargs)

            # zero_mask
            for edge_mask in self.edge_mask:
                edge_mask.data = - float('inf') * torch.ones(mask.size(), device=data_args.device)
            zero_mask_pred = self.model(x=x, edge_index=edge_index, **kwargs)

            # Store related predictions for further evaluation.
            related_preds.append({'zero': zero_mask_pred[node_idx],
                                  'masked': masked_pred[node_idx],
                                  'maskout': maskout_pred[node_idx],
                                  'origin': ori_pred[node_idx]})

            # Adding proper activation function to the models' outputs.
            if 'cs' in Metric.cur_task:
                related_preds[label] = {key: pred.softmax(0)[label].item()
                                        for key, pred in related_preds[label].items()}

        return related_preds


    def explain_edges_with_loop(self, x: Tensor, walks: Dict[Tensor, Tensor], ex_label):

        walks_ids = walks['ids']
        walks_score = walks['score'][:walks_ids.shape[0], ex_label].reshape(-1)
        idx_ensemble = torch.cat([(walks_ids == i).int().sum(dim=1).unsqueeze(0) for i in range(self.num_edges + self.num_nodes)], dim=0)
        hard_edge_attr_mask = (idx_ensemble.sum(1) > 0).long()
        hard_edge_attr_mask_value = torch.tensor([float('inf'), 0], dtype=torch.float, device=self.device)[hard_edge_attr_mask]
        edge_attr = (idx_ensemble * (walks_score.unsqueeze(0))).sum(1)
        # idx_ensemble1 = torch.cat(
        #     [(walks_ids == i).int().sum(dim=1).unsqueeze(1) for i in range(self.num_edges + self.num_nodes)], dim=1)
        # edge_attr1 = (idx_ensemble1 * (walks_score.unsqueeze(1))).sum(0)

        return edge_attr - hard_edge_attr_mask_value

    def batch_input(self, x, edge_index, batch_size):
        batch_x = x.repeat(batch_size, 1)
        batch_batch = torch.arange(batch_size, device=self.device).unsqueeze(1).repeat(1, self.num_nodes).view(-1)
        batch_edge_batch = torch.arange(batch_size, device=self.device).unsqueeze(1).repeat(1, self.num_edges).view(-1)
        batch_edge_index = edge_index.repeat(1, batch_size) + batch_edge_batch * self.num_nodes
        batch = Batch(x=batch_x, edge_index=batch_edge_index, batch=batch_batch)
        return batch

    class temp_mask(object):

        def __init__(self, cls, temp_edge_mask):
            self.cls = cls
            self.temp_edge_mask = temp_edge_mask

        def __enter__(self):

            for idx, module in enumerate(self.cls.mp_layers):
                module.__explain_flow__ = True
                module.layer_edge_mask = self.temp_edge_mask[idx]

        def __exit__(self, *args):
            for idx, module in enumerate(self.cls.mp_layers):
                module.__explain_flow__ = False

    class connect_mask(object):

        def __init__(self, cls):
            self.cls = cls

        def __enter__(self):

            self.cls.edge_mask = [nn.Parameter(torch.randn(self.cls.x_batch_size * (self.cls.num_edges + self.cls.num_nodes))) for _ in
                             range(self.cls.num_layers)] if hasattr(self.cls, 'x_batch_size') else \
                                 [nn.Parameter(torch.randn(1 * (self.cls.num_edges + self.cls.num_nodes))) for _ in
                             range(self.cls.num_layers)]

            for idx, module in enumerate(self.cls.mp_layers):
                module.__explain__ = True
                module.__edge_mask__ = self.cls.edge_mask[idx]

        def __exit__(self, *args):
            for idx, module in enumerate(self.cls.mp_layers):
                module.__explain__ = False



class FlowExplainer(FlowBase):

    r"""
    Walks Explainer is a trial to do essential Graph Neural Network explanation which kernel
    concept is to extract path/walk relevant score/energies from the total Graph Model.
    This method generally applies on common Massage Passing models.
    """

    def __init__(self, model: nn.Module, epochs=0, lr=0, explain_graph=False, molecule=False):
        super().__init__(model=model, epochs=epochs, lr=lr, explain_graph=explain_graph, molecule=molecule)


    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                **kwargs
                ):

        # --- run the model once ---
        super().forward(x=x, edge_index=edge_index, **kwargs)
        self.model.eval()
        self_loop_edge_index, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)

        walk_steps, fc_step = self.extract_step(x, edge_index)

        labels = tuple(i for i in range(data_args.num_classes))
        walk_scores_tensor_list = [None for i in labels]
        for label in labels:
            # --- get the shapley value of the last layer nodes (go back to the fc input layer) ---
            # ATT: Remember to merge baseline and input in a batch (using pyg way)
            layer_deeplift_shap = GraphLayerDeepLift(GraphSequential(*fc_step['module']), fc_step['module'][0])
            last_attrs = layer_deeplift_shap.attribute(fc_step['input'], baselines=torch.zeros(fc_step['input'].shape, device=x.device, requires_grad=True),
                                                       target=label,
                                                       additional_forward_args=torch.zeros(x.shape[0], dtype=torch.int64, device=x.device),
                                                       attribute_to_layer_input=True)

            # --- iteratively BP energies back ---
            walk_steps.reverse()
            walk_scores = {tuple([i]): score.item() for i, score in enumerate(last_attrs[0].sum(1))}
            for i, step in enumerate(walk_steps):
                layer_deeplift = GraphLayerDeepLift(GraphSequential(*step['module']), step['module'][0])
                attrs = layer_deeplift.attribute(step['input'], baselines=torch.zeros(step['input'].shape, device=x.device, requires_grad=True),
                                                 target=None,
                                                 additional_forward_args=edge_index,
                                                 attribute_to_layer_input=True)
                new_walk_scores = {}
                for tgt_node, attr in enumerate(attrs):
                    attr = attr[0].sum(1)
                    attr_sum = attr.sum()
                    src_nodes = self_loop_edge_index[0, torch.where(self_loop_edge_index[1] == tgt_node)[0]]
                    for walk in walk_scores.keys():
                        if walk[0] == tgt_node:
                            for src_node in src_nodes:
                                new_walk_scores[tuple([src_node.item()] + list(walk))] = \
                                    (attr[src_node] / attr_sum * walk_scores[walk]).item()
                walk_scores = new_walk_scores
            walk_indices_tensor, walk_scores_tensor_list[label] = self.node2edge(walk_scores, self_loop_edge_index)

        walks = {'ids': walk_indices_tensor, 'score': torch.cat(walk_scores_tensor_list, dim=1)}


        # --- Apply edge mask evaluation ---
        with torch.no_grad():
            with self.connect_mask(self):
                ex_labels = tuple(torch.tensor([label]).to(data_args.device) for label in labels)
                masks = []
                for ex_label in ex_labels:
                    edge_attr = self.explain_edges_with_loop(x, walks, ex_label)
                    mask = edge_attr
                    mask = self.control_sparsity(mask, kwargs.get('sparsity'))
                    masks.append(mask.detach())

                related_preds = self.eval_related_pred(x, edge_index, masks, **kwargs)

        return None, masks, related_preds

    def node2edge(self, walk_scores, edge_index_loop):
        walk_indices_list = []
        walk_scores_list = []
        for node_tuple, score in walk_scores.items():
            # - indices -
            src_node = None
            edges = []
            for node in node_tuple:
                if src_node is None:
                    src_node = node
                    continue
                tgt_node = node
                edges.append(torch.where((edge_index_loop[0] == src_node).int() +
                                         (edge_index_loop[1] == tgt_node).int() > 1)[0].squeeze().item())
                src_node = tgt_node
            walk_indices_list.append(edges)

            # - score -
            walk_scores_list.append(score)

        return torch.tensor(walk_indices_list, dtype=torch.long, device=self.device), \
               torch.tensor(walk_scores_list, dtype=torch.float32, device=self.device).unsqueeze(1)




class FlowEraser(FlowBase):

    r"""
    Walks Explainer is a trial to do essential Graph Neural Network explanation which kernel
    concept is to extract path/walk relevant score/contribution from the total Graph Model.
    This method generally applies on common Massage Passing models.
    """

    def __init__(self, model, epochs=0, lr=0, explain_graph=False, molecule=False):
        super().__init__(model=model, epochs=epochs, lr=lr, explain_graph=explain_graph, molecule=molecule)

        self.score_structure = [(i % 2, term_idx)
                                for i in range(1, self.num_layers + 1)
                                for term_idx in combinations(range(self.num_layers), i)
                                ]

        self.x_batch_size = 100


    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                **kwargs
                ) -> Union[Tuple[None, List, List[Dict]], Tuple[Dict, List, List[Dict]]]:
        r"""
        WalksExplainer Main: Pay attention that the edge goes to it self is the weight of the node!
        :param x: Tensor - Hiden features of all vertexes
        :param edge_index: Tensor - All connected edge between vertexes/nodes
        :param kwargs:
        :return:
        """
        super().forward(x, edge_index, **kwargs)
        with torch.no_grad():
            # Explanation initial process
            self.model.eval()

            # Initial original prediction
            self.ori_logits_pred = self.model(x, edge_index).softmax(1)


            # Connect mask
            with self.connect_mask(self):
                # self.__connect_mask__()

                # Edge Index with self loop
                edge_index, _ = remove_self_loops(edge_index)
                edge_index_with_loop, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)
                walk_indices_list = torch.tensor(self.walks_pick(edge_index_with_loop.cpu(), list(range(edge_index_with_loop.shape[1])),
                                                                 num_layers=self.num_layers), device=self.device)

                if data_args.model_level == 'node':
                    node_idx = kwargs.get('node_idx')
                    self.node_idx = node_idx
                    assert node_idx is not None
                    _, _, _, self.hard_edge_mask = subgraph(
                        node_idx, self.__num_hops__, edge_index_with_loop, relabel_nodes=True,
                        num_nodes=None, flow=self.__flow__())

                    # walk indices list mask
                    edge2node_idx = edge_index_with_loop[1] == node_idx
                    walk_indices_list_mask = edge2node_idx[walk_indices_list[:, -1]]
                    walk_indices_list = walk_indices_list[walk_indices_list_mask]


                import time
                start = time.time()
                walk_score_list = []
                self.time_list = []
                for batch_idx in range((walk_indices_list.shape[0] // self.x_batch_size) + 1):
                    walk_indices: Tensor = walk_indices_list[batch_idx * self.x_batch_size: (batch_idx + 1) * self.x_batch_size]
                    # print(walk_indices)
                    # Kernel algorithm
                    walk_score: Tensor = self.compute_walk_score(x, edge_index, walk_indices)
                    walk_score_list.append(walk_score)
                    # print(f'#D#{walk_indices}\n{walk_score}')

                # Attention: we may do SHAPLEY here on walk_score_list with the help of walk_indices_list as player_list
                walks = {'ids': walk_indices_list, 'score': torch.cat(walk_score_list, dim=0)}
                print(f'Walk scores summation: {torch.cat(walk_score_list, dim=0).sum(0)}')
                print(f'#D#WalkScore total time: {time.time() - start}\n'
                      f'predict time: {sum(self.time_list)}')
                # exit()

                # specify to edge with self-loop mask prediction
                labels = tuple(i for i in range(data_args.num_classes))
                ex_labels = tuple(torch.tensor([label]).to(data_args.device) for label in labels)
                start = time.time()
                masks = []
                for ex_label in ex_labels:
                    edge_attr = self.explain_edges_with_loop(x, walks, ex_label)
                    mask = edge_attr
                    mask = self.control_sparsity(mask, kwargs.get('sparsity'))
                    # mask[mask >= 1e-1] = float('inf')
                    # mask[mask < 1e-1] = - float('inf')
                    masks.append(mask.detach())
                print(f'#D#Edge mask predict total time: {time.time() - start}')

                related_preds = self.eval_related_pred(x, edge_index, masks, **kwargs)


        return walks, masks, related_preds

    def compute_walk_score(self,
                           x: Tensor,
                           edge_index: Tensor,
                           walk_indices: Tensor
                           ) -> Union[Tensor, int]:

        # First-order removement prediction
        first_order_score = 0
        for sig, term_idx in self.score_structure:
            term = self.drop_score(x, edge_index, walk_indices, term_idx).detach()

            if sig:
                first_order_score += term
            else:
                first_order_score -= term

        return first_order_score


    def drop_score(self,
                   x: Tensor,
                   edge_index: Tensor,
                   walk_indices: Tensor,
                   term_idx
                   ):

        # construct edge_mask
        import time
        for layer_idx, edges_in_layer in enumerate(walk_indices.T):

            # This edge mask includes self-loop mask that applies to [edge_index || self-loop_edge ]
            edge_mask = float('inf') * torch.ones((self.x_batch_size, self.num_edges + self.num_nodes), device=self.device)
            if layer_idx in term_idx:
                # only applies to the edge in the term, i.e. if term is (0, 2) then we only remove the edges from layer
                # 0 and layer 2.

                sig = torch.ones((self.x_batch_size, self.num_edges + self.num_nodes), device=self.device)
                eye = torch.eye(self.num_edges + self.num_nodes, device=self.device)
                onehot = torch.cat([eye[edges_in_layer],
                                   torch.zeros((self.x_batch_size - edges_in_layer.shape[0], self.num_edges + self.num_nodes), device=self.device)],
                                   dim=0)
                sig -= 2 * onehot
                edge_mask = edge_mask * sig
                # discarded slow code
                # for walk_idx, edge_in_layer in enumerate(edges_in_layer):
                #     edge_mask[walk_idx, edge_in_layer] = - float('inf')

            # move self-loop edges to the last
            self.edge_mask[layer_idx].data = torch.cat([edge_mask[:, :self.num_edges].reshape(-1),
                                                        edge_mask[:, self.num_edges:].reshape(-1)])

        # predict without some walks Union

        # stack repeated inputs
        batch = self.batch_input(x, edge_index, self.x_batch_size)

        # Discard: [slow code]
        # data_list = [Data(x, edge_index) for _ in range(self.x_batch_size)]
        # batch = Batch.from_data_list(data_list).to(self.device)

        # Attention: if self.x_batch_size > len(edges_in_layer){real this batch's size}
        # self.ori_logits_pred - remove_logits_pred will be 0 there, so there is no inverse effect
        start = time.time()
        remove_logits_pred = self.model(data=batch).softmax(1)
        self.time_list.append(time.time() - start)

        if data_args.model_level == 'node':
            batch_node_idx = torch.arange(self.x_batch_size) * x.shape[0] + self.node_idx
            return (self.ori_logits_pred.repeat(self.x_batch_size, 1) - remove_logits_pred)[batch_node_idx]
        else:
            return self.ori_logits_pred - remove_logits_pred

        # return 0


class FlowShap_orig(FlowBase):
    coeffs = {
        'edge_size': 5e-4,
        'edge_ent': 1e-1
    }

    def __init__(self, model, epochs=500, lr=3e-1, explain_graph=False, molecule=False):
    # def __init__(self, model, epochs=500, lr=1e-1, explain_graph=False, molecule=False):
        super().__init__(model=model, epochs=epochs, lr=lr, explain_graph=explain_graph, molecule=molecule)

        self.score_structure = [(i % 2, term_idx)
                                for i in range(1, self.num_layers + 1)
                                for term_idx in combinations(range(self.num_layers), i)
                                ]

        self.ns_iter = 30
        self.ns_per_iter = None
        self.fidelity_plus = True
        self.score_lr = 0e-5 #2e-5
        # self.alpha = 0.5

        self.no_mask = True
        if self.no_mask:
            self.epochs = 1
            self.lr = 0
            self.score_lr = 0


    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                **kwargs
                ) -> Union[Tuple[None, List, List[Dict]], Tuple[Dict, List, List[Dict]]]:

        super().forward(x, edge_index, **kwargs)

        # Explanation initial process
        self.model.eval()

        # Initial original prediction
        self.ori_logits_pred = self.model(x, edge_index).softmax(1)
        print(f'#I#pred label: {torch.argmax(self.ori_logits_pred)}')


        # Edge Index with self loop
        edge_index, _ = remove_self_loops(edge_index)
        edge_index_with_loop, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)
        walk_indices_list = torch.tensor(self.walks_pick(edge_index_with_loop.cpu(), list(range(edge_index_with_loop.shape[1])),
                                                         num_layers=self.num_layers), device=self.device)

        if data_args.model_level == 'node':
            node_idx = kwargs.get('node_idx')
            self.node_idx = node_idx
            assert node_idx is not None
            _, _, _, self.hard_edge_mask = subgraph(
                node_idx, self.__num_hops__, edge_index_with_loop, relabel_nodes=True,
                num_nodes=None, flow=self.__flow__())

            # walk indices list mask
            edge2node_idx = edge_index_with_loop[1] == node_idx
            walk_indices_list_mask = edge2node_idx[walk_indices_list[:, -1]]
            walk_indices_list = walk_indices_list[walk_indices_list_mask]


        import time
        start = time.time()
        self.time_list = []
        labels = tuple(i for i in range(data_args.num_classes))
        ex_labels = tuple(torch.tensor([label]).to(data_args.device) for label in labels)

        force_recalculate = x_args.force_recalculate
        explain_index = kwargs.get('index')
        store_path = os.path.join(ROOT_DIR, 'masks', f'{x_args.explainer}_tmp', f'{x_args.dataset_name}_{x_args.model_name}')
        store_file = os.path.join(store_path, f'{explain_index}.pt')

        if not os.path.exists(store_path):
            os.makedirs(store_path)
            print(f'#W#create dirs {store_path}')

        if explain_index is None\
            or not os.path.exists(store_file) \
            or force_recalculate:
            # --- without saving output mask before ---

            # Connect mask
            with self.connect_mask(self):
                iter_weighted_change_walks_list, iter_changed_subsets_score_list, walk_sample_count = \
                    self.flow_shap(x, edge_index, edge_index_with_loop, walk_indices_list)

            walk_score_list = []
            for ex_label in ex_labels:
                # --- training ---
                self.train_mask(x,
                                edge_index,
                                ex_label,
                                walk_indices_list,
                                edge_index_with_loop,
                                iter_weighted_change_walks_list,
                                iter_changed_subsets_score_list,
                                walk_sample_count)

                walk_score_list.append(self.flow_mask.data)

            walks = {'ids': walk_indices_list, 'score': torch.cat(walk_score_list, dim=1)}
            # print(f'Walk scores summation: {torch.cat(walk_score_list, dim=0).sum(0)}')
            print(f'#D#WalkScore total time: {time.time() - start}\n'
                  f'predict time: {sum(self.time_list)}')

            # --- save results for different Sparsity ---
            torch.save(walks, store_file)
        else:
            walks = torch.load(store_file)

        # specify to edge with self-loop mask prediction
        labels = tuple(i for i in range(data_args.num_classes))
        ex_labels = tuple(torch.tensor([label]).to(data_args.device) for label in labels)
        start = time.time()
        masks = []
        for ex_label in ex_labels:
            edge_attr = self.explain_edges_with_loop(x, walks, ex_label)
            mask = edge_attr
            mask = self.control_sparsity(mask, kwargs.get('sparsity'))
            masks.append(mask.detach())
        print(f'#D#Edge mask predict total time: {time.time() - start}')

        # Connect mask
        with self.connect_mask(self):
            related_preds = self.eval_related_pred(x, edge_index, masks, **kwargs)


        return walks, masks, related_preds

    def __loss__(self, raw_preds, x_label):
        if self.explain_graph:
            loss = Metric.loss_func(raw_preds, x_label)
        else:
            loss = Metric.loss_func(raw_preds[self.node_idx].unsqueeze(0), x_label)

        if self.fidelity_plus:
            loss = - loss

        # Option 2: make it hard: higher std, and closer to Sparsity
        # loss = loss - 10 * torch.square(self.mask - 0.5).mean()

        # loss = loss + 1e-3 * torch.square(self.mask.sum() - self.mask.shape[0] * (1 - x_args.sparsity))
        # m = self.nec_suf_mask.sigmoid()
        # loss = loss + self.coeffs['edge_size'] * m.sum()
        # ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        # loss = loss + self.coeffs['edge_ent'] * ent.mean()

        return loss

    def train_mask(self,
                   x: Tensor,
                   edge_index: Tensor,
                   ex_label: Tensor,
                   walk_indices_list,
                   edge_index_with_loop,
                   iter_weighted_change_walks_list,
                   iter_changed_subsets_score_list,
                   walk_sample_count,
                   t0=7.,
                   t1=0.5,
                   **kwargs
                   ) -> None:
        # initialize a mask
        self.to(x.device)

        # --- necesufy mask ---
        self.nec_suf_mask = nn.Parameter(1e-1 * nn.init.uniform_(torch.empty((1, iter_weighted_change_walks_list.shape[1], 1), device=self.device)))

        # --- force higher Sparsity ---
        # self.nec_suf_mask.data = self.nec_suf_mask.data - self.nec_suf_mask.data.min()
        # self.nec_suf_mask.data = self.nec_suf_mask.data / (self.nec_suf_mask.data.max() + 1e-20)
        # self.nec_suf_mask.data = self.nec_suf_mask.data ** 8
        # self.nec_suf_mask.data = self.nec_suf_mask.data - self.nec_suf_mask.data.min()
        # self.nec_suf_mask.data = self.nec_suf_mask.data / (self.nec_suf_mask.data.max() + 1e-20)
        # self.nec_suf_mask.data = self.nec_suf_mask.data * 1 - 0.5

        if self.no_mask:
            self.nec_suf_mask = nn.Parameter(100 * torch.ones((1, iter_weighted_change_walks_list.shape[1], 1), device=self.device))
        self.iter_weighted_change_walks_list = nn.Parameter(iter_weighted_change_walks_list.clone().detach())

        # --- Training ---
        walk_plain_indices_list = walk_indices_list + \
                                  (edge_index_with_loop.shape[1]
                                   * torch.arange(self.num_layers, device=self.device)).repeat(
                                      walk_indices_list.shape[0], 1)

        self.flow2layeredge_matrix = torch.stack([(walk_plain_indices_list == i).float().sum(dim=1)
                                                  for i in
                                                  range(self.num_layers * (self.num_edges + self.num_nodes))],
                                                 dim=1).detach()

        # train to get the mask
        optimizer = torch.optim.Adam([{'params': self.nec_suf_mask}],
                                      # {'params': self.iter_weighted_change_walks_list, 'lr': self.score_lr}],
                                     lr=self.lr)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300, 400], gamma=0.05)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300], gamma=0.05)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300, 450], gamma=1e-1)
        print('#I#begin this ex_label')
        for epoch in range(1, self.epochs + 1):

            masked_iter_weighted_change_walks_list = self.iter_weighted_change_walks_list * self.nec_suf_mask.sigmoid()

            walk_scores = (masked_iter_weighted_change_walks_list.unsqueeze(3).repeat(1, 1, 1,
                                                                                      iter_changed_subsets_score_list.shape[
                                                                                          2]) * iter_changed_subsets_score_list.unsqueeze(2)).sum(1).sum(0)
            # EPS will affect the stability of training
            EPS = 1e-22
            shap_flow_score = (walk_scores / (walk_sample_count.unsqueeze(1) + EPS))

            # --- score/mask transformer ---
            self.flow_mask = shap_flow_score[:, ex_label]

            # --- setting layer edge masks ---
            self.layer_edge_mask = (self.flow_mask * self.flow2layeredge_matrix).view(self.flow_mask.shape[0],
                                                                                      self.num_layers,
                                                                                      -1).sum(0)
            mask = self.layer_edge_mask.sum(0)
            mask = mask - mask.min()
            mask = mask / (mask.max() + EPS)

            # Option 1: make it harder: draw back: cannot control sparsity
            # mask = mask * 500 - 250
            # mask = mask.sigmoid()
            climb = True
            if climb:
                mask = mask ** 8
            else:
                end_epoch = 300
                temperature = float(t0 * ((t1 / t0) ** (epoch / end_epoch))) if epoch < end_epoch else t1
                mask = gumbel_softmax(mask, temperature, training=True)

            mask = mask - mask.min()
            mask = mask / (mask.max() + EPS)
            cur_sparsity = (mask < 0.5).sum().float() / mask.shape[0]
            # if cur_sparsity < x_args.sparsity:
            #     # --- early stop ---
            #     break
            if epoch % 20 == 0:
                print(f'Epoch: {epoch} --- training mask Sparsity: {cur_sparsity}')
            if self.fidelity_plus:
                mask = 1 - mask # Fidelity +
            self.mask = mask
            isig_mask = torch.log(self.mask / (1 - self.mask + EPS) + EPS)


            # --- temp update non-leaf edge_mask
            temp_edge_mask = []
            for layer_idx in range(self.num_layers):
                # --- Attention self-loop will be put at last because of the model will do it ---
                temp_edge_mask.append(isig_mask)

            # debug:
            with self.temp_mask(self, temp_edge_mask):
                raw_preds = self.model(x, edge_index, **kwargs)
            loss = self.__loss__(raw_preds, ex_label)

            if epoch % 20 == 0:
                print(f'#D#Loss:{loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()
        return

    def flow_shap(self,
                  x,
                  edge_index,
                  edge_index_with_loop,
                  walk_indices_list
                  ):
        # --- Kernel algorithm ---

        # --- row: walk index, column: total score
        walk_sample_count = torch.zeros(walk_indices_list.shape[0], dtype=torch.float, device=self.device)

        # General setting
        iter_weighted_change_walks_list = []
        iter_changed_subsets_score_list = []

        # Random sample ns_iter * ns_per_iter times
        for iter_idx in range(self.ns_iter):

            # --- random index ---
            unmask_pool = torch.cat([walk_indices_list[:, layer].unique() + layer * edge_index_with_loop.shape[1]
                                     for layer in range(self.num_layers)])
            self.ns_per_iter = unmask_pool.shape[0] if data_args.model_level != 'node' or unmask_pool.shape[
                0] <= 100 else 100
            idx = torch.randperm(unmask_pool.nelement())
            unmask_pool = unmask_pool.view(-1)[idx].view(unmask_pool.size())

            mask_per_sub = unmask_pool.shape[0] // self.ns_per_iter
            weighted_change_walks_list = []
            last_eliminated_walks = torch.zeros(walk_indices_list.shape[0], dtype=torch.bool, device=self.device)
            layer_edge_mask_list = []
            for sub_idx in range(self.ns_per_iter):
                # --- sub random index ---
                mask_pool = unmask_pool[: mask_per_sub * (sub_idx + 1)]

                # --- obtain changed walk idx ---
                eliminated_layer_edges = unmask_pool[mask_per_sub * sub_idx: mask_per_sub * (sub_idx + 1)]
                walk_plain_indices_list = walk_indices_list + \
                                          (edge_index_with_loop.shape[1] * torch.arange(self.num_layers,
                                                                                        device=self.device)).repeat(
                                              walk_indices_list.shape[0], 1)
                eliminated_walks = torch.stack([walk_plain_indices_list == edge for edge in eliminated_layer_edges],
                                               dim=0).long().sum(0).sum(1).bool().long()
                weighted_changed_walks = eliminated_walks.clone().float()
                weighted_changed_walks[eliminated_walks == last_eliminated_walks] = 0.
                weighted_changed_walks /= (weighted_changed_walks > 1e-20).sum() + 1e-30
                weighted_change_walks_list.append(weighted_changed_walks)
                last_eliminated_walks = eliminated_walks

                # --- setting a subset mask ---
                layer_edge_masks = torch.ones((self.num_layers, edge_index_with_loop.shape[1]),
                                              device=self.device)
                layer_edge_masks.view(-1)[mask_pool] -= 2
                layer_edge_mask_list.append(layer_edge_masks)

            weighted_change_walks_list = torch.stack(weighted_change_walks_list, dim=0)
            iter_weighted_change_walks_list.append(weighted_change_walks_list.detach())
            layer_edge_mask_list = torch.stack(layer_edge_mask_list, dim=0) * float('inf')

            # --- compute subsets' outputs of current iteration ---
            for layer_idx in range(self.num_layers):
                # --- Attention self-loop will be put at last because of the model will do it ---
                self.edge_mask[layer_idx].data = torch.cat(
                    [layer_edge_mask_list[:, layer_idx, :self.num_edges].reshape(-1),
                     layer_edge_mask_list[:, layer_idx, self.num_edges:].reshape(-1)])

            batch = self.batch_input(x, edge_index, self.ns_per_iter)
            subsets_output = self.model(data=batch).softmax(1).detach()
            if data_args.model_level == 'node':
                subsets_output = subsets_output.view(self.ns_per_iter, -1, data_args.num_classes)[:, self.node_idx]
                last_subsets_output = torch.cat(
                    [self.ori_logits_pred[self.node_idx].unsqueeze(0), subsets_output.clone()[:-1]], dim=0)
            else:
                last_subsets_output = torch.cat([self.ori_logits_pred, subsets_output.clone()[:-1]], dim=0)

            changed_subsets_score_list = (last_subsets_output - subsets_output).detach()
            iter_changed_subsets_score_list.append(changed_subsets_score_list)

            walk_sample_count += (weighted_change_walks_list > 1e-30).float().sum(0)

        # iter x subset_idx x flow_idx
        iter_weighted_change_walks_list = torch.stack(iter_weighted_change_walks_list, dim=0)
        iter_changed_subsets_score_list = torch.stack(iter_changed_subsets_score_list, dim=0)

        return iter_weighted_change_walks_list, iter_changed_subsets_score_list, walk_sample_count


class FlowShap_plus(FlowBase):
    coeffs = {
        'edge_size': 5e-4,
        'edge_ent': 1e-1
    }

    def __init__(self, model, epochs=500, lr=3e-1, explain_graph=False, molecule=False):
    # def __init__(self, model, epochs=500, lr=1e-1, explain_graph=False, molecule=False):
        super().__init__(model=model, epochs=epochs, lr=lr, explain_graph=explain_graph, molecule=molecule)

        self.score_structure = [(i % 2, term_idx)
                                for i in range(1, self.num_layers + 1)
                                for term_idx in combinations(range(self.num_layers), i)
                                ]

        self.ns_iter = 30
        self.ns_per_iter = None
        self.fidelity_plus = True
        self.score_lr = 0e-5 #2e-5
        # self.alpha = 0.5

        self.no_mask = False
        if self.no_mask:
            self.epochs = 1
            self.lr = 0
            self.score_lr = 0


    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                **kwargs
                ) -> Union[Tuple[None, List, List[Dict]], Tuple[Dict, List, List[Dict]]]:

        super().forward(x, edge_index, **kwargs)

        # Explanation initial process
        self.model.eval()

        # Initial original prediction
        self.ori_logits_pred = self.model(x, edge_index).softmax(1)
        print(f'#I#pred label: {torch.argmax(self.ori_logits_pred)}')


        # Edge Index with self loop
        edge_index, _ = remove_self_loops(edge_index)
        edge_index_with_loop, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)
        walk_indices_list = torch.tensor(self.walks_pick(edge_index_with_loop.cpu(), list(range(edge_index_with_loop.shape[1])),
                                                         num_layers=self.num_layers), device=self.device)

        if data_args.model_level == 'node':
            node_idx = kwargs.get('node_idx')
            self.node_idx = node_idx
            assert node_idx is not None
            _, _, _, self.hard_edge_mask = subgraph(
                node_idx, self.__num_hops__, edge_index_with_loop, relabel_nodes=True,
                num_nodes=None, flow=self.__flow__())

            # walk indices list mask
            edge2node_idx = edge_index_with_loop[1] == node_idx
            walk_indices_list_mask = edge2node_idx[walk_indices_list[:, -1]]
            walk_indices_list = walk_indices_list[walk_indices_list_mask]


        import time
        start = time.time()
        self.time_list = []
        labels = tuple(i for i in range(data_args.num_classes))
        ex_labels = tuple(torch.tensor([label]).to(data_args.device) for label in labels)

        force_recalculate = x_args.force_recalculate
        explain_index = kwargs.get('index')
        store_path = os.path.join(ROOT_DIR, 'masks', f'{x_args.explainer}_tmp', f'{x_args.dataset_name}_{x_args.model_name}')
        store_file = os.path.join(store_path, f'{explain_index}.pt')

        if not os.path.exists(store_path):
            os.makedirs(store_path)
            print(f'#W#create dirs {store_path}')

        if explain_index is None\
            or not os.path.exists(store_file) \
            or force_recalculate:
            # --- without saving output mask before ---

            # Connect mask
            with self.connect_mask(self):
                iter_weighted_change_walks_list, iter_changed_subsets_score_list, walk_sample_count = \
                    self.flow_shap(x, edge_index, edge_index_with_loop, walk_indices_list)

            walk_score_list = []
            for ex_label in ex_labels:
                # --- training ---
                self.train_mask(x,
                                edge_index,
                                ex_label,
                                walk_indices_list,
                                edge_index_with_loop,
                                iter_weighted_change_walks_list,
                                iter_changed_subsets_score_list,
                                walk_sample_count)

                walk_score_list.append(self.flow_mask.data)

            walks = {'ids': walk_indices_list, 'score': torch.cat(walk_score_list, dim=1)}
            # print(f'Walk scores summation: {torch.cat(walk_score_list, dim=0).sum(0)}')
            print(f'#D#WalkScore total time: {time.time() - start}\n'
                  f'predict time: {sum(self.time_list)}')

            # --- save results for different Sparsity ---
            torch.save(walks, store_file)
        else:
            walks = torch.load(store_file)

        # specify to edge with self-loop mask prediction
        labels = tuple(i for i in range(data_args.num_classes))
        ex_labels = tuple(torch.tensor([label]).to(data_args.device) for label in labels)
        start = time.time()
        masks = []
        for ex_label in ex_labels:
            edge_attr = self.explain_edges_with_loop(x, walks, ex_label)
            mask = edge_attr
            mask = self.control_sparsity(mask, kwargs.get('sparsity'))
            masks.append(mask.detach())
        print(f'#D#Edge mask predict total time: {time.time() - start}')

        # Connect mask
        with self.connect_mask(self):
            related_preds = self.eval_related_pred(x, edge_index, masks, **kwargs)


        return walks, masks, related_preds

    def __loss__(self, raw_preds, x_label):
        if self.explain_graph:
            loss = Metric.loss_func(raw_preds, x_label)
        else:
            loss = Metric.loss_func(raw_preds[self.node_idx].unsqueeze(0), x_label)

        if self.fidelity_plus:
            loss = - loss

        # Option 2: make it hard: higher std, and closer to Sparsity
        # loss = loss - 10 * torch.square(self.mask - 0.5).mean()

        # loss = loss + 1e-3 * torch.square(self.mask.sum() - self.mask.shape[0] * (1 - x_args.sparsity))
        # m = self.nec_suf_mask.sigmoid()
        # loss = loss + self.coeffs['edge_size'] * m.sum()
        # ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        # loss = loss + self.coeffs['edge_ent'] * ent.mean()

        return loss

    def train_mask(self,
                   x: Tensor,
                   edge_index: Tensor,
                   ex_label: Tensor,
                   walk_indices_list,
                   edge_index_with_loop,
                   iter_weighted_change_walks_list,
                   iter_changed_subsets_score_list,
                   walk_sample_count,
                   t0=7.,
                   t1=0.5,
                   **kwargs
                   ) -> None:
        # initialize a mask
        self.to(x.device)

        # --- necesufy mask --- wrong!!!!! because of testing!!!!
        self.nec_suf_mask = nn.Parameter(1e-1 * nn.init.uniform_(torch.empty((1, iter_weighted_change_walks_list.shape[1], 1), device=self.device)))
        # self.nec_suf_mask = nn.Parameter(1 * torch.ones((1, iter_weighted_change_walks_list.shape[1], 1), device=self.device))

        # --- force higher Sparsity ---
        # self.nec_suf_mask.data = self.nec_suf_mask.data - self.nec_suf_mask.data.min()
        # self.nec_suf_mask.data = self.nec_suf_mask.data / (self.nec_suf_mask.data.max() + 1e-20)
        # self.nec_suf_mask.data = self.nec_suf_mask.data ** 8
        # self.nec_suf_mask.data = self.nec_suf_mask.data - self.nec_suf_mask.data.min()
        # self.nec_suf_mask.data = self.nec_suf_mask.data / (self.nec_suf_mask.data.max() + 1e-20)
        # self.nec_suf_mask.data = self.nec_suf_mask.data * 1 - 0.5

        if self.no_mask:
            self.nec_suf_mask = nn.Parameter(100 * torch.ones((1, iter_weighted_change_walks_list.shape[1], 1), device=self.device))
        self.iter_weighted_change_walks_list = nn.Parameter(iter_weighted_change_walks_list.clone().detach())

        # --- Training ---
        walk_plain_indices_list = walk_indices_list + \
                                  (edge_index_with_loop.shape[1]
                                   * torch.arange(self.num_layers, device=self.device)).repeat(
                                      walk_indices_list.shape[0], 1)

        self.flow2layeredge_matrix = torch.stack([(walk_plain_indices_list == i).float().sum(dim=1)
                                                  for i in
                                                  range(self.num_layers * (self.num_edges + self.num_nodes))],
                                                 dim=1).detach()

        # train to get the mask
        optimizer = torch.optim.Adam([{'params': self.nec_suf_mask}],
                                      # {'params': self.iter_weighted_change_walks_list, 'lr': self.score_lr}],
                                     lr=self.lr)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300, 400], gamma=0.05)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300], gamma=0.05)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300, 450], gamma=1e-1)
        print('#I#begin this ex_label')
        for epoch in range(1, self.epochs + 1):

            masked_iter_weighted_change_walks_list = self.iter_weighted_change_walks_list * self.nec_suf_mask.sigmoid()

            walk_scores = (masked_iter_weighted_change_walks_list.unsqueeze(3).repeat(1, 1, 1,
                                                                                      iter_changed_subsets_score_list.shape[
                                                                                          2]) * iter_changed_subsets_score_list.unsqueeze(2)).sum(1).sum(0)
            # EPS will affect the stability of training
            EPS = 1e-18
            shap_flow_score = (walk_scores / (walk_sample_count.unsqueeze(1) + EPS))

            # --- score/mask transformer ---
            self.flow_mask = shap_flow_score[:, ex_label]

            # --- setting layer edge masks ---
            self.layer_edge_mask = (self.flow_mask * self.flow2layeredge_matrix).view(self.flow_mask.shape[0],
                                                                                      self.num_layers,
                                                                                      -1).sum(0)
            mask = self.layer_edge_mask.sum(0)
            mask = mask - mask.min()
            mask = mask / (mask.max() + EPS)

            # Option 1: make it harder: draw back: cannot control sparsity
            # mask = mask * 500 - 250
            # mask = mask.sigmoid()
            climb = True
            if climb:
                # pass
                mask = mask ** 8
            else:
                end_epoch = 300
                temperature = float(t0 * ((t1 / t0) ** (epoch / end_epoch))) if epoch < end_epoch else t1
                mask = gumbel_softmax(mask, temperature, training=True)

            mask = mask - mask.min()
            mask = mask / (mask.max() + EPS)
            cur_sparsity = (mask < 0.5).sum().float() / mask.shape[0]
            # if cur_sparsity < x_args.sparsity:
            #     # --- early stop ---
            #     break
            if epoch % 20 == 0:
                print(f'Epoch: {epoch} --- training mask Sparsity: {cur_sparsity}')
            if self.fidelity_plus:
                mask = 1 - mask # Fidelity +
            self.mask = mask
            isig_mask = torch.log(self.mask / (1 - self.mask + EPS) + EPS)


            # --- temp update non-leaf edge_mask
            temp_edge_mask = []
            for layer_idx in range(self.num_layers):
                # --- Attention self-loop will be put at last because of the model will do it ---
                temp_edge_mask.append(isig_mask)

            # debug:
            with self.temp_mask(self, temp_edge_mask):
                raw_preds = self.model(x, edge_index, **kwargs)
            loss = self.__loss__(raw_preds, ex_label)

            if epoch % 20 == 0:
                print(f'#D#Loss:{loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()
        return

    def flow_shap(self,
                  x,
                  edge_index,
                  edge_index_with_loop,
                  walk_indices_list
                  ):
        # --- Kernel algorithm ---

        # --- row: walk index, column: total score
        walk_sample_count = torch.zeros(walk_indices_list.shape[0], dtype=torch.float, device=self.device)

        # General setting
        iter_weighted_change_walks_list = []
        iter_changed_subsets_score_list = []

        # Random sample ns_iter * ns_per_iter times
        for iter_idx in range(self.ns_iter):

            # --- random index ---
            unmask_pool = torch.cat([walk_indices_list[:, layer].unique() + layer * edge_index_with_loop.shape[1]
                                     for layer in range(self.num_layers)])
            self.ns_per_iter = unmask_pool.shape[0] if data_args.model_level != 'node' or unmask_pool.shape[
                0] <= 100 else 100
            idx = torch.randperm(unmask_pool.nelement())
            unmask_pool = unmask_pool.view(-1)[idx].view(unmask_pool.size())

            mask_per_sub = unmask_pool.shape[0] // self.ns_per_iter
            weighted_change_walks_list = []
            last_eliminated_walks = torch.zeros(walk_indices_list.shape[0], dtype=torch.bool, device=self.device)
            layer_edge_mask_list = []
            for sub_idx in range(self.ns_per_iter):
                # --- sub random index ---
                mask_pool = unmask_pool[: mask_per_sub * (sub_idx + 1)]

                # --- obtain changed walk idx ---
                eliminated_layer_edges = unmask_pool[mask_per_sub * sub_idx: mask_per_sub * (sub_idx + 1)]
                walk_plain_indices_list = walk_indices_list + \
                                          (edge_index_with_loop.shape[1] * torch.arange(self.num_layers,
                                                                                        device=self.device)).repeat(
                                              walk_indices_list.shape[0], 1)
                eliminated_walks = torch.stack([walk_plain_indices_list == edge for edge in eliminated_layer_edges],
                                               dim=0).long().sum(0).sum(1).bool().long()
                weighted_changed_walks = eliminated_walks.clone().float()
                weighted_changed_walks[eliminated_walks == last_eliminated_walks] = 0.
                weighted_changed_walks /= (weighted_changed_walks > 1e-20).sum() + 1e-30
                weighted_change_walks_list.append(weighted_changed_walks)
                last_eliminated_walks = eliminated_walks

                # --- setting a subset mask ---
                layer_edge_masks = torch.ones((self.num_layers, edge_index_with_loop.shape[1]),
                                              device=self.device)
                layer_edge_masks.view(-1)[mask_pool] -= 2
                layer_edge_mask_list.append(layer_edge_masks)

            weighted_change_walks_list = torch.stack(weighted_change_walks_list, dim=0)
            iter_weighted_change_walks_list.append(weighted_change_walks_list.detach())
            layer_edge_mask_list = torch.stack(layer_edge_mask_list, dim=0) * float('inf')

            # --- compute subsets' outputs of current iteration ---
            for layer_idx in range(self.num_layers):
                # --- Attention self-loop will be put at last because of the model will do it ---
                self.edge_mask[layer_idx].data = torch.cat(
                    [layer_edge_mask_list[:, layer_idx, :self.num_edges].reshape(-1),
                     layer_edge_mask_list[:, layer_idx, self.num_edges:].reshape(-1)])

            batch = self.batch_input(x, edge_index, self.ns_per_iter)
            subsets_output = self.model(data=batch).softmax(1).detach()
            if data_args.model_level == 'node':
                subsets_output = subsets_output.view(self.ns_per_iter, -1, data_args.num_classes)[:, self.node_idx]
                last_subsets_output = torch.cat(
                    [self.ori_logits_pred[self.node_idx].unsqueeze(0), subsets_output.clone()[:-1]], dim=0)
            else:
                last_subsets_output = torch.cat([self.ori_logits_pred, subsets_output.clone()[:-1]], dim=0)

            changed_subsets_score_list = (last_subsets_output - subsets_output).detach()
            iter_changed_subsets_score_list.append(changed_subsets_score_list)

            walk_sample_count += (weighted_change_walks_list > 1e-30).float().sum(0)

        # iter x subset_idx x flow_idx
        iter_weighted_change_walks_list = torch.stack(iter_weighted_change_walks_list, dim=0)
        iter_changed_subsets_score_list = torch.stack(iter_changed_subsets_score_list, dim=0)

        return iter_weighted_change_walks_list, iter_changed_subsets_score_list, walk_sample_count

class FlowShap_plus_r(FlowBase):
    coeffs = {
        'edge_size': 5e-4,
        'edge_ent': 1e-1
    }

    def __init__(self, model, epochs=500, lr=3e-1, explain_graph=False, molecule=False):
    # def __init__(self, model, epochs=500, lr=1e-1, explain_graph=False, molecule=False):
        super().__init__(model=model, epochs=epochs, lr=lr, explain_graph=explain_graph, molecule=molecule)

        self.score_structure = [(i % 2, term_idx)
                                for i in range(1, self.num_layers + 1)
                                for term_idx in combinations(range(self.num_layers), i)
                                ]

        self.ns_iter = 30
        self.ns_per_iter = None
        self.fidelity_plus = True
        self.score_lr = 0e-5 #2e-5
        # self.alpha = 0.5

        self.no_mask = False
        if self.no_mask:
            self.epochs = 1
            self.lr = 0
            self.score_lr = 0


    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                **kwargs
                ) -> Union[Tuple[None, List, List[Dict]], Tuple[Dict, List, List[Dict]]]:

        super().forward(x, edge_index, **kwargs)

        # Explanation initial process
        self.model.eval()

        # Initial original prediction
        self.ori_logits_pred = self.model(x, edge_index).softmax(1)
        print(f'#I#pred label: {torch.argmax(self.ori_logits_pred)}')


        # Edge Index with self loop
        edge_index, _ = remove_self_loops(edge_index)
        edge_index_with_loop, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)
        walk_indices_list = torch.tensor(self.walks_pick(edge_index_with_loop.cpu(), list(range(edge_index_with_loop.shape[1])),
                                                         num_layers=self.num_layers), device=self.device)

        if data_args.model_level == 'node':
            node_idx = kwargs.get('node_idx')
            self.node_idx = node_idx
            assert node_idx is not None
            _, _, _, self.hard_edge_mask = subgraph(
                node_idx, self.__num_hops__, edge_index_with_loop, relabel_nodes=True,
                num_nodes=None, flow=self.__flow__())

            # walk indices list mask
            edge2node_idx = edge_index_with_loop[1] == node_idx
            walk_indices_list_mask = edge2node_idx[walk_indices_list[:, -1]]
            walk_indices_list = walk_indices_list[walk_indices_list_mask]


        import time
        start = time.time()
        self.time_list = []
        labels = tuple(i for i in range(data_args.num_classes))
        ex_labels = tuple(torch.tensor([label]).to(data_args.device) for label in labels)

        force_recalculate = x_args.force_recalculate
        explain_index = kwargs.get('index')
        store_path = os.path.join(ROOT_DIR, 'masks', f'{x_args.explainer}_tmp', f'{x_args.dataset_name}_{x_args.model_name}')
        store_file = os.path.join(store_path, f'{explain_index}.pt')

        if not os.path.exists(store_path):
            os.makedirs(store_path)
            print(f'#W#create dirs {store_path}')

        if explain_index is None\
            or not os.path.exists(store_file) \
            or force_recalculate:
            # --- without saving output mask before ---

            # Connect mask
            with self.connect_mask(self):
                iter_weighted_change_walks_list, iter_changed_subsets_score_list, walk_sample_count = \
                    self.flow_shap(x, edge_index, edge_index_with_loop, walk_indices_list)

            walk_score_list = []
            for ex_label in ex_labels:
                # --- training ---
                self.train_mask(x,
                                edge_index,
                                ex_label,
                                walk_indices_list,
                                edge_index_with_loop,
                                iter_weighted_change_walks_list,
                                iter_changed_subsets_score_list,
                                walk_sample_count)

                walk_score_list.append(self.flow_mask.data)

            walks = {'ids': walk_indices_list, 'score': torch.cat(walk_score_list, dim=1)}
            # print(f'Walk scores summation: {torch.cat(walk_score_list, dim=0).sum(0)}')
            print(f'#D#WalkScore total time: {time.time() - start}\n'
                  f'predict time: {sum(self.time_list)}')

            # --- save results for different Sparsity ---
            torch.save(walks, store_file)
        else:
            walks = torch.load(store_file)

        # specify to edge with self-loop mask prediction
        labels = tuple(i for i in range(data_args.num_classes))
        ex_labels = tuple(torch.tensor([label]).to(data_args.device) for label in labels)
        start = time.time()
        masks = []
        for ex_label in ex_labels:
            edge_attr = self.explain_edges_with_loop(x, walks, ex_label)
            mask = edge_attr
            mask = self.control_sparsity(mask, kwargs.get('sparsity'))
            masks.append(mask.detach())
        print(f'#D#Edge mask predict total time: {time.time() - start}')

        # Connect mask
        with self.connect_mask(self):
            related_preds = self.eval_related_pred(x, edge_index, masks, **kwargs)


        return walks, masks, related_preds

    def __loss__(self, raw_preds, x_label):
        if self.explain_graph:
            loss = Metric.loss_func(raw_preds, x_label)
        else:
            loss = Metric.loss_func(raw_preds[self.node_idx].unsqueeze(0), x_label)

        if self.fidelity_plus:
            loss = - loss

        # Option 2: make it hard: higher std, and closer to Sparsity
        # loss = loss - 10 * torch.square(self.mask - 0.5).mean()

        # loss = loss + 1e-3 * torch.square(self.mask.sum() - self.mask.shape[0] * (1 - x_args.sparsity))
        m = self.nec_suf_mask.sigmoid()
        loss = loss + self.coeffs['edge_size'] * m.sum()
        # ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        # loss = loss + self.coeffs['edge_ent'] * ent.mean()

        return loss

    def train_mask(self,
                   x: Tensor,
                   edge_index: Tensor,
                   ex_label: Tensor,
                   walk_indices_list,
                   edge_index_with_loop,
                   iter_weighted_change_walks_list,
                   iter_changed_subsets_score_list,
                   walk_sample_count,
                   t0=7.,
                   t1=0.5,
                   **kwargs
                   ) -> None:
        # initialize a mask
        self.to(x.device)

        # --- necesufy mask ---
        self.nec_suf_mask = nn.Parameter(1e-1 * nn.init.uniform_(torch.empty((1, iter_weighted_change_walks_list.shape[1], 1), device=self.device)))
        # self.nec_suf_mask = nn.Parameter(1 * torch.ones((1, iter_weighted_change_walks_list.shape[1], 1), device=self.device))

        # --- force higher Sparsity ---
        # self.nec_suf_mask.data = self.nec_suf_mask.data - self.nec_suf_mask.data.min()
        # self.nec_suf_mask.data = self.nec_suf_mask.data / (self.nec_suf_mask.data.max() + 1e-20)
        # self.nec_suf_mask.data = self.nec_suf_mask.data ** 8
        # self.nec_suf_mask.data = self.nec_suf_mask.data - self.nec_suf_mask.data.min()
        # self.nec_suf_mask.data = self.nec_suf_mask.data / (self.nec_suf_mask.data.max() + 1e-20)
        # self.nec_suf_mask.data = self.nec_suf_mask.data * 1 - 0.5

        if self.no_mask:
            self.nec_suf_mask = nn.Parameter(100 * torch.ones((1, iter_weighted_change_walks_list.shape[1], 1), device=self.device))
        self.iter_weighted_change_walks_list = nn.Parameter(iter_weighted_change_walks_list.clone().detach())

        # --- Training ---
        walk_plain_indices_list = walk_indices_list + \
                                  (edge_index_with_loop.shape[1]
                                   * torch.arange(self.num_layers, device=self.device)).repeat(
                                      walk_indices_list.shape[0], 1)

        self.flow2layeredge_matrix = torch.stack([(walk_plain_indices_list == i).float().sum(dim=1)
                                                  for i in
                                                  range(self.num_layers * (self.num_edges + self.num_nodes))],
                                                 dim=1).detach()

        # train to get the mask
        optimizer = torch.optim.Adam([{'params': self.nec_suf_mask}],
                                      # {'params': self.iter_weighted_change_walks_list, 'lr': self.score_lr}],
                                     lr=self.lr)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300, 400], gamma=0.05)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300], gamma=0.05)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300, 450], gamma=1e-1)
        print('#I#begin this ex_label')
        for epoch in range(1, self.epochs + 1):

            masked_iter_weighted_change_walks_list = self.iter_weighted_change_walks_list * self.nec_suf_mask.sigmoid()

            walk_scores = (masked_iter_weighted_change_walks_list.unsqueeze(3).repeat(1, 1, 1,
                                                                                      iter_changed_subsets_score_list.shape[
                                                                                          2]) * iter_changed_subsets_score_list.unsqueeze(2)).sum(1).sum(0)
            # EPS will affect the stability of training
            EPS = 1e-18
            shap_flow_score = (walk_scores / (walk_sample_count.unsqueeze(1) + EPS))

            # --- score/mask transformer ---
            self.flow_mask = shap_flow_score[:, ex_label]

            # --- setting layer edge masks ---
            self.layer_edge_mask = (self.flow_mask * self.flow2layeredge_matrix).view(self.flow_mask.shape[0],
                                                                                      self.num_layers,
                                                                                      -1).sum(0)
            mask = self.layer_edge_mask.sum(0)
            mask = mask - mask.min()
            mask = mask / (mask.max() + EPS)

            # Option 1: make it harder: draw back: cannot control sparsity
            # mask = mask * 500 - 250
            # mask = mask.sigmoid()
            climb = True
            if climb:
                # pass
                mask = mask ** 8
            else:
                end_epoch = 300
                temperature = float(t0 * ((t1 / t0) ** (epoch / end_epoch))) if epoch < end_epoch else t1
                mask = gumbel_softmax(mask, temperature, training=True)

            mask = mask - mask.min()
            mask = mask / (mask.max() + EPS)
            cur_sparsity = (mask < 0.5).sum().float() / mask.shape[0]
            # if cur_sparsity < x_args.sparsity:
            #     # --- early stop ---
            #     break
            if epoch % 20 == 0:
                print(f'Epoch: {epoch} --- training mask Sparsity: {cur_sparsity}')
            if self.fidelity_plus:
                mask = 1 - mask # Fidelity +
            self.mask = mask
            isig_mask = torch.log(self.mask / (1 - self.mask + EPS) + EPS)


            # --- temp update non-leaf edge_mask
            temp_edge_mask = []
            for layer_idx in range(self.num_layers):
                # --- Attention self-loop will be put at last because of the model will do it ---
                temp_edge_mask.append(isig_mask)

            # debug:
            with self.temp_mask(self, temp_edge_mask):
                raw_preds = self.model(x, edge_index, **kwargs)
            loss = self.__loss__(raw_preds, ex_label)

            if epoch % 20 == 0:
                print(f'#D#Loss:{loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()
        return

    def flow_shap(self,
                  x,
                  edge_index,
                  edge_index_with_loop,
                  walk_indices_list
                  ):
        # --- Kernel algorithm ---

        # --- row: walk index, column: total score
        walk_sample_count = torch.zeros(walk_indices_list.shape[0], dtype=torch.float, device=self.device)

        # General setting
        iter_weighted_change_walks_list = []
        iter_changed_subsets_score_list = []

        # Random sample ns_iter * ns_per_iter times
        for iter_idx in range(self.ns_iter):

            # --- random index ---
            unmask_pool = torch.cat([walk_indices_list[:, layer].unique() + layer * edge_index_with_loop.shape[1]
                                     for layer in range(self.num_layers)])
            self.ns_per_iter = unmask_pool.shape[0] if data_args.model_level != 'node' or unmask_pool.shape[
                0] <= 100 else 100
            idx = torch.randperm(unmask_pool.nelement())
            unmask_pool = unmask_pool.view(-1)[idx].view(unmask_pool.size())

            mask_per_sub = unmask_pool.shape[0] // self.ns_per_iter
            weighted_change_walks_list = []
            last_eliminated_walks = torch.zeros(walk_indices_list.shape[0], dtype=torch.bool, device=self.device)
            layer_edge_mask_list = []
            for sub_idx in range(self.ns_per_iter):
                # --- sub random index ---
                mask_pool = unmask_pool[: mask_per_sub * (sub_idx + 1)]

                # --- obtain changed walk idx ---
                eliminated_layer_edges = unmask_pool[mask_per_sub * sub_idx: mask_per_sub * (sub_idx + 1)]
                walk_plain_indices_list = walk_indices_list + \
                                          (edge_index_with_loop.shape[1] * torch.arange(self.num_layers,
                                                                                        device=self.device)).repeat(
                                              walk_indices_list.shape[0], 1)
                eliminated_walks = torch.stack([walk_plain_indices_list == edge for edge in eliminated_layer_edges],
                                               dim=0).long().sum(0).sum(1).bool().long()
                weighted_changed_walks = eliminated_walks.clone().float()
                weighted_changed_walks[eliminated_walks == last_eliminated_walks] = 0.
                weighted_changed_walks /= (weighted_changed_walks > 1e-20).sum() + 1e-30
                weighted_change_walks_list.append(weighted_changed_walks)
                last_eliminated_walks = eliminated_walks

                # --- setting a subset mask ---
                layer_edge_masks = torch.ones((self.num_layers, edge_index_with_loop.shape[1]),
                                              device=self.device)
                layer_edge_masks.view(-1)[mask_pool] -= 2
                layer_edge_mask_list.append(layer_edge_masks)

            weighted_change_walks_list = torch.stack(weighted_change_walks_list, dim=0)
            iter_weighted_change_walks_list.append(weighted_change_walks_list.detach())
            layer_edge_mask_list = torch.stack(layer_edge_mask_list, dim=0) * float('inf')

            # --- compute subsets' outputs of current iteration ---
            for layer_idx in range(self.num_layers):
                # --- Attention self-loop will be put at last because of the model will do it ---
                self.edge_mask[layer_idx].data = torch.cat(
                    [layer_edge_mask_list[:, layer_idx, :self.num_edges].reshape(-1),
                     layer_edge_mask_list[:, layer_idx, self.num_edges:].reshape(-1)])

            batch = self.batch_input(x, edge_index, self.ns_per_iter)
            subsets_output = self.model(data=batch).softmax(1).detach()
            if data_args.model_level == 'node':
                subsets_output = subsets_output.view(self.ns_per_iter, -1, data_args.num_classes)[:, self.node_idx]
                last_subsets_output = torch.cat(
                    [self.ori_logits_pred[self.node_idx].unsqueeze(0), subsets_output.clone()[:-1]], dim=0)
            else:
                last_subsets_output = torch.cat([self.ori_logits_pred, subsets_output.clone()[:-1]], dim=0)

            changed_subsets_score_list = (last_subsets_output - subsets_output).detach()
            iter_changed_subsets_score_list.append(changed_subsets_score_list)

            walk_sample_count += (weighted_change_walks_list > 1e-30).float().sum(0)

        # iter x subset_idx x flow_idx
        iter_weighted_change_walks_list = torch.stack(iter_weighted_change_walks_list, dim=0)
        iter_changed_subsets_score_list = torch.stack(iter_changed_subsets_score_list, dim=0)

        return iter_weighted_change_walks_list, iter_changed_subsets_score_list, walk_sample_count

class FlowShap_minus(FlowBase):
    coeffs = {
        'edge_size': 5e-4,
        'edge_ent': 1e-1
    }

    def __init__(self, model, epochs=500, lr=3e-1, explain_graph=False, molecule=False):
    # def __init__(self, model, epochs=500, lr=1e-1, explain_graph=False, molecule=False):
        super().__init__(model=model, epochs=epochs, lr=lr, explain_graph=explain_graph, molecule=molecule)

        self.score_structure = [(i % 2, term_idx)
                                for i in range(1, self.num_layers + 1)
                                for term_idx in combinations(range(self.num_layers), i)
                                ]

        self.ns_iter = 30
        self.ns_per_iter = None
        self.fidelity_plus = False
        self.score_lr = 0e-5 #2e-5
        # self.alpha = 0.5

        self.no_mask = False
        if self.no_mask:
            self.epochs = 1
            self.lr = 0
            self.score_lr = 0


    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                **kwargs
                ) -> Union[Tuple[None, List, List[Dict]], Tuple[Dict, List, List[Dict]]]:

        super().forward(x, edge_index, **kwargs)

        # Explanation initial process
        self.model.eval()

        # Initial original prediction
        self.ori_logits_pred = self.model(x, edge_index).softmax(1)
        print(f'#I#pred label: {torch.argmax(self.ori_logits_pred)}')


        # Edge Index with self loop
        edge_index, _ = remove_self_loops(edge_index)
        edge_index_with_loop, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)
        walk_indices_list = torch.tensor(self.walks_pick(edge_index_with_loop.cpu(), list(range(edge_index_with_loop.shape[1])),
                                                         num_layers=self.num_layers), device=self.device)

        if data_args.model_level == 'node':
            node_idx = kwargs.get('node_idx')
            self.node_idx = node_idx
            assert node_idx is not None
            _, _, _, self.hard_edge_mask = subgraph(
                node_idx, self.__num_hops__, edge_index_with_loop, relabel_nodes=True,
                num_nodes=None, flow=self.__flow__())

            # walk indices list mask
            edge2node_idx = edge_index_with_loop[1] == node_idx
            walk_indices_list_mask = edge2node_idx[walk_indices_list[:, -1]]
            walk_indices_list = walk_indices_list[walk_indices_list_mask]


        import time
        start = time.time()
        self.time_list = []
        labels = tuple(i for i in range(data_args.num_classes))
        ex_labels = tuple(torch.tensor([label]).to(data_args.device) for label in labels)

        force_recalculate = x_args.force_recalculate
        explain_index = kwargs.get('index')
        store_path = os.path.join(ROOT_DIR, 'masks', f'{x_args.explainer}_tmp', f'{x_args.dataset_name}_{x_args.model_name}')
        store_file = os.path.join(store_path, f'{explain_index}.pt')

        if not os.path.exists(store_path):
            os.makedirs(store_path)
            print(f'#W#create dirs {store_path}')

        if explain_index is None\
            or not os.path.exists(store_file) \
            or force_recalculate:
            # --- without saving output mask before ---

            # Connect mask
            with self.connect_mask(self):
                iter_weighted_change_walks_list, iter_changed_subsets_score_list, walk_sample_count = \
                    self.flow_shap(x, edge_index, edge_index_with_loop, walk_indices_list)

            walk_score_list = []
            for ex_label in ex_labels:
                # --- training ---
                self.train_mask(x,
                                edge_index,
                                ex_label,
                                walk_indices_list,
                                edge_index_with_loop,
                                iter_weighted_change_walks_list,
                                iter_changed_subsets_score_list,
                                walk_sample_count)

                walk_score_list.append(self.flow_mask.data)

            walks = {'ids': walk_indices_list, 'score': torch.cat(walk_score_list, dim=1)}
            # print(f'Walk scores summation: {torch.cat(walk_score_list, dim=0).sum(0)}')
            print(f'#D#WalkScore total time: {time.time() - start}\n'
                  f'predict time: {sum(self.time_list)}')

            # --- save results for different Sparsity ---
            torch.save(walks, store_file)
        else:
            walks = torch.load(store_file)

        # specify to edge with self-loop mask prediction
        labels = tuple(i for i in range(data_args.num_classes))
        ex_labels = tuple(torch.tensor([label]).to(data_args.device) for label in labels)
        start = time.time()
        masks = []
        for ex_label in ex_labels:
            edge_attr = self.explain_edges_with_loop(x, walks, ex_label)
            mask = edge_attr
            mask = self.control_sparsity(mask, kwargs.get('sparsity'))
            masks.append(mask.detach())
        print(f'#D#Edge mask predict total time: {time.time() - start}')

        # Connect mask
        with self.connect_mask(self):
            related_preds = self.eval_related_pred(x, edge_index, masks, **kwargs)


        return walks, masks, related_preds

    def __loss__(self, raw_preds, x_label):
        if self.explain_graph:
            loss = Metric.loss_func(raw_preds, x_label)
        else:
            loss = Metric.loss_func(raw_preds[self.node_idx].unsqueeze(0), x_label)

        if self.fidelity_plus:
            loss = - loss

        # Option 2: make it hard: higher std, and closer to Sparsity
        # loss = loss - 10 * torch.square(self.mask - 0.5).mean()

        # loss = loss + 1e-3 * torch.square(self.mask.sum() - self.mask.shape[0] * (1 - x_args.sparsity))
        m = self.nec_suf_mask.sigmoid()
        loss = loss + self.coeffs['edge_size'] * m.sum()
        # ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        # loss = loss + self.coeffs['edge_ent'] * ent.mean()

        return loss

    def train_mask(self,
                   x: Tensor,
                   edge_index: Tensor,
                   ex_label: Tensor,
                   walk_indices_list,
                   edge_index_with_loop,
                   iter_weighted_change_walks_list,
                   iter_changed_subsets_score_list,
                   walk_sample_count,
                   t0=7.,
                   t1=0.5,
                   **kwargs
                   ) -> None:
        # initialize a mask
        self.to(x.device)

        # --- necesufy mask --- wrong!!!!! because of testing!!!!
        self.nec_suf_mask = nn.Parameter(1e-1 * nn.init.uniform_(torch.empty((1, iter_weighted_change_walks_list.shape[1], 1), device=self.device)))
        # self.nec_suf_mask = nn.Parameter(1 * torch.ones((1, iter_weighted_change_walks_list.shape[1], 1), device=self.device))

        # --- force higher Sparsity ---
        # self.nec_suf_mask.data = self.nec_suf_mask.data - self.nec_suf_mask.data.min()
        # self.nec_suf_mask.data = self.nec_suf_mask.data / (self.nec_suf_mask.data.max() + 1e-20)
        # self.nec_suf_mask.data = self.nec_suf_mask.data ** 8
        # self.nec_suf_mask.data = self.nec_suf_mask.data - self.nec_suf_mask.data.min()
        # self.nec_suf_mask.data = self.nec_suf_mask.data / (self.nec_suf_mask.data.max() + 1e-20)
        # self.nec_suf_mask.data = self.nec_suf_mask.data * 1 - 0.5

        if self.no_mask:
            self.nec_suf_mask = nn.Parameter(100 * torch.ones((1, iter_weighted_change_walks_list.shape[1], 1), device=self.device))
        self.iter_weighted_change_walks_list = nn.Parameter(iter_weighted_change_walks_list.clone().detach())

        # --- Training ---
        walk_plain_indices_list = walk_indices_list + \
                                  (edge_index_with_loop.shape[1]
                                   * torch.arange(self.num_layers, device=self.device)).repeat(
                                      walk_indices_list.shape[0], 1)

        self.flow2layeredge_matrix = torch.stack([(walk_plain_indices_list == i).float().sum(dim=1)
                                                  for i in
                                                  range(self.num_layers * (self.num_edges + self.num_nodes))],
                                                 dim=1).detach()

        # train to get the mask
        optimizer = torch.optim.Adam([{'params': self.nec_suf_mask}],
                                      # {'params': self.iter_weighted_change_walks_list, 'lr': self.score_lr}],
                                     lr=self.lr)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300, 400], gamma=0.05)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300], gamma=0.05)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300, 450], gamma=1e-1)
        print('#I#begin this ex_label')
        for epoch in range(1, self.epochs + 1):

            masked_iter_weighted_change_walks_list = self.iter_weighted_change_walks_list * self.nec_suf_mask.sigmoid()

            walk_scores = (masked_iter_weighted_change_walks_list.unsqueeze(3).repeat(1, 1, 1,
                                                                                      iter_changed_subsets_score_list.shape[
                                                                                          2]) * iter_changed_subsets_score_list.unsqueeze(2)).sum(1).sum(0)
            # EPS will affect the stability of training
            EPS = 1e-18
            shap_flow_score = (walk_scores / (walk_sample_count.unsqueeze(1) + EPS))

            # --- score/mask transformer ---
            self.flow_mask = shap_flow_score[:, ex_label]

            # --- setting layer edge masks ---
            self.layer_edge_mask = (self.flow_mask * self.flow2layeredge_matrix).view(self.flow_mask.shape[0],
                                                                                      self.num_layers,
                                                                                      -1).sum(0)
            mask = self.layer_edge_mask.sum(0)
            mask = mask - mask.min()
            mask = mask / (mask.max() + EPS)

            # Option 1: make it harder: draw back: cannot control sparsity
            # mask = mask * 500 - 250
            # mask = mask.sigmoid()
            climb = True
            if climb:
                # pass
                mask = mask ** 8
            else:
                end_epoch = 300
                temperature = float(t0 * ((t1 / t0) ** (epoch / end_epoch))) if epoch < end_epoch else t1
                mask = gumbel_softmax(mask, temperature, training=True)

            mask = mask - mask.min()
            mask = mask / (mask.max() + EPS)
            cur_sparsity = (mask < 0.5).sum().float() / mask.shape[0]
            # if cur_sparsity < x_args.sparsity:
            #     # --- early stop ---
            #     break
            if epoch % 20 == 0:
                print(f'Epoch: {epoch} --- training mask Sparsity: {cur_sparsity}')
            if self.fidelity_plus:
                mask = 1 - mask # Fidelity +
            self.mask = mask
            isig_mask = torch.log(self.mask / (1 - self.mask + EPS) + EPS)


            # --- temp update non-leaf edge_mask
            temp_edge_mask = []
            for layer_idx in range(self.num_layers):
                # --- Attention self-loop will be put at last because of the model will do it ---
                temp_edge_mask.append(isig_mask)

            # debug:
            with self.temp_mask(self, temp_edge_mask):
                raw_preds = self.model(x, edge_index, **kwargs)
            loss = self.__loss__(raw_preds, ex_label)

            if epoch % 20 == 0:
                print(f'#D#Loss:{loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()
        return

    def flow_shap(self,
                  x,
                  edge_index,
                  edge_index_with_loop,
                  walk_indices_list
                  ):
        # --- Kernel algorithm ---

        # --- row: walk index, column: total score
        walk_sample_count = torch.zeros(walk_indices_list.shape[0], dtype=torch.float, device=self.device)

        # General setting
        iter_weighted_change_walks_list = []
        iter_changed_subsets_score_list = []

        # Random sample ns_iter * ns_per_iter times
        for iter_idx in range(self.ns_iter):

            # --- random index ---
            unmask_pool = torch.cat([walk_indices_list[:, layer].unique() + layer * edge_index_with_loop.shape[1]
                                     for layer in range(self.num_layers)])
            self.ns_per_iter = unmask_pool.shape[0] if data_args.model_level != 'node' or unmask_pool.shape[
                0] <= 100 else 100
            idx = torch.randperm(unmask_pool.nelement())
            unmask_pool = unmask_pool.view(-1)[idx].view(unmask_pool.size())

            mask_per_sub = unmask_pool.shape[0] // self.ns_per_iter
            weighted_change_walks_list = []
            last_eliminated_walks = torch.zeros(walk_indices_list.shape[0], dtype=torch.bool, device=self.device)
            layer_edge_mask_list = []
            for sub_idx in range(self.ns_per_iter):
                # --- sub random index ---
                mask_pool = unmask_pool[: mask_per_sub * (sub_idx + 1)]

                # --- obtain changed walk idx ---
                eliminated_layer_edges = unmask_pool[mask_per_sub * sub_idx: mask_per_sub * (sub_idx + 1)]
                walk_plain_indices_list = walk_indices_list + \
                                          (edge_index_with_loop.shape[1] * torch.arange(self.num_layers,
                                                                                        device=self.device)).repeat(
                                              walk_indices_list.shape[0], 1)
                eliminated_walks = torch.stack([walk_plain_indices_list == edge for edge in eliminated_layer_edges],
                                               dim=0).long().sum(0).sum(1).bool().long()
                weighted_changed_walks = eliminated_walks.clone().float()
                weighted_changed_walks[eliminated_walks == last_eliminated_walks] = 0.
                weighted_changed_walks /= (weighted_changed_walks > 1e-20).sum() + 1e-30
                weighted_change_walks_list.append(weighted_changed_walks)
                last_eliminated_walks = eliminated_walks

                # --- setting a subset mask ---
                layer_edge_masks = torch.ones((self.num_layers, edge_index_with_loop.shape[1]),
                                              device=self.device)
                layer_edge_masks.view(-1)[mask_pool] -= 2
                layer_edge_mask_list.append(layer_edge_masks)

            weighted_change_walks_list = torch.stack(weighted_change_walks_list, dim=0)
            iter_weighted_change_walks_list.append(weighted_change_walks_list.detach())
            layer_edge_mask_list = torch.stack(layer_edge_mask_list, dim=0) * float('inf')

            # --- compute subsets' outputs of current iteration ---
            for layer_idx in range(self.num_layers):
                # --- Attention self-loop will be put at last because of the model will do it ---
                self.edge_mask[layer_idx].data = torch.cat(
                    [layer_edge_mask_list[:, layer_idx, :self.num_edges].reshape(-1),
                     layer_edge_mask_list[:, layer_idx, self.num_edges:].reshape(-1)])

            batch = self.batch_input(x, edge_index, self.ns_per_iter)
            subsets_output = self.model(data=batch).softmax(1).detach()
            if data_args.model_level == 'node':
                subsets_output = subsets_output.view(self.ns_per_iter, -1, data_args.num_classes)[:, self.node_idx]
                last_subsets_output = torch.cat(
                    [self.ori_logits_pred[self.node_idx].unsqueeze(0), subsets_output.clone()[:-1]], dim=0)
            else:
                last_subsets_output = torch.cat([self.ori_logits_pred, subsets_output.clone()[:-1]], dim=0)

            changed_subsets_score_list = (last_subsets_output - subsets_output).detach()
            iter_changed_subsets_score_list.append(changed_subsets_score_list)

            walk_sample_count += (weighted_change_walks_list > 1e-30).float().sum(0)

        # iter x subset_idx x flow_idx
        iter_weighted_change_walks_list = torch.stack(iter_weighted_change_walks_list, dim=0)
        iter_changed_subsets_score_list = torch.stack(iter_changed_subsets_score_list, dim=0)

        return iter_weighted_change_walks_list, iter_changed_subsets_score_list, walk_sample_count

class FlowShap_gumbel(FlowBase):
    coeffs = {
        'edge_size': 5e-4,
        'edge_ent': 1e-1
    }

    # def __init__(self, model, epochs=500, lr=3e-1, explain_graph=False, molecule=False):
    def __init__(self, model, epochs=500, lr=1e-1, explain_graph=False, molecule=False):
        super().__init__(model=model, epochs=epochs, lr=lr, explain_graph=explain_graph, molecule=molecule)

        self.score_structure = [(i % 2, term_idx)
                                for i in range(1, self.num_layers + 1)
                                for term_idx in combinations(range(self.num_layers), i)
                                ]

        self.ns_iter = 30
        self.ns_per_iter = None
        self.fidelity_plus = True
        self.score_lr = 0e-5 #2e-5
        # self.alpha = 0.5

        self.no_mask = False
        if self.no_mask:
            self.epochs = 1
            self.lr = 0
            self.score_lr = 0


    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                **kwargs
                ) -> Union[Tuple[None, List, List[Dict]], Tuple[Dict, List, List[Dict]]]:

        super().forward(x, edge_index, **kwargs)

        # Explanation initial process
        self.model.eval()

        # Initial original prediction
        self.ori_logits_pred = self.model(x, edge_index).softmax(1)
        print(f'#I#pred label: {torch.argmax(self.ori_logits_pred)}')


        # Edge Index with self loop
        edge_index, _ = remove_self_loops(edge_index)
        edge_index_with_loop, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)
        walk_indices_list = torch.tensor(self.walks_pick(edge_index_with_loop.cpu(), list(range(edge_index_with_loop.shape[1])),
                                                         num_layers=self.num_layers), device=self.device)

        if data_args.model_level == 'node':
            node_idx = kwargs.get('node_idx')
            self.node_idx = node_idx
            assert node_idx is not None
            _, _, _, self.hard_edge_mask = subgraph(
                node_idx, self.__num_hops__, edge_index_with_loop, relabel_nodes=True,
                num_nodes=None, flow=self.__flow__())

            # walk indices list mask
            edge2node_idx = edge_index_with_loop[1] == node_idx
            walk_indices_list_mask = edge2node_idx[walk_indices_list[:, -1]]
            walk_indices_list = walk_indices_list[walk_indices_list_mask]


        import time
        start = time.time()
        self.time_list = []

        # Connect mask
        with self.connect_mask(self):
            iter_weighted_change_walks_list, iter_changed_subsets_score_list, walk_sample_count = \
                self.flow_shap(x, edge_index, edge_index_with_loop, walk_indices_list)


        labels = tuple(i for i in range(data_args.num_classes))
        ex_labels = tuple(torch.tensor([label]).to(data_args.device) for label in labels)

        walk_score_list = []
        for ex_label in ex_labels:
            # --- training ---
            self.train_mask(x,
                   edge_index,
                   ex_label,
                   walk_indices_list,
                   edge_index_with_loop,
                   iter_weighted_change_walks_list,
                   iter_changed_subsets_score_list,
                   walk_sample_count)

            walk_score_list.append(self.flow_mask.data)


        walks = {'ids': walk_indices_list, 'score': torch.cat(walk_score_list, dim=1)}
        # print(f'Walk scores summation: {torch.cat(walk_score_list, dim=0).sum(0)}')
        print(f'#D#WalkScore total time: {time.time() - start}\n'
              f'predict time: {sum(self.time_list)}')
        # exit()

        # specify to edge with self-loop mask prediction
        labels = tuple(i for i in range(data_args.num_classes))
        ex_labels = tuple(torch.tensor([label]).to(data_args.device) for label in labels)
        start = time.time()
        masks = []
        for ex_label in ex_labels:
            edge_attr = self.explain_edges_with_loop(x, walks, ex_label)
            mask = edge_attr
            mask = self.control_sparsity(mask, kwargs.get('sparsity'))
            masks.append(mask.detach())
        print(f'#D#Edge mask predict total time: {time.time() - start}')

        # Connect mask
        with self.connect_mask(self):
            related_preds = self.eval_related_pred(x, edge_index, masks, **kwargs)


        return walks, masks, related_preds

    def __loss__(self, raw_preds, x_label):
        if self.explain_graph:
            loss = Metric.loss_func(raw_preds, x_label)
        else:
            loss = Metric.loss_func(raw_preds[self.node_idx].unsqueeze(0), x_label)

        if self.fidelity_plus:
            loss = - loss

        # Option 2: make it hard: higher std, and closer to Sparsity
        # loss = loss - 10 * torch.square(self.mask - 0.5).mean()

        # loss = loss + 1e-3 * torch.square(self.mask.sum() - self.mask.shape[0] * (1 - x_args.sparsity))
        # m = self.nec_suf_mask.sigmoid()
        # loss = loss + self.coeffs['edge_size'] * m.sum()
        # ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        # loss = loss + self.coeffs['edge_ent'] * ent.mean()

        return loss

    def train_mask(self,
                   x: Tensor,
                   edge_index: Tensor,
                   ex_label: Tensor,
                   walk_indices_list,
                   edge_index_with_loop,
                   iter_weighted_change_walks_list,
                   iter_changed_subsets_score_list,
                   walk_sample_count,
                   t0=7.,
                   t1=0.5,
                   **kwargs
                   ) -> None:
        # initialize a mask
        self.to(x.device)

        # --- necesufy mask ---
        self.nec_suf_mask = nn.Parameter(1e-1 * nn.init.uniform_(torch.empty((1, iter_weighted_change_walks_list.shape[1], 1), device=self.device)))

        # --- force higher Sparsity ---
        # self.nec_suf_mask.data = self.nec_suf_mask.data - self.nec_suf_mask.data.min()
        # self.nec_suf_mask.data = self.nec_suf_mask.data / (self.nec_suf_mask.data.max() + 1e-20)
        # self.nec_suf_mask.data = self.nec_suf_mask.data ** 8
        # self.nec_suf_mask.data = self.nec_suf_mask.data - self.nec_suf_mask.data.min()
        # self.nec_suf_mask.data = self.nec_suf_mask.data / (self.nec_suf_mask.data.max() + 1e-20)
        # self.nec_suf_mask.data = self.nec_suf_mask.data * 1 - 0.5

        if self.no_mask:
            self.nec_suf_mask = nn.Parameter(100 * torch.ones((1, iter_weighted_change_walks_list.shape[1], 1), device=self.device))
        self.iter_weighted_change_walks_list = nn.Parameter(iter_weighted_change_walks_list.clone().detach())

        # --- Training ---
        walk_plain_indices_list = walk_indices_list + \
                                  (edge_index_with_loop.shape[1]
                                   * torch.arange(self.num_layers, device=self.device)).repeat(
                                      walk_indices_list.shape[0], 1)

        self.flow2layeredge_matrix = torch.stack([(walk_plain_indices_list == i).float().sum(dim=1)
                                                  for i in
                                                  range(self.num_layers * (self.num_edges + self.num_nodes))],
                                                 dim=1).detach()

        # train to get the mask
        optimizer = torch.optim.Adam([{'params': self.nec_suf_mask}],
                                      # {'params': self.iter_weighted_change_walks_list, 'lr': self.score_lr}],
                                     lr=self.lr)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300, 400], gamma=0.05)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300], gamma=0.05)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300, 450], gamma=1e-1)
        print('#I#begin this ex_label')
        for epoch in range(1, self.epochs + 1):

            masked_iter_weighted_change_walks_list = self.iter_weighted_change_walks_list * self.nec_suf_mask.sigmoid()

            walk_scores = (masked_iter_weighted_change_walks_list.unsqueeze(3).repeat(1, 1, 1,
                                                                                      iter_changed_subsets_score_list.shape[
                                                                                          2]) * iter_changed_subsets_score_list.unsqueeze(2)).sum(1).sum(0)
            # EPS will affect the stability of training
            EPS = 1e-22
            shap_flow_score = (walk_scores / (walk_sample_count.unsqueeze(1) + EPS))

            # --- score/mask transformer ---
            self.flow_mask = shap_flow_score[:, ex_label]

            # --- setting layer edge masks ---
            self.layer_edge_mask = (self.flow_mask * self.flow2layeredge_matrix).view(self.flow_mask.shape[0],
                                                                                      self.num_layers,
                                                                                      -1).sum(0)
            mask = self.layer_edge_mask.sum(0)
            mask = mask - mask.min()
            mask = mask / (mask.max() + EPS)

            # Option 1: make it harder: draw back: cannot control sparsity
            # mask = mask * 500 - 250
            # mask = mask.sigmoid()
            climb = False
            if climb:
                mask = mask ** 8
            else:
                end_epoch = 300
                temperature = float(t0 * ((t1 / t0) ** (epoch / end_epoch))) if epoch < end_epoch else t1
                mask = gumbel_softmax(mask, temperature, training=True)

            mask = mask - mask.min()
            mask = mask / (mask.max() + EPS)
            cur_sparsity = (mask < 0.5).sum().float() / mask.shape[0]
            # if cur_sparsity < x_args.sparsity:
            #     # --- early stop ---
            #     break
            if epoch % 20 == 0:
                print(f'Epoch: {epoch} --- training mask Sparsity: {cur_sparsity}')
            if self.fidelity_plus:
                mask = 1 - mask # Fidelity +
            self.mask = mask
            isig_mask = torch.log(self.mask / (1 - self.mask + EPS) + EPS)


            # --- temp update non-leaf edge_mask
            temp_edge_mask = []
            for layer_idx in range(self.num_layers):
                # --- Attention self-loop will be put at last because of the model will do it ---
                temp_edge_mask.append(isig_mask)

            # debug:
            with self.temp_mask(self, temp_edge_mask):
                raw_preds = self.model(x, edge_index, **kwargs)
            loss = self.__loss__(raw_preds, ex_label)

            if epoch % 20 == 0:
                print(f'#D#Loss:{loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        return

    def flow_shap(self,
                  x,
                  edge_index,
                  edge_index_with_loop,
                  walk_indices_list
                  ):
        # --- Kernel algorithm ---

        # --- row: walk index, column: total score
        walk_sample_count = torch.zeros(walk_indices_list.shape[0], dtype=torch.float, device=self.device)

        # General setting
        iter_weighted_change_walks_list = []
        iter_changed_subsets_score_list = []

        # Random sample ns_iter * ns_per_iter times
        for iter_idx in range(self.ns_iter):

            # --- random index ---
            unmask_pool = torch.cat([walk_indices_list[:, layer].unique() + layer * edge_index_with_loop.shape[1]
                                     for layer in range(self.num_layers)])
            self.ns_per_iter = unmask_pool.shape[0] if data_args.model_level != 'node' or unmask_pool.shape[
                0] <= 100 else 100
            idx = torch.randperm(unmask_pool.nelement())
            unmask_pool = unmask_pool.view(-1)[idx].view(unmask_pool.size())

            mask_per_sub = unmask_pool.shape[0] // self.ns_per_iter
            weighted_change_walks_list = []
            last_eliminated_walks = torch.zeros(walk_indices_list.shape[0], dtype=torch.bool, device=self.device)
            layer_edge_mask_list = []
            for sub_idx in range(self.ns_per_iter):
                # --- sub random index ---
                mask_pool = unmask_pool[: mask_per_sub * (sub_idx + 1)]

                # --- obtain changed walk idx ---
                eliminated_layer_edges = unmask_pool[mask_per_sub * sub_idx: mask_per_sub * (sub_idx + 1)]
                walk_plain_indices_list = walk_indices_list + \
                                          (edge_index_with_loop.shape[1] * torch.arange(self.num_layers,
                                                                                        device=self.device)).repeat(
                                              walk_indices_list.shape[0], 1)
                eliminated_walks = torch.stack([walk_plain_indices_list == edge for edge in eliminated_layer_edges],
                                               dim=0).long().sum(0).sum(1).bool().long()
                weighted_changed_walks = eliminated_walks.clone().float()
                weighted_changed_walks[eliminated_walks == last_eliminated_walks] = 0.
                weighted_changed_walks /= (weighted_changed_walks > 1e-20).sum() + 1e-30
                weighted_change_walks_list.append(weighted_changed_walks)
                last_eliminated_walks = eliminated_walks

                # --- setting a subset mask ---
                layer_edge_masks = torch.ones((self.num_layers, edge_index_with_loop.shape[1]),
                                              device=self.device)
                layer_edge_masks.view(-1)[mask_pool] -= 2
                layer_edge_mask_list.append(layer_edge_masks)

            weighted_change_walks_list = torch.stack(weighted_change_walks_list, dim=0)
            iter_weighted_change_walks_list.append(weighted_change_walks_list.detach())
            layer_edge_mask_list = torch.stack(layer_edge_mask_list, dim=0) * float('inf')

            # --- compute subsets' outputs of current iteration ---
            for layer_idx in range(self.num_layers):
                # --- Attention self-loop will be put at last because of the model will do it ---
                self.edge_mask[layer_idx].data = torch.cat(
                    [layer_edge_mask_list[:, layer_idx, :self.num_edges].reshape(-1),
                     layer_edge_mask_list[:, layer_idx, self.num_edges:].reshape(-1)])

            batch = self.batch_input(x, edge_index, self.ns_per_iter)
            subsets_output = self.model(data=batch).softmax(1).detach()
            if data_args.model_level == 'node':
                subsets_output = subsets_output.view(self.ns_per_iter, -1, data_args.num_classes)[:, self.node_idx]
                last_subsets_output = torch.cat(
                    [self.ori_logits_pred[self.node_idx].unsqueeze(0), subsets_output.clone()[:-1]], dim=0)
            else:
                last_subsets_output = torch.cat([self.ori_logits_pred, subsets_output.clone()[:-1]], dim=0)

            changed_subsets_score_list = (last_subsets_output - subsets_output).detach()
            iter_changed_subsets_score_list.append(changed_subsets_score_list)

            walk_sample_count += (weighted_change_walks_list > 1e-30).float().sum(0)

        # iter x subset_idx x flow_idx
        iter_weighted_change_walks_list = torch.stack(iter_weighted_change_walks_list, dim=0)
        iter_changed_subsets_score_list = torch.stack(iter_changed_subsets_score_list, dim=0)

        return iter_weighted_change_walks_list, iter_changed_subsets_score_list, walk_sample_count

from benchmark.models.ext.Gem.generate_ground_truth_graph_classification import gen_gt
from benchmark.models.ext.Gem.explainer_gae_graph import train_vae
class ArgsStorage(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class Gem(FlowBase):

    def __init__(self, model: nn.Module, epochs=0, lr=0, explain_graph=False, molecule=False):
        super().__init__(model=model, epochs=epochs, lr=lr, explain_graph=explain_graph, molecule=molecule)
        self.gen_gt = gen_gt
        self.train_args = {'graph': ArgsStorage(distillation=None, dataset=None, output=None, graph_labeling=True, degree_feat=False,
                                      neigh_degree_feat=0, gae3=True, hidden1=32, hidden2=16, dropout=0,
                                      explain_class=None, loss='mse', load_checkpoint=None, plot=False,
                                      batch_size=128, epochs=300, lr=0.01, early_stop=True, normalize_feat=False,
                                      weighted=True),
                           'node': ArgsStorage(distillation=None, dataset=None, output=None, graph_labeling=False, degree_feat=False,
                                      neigh_degree_feat=0, gae3=False, hidden1=32, hidden2=16, dropout=0,
                                      explain_class=None, loss='mse', load_checkpoint=None, plot=False,
                                      batch_size=32, epochs=100, lr=0.001, early_stop=False, normalize_feat=False,
                                      weighted=False)}
        self.train_vae = train_vae

    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                **kwargs
                ):

        # --- run the model once ---
        super().forward(x=x, edge_index=edge_index, **kwargs)
        self.model.eval()
        self_loop_edge_index, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)

        if data_args.model_level == 'node':
            node_idx = kwargs.get('node_idx')
            assert node_idx is not None
            _, _, _, self.hard_edge_mask = subgraph(
                node_idx, self.__num_hops__, self_loop_edge_index, relabel_nodes=True,
                num_nodes=None, flow=self.__flow__())


        labels = tuple(i for i in range(data_args.num_classes))
        ex_labels = tuple(torch.tensor([label]).to(data_args.device) for label in labels)

        print('#D#Mask Calculate...')
        masks = []
        for ex_label in ex_labels:
            graph_idx = kwargs.get('index')
            recovered_file = os.path.join(self.train_args[data_args.model_level].output, 'test', f'graph_idx_{graph_idx}_pred.csv')
            recovered = np.loadtxt(recovered_file, delimiter=',')
            recovered = torch.tensor(recovered, dtype=torch.float32, device=self.device)
            mask = recovered[self_loop_edge_index[0], self_loop_edge_index[1]]
            if mask.shape.__len__() == 0:
                mask = mask.unsqueeze(0)
            mask = self.control_sparsity(mask, kwargs.get('sparsity'))
            masks.append(mask.detach())

        # Store related predictions for further evaluation.
        print('#D#Predict...')

        with torch.no_grad():
            with self.connect_mask(self):
                related_preds = self.eval_related_pred(x, edge_index, masks, **kwargs)


        return None, masks, related_preds


class PGExplainer(ExplainerBase):

    def __init__(self, model, epochs=100, lr=3e-1, explain_graph=False, molecule=False):
        super().__init__(model=model, epochs=epochs, lr=lr, explain_graph=explain_graph, molecule=molecule)
        self.pg = pgex(model, in_channels=600, device=self.device, explain_graph=explain_graph)

    def forward(self, x, edge_index, mask_features=False,
                positive=True, **kwargs):
        r"""Learns and returns a node feature mask and an edge mask that play a
        crucial role to explain the prediction made by the GNN for node
        :attr:`node_idx`.

        Args:
            data (Batch): batch from dataloader
            edge_index (LongTensor): The edge indices.
            pos_neg (Literal['pos', 'neg']) : get positive or negative mask
            **kwargs (optional): Additional arguments passed to the GNN module.

        :rtype: (:class:`Tensor`, :class:`Tensor`)
        """
        self.model.eval()
        super().forward(x, edge_index)

        # --- substitute edge_index with self loops ---
        edge_index, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)

        # Only operate on a k-hop subgraph around `node_idx`.
        # Get subgraph and relabel the node, mapping is the relabeled given node_idx.
        if data_args.model_level == 'node':
            node_idx = kwargs.get('node_idx')
            self.node_idx = node_idx
            assert node_idx is not None
            _, _, _, self.hard_edge_mask = subgraph(
                node_idx, self.__num_hops__, edge_index, relabel_nodes=True,
                num_nodes=None, flow=self.__flow__())

        # Assume the mask we will predict
        labels = tuple(i for i in range(data_args.num_classes))
        ex_labels = tuple(torch.tensor([label]).to(data_args.device) for label in labels)

        # Calculate mask
        print('#D#Masks calculate...')
        edge_masks = []
        for ex_label in ex_labels:

            edge_masks.append(self.control_sparsity(self.pg(x, edge_index, **kwargs)[1][0], sparsity=kwargs.get('sparsity')))
            # edge_masks.append(self.gnn_explainer_alg(x, edge_index, ex_label))


        print('#D#Predict...')
        self.__clear_masks__()
        self.__set_masks__(x, edge_index)
        with torch.no_grad():
            related_preds = self.eval_related_pred(x, edge_index, edge_masks, **kwargs)

        self.__clear_masks__()

        return None, edge_masks, related_preds


class PGMExplainer(FlowBase):

    def __init__(self, model, epochs=100, lr=0.01, explain_graph=False, molecule=False):
        super().__init__(model, epochs, lr, explain_graph, molecule)

    def forward(self, x: Tensor, edge_index: Tensor, **kwargs)\
            -> Union[Tuple[None, List, List[Dict]], Tuple[Dict, List, List[Dict]]]:
        """
        Given a sample, this function will return its predicted masks and corresponding predictions
        for evaluation
        :param x: Tensor - Hiden features of all vertexes
        :param edge_index: Tensor - All connected edge between vertexes/nodes
        :param kwargs:
        :return:
        """
        self.model.eval()
        super().forward(x, edge_index)

        labels = tuple(i for i in range(data_args.num_classes))
        ex_labels = tuple(torch.tensor([label]).to(data_args.device) for label in labels)


        self_loop_edge_index, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)

        if data_args.model_level == 'node':
            node_idx = kwargs.get('node_idx')
            assert node_idx is not None
            subset, _, _, self.hard_edge_mask = subgraph(
                node_idx, self.__num_hops__, self_loop_edge_index, relabel_nodes=True,
                num_nodes=None, flow=self.__flow__())

        force_recalculate = x_args.force_recalculate
        explain_index = kwargs.get('index')
        store_path = os.path.join(ROOT_DIR, 'masks', f'{x_args.explainer}_tmp', f'{x_args.dataset_name}_{x_args.model_name}')
        store_file = os.path.join(store_path, f'{explain_index}.pt')

        if not os.path.exists(store_path):
            os.makedirs(store_path)
            print(f'#W#create dirs {store_path}')

        if explain_index is None \
                or not os.path.exists(store_file) \
                or force_recalculate:
            # --- without saving output mask before ---

            print('#D#Mask Calculate...')
            if data_args.model_level == 'node':
                e = pgmn.Node_Explainer(self.model, self_loop_edge_index, x, self.num_layers)
                explanation = e.explain(node_idx, subset, self.device, num_samples=100,
                                                top_node=None)
                subset_p_values = np.array(explanation[3])
                p_values = np.zeros(x.shape[0], dtype=np.float)
                p_values[subset.cpu()] = subset_p_values
            else:
                perturb_features_list = [i for i in range(x.shape[1])]
                pred = self.model(x, self_loop_edge_index)
                soft_pred = torch.softmax(pred[0], 0)
                pred_threshold = 0.1 * torch.max(soft_pred)
                e = pgmg.Graph_Explainer(self.model, x, self_loop_edge_index,
                                       perturb_feature_list=perturb_features_list)
                pgm_nodes, p_values, candidates = e.explain(num_samples=1000, percentage=10,
                                                            top_node=None, p_threshold=0.05,
                                                            pred_threshold=pred_threshold)
            mask = - torch.tensor(p_values, dtype=torch.float32, device=self.device).sigmoid() # p_values: less is more important
            if mask.shape.__len__() == 0:
                mask = mask.unsqueeze(0)
            mask = (mask[self_loop_edge_index[0]] + mask[self_loop_edge_index[1]]) / 2

            # --- save results for different Sparsity ---
            torch.save(mask, store_file)
        else:
            mask = torch.load(store_file)

        mask = self.control_sparsity(mask, kwargs.get('sparsity'))
        masks = [mask.detach() for _ in ex_labels]


        # Store related predictions for further evaluation.
        print('#D#Predict...')

        with torch.no_grad():
            with self.connect_mask(self):
                related_preds = self.eval_related_pred(x, edge_index, masks, **kwargs)



        return None, masks, related_preds


class FlowMask(FlowBase):
    coeffs = {
        'edge_size': 0.005,
        'edge_ent': 1.0
    }

    def __init__(self, model, epochs=100, lr=3e-1, explain_graph=False, molecule=False):
        super().__init__(model=model, epochs=epochs, lr=lr, explain_graph=explain_graph, molecule=molecule)

    def __loss__(self, raw_preds, x_label):
        if self.explain_graph:
            loss = - Metric.loss_func(raw_preds, x_label)
        else:
            loss = - Metric.loss_func(raw_preds[self.node_idx].unsqueeze(0), x_label)

        m = self.flow_mask.sigmoid()
        loss = loss + self.coeffs['edge_size'] * m.sum()
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        loss = loss + self.coeffs['edge_ent'] * ent.mean()

        return loss

    def train_mask(self,
                  x: Tensor,
                  edge_index: Tensor,
                  ex_label: Tensor,
                  **kwargs
                  ) -> None:

        # initialize a mask
        self.to(x.device)

        # train to get the mask
        optimizer = torch.optim.Adam([self.flow_mask],
                                     lr=self.lr)

        for epoch in range(1, self.epochs + 1):
            # --- setting layer edge masks ---
            self.layer_edge_mask = (self.flow_mask * self.flow2layeredge_matrix).view(self.flow_mask.shape[0], self.num_layers,
                                                                            -1).sum(0) * self.layer_mask

            # --- temp update non-leaf edge_mask
            temp_edge_mask = []
            for layer_idx in range(self.num_layers):
                # --- Attention self-loop will be put at last because of the model will do it ---
                # --- Minus ---
                temp_edge_mask.append(- torch.cat(
                    [self.layer_edge_mask[layer_idx, :self.num_edges].reshape(-1),
                     self.layer_edge_mask[layer_idx, self.num_edges:].reshape(-1)]))

            # debug:
            with self.temp_mask(self, temp_edge_mask):
                raw_preds = self.model(x, edge_index, **kwargs)
            loss = self.__loss__(raw_preds, ex_label)

            if epoch % 20 == 0:
                print(f'#D#Loss:{loss.item()}')


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return

    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                **kwargs
                ) -> Union[Tuple[None, List, List[Dict]], Tuple[Dict, List, List[Dict]]]:

        super().forward(x, edge_index, **kwargs)

        # Explanation initial process
        self.model.eval()

        # Initial original prediction
        self.ori_logits_pred = self.model(x, edge_index).softmax(1)


        # Connect mask
        # self.__connect_mask__()

        # Edge Index with self loop
        edge_index, _ = remove_self_loops(edge_index)
        edge_index_with_loop, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)
        walk_indices_list = torch.tensor(self.walks_pick(edge_index_with_loop.cpu(), list(range(edge_index_with_loop.shape[1])),
                                                         num_layers=self.num_layers), device=self.device)

        if data_args.model_level == 'node':
            node_idx = kwargs.get('node_idx')
            self.node_idx = node_idx
            assert node_idx is not None
            _, _, _, self.hard_edge_mask = subgraph(
                node_idx, self.__num_hops__, edge_index_with_loop, relabel_nodes=True,
                num_nodes=None, flow=self.__flow__())

            # walk indices list mask
            edge2node_idx = edge_index_with_loop[1] == node_idx
            walk_indices_list_mask = edge2node_idx[walk_indices_list[:, -1]]
            walk_indices_list = walk_indices_list[walk_indices_list_mask]


        import time
        start = time.time()
        walk_score_list = []
        self.time_list = []

        force_recalculate = x_args.force_recalculate
        explain_index = kwargs.get('index')
        store_path = os.path.join(ROOT_DIR, 'masks', f'{x_args.explainer}_tmp', f'{x_args.dataset_name}_{x_args.model_name}')
        store_file = os.path.join(store_path, f'{explain_index}.pt')

        if not os.path.exists(store_path):
            os.makedirs(store_path)
            print(f'#W#create dirs {store_path}')

        if explain_index is None \
                or not os.path.exists(store_file) \
                or force_recalculate:
        # --- without saving output mask before ---
            # --- Kernel algorithm ---

            # --- Flow mask initialize ---

            walk_plain_indices_list = walk_indices_list + \
                                      (edge_index_with_loop.shape[1]
                                       * torch.arange(self.num_layers, device=self.device)).repeat(
                                          walk_indices_list.shape[0], 1)

            self.flow2layeredge_matrix = torch.stack([(walk_plain_indices_list == i).float().sum(dim=1)
                                                 for i in
                                                 range(self.num_layers * (self.num_edges + self.num_nodes))],
                                                dim=1).detach()

            labels = tuple(i for i in range(data_args.num_classes))
            ex_labels = tuple(torch.tensor([label]).to(data_args.device) for label in labels)

            for ex_label in ex_labels:
                self.flow_mask = nn.Parameter(
                    torch.nn.init.xavier_normal_(torch.empty((walk_indices_list.shape[0], 1), dtype=torch.float, device=self.device))
                    * 10,
                    requires_grad=True
                )

                self.layer_mask = nn.Parameter(torch.ones((self.num_layers, 1), dtype=torch.float, device=self.device),
                                          requires_grad=True)

                # --- training ---
                self.train_mask(x, edge_index, ex_label=ex_label)

                walk_score_list.append(self.flow_mask.data)


            # Attention: we may do SHAPLEY here on walk_score_list with the help of walk_indices_list as player_list
            walks = {'ids': walk_indices_list, 'score': torch.cat(walk_score_list, dim=1)}
            # print(f'Walk scores summation: {torch.cat(walk_score_list, dim=0).sum(0)}')
            print(f'#D#WalkScore total time: {time.time() - start}\n'
                  f'predict time: {sum(self.time_list)}')

            # --- save results for different Sparsity ---
            torch.save(walks, store_file)
        else:
            walks = torch.load(store_file)

        # specify to edge with self-loop mask prediction
        labels = tuple(i for i in range(data_args.num_classes))
        ex_labels = tuple(torch.tensor([label]).to(data_args.device) for label in labels)
        start = time.time()
        masks = []
        for ex_label in ex_labels:
            edge_attr = self.explain_edges_with_loop(x, walks, ex_label)
            mask = edge_attr
            mask = self.control_sparsity(mask, kwargs.get('sparsity'))
            # mask[mask >= 1e-1] = float('inf')
            # mask[mask < 1e-1] = - float('inf')
            masks.append(mask.detach())
        print(f'#D#Edge mask predict total time: {time.time() - start}')
        with self.connect_mask(self):
            related_preds = self.eval_related_pred(x, edge_index, masks, **kwargs)


        return walks, masks, related_preds


class GNN_GI(FlowBase):

    def __init__(self, model: nn.Module, epochs=0, lr=0, explain_graph=False, molecule=False):
        super().__init__(model=model, epochs=epochs, lr=lr, explain_graph=explain_graph, molecule=molecule)

    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                **kwargs
                ):
        super().forward(x, edge_index, **kwargs)
        self.model.eval()
        self_loop_edge_index, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)

        walk_steps, fc_step = self.extract_step(x, edge_index, detach=False)


        if data_args.model_level == 'node':
            node_idx = kwargs.get('node_idx')
            assert node_idx is not None
            _, _, _, self.hard_edge_mask = subgraph(
                node_idx, self.__num_hops__, self_loop_edge_index, relabel_nodes=True,
                num_nodes=None, flow=self.__flow__())


        def compute_walk_score(adjs, r, allow_edges, walk_idx=[]):
            if not adjs:
                walk_indices.append(walk_idx)
                walk_scores.append(r.detach())
                return
            (grads,) = torch.autograd.grad(outputs=r, inputs=adjs[0], create_graph=True)
            for i in allow_edges:
                allow_edges= torch.where(self_loop_edge_index[1] == self_loop_edge_index[0][i])[0].tolist()
                new_r = grads[i] * adjs[0][i]
                compute_walk_score(adjs[1:], new_r, allow_edges, [i] + walk_idx)


        labels = tuple(i for i in range(data_args.num_classes))
        walk_scores_tensor_list = [None for i in labels]

        force_recalculate = x_args.force_recalculate
        explain_index = kwargs.get('index')
        store_path = os.path.join(ROOT_DIR, 'masks', f'{x_args.explainer}_tmp', f'{x_args.dataset_name}_{x_args.model_name}')
        store_file = os.path.join(store_path, f'{explain_index}.pt')

        if not os.path.exists(store_path):
            os.makedirs(store_path)
            print(f'#W#create dirs {store_path}')

        if explain_index is None \
                or not os.path.exists(store_file) \
                or force_recalculate:

            for label in labels:

                if self.explain_graph:
                    f = torch.unbind(fc_step['output'][0, label].unsqueeze(0))
                    allow_edges = [i for i in range(self_loop_edge_index.shape[1])]
                else:
                    f = torch.unbind(fc_step['output'][node_idx, label].unsqueeze(0))
                    allow_edges = torch.where(self_loop_edge_index[1] == node_idx)[0].tolist()

                adjs = [walk_step['module'][0].edge_weight for walk_step in walk_steps]

                reverse_adjs = adjs.reverse()
                walk_indices = []
                walk_scores = []

                compute_walk_score(adjs, f, allow_edges)
                walk_scores_tensor_list[label] = torch.stack(walk_scores, dim=0).view(-1, 1)

            walks = {'ids': torch.tensor(walk_indices, device=self.device), 'score': torch.cat(walk_scores_tensor_list, dim=1)}

            # --- save results for different Sparsity ---
            torch.save(walks, store_file)
        else:
            print('skip predict')
            walks = torch.load(store_file)

        # --- Apply edge mask evaluation ---
        with torch.no_grad():
            with self.connect_mask(self):
                ex_labels = tuple(torch.tensor([label]).to(data_args.device) for label in labels)
                masks = []
                for ex_label in ex_labels:
                    edge_attr = self.explain_edges_with_loop(x, walks, ex_label)
                    mask = edge_attr
                    mask = self.control_sparsity(mask, kwargs.get('sparsity'))
                    masks.append(mask.detach())

                related_preds = self.eval_related_pred(x, edge_index, masks, **kwargs)

        return walks, masks, related_preds


class GNN_GIR(FlowBase):

    def __init__(self, model: nn.Module, epochs=0, lr=0, explain_graph=False, molecule=False):
        super().__init__(model=model, epochs=epochs, lr=lr, explain_graph=explain_graph, molecule=molecule)

    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                **kwargs
                ):
        super().forward(x, edge_index, **kwargs)
        self.model.eval()
        self_loop_edge_index, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)

        walk_steps, fc_step = self.extract_step(x, edge_index, detach=False)
        adjs = [walk_step['module'][0].edge_weight.clone().detach() for walk_step in walk_steps]

        ref_walk_steps, ref_fc_step = walk_steps, fc_step
        for i, ref_walk_step in enumerate(ref_walk_steps):
            ref_walk_step['input'] = ref_walk_steps[i - 1]['output'] if i != 0 else ref_walk_step['input']
            edge_weight = torch.zeros(ref_walk_step['module'][0].edge_weight.shape, device=self.device)
            ref_walk_step['output'] = GraphSequential(*ref_walk_step['module'])(ref_walk_step['input'],
                                                                                self_loop_edge_index, edge_weight)
        ref_fc_step['input'] = ref_walk_step['output']
        ref_fc_step['output'] = GraphSequential(*ref_fc_step['module'])(ref_fc_step['input'], torch.zeros(ref_fc_step['input'].shape[0], dtype=torch.long, device=self.device))


        def compute_walk_score(ref_adjs, adjs, r, allow_edges, walk_idx=[]):
            if not adjs:
                walk_indices.append(walk_idx)
                walk_scores.append(r.detach())
                return
            (grads,) = torch.autograd.grad(outputs=r, inputs=ref_adjs[0], create_graph=True)
            for i in allow_edges:
                allow_edges= torch.where(self_loop_edge_index[1] == self_loop_edge_index[0][i])[0].tolist()
                new_r = grads[i] * adjs[0][i]
                compute_walk_score(ref_adjs[1:], adjs[1:], new_r, allow_edges, [i] + walk_idx)


        labels = tuple(i for i in range(data_args.num_classes))
        walk_scores_tensor_list = [None for i in labels]
        for label in labels:
            ref_f = ref_fc_step['output'][0, label]

            ref_adjs = [ref_walk_step['module'][0].edge_weight for ref_walk_step in ref_walk_steps]

            reverse_ref_adjs = ref_adjs.reverse()
            reverse_adjs = adjs.reverse()
            walk_indices = []
            walk_scores = []
            allow_edges = [i for i in range(len(adjs[0]))]

            compute_walk_score(ref_adjs, adjs, ref_f, allow_edges)
            walk_scores_tensor_list[label] = torch.stack(walk_scores, dim=0).view(-1, 1)

        walks = {'ids': torch.tensor(walk_indices, device=self.device), 'score': torch.cat(walk_scores_tensor_list, dim=1)}

        # --- Apply edge mask evaluation ---
        with torch.no_grad():
            with self.connect_mask(self):
                ex_labels = tuple(torch.tensor([label]).to(data_args.device) for label in labels)
                masks = []
                for ex_label in ex_labels:
                    edge_attr = self.explain_edges_with_loop(x, walks, ex_label)
                    mask = edge_attr
                    mask = self.control_sparsity(mask, kwargs.get('sparsity'))
                    masks.append(mask.detach())

                related_preds = self.eval_related_pred(x, edge_index, masks, **kwargs)

        return None, masks, related_preds


class GNN_LRP(FlowBase):

    def __init__(self, model: nn.Module, epochs=0, lr=0, explain_graph=False, molecule=False):
        super().__init__(model=model, epochs=epochs, lr=lr, explain_graph=explain_graph, molecule=molecule)

    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                **kwargs
                ):
        super().forward(x, edge_index, **kwargs)
        self.model.eval()

        walk_steps, fc_steps = self.extract_step(x, edge_index, detach=False, split_fc=True)


        edge_index_with_loop, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)


        walk_indices_list = torch.tensor(
            self.walks_pick(edge_index_with_loop.cpu(), list(range(edge_index_with_loop.shape[1])),
                            num_layers=self.num_layers), device=self.device)
        if data_args.model_level == 'node':
            node_idx = kwargs.get('node_idx')
            assert node_idx is not None
            _, _, _, self.hard_edge_mask = subgraph(
                node_idx, self.__num_hops__, edge_index_with_loop, relabel_nodes=True,
                num_nodes=None, flow=self.__flow__())

            # walk indices list mask
            edge2node_idx = edge_index_with_loop[1] == node_idx
            walk_indices_list_mask = edge2node_idx[walk_indices_list[:, -1]]
            walk_indices_list = walk_indices_list[walk_indices_list_mask]


        def compute_walk_score():

            # hyper-parameter gamma
            epsilon = 1e-20   # prevent from zero division
            gamma = [2, 1, 1]

            # --- record original weights of GNN ---
            ori_gnn_weights = []
            gnn_gamma_modules = []
            clear_probe = x
            for i, walk_step in enumerate(walk_steps):
                modules = walk_step['module']
                gamma_ = gamma[i] if i <= 1 else 1
                if hasattr(modules[0], 'nn'):
                    clear_probe = modules[0](clear_probe, edge_index, probe=False)
                    # clear nodes that are not created by user
                gamma_module = copy.deepcopy(modules[0])
                if hasattr(modules[0], 'nn'):
                    for j, fc_step in enumerate(gamma_module.fc_steps):
                        fc_modules = fc_step['module']
                        if hasattr(fc_modules[0], 'weight'):
                            ori_fc_weight = fc_modules[0].weight.data
                            fc_modules[0].weight.data = ori_fc_weight + gamma_ * ori_fc_weight
                else:
                    ori_gnn_weights.append(modules[0].weight.data)
                    gamma_module.weight.data = ori_gnn_weights[i] + gamma_ * ori_gnn_weights[i].relu()
                gnn_gamma_modules.append(gamma_module)

            # --- record original weights of fc layer ---
            ori_fc_weights = []
            fc_gamma_modules = []
            for i, fc_step in enumerate(fc_steps):
                modules = fc_step['module']
                gamma_module = copy.deepcopy(modules[0])
                if hasattr(modules[0], 'weight'):
                    ori_fc_weights.append(modules[0].weight.data)
                    gamma_ = 1
                    gamma_module.weight.data = ori_fc_weights[i] + gamma_ * ori_fc_weights[i].relu()
                else:
                    ori_fc_weights.append(None)
                fc_gamma_modules.append(gamma_module)

            # --- GNN_LRP implementation ---
            for walk_indices in walk_indices_list:
                walk_node_indices = [edge_index_with_loop[0, walk_indices[0]]]
                for walk_idx in walk_indices:
                    walk_node_indices.append(edge_index_with_loop[1, walk_idx])

                h = x.requires_grad_(True)
                for i, walk_step in enumerate(walk_steps):
                    modules = walk_step['module']
                    if hasattr(modules[0], 'nn'):
                        # for the specific 2-layer nn GINs.
                        gin = modules[0]
                        run1 = gin(h, edge_index, probe=True)
                        std_h1 = gin.fc_steps[0]['output']
                        gamma_run1 = gnn_gamma_modules[i](h, edge_index, probe=True)
                        p1 = gnn_gamma_modules[i].fc_steps[0]['output']
                        q1 = (p1 + epsilon) * (std_h1 / (p1 + epsilon)).detach()

                        std_h2 = GraphSequential(*gin.fc_steps[1]['module'])(q1)
                        p2 = GraphSequential(*gnn_gamma_modules[i].fc_steps[1]['module'])(q1)
                        q2 = (p2 + epsilon) * (std_h2 / (p2 + epsilon)).detach()
                        q = q2
                    else:

                        std_h = GraphSequential(*modules)(h, edge_index)

                        # --- LRP-gamma ---
                        p = gnn_gamma_modules[i](h, edge_index)
                        q = (p + epsilon) * (std_h / (p + epsilon)).detach()

                    # --- pick a path ---
                    mk = torch.zeros((h.shape[0], 1), device=self.device)
                    k = walk_node_indices[i + 1]
                    mk[k] = 1
                    ht = q * mk + q.detach() * (1 - mk)
                    h = ht

                # --- FC LRP_gamma ---
                for i, fc_step in enumerate(fc_steps):
                    modules = fc_step['module']
                    std_h = nn.Sequential(*modules)(h) if i != 0 \
                        else GraphSequential(*modules)(h, torch.zeros(h.shape[0], dtype=torch.long, device=self.device))

                    # --- gamma ---
                    s = fc_gamma_modules[i](h) if i != 0 \
                        else fc_gamma_modules[i](h, torch.zeros(h.shape[0], dtype=torch.long, device=self.device))
                    ht = (s + epsilon) * (std_h / (s + epsilon)).detach()
                    h = ht

                if data_args.model_level == 'node':
                    f = h[node_idx, label]
                else:
                    f = h[0, label]
                x_grads = torch.autograd.grad(outputs=f, inputs=x)[0]
                I = walk_node_indices[0]
                r = x_grads[I, :] @ x[I].T
                walk_scores.append(r)

        labels = tuple(i for i in range(data_args.num_classes))
        walk_scores_tensor_list = [None for i in labels]

        force_recalculate = x_args.force_recalculate
        explain_index = kwargs.get('index')
        store_path = os.path.join(ROOT_DIR, 'masks', f'{x_args.explainer}_tmp', f'{x_args.dataset_name}_{x_args.model_name}')
        store_file = os.path.join(store_path, f'{explain_index}.pt')
        # D:\OneDrive - mail.ustc.edu.cn\Code\PythonProjects\GNN_benchmark\masks\GNN_LRP_tmp\ba_shapes_GCN_2l\0.pt
        if not os.path.exists(store_path):
            os.makedirs(store_path)
            print(f'#W#create dirs {store_path}')

        if explain_index is None \
                or not os.path.exists(store_file) \
                or force_recalculate:

            for label in labels:

                walk_scores = []

                compute_walk_score()
                walk_scores_tensor_list[label] = torch.stack(walk_scores, dim=0).view(-1, 1)

            walks = {'ids': walk_indices_list, 'score': torch.cat(walk_scores_tensor_list, dim=1)}

            # --- save results for different Sparsity ---
            torch.save(walks, store_file)
        else:
            print('skip predict')
            walks = torch.load(store_file)

        # --- Debug ---
        # walk_node_indices_list = []
        # for walk_indices in walk_indices_list:
        #     walk_node_indices = [edge_index_with_loop[0, walk_indices[0]]]
        #     for walk_idx in walk_indices:
        #         walk_node_indices.append(edge_index_with_loop[1, walk_idx])
        #     walk_node_indices_list.append(torch.stack(walk_node_indices))
        # walk_node_indices_list = torch.stack(walk_node_indices_list, dim=0)
        # --- Debug end ---

        # --- Apply edge mask evaluation ---
        with torch.no_grad():
            with self.connect_mask(self):
                ex_labels = tuple(torch.tensor([label]).to(data_args.device) for label in labels)
                masks = []
                for ex_label in ex_labels:
                    edge_attr = self.explain_edges_with_loop(x, walks, ex_label)
                    mask = edge_attr
                    mask = self.control_sparsity(mask, kwargs.get('sparsity'))
                    masks.append(mask.detach())

                related_preds = self.eval_related_pred(x, edge_index, masks, **kwargs)

        return walks, masks, related_preds


class GNN_LRP_v2(FlowBase):

    def __init__(self, model: nn.Module, epochs=0, lr=0, explain_graph=False, molecule=False):
        super().__init__(model=model, epochs=epochs, lr=lr, explain_graph=explain_graph, molecule=molecule)

    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                **kwargs
                ):
        super().forward(x, edge_index, **kwargs)
        self.model.eval()
        self_loop_edge_index, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)

        walk_steps, fc_step = self.extract_step(x, edge_index, detach=False)

        linear_num = []

        def count_linear_num(module: nn.Module):
            if hasattr(module, 'weight'):
                linear_num.append(0)

        self.model.apply(count_linear_num)

        lrp = LRP_gamma(model=self.model, gamma_list=[2] + [1 for _ in range(len(linear_num))])
        self.model.apply(lrp._register_hooks)


        for i, walk_step in enumerate(walk_steps):
            walk_step['input'] = walk_steps[i - 1]['output'] if i != 0 else \
                walk_step['input'].requires_grad_(True)
            walk_step['output'] = GraphSequential(*walk_step['module'])(walk_step['input'],
                                                                        self_loop_edge_index,
                                                                        walk_step['module'][0].edge_weight)
        fc_step['input'] = walk_step['output']
        fc_step['output'] = GraphSequential(*fc_step['module'])(fc_step['input'],
                                                                torch.arange(1, dtype=torch.long,
                                                                             device=self.device).view(1, 1).repeat(1, x.shape[0]).reshape(-1))

        if data_args.model_level == 'node':
            node_idx = kwargs.get('node_idx')
            assert node_idx is not None
            _, _, _, self.hard_edge_mask = subgraph(
                node_idx, self.__num_hops__, self_loop_edge_index, relabel_nodes=True,
                num_nodes=None, flow=self.__flow__())

        def compute_walk_score(adjs, r, allow_edges, walk_idx=[]):
            if not adjs:
                walk_indices.append(walk_idx)
                walk_scores.append(r.detach())
                return
            (grads,) = torch.autograd.grad(outputs=r, inputs=adjs[0], create_graph=True)
            for i in allow_edges:
                if not walk_idx:
                    lrp.gamma = [2]
                else:
                    lrp.gamma = [1]
                allow_edges = torch.where(self_loop_edge_index[1] == self_loop_edge_index[0][i])[0].tolist()
                new_r = grads[i] * adjs[0][i]
                compute_walk_score(adjs[1:], new_r, allow_edges, [i] + walk_idx)

        labels = tuple(i for i in range(data_args.num_classes))
        walk_scores_tensor_list = [None for i in labels]
        for label in labels:

            if self.explain_graph:
                f = torch.unbind(fc_step['output'][0, label].unsqueeze(0))
                allow_edges = [i for i in range(self_loop_edge_index.shape[1])]
            else:
                f = torch.unbind(fc_step['output'][node_idx, label].unsqueeze(0))
                allow_edges = torch.where(self_loop_edge_index[1] == node_idx)[0].tolist()

            adjs = [walk_step['module'][0].edge_weight for walk_step in walk_steps]

            reverse_adjs = adjs.reverse()
            walk_indices = []
            walk_scores = []

            compute_walk_score(adjs, f, allow_edges)
            walk_scores_tensor_list[label] = torch.stack(walk_scores, dim=0).view(-1, 1)

        lrp._remove_hooks()
        walks = {'ids': torch.tensor(walk_indices, device=self.device),
                 'score': torch.cat(walk_scores_tensor_list, dim=1)}

        # --- Apply edge mask evaluation ---
        with torch.no_grad():
            with self.connect_mask(self):
                ex_labels = tuple(torch.tensor([label]).to(data_args.device) for label in labels)
                masks = []
                for ex_label in ex_labels:
                    edge_attr = self.explain_edges_with_loop(x, walks, ex_label)
                    mask = edge_attr
                    mask = self.control_sparsity(mask, kwargs.get('sparsity'))
                    masks.append(mask.detach())

                related_preds = self.eval_related_pred(x, edge_index, masks, **kwargs)

        return walks, masks, related_preds

class GNN_LRPB(FlowBase):

    def __init__(self, model: nn.Module, epochs=0, lr=0, explain_graph=False, molecule=False):
        super().__init__(model=model, epochs=epochs, lr=lr, explain_graph=explain_graph, molecule=molecule)

    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                **kwargs
                ):
        super().forward(x, edge_index, **kwargs)
        self.model.eval()

        walk_steps, fc_steps = self.extract_step(x, edge_index, detach=False, split_fc=True)


        edge_index_with_loop, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)
        walk_indices_list = [torch.tensor(
            self.walks_pick(edge_index_with_loop.cpu(), list(range(edge_index_with_loop.shape[1])),
                            num_layers=len_walk), device=self.device)
            for len_walk in range(1, self.num_layers + 1)]


        def compute_walk_score(len_walk):

            walk_scores = []

            # hyper-parameter gamma
            epsilon = 1e-30   # prevent from zero division
            gamma = [2, 1]

            # --- record original weights of GNN ---
            ori_gnn_weights = []
            gnn_gamma_modules = []
            for i, walk_step in enumerate(walk_steps):
                modules = walk_step['module']
                ori_gnn_weights.append(modules[0].weight.data)
                gamma_ = gamma[i] if i <= 1 else 0
                gamma_module = copy.deepcopy(modules[0])
                gamma_module.weight.data = ori_gnn_weights[i] + gamma_ * ori_gnn_weights[i].relu()
                gnn_gamma_modules.append(gamma_module)

            # --- record original weights of fc layer ---
            ori_fc_weights = []
            fc_gamma_modules = []
            for i, fc_step in enumerate(fc_steps):
                modules = fc_step['module']
                gamma_module = copy.deepcopy(modules[0])
                if hasattr(modules[0], 'weight'):
                    ori_fc_weights.append(modules[0].weight.data)
                    gamma_ = 0
                    gamma_module.weight.data = ori_fc_weights[i] + gamma_ * ori_fc_weights[i].relu()
                else:
                    ori_fc_weights.append(None)
                fc_gamma_modules.append(gamma_module)

            # --- GNN_LRP implementation ---
            for walk_indices in walk_indices_list[len_walk]:
                x.requires_grad_(True)

                # --- for bias walks ---
                step_less = len(walk_steps) - len(walk_indices)
                if step_less > 0:
                    # 0 has no meaning here
                    walk_indices = [0 for _ in range(step_less)] + walk_indices.tolist()
                    walk_start_node = gnn_gamma_modules[step_less - 1].bias
                else:
                    walk_start_node = x

                walk_node_indices = [edge_index_with_loop[0, walk_indices[0]]]
                for walk_idx in walk_indices:
                    walk_node_indices.append(edge_index_with_loop[1, walk_idx])

                h = x
                for i, walk_step in enumerate(walk_steps):
                    modules = walk_step['module']
                    std_h = GraphSequential(*modules)(h, edge_index)

                    # --- LRP-gamma ---
                    p = gnn_gamma_modules[i](h, edge_index)
                    q = (p + epsilon) * (std_h / (p + epsilon)).detach()

                    # --- pick a path ---
                    mk = torch.zeros((h.shape[0], 1), device=self.device)
                    k = walk_node_indices[i + 1]
                    mk[k] = 1
                    ht = q * mk + q.detach() * (1 - mk)
                    h = ht

                # --- FC LRP_gamma ---
                for i, fc_step in enumerate(fc_steps):
                    modules = fc_step['module']
                    std_h = nn.Sequential(*modules)(h) if i != 0 \
                        else GraphSequential(*modules)(h, torch.zeros(h.shape[0], dtype=torch.long, device=self.device))

                    # --- gamma ---
                    s = fc_gamma_modules[i](h) if i != 0 \
                        else fc_gamma_modules[i](h, torch.zeros(h.shape[0], dtype=torch.long, device=self.device))
                    ht = (s + epsilon) * (std_h / (s + epsilon)).detach()
                    h = ht

                f = h[0, label]
                grads = torch.autograd.grad(outputs=f, inputs=walk_start_node)[0]
                if walk_start_node is x:
                    I = walk_node_indices[0]
                    r = grads[I, :] @ x[I].T
                else:
                    r = grads @ walk_start_node.T
                walk_scores.append(r)

            return walk_scores


        labels = tuple(i for i in range(data_args.num_classes))
        walk_scores_tensor_list = [[None for i in labels] for _ in range(self.num_layers)]
        for label in labels:

            for len_walk in range(self.num_layers):

                walk_scores = compute_walk_score(len_walk)
                walk_scores_tensor_list[len_walk][label] = torch.stack(walk_scores, dim=0).view(-1, 1)

        walks = [{'ids': walk_indices_list[len_walk], 'score': torch.cat(walk_scores_tensor_list[len_walk], dim=1)}
                 for len_walk in range(self.num_layers)]

        # --- Apply edge mask evaluation ---
        with torch.no_grad():
            with self.connect_mask(self):
                ex_labels = tuple(torch.tensor([label]).to(data_args.device) for label in labels)
                masks = []
                for ex_label in ex_labels:
                    edge_attr = []
                    for len_walk in range(self.num_layers):
                        edge_attr.append(self.explain_edges_with_loop(x, walks[len_walk], ex_label))
                    edge_attr = torch.stack(edge_attr, dim=1).sum(dim=1)
                    mask = edge_attr
                    mask = self.control_sparsity(mask, kwargs.get('sparsity'))
                    masks.append(mask.detach())

                related_preds = self.eval_related_pred(x, edge_index, masks, **kwargs)

        return None, masks, related_preds


class GradCAM(FlowBase):

    def __init__(self, model, epochs=100, lr=0.01, explain_graph=False, molecule=False):
        super().__init__(model, epochs, lr, explain_graph, molecule)

    def forward(self, x: Tensor, edge_index: Tensor, **kwargs)\
            -> Union[Tuple[None, List, List[Dict]], Tuple[Dict, List, List[Dict]]]:
        """
        Given a sample, this function will return its predicted masks and corresponding predictions
        for evaluation
        :param x: Tensor - Hiden features of all vertexes
        :param edge_index: Tensor - All connected edge between vertexes/nodes
        :param kwargs:
        :return:
        """
        self.model.eval()
        super().forward(x, edge_index)

        labels = tuple(i for i in range(data_args.num_classes))
        ex_labels = tuple(torch.tensor([label]).to(data_args.device) for label in labels)


        self_loop_edge_index, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)

        if data_args.model_level == 'node':
            node_idx = kwargs.get('node_idx')
            assert node_idx is not None
            _, _, _, self.hard_edge_mask = subgraph(
                node_idx, self.__num_hops__, self_loop_edge_index, relabel_nodes=True,
                num_nodes=None, flow=self.__flow__())

        # --- setting GradCAM ---
        class model_node(nn.Module):
            def __init__(self, cls):
                super().__init__()
                self.cls = cls
                self.convs = cls.model.convs
            def forward(self, *args, **kwargs):
                return self.cls.model(*args, **kwargs)[node_idx].unsqueeze(0)
        if self.explain_graph:
            model = self.model
        else:
            model = model_node(self)
        self.explain_method = GraphLayerGradCam(model, model.convs[-1])
        # --- setting end ---

        print('#D#Mask Calculate...')
        masks = []
        for ex_label in ex_labels:
            attr_wo_relu = self.explain_method.attribute(x, ex_label, additional_forward_args=edge_index)
            mask = normalize(attr_wo_relu.relu())
            mask = mask.squeeze()
            if mask.shape.__len__() == 0:
                mask = mask.unsqueeze(0)
            mask = (mask[self_loop_edge_index[0]] + mask[self_loop_edge_index[1]]) / 2
            mask = self.control_sparsity(mask, kwargs.get('sparsity'))
            masks.append(mask.detach())

        # Store related predictions for further evaluation.
        print('#D#Predict...')

        with torch.no_grad():
            with self.connect_mask(self):
                related_preds = self.eval_related_pred(x, edge_index, masks, **kwargs)



        return None, masks, related_preds





class GNNExplainer(ExplainerBase):
    r"""The GNN-Explainer model from the `"GNNExplainer: Generating
    Explanations for Graph Neural Networks"
    <https://arxiv.org/abs/1903.03894>`_ paper for identifying compact subgraph
    structures and small subsets node features that play a crucial role in a
    GNN’s node-predictions.

    .. note::

        For an example of using GNN-Explainer, see `examples/gnn_explainer.py
        <https://github.com/rusty1s/pytorch_geometric/blob/master/examples/
        gnn_explainer.py>`_.

    Args:
        model (torch.nn.Module): The GNN module to explain.
        epochs (int, optional): The number of epochs to train.
            (default: :obj:`100`)
        lr (float, optional): The learning rate to apply.
            (default: :obj:`0.01`)
        log (bool, optional): If set to :obj:`False`, will not log any learning
            progress. (default: :obj:`True`)
    """

    coeffs = {
        'edge_size': 0.005,
        'node_feat_size': 1.0,
        'edge_ent': 1.0,
        'node_feat_ent': 0.1,
    }

    def __init__(self, model, epochs=100, lr=0.01, explain_graph=False, molecule=False):
        super(GNNExplainer, self).__init__(model, epochs, lr, explain_graph, molecule)



    def __loss__(self, raw_preds, x_label):
        if self.explain_graph:
            loss = Metric.loss_func(raw_preds, x_label)
        else:
            loss = Metric.loss_func(raw_preds[self.node_idx].unsqueeze(0), x_label)

        m = self.edge_mask.sigmoid()
        loss = loss + self.coeffs['edge_size'] * m.sum()
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        loss = loss + self.coeffs['edge_ent'] * ent.mean()

        if self.mask_features:
            m = self.node_feat_mask.sigmoid()
            loss = loss + self.coeffs['node_feat_size'] * m.sum()
            ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
            loss = loss + self.coeffs['node_feat_ent'] * ent.mean()

        return loss

    def gnn_explainer_alg(self,
                          x: Tensor,
                          edge_index: Tensor,
                          ex_label: Tensor,
                          mask_features: bool = False,
                          **kwargs
                          ) -> None:

        # initialize a mask
        self.to(x.device)
        self.mask_features = mask_features

        # train to get the mask
        optimizer = torch.optim.Adam([self.node_feat_mask, self.edge_mask],
                                     lr=self.lr)

        for epoch in range(1, self.epochs + 1):

            if mask_features:
                h = x * self.node_feat_mask.view(1, -1).sigmoid()
            else:
                h = x
            raw_preds = self.model(x=h, edge_index=edge_index, **kwargs)
            loss = self.__loss__(raw_preds, ex_label)
            if epoch % 20 == 0:
                print(f'#D#Loss:{loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return self.edge_mask.data

    def forward(self, x, edge_index, mask_features=False,
                positive=True, **kwargs):
        r"""Learns and returns a node feature mask and an edge mask that play a
        crucial role to explain the prediction made by the GNN for node
        :attr:`node_idx`.

        Args:
            data (Batch): batch from dataloader
            edge_index (LongTensor): The edge indices.
            pos_neg (Literal['pos', 'neg']) : get positive or negative mask
            **kwargs (optional): Additional arguments passed to the GNN module.

        :rtype: (:class:`Tensor`, :class:`Tensor`)
        """
        self.model.eval()
        super().forward(x, edge_index)

        self_loop_edge_index, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)

        # Only operate on a k-hop subgraph around `node_idx`.
        # Get subgraph and relabel the node, mapping is the relabeled given node_idx.
        if data_args.model_level == 'node':
            node_idx = kwargs.get('node_idx')
            self.node_idx = node_idx
            assert node_idx is not None
            _, _, _, self.hard_edge_mask = subgraph(
                node_idx, self.__num_hops__, self_loop_edge_index, relabel_nodes=True,
                num_nodes=None, flow=self.__flow__())

        # Assume the mask we will predict
        labels = tuple(i for i in range(data_args.num_classes))
        ex_labels = tuple(torch.tensor([label]).to(data_args.device) for label in labels)

        force_recalculate = x_args.force_recalculate
        explain_index = kwargs.get('index')
        store_path = os.path.join(ROOT_DIR, 'masks', f'{x_args.explainer}_tmp', f'{x_args.dataset_name}_{x_args.model_name}')
        store_file = os.path.join(store_path, f'{explain_index}.pt')

        if not os.path.exists(store_path):
            os.makedirs(store_path)
            print(f'#W#create dirs {store_path}')

        if explain_index is None \
                or not os.path.exists(store_file) \
                or force_recalculate:

            # Calculate mask
            print('#D#Masks calculate...')
            edge_masks = []
            for ex_label in ex_labels:
                self.__clear_masks__()
                self.__set_masks__(x, self_loop_edge_index)
                edge_masks.append(self.control_sparsity(self.gnn_explainer_alg(x, edge_index, ex_label), sparsity=kwargs.get('sparsity')))
                # edge_masks.append(self.gnn_explainer_alg(x, edge_index, ex_label))

            torch.save(edge_masks, store_file)
        else:
            self.__clear_masks__()
            self.__set_masks__(x, self_loop_edge_index)
            edge_masks = torch.load(store_file)

        edge_masks = [self.control_sparsity(edge_mask, sparsity=kwargs.get('sparsity'))for edge_mask in edge_masks]
        print('#D#Predict...')

        with torch.no_grad():
            related_preds = self.eval_related_pred(x, edge_index, edge_masks, **kwargs)

        self.__clear_masks__()

        return None, edge_masks, related_preds




    def __repr__(self):
        return f'{self.__class__.__name__}()'






class GraphLayerGradCam(ca.LayerGradCam):

    def __init__(
            self,
            forward_func: Callable,
            layer: Module,
            device_ids: Union[None, List[int]] = None,
    ) -> None:
        super().__init__(forward_func, layer, device_ids)

    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        target: TargetType = None,
        additional_forward_args: Any = None,
        attribute_to_layer_input: bool = False,
        relu_attributions: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        r"""
        Args:

            inputs (tensor or tuple of tensors):  Input for which attributions
                        are computed. If forward_func takes a single
                        tensor as input, a single input tensor should be provided.
                        If forward_func takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples, and if multiple input tensors
                        are provided, the examples must be aligned appropriately.
            target (int, tuple, tensor or list, optional):  Output indices for
                        which gradients are computed (for classification cases,
                        this is usually the target class).
                        If the network returns a scalar value per example,
                        no target index is necessary.
                        For general 2D outputs, targets can be either:

                        - a single integer or a tensor containing a single
                          integer, which is applied to all input examples

                        - a list of integers or a 1D tensor, with length matching
                          the number of examples in inputs (dim 0). Each integer
                          is applied as the target for the corresponding example.

                        For outputs with > 2 dimensions, targets can be either:

                        - A single tuple, which contains #output_dims - 1
                          elements. This target index is applied to all examples.

                        - A list of tuples with length equal to the number of
                          examples in inputs (dim 0), and each tuple containing
                          #output_dims - 1 elements. Each tuple is applied as the
                          target for the corresponding example.

                        Default: None
            additional_forward_args (any, optional): If the forward function
                        requires additional arguments other than the inputs for
                        which attributions should not be computed, this argument
                        can be provided. It must be either a single additional
                        argument of a Tensor or arbitrary (non-tuple) type or a
                        tuple containing multiple additional arguments including
                        tensors or any arbitrary python types. These arguments
                        are provided to forward_func in order following the
                        arguments in inputs.
                        Note that attributions are not computed with respect
                        to these arguments.
                        Default: None
            attribute_to_layer_input (bool, optional): Indicates whether to
                        compute the attributions with respect to the layer input
                        or output. If `attribute_to_layer_input` is set to True
                        then the attributions will be computed with respect to the
                        layer input, otherwise it will be computed with respect
                        to layer output.
                        Note that currently it is assumed that either the input
                        or the outputs of internal layers, depending on whether we
                        attribute to the input or output, are single tensors.
                        Support for multiple tensors will be added later.
                        Default: False
            relu_attributions (bool, optional): Indicates whether to
                        apply a ReLU operation on the final attribution,
                        returning only non-negative attributions. Setting this
                        flag to True matches the original GradCAM algorithm,
                        otherwise, by default, both positive and negative
                        attributions are returned.
                        Default: False

        Returns:
            *tensor* or tuple of *tensors* of **attributions**:
            - **attributions** (*tensor* or tuple of *tensors*):
                        Attributions based on GradCAM method.
                        Attributions will be the same size as the
                        output of the given layer, except for dimension 2,
                        which will be 1 due to summing over channels.
                        Attributions are returned in a tuple based on whether
                        the layer inputs / outputs are contained in a tuple
                        from a forward hook. For standard modules, inputs of
                        a single tensor are usually wrapped in a tuple, while
                        outputs of a single tensor are not.
        Examples::

            # >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
            # >>> # and returns an Nx10 tensor of class probabilities.
            # >>> # It contains a layer conv4, which is an instance of nn.conv2d,
            # >>> # and the output of this layer has dimensions Nx50x8x8.
            # >>> # It is the last convolution layer, which is the recommended
            # >>> # use case for GradCAM.
            # >>> net = ImageClassifier()
            # >>> layer_gc = LayerGradCam(net, net.conv4)
            # >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
            # >>> # Computes layer GradCAM for class 3.
            # >>> # attribution size matches layer output except for dimension
            # >>> # 1, so dimensions of attr would be Nx1x8x8.
            # >>> attr = layer_gc.attribute(input, 3)
            # >>> # GradCAM attributions are often upsampled and viewed as a
            # >>> # mask to the input, since the convolutional layer output
            # >>> # spatially matches the original input image.
            # >>> # This can be done with LayerAttribution's interpolate method.
            # >>> upsampled_attr = LayerAttribution.interpolate(attr, (32, 32))
        """
        inputs = _format_input(inputs)
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        gradient_mask = apply_gradient_requirements(inputs)
        # Returns gradient of output with respect to
        # hidden layer and hidden layer evaluated at each input.
        layer_gradients, layer_evals, is_layer_tuple = compute_layer_gradients_and_eval(
            self.forward_func,
            self.layer,
            inputs,
            target,
            additional_forward_args,
            device_ids=self.device_ids,
            attribute_to_layer_input=attribute_to_layer_input,
        )
        undo_gradient_requirements(inputs, gradient_mask)

        # Gradient Calculation end

        # what I add: shape from PyG to General PyTorch
        layer_gradients = tuple(layer_grad.transpose(0, 1).unsqueeze(0)
                                for layer_grad in layer_gradients)

        layer_evals = tuple(layer_eval.transpose(0, 1).unsqueeze(0)
                            for layer_eval in layer_evals)
        # end

        summed_grads = tuple(
            torch.mean(
                layer_grad,
                dim=tuple(x for x in range(2, len(layer_grad.shape))),
                keepdim=True,
            )
            for layer_grad in layer_gradients
        )

        scaled_acts = tuple(
            torch.sum(summed_grad * layer_eval, dim=1, keepdim=True)
            for summed_grad, layer_eval in zip(summed_grads, layer_evals)
        )
        if relu_attributions:
            scaled_acts = tuple(F.relu(scaled_act) for scaled_act in scaled_acts)

        # what I add: shape from General PyTorch to PyG

        scaled_acts = tuple(scaled_act.squeeze(0).transpose(0, 1)
                            for scaled_act in scaled_acts)

        # end

        return _format_attributions(is_layer_tuple, scaled_acts)


class FlowX(FlowBase):

    def __init__(self, model: nn.Module, epochs=0, lr=0, explain_graph=False, molecule=False):
        super().__init__(model=model, epochs=epochs, lr=lr, explain_graph=explain_graph, molecule=molecule)

    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                **kwargs
                ):

        # --- run the model once ---
        super().forward(x=x, edge_index=edge_index, **kwargs)
        self.model.eval()
        self_loop_edge_index, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)

        if data_args.model_level == 'node':
            node_idx = kwargs.get('node_idx')
            assert node_idx is not None
            _, _, _, self.hard_edge_mask = subgraph(
                node_idx, self.__num_hops__, self_loop_edge_index, relabel_nodes=True,
                num_nodes=None, flow=self.__flow__())

        labels = tuple(i for i in range(data_args.num_classes))

        force_recalculate = x_args.force_recalculate
        explain_index = kwargs.get('index')
        store_path = os.path.join(ROOT_DIR, 'masks', f'{x_args.explainer}_tmp', f'{x_args.dataset_name}_{x_args.model_name}')
        store_file = os.path.join(store_path, f'{explain_index}.pt')

        if not os.path.exists(store_path):
            os.makedirs(store_path)
            print(f'#W#create dirs {store_path}')

        if explain_index is None \
                or not os.path.exists(store_file) \
                or force_recalculate:

            walk_steps, fc_step = self.extract_step(x, self_loop_edge_index)

            # --- add shap calculation hook ---
            shap = DeepLift(self.model)
            self.model.apply(shap._register_hooks)

            for i, walk_step in enumerate(walk_steps):
                walk_step['input'] = walk_steps[i - 1]['output'] if i != 0 else \
                    torch.cat([walk_step['input'], walk_step['input']], dim=0).requires_grad_(True)
                edge_weight_with_ref = torch.cat([walk_step['module'][0].edge_weight,
                                         torch.zeros(walk_step['module'][0].edge_weight.shape, device=self.device)], dim=0)
                edge_index_with_ref = torch.cat([self_loop_edge_index, self_loop_edge_index + x.shape[0]], dim=1)
                walk_step['output'] = GraphSequential(*walk_step['module'])(walk_step['input'],
                                                                            edge_index_with_ref,
                                                                            edge_weight_with_ref)
            fc_step['input'] = walk_step['output']
            fc_step['output'] = GraphSequential(*fc_step['module'])(fc_step['input'],
                                                                    torch.arange(2, dtype=torch.long,
                                                                                 device=self.device).view(2, 1).repeat(1, x.shape[0]).reshape(-1))

            # --- Back distribute scores ---
            def compute_walk_score(adjs, C, allow_edges, walk_idx=[]):
                if not adjs:
                    walk_indices.append(walk_idx)
                    walk_scores.append(C.detach())
                    return
                (m, ) = torch.autograd.grad(outputs=C, inputs=adjs[0], create_graph=True)
                edge_weight, edge_weight_ref = torch.chunk(adjs[0], 2)
                Cs = torch.chunk(m, 2)[0] * (edge_weight - edge_weight_ref)
                for i in allow_edges:
                    allow_edges= torch.where(self_loop_edge_index[1] == self_loop_edge_index[0][i])[0].tolist()
                    new_C = Cs[i]
                    compute_walk_score(adjs[1:], new_C, allow_edges, [i] + walk_idx)


            walk_scores_tensor_list = [None for i in labels]
            for label in labels:
                if self.explain_graph:
                    f = torch.unbind(fc_step['output'][:, label])
                    allow_edges = [i for i in range(self_loop_edge_index.shape[1])]
                else:
                    f = torch.unbind(fc_step['output'][[node_idx, node_idx + x.shape[0]], label])
                    allow_edges = torch.where(self_loop_edge_index[1] == node_idx)[0].tolist()


                adjs = [walk_step['module'][0].edge_weight for walk_step in walk_steps]

                adjs.reverse()
                walk_indices = []
                walk_scores = []

                compute_walk_score(adjs, f, allow_edges)
                walk_scores_tensor_list[label] = torch.stack(walk_scores, dim=0).view(-1, 1)

            # --- release hooks ---
            shap._remove_hooks()
            walks = {'ids': torch.tensor(walk_indices, device=self.device), 'score': torch.cat(walk_scores_tensor_list, dim=1)}

            # --- save results for different Sparsity ---
            torch.save(walks, store_file)
        else:
            walks = torch.load(store_file)


        # --- Apply edge mask evaluation ---
        with torch.no_grad():
            with self.connect_mask(self):
                ex_labels = tuple(torch.tensor([label]).to(data_args.device) for label in labels)
                masks = []
                for ex_label in ex_labels:
                    edge_attr = self.explain_edges_with_loop(x, walks, ex_label)
                    mask = edge_attr
                    mask = self.control_sparsity(mask, kwargs.get('sparsity'))
                    masks.append(mask.detach())

                related_preds = self.eval_related_pred(x, edge_index, masks, **kwargs)

        return walks, masks, related_preds


class DeepLIFT(FlowBase):

    def __init__(self, model: nn.Module, epochs=0, lr=0, explain_graph=False, molecule=False):
        super().__init__(model=model, epochs=epochs, lr=lr, explain_graph=explain_graph, molecule=molecule)

    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                **kwargs
                ):

        # --- run the model once ---
        super().forward(x=x, edge_index=edge_index, **kwargs)
        self.model.eval()
        self_loop_edge_index, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)

        if data_args.model_level == 'node':
            node_idx = kwargs.get('node_idx')
            assert node_idx is not None
            _, _, _, self.hard_edge_mask = subgraph(
                node_idx, self.__num_hops__, self_loop_edge_index, relabel_nodes=True,
                num_nodes=None, flow=self.__flow__())

        # --- add shap calculation hook ---
        shap = DeepLift(self.model)
        self.model.apply(shap._register_hooks)

        inp_with_ref = torch.cat([x, torch.zeros(x.shape, device=self.device, dtype=torch.float)], dim=0).requires_grad_(True)
        edge_index_with_ref = torch.cat([edge_index, edge_index + x.shape[0]], dim=1)
        batch = torch.arange(2, dtype=torch.long, device=self.device).view(2, 1).repeat(1, x.shape[0]).reshape(-1)
        out = self.model(inp_with_ref, edge_index_with_ref, batch)


        labels = tuple(i for i in range(data_args.num_classes))
        ex_labels = tuple(torch.tensor([label]).to(data_args.device) for label in labels)

        print('#D#Mask Calculate...')
        masks = []
        for ex_label in ex_labels:

            if self.explain_graph:
                f = torch.unbind(out[:, ex_label])
            else:
                f = torch.unbind(out[[node_idx, node_idx + x.shape[0]], ex_label])

            (m, ) = torch.autograd.grad(outputs=f, inputs=inp_with_ref, retain_graph=True)
            inp, inp_ref = torch.chunk(inp_with_ref, 2)
            attr_wo_relu = (torch.chunk(m, 2)[0] * (inp - inp_ref)).sum(1)

            mask = attr_wo_relu.squeeze()
            if mask.shape.__len__() == 0:
                mask = mask.unsqueeze(0)
            mask = (mask[self_loop_edge_index[0]] + mask[self_loop_edge_index[1]]) / 2
            mask = self.control_sparsity(mask, kwargs.get('sparsity'))
            masks.append(mask.detach())

        # Store related predictions for further evaluation.
        shap._remove_hooks()
        print('#D#Predict...')

        with torch.no_grad():
            with self.connect_mask(self):
                related_preds = self.eval_related_pred(x, edge_index, masks, **kwargs)


        return None, masks, related_preds


class TransSha(FlowBase):

    def __init__(self, model: nn.Module, epochs=0, lr=0, explain_graph=False, molecule=False):
        super().__init__(model=model, epochs=epochs, lr=lr, explain_graph=explain_graph, molecule=molecule)

    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                **kwargs
                ):

        # --- run the model once ---
        super().forward(x=x, edge_index=edge_index, **kwargs)
        self.model.eval()
        self_loop_edge_index, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)

        walk_steps, fc_step = self.extract_step(x, self_loop_edge_index)

        # --- add shap calculation hook ---
        shap = DeepLift(self.model)
        self.model.apply(shap._register_hooks)

        for i, walk_step in enumerate(walk_steps):
            walk_step['input'] = walk_steps[i - 1]['output'] if i != 0 else \
                torch.cat([walk_step['input'], walk_step['input']], dim=0).requires_grad_(True)
            edge_weight_with_ref = torch.cat([walk_step['module'][0].edge_weight,
                                     torch.zeros(walk_step['module'][0].edge_weight.shape, device=self.device)], dim=0)
            edge_index_with_ref = torch.cat([self_loop_edge_index, self_loop_edge_index + x.shape[0]], dim=1)
            walk_step['output'] = GraphSequential(*walk_step['module'])(walk_step['input'],
                                                                        edge_index_with_ref,
                                                                        edge_weight_with_ref)
        fc_step['input'] = walk_step['output']
        fc_step['output'] = GraphSequential(*fc_step['module'])(fc_step['input'],
                                                                torch.arange(2, dtype=torch.long,
                                                                             device=self.device).view(2, 1).repeat(1, x.shape[0]).reshape(-1))

        # --- Back distribute scores ---

        def compute_walk_score(adjs, C, allow_edges, walk_idx=[]):
            if not adjs:
                walk_indices.append(walk_idx)
                walk_scores.append(C.detach())
                return
            (m, ) = torch.autograd.grad(outputs=C, inputs=adjs[0], create_graph=True)
            edge_weight, edge_weight_ref = torch.chunk(adjs[0], 2)
            Cs = torch.chunk(m, 2)[0] * (edge_weight - edge_weight_ref)
            for i in allow_edges:
                allow_edges= torch.where(self_loop_edge_index[1] == self_loop_edge_index[0][i])[0].tolist()
                new_C = Cs[i]
                compute_walk_score(adjs[1:], new_C, allow_edges, [i] + walk_idx)


        labels = tuple(i for i in range(data_args.num_classes))
        walk_scores_tensor_list = [None for i in labels]
        for label in labels:
            f = torch.unbind(fc_step['output'][:, label])

            adjs = [walk_step['module'][0].edge_weight for walk_step in walk_steps]

            adjs.reverse()
            walk_indices = []
            walk_scores = []
            allow_edges = [i for i in range(len(adjs[0]) // 2)]

            compute_walk_score(adjs, f, allow_edges)
            walk_scores_tensor_list[label] = torch.stack(walk_scores, dim=0).view(-1, 1)

        shap._remove_hooks()
        walks = {'ids': torch.tensor(walk_indices, device=self.device), 'score': torch.cat(walk_scores_tensor_list, dim=1)}

        # --- Transformable walks ---
        for label in labels:
            step_weights = [nn.Parameter(torch.rand(1, device=self.device)) for _ in range(len(walk_steps))]
            optimizer = torch.optim.Adam(step_weights, lr=1e-3)
            pool = GlobalMeanPool()
            lin1 = nn.Linear(x.shape[0] * len(walk_steps), 128)
            lin2 = nn.Linear(128, fc_step['output'].shape[1])
            relu = nn.ReLU()
            for it in range(100):
                pooled = []
                for i, walk_step in enumerate(walk_steps):
                    pooled.append(pool(walk_step['output']) * step_weights[i])
                pooled = torch.cat(pooled, dim=0)
                output = lin2(relu(lin1(pooled)))
                loss = nn.CrossEntropyLoss()(output, torch.tensor([label], dtype=torch.long, device=self.device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


        # --- Apply edge mask evaluation ---
        with torch.no_grad():
            with self.connect_mask(self):
                ex_labels = tuple(torch.tensor([label]).to(data_args.device) for label in labels)
                masks = []
                for ex_label in ex_labels:
                    edge_attr = self.explain_edges_with_loop(x, walks, ex_label)
                    mask = edge_attr
                    mask = self.control_sparsity(mask, kwargs.get('sparsity'))
                    masks.append(mask.detach())

                related_preds = self.eval_related_pred(x, edge_index, masks, **kwargs)

        return None, masks, related_preds


class GraphLayerDeepLift(LayerDeepLift):

    def __init__(self, model: Module, layer: Module):
        super().__init__(model, layer)

    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        return_convergence_delta: bool = False,
        attribute_to_layer_input: bool = False,
        custom_attribution_func: Union[None, Callable[..., Tuple[Tensor, ...]]] = None,
    ) -> Union[
        Tensor, Tuple[Tensor, ...], Tuple[Union[Tensor, Tuple[Tensor, ...]], Tensor],
    ]:
        r"""
        Args:

            inputs (tensor or tuple of tensors):  Input for which layer
                        attributions are computed. If forward_func takes a
                        single tensor as input, a single input tensor should be
                        provided. If forward_func takes multiple tensors as input,
                        a tuple of the input tensors should be provided. It is
                        assumed that for all given input tensors, dimension 0
                        corresponds to the number of examples (aka batch size),
                        and if multiple input tensors are provided, the examples
                        must be aligned appropriately.
            baselines (scalar, tensor, tuple of scalars or tensors, optional):
                        Baselines define reference samples that are compared with
                        the inputs. In order to assign attribution scores DeepLift
                        computes the differences between the inputs/outputs and
                        corresponding references.
                        Baselines can be provided as:

                        - a single tensor, if inputs is a single tensor, with
                          exactly the same dimensions as inputs or the first
                          dimension is one and the remaining dimensions match
                          with inputs.

                        - a single scalar, if inputs is a single tensor, which will
                          be broadcasted for each input value in input tensor.

                        - a tuple of tensors or scalars, the baseline corresponding
                          to each tensor in the inputs' tuple can be:

                          - either a tensor with matching dimensions to
                            corresponding tensor in the inputs' tuple
                            or the first dimension is one and the remaining
                            dimensions match with the corresponding
                            input tensor.

                          - or a scalar, corresponding to a tensor in the
                            inputs' tuple. This scalar value is broadcasted
                            for corresponding input tensor.
                        In the cases when `baselines` is not provided, we internally
                        use zero scalar corresponding to each input tensor.

                        Default: None
            target (int, tuple, tensor or list, optional):  Output indices for
                        which gradients are computed (for classification cases,
                        this is usually the target class).
                        If the network returns a scalar value per example,
                        no target index is necessary.
                        For general 2D outputs, targets can be either:

                        - a single integer or a tensor containing a single
                          integer, which is applied to all input examples

                        - a list of integers or a 1D tensor, with length matching
                          the number of examples in inputs (dim 0). Each integer
                          is applied as the target for the corresponding example.

                        For outputs with > 2 dimensions, targets can be either:

                        - A single tuple, which contains #output_dims - 1
                          elements. This target index is applied to all examples.

                        - A list of tuples with length equal to the number of
                          examples in inputs (dim 0), and each tuple containing
                          #output_dims - 1 elements. Each tuple is applied as the
                          target for the corresponding example.

                        Default: None
            additional_forward_args (any, optional): If the forward function
                        requires additional arguments other than the inputs for
                        which attributions should not be computed, this argument
                        can be provided. It must be either a single additional
                        argument of a Tensor or arbitrary (non-tuple) type or a tuple
                        containing multiple additional arguments including tensors
                        or any arbitrary python types. These arguments are provided to
                        forward_func in order, following the arguments in inputs.
                        Note that attributions are not computed with respect
                        to these arguments.
                        Default: None
            return_convergence_delta (bool, optional): Indicates whether to return
                        convergence delta or not. If `return_convergence_delta`
                        is set to True convergence delta will be returned in
                        a tuple following attributions.
                        Default: False
            attribute_to_layer_input (bool, optional): Indicates whether to
                        compute the attribution with respect to the layer input
                        or output. If `attribute_to_layer_input` is set to True
                        then the attributions will be computed with respect to
                        layer input, otherwise it will be computed with respect
                        to layer output.
                        Note that currently it is assumed that either the input
                        or the output of internal layer, depending on whether we
                        attribute to the input or output, is a single tensor.
                        Support for multiple tensors will be added later.
                        Default: False
            custom_attribution_func (callable, optional): A custom function for
                        computing final attribution scores. This function can take
                        at least one and at most three arguments with the
                        following signature:

                        - custom_attribution_func(multipliers)
                        - custom_attribution_func(multipliers, inputs)
                        - custom_attribution_func(multipliers, inputs, baselines)

                        In case this function is not provided, we use the default
                        logic defined as: multipliers * (inputs - baselines)
                        It is assumed that all input arguments, `multipliers`,
                        `inputs` and `baselines` are provided in tuples of same length.
                        `custom_attribution_func` returns a tuple of attribution
                        tensors that have the same length as the `inputs`.
                        Default: None

        Returns:
            **attributions** or 2-element tuple of **attributions**, **delta**:
            - **attributions** (*tensor* or tuple of *tensors*):
                Attribution score computed based on DeepLift's rescale rule with
                respect to layer's inputs or outputs. Attributions will always be the
                same size as the provided layer's inputs or outputs, depending on
                whether we attribute to the inputs or outputs of the layer.
                If the layer input / output is a single tensor, then
                just a tensor is returned; if the layer input / output
                has multiple tensors, then a corresponding tuple
                of tensors is returned.
            - **delta** (*tensor*, returned if return_convergence_delta=True):
                This is computed using the property that the total sum of
                forward_func(inputs) - forward_func(baselines) must equal the
                total sum of the attributions computed based on DeepLift's
                rescale rule.
                Delta is calculated per example, meaning that the number of
                elements in returned delta tensor is equal to the number of
                of examples in input.
                Note that the logic described for deltas is guaranteed
                when the default logic for attribution computations is used,
                meaning that the `custom_attribution_func=None`, otherwise
                it is not guaranteed and depends on the specifics of the
                `custom_attribution_func`.

        Examples::

            >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
            >>> # and returns an Nx10 tensor of class probabilities.
            >>> net = ImageClassifier()
            >>> # creates an instance of LayerDeepLift to interpret target
            >>> # class 1 with respect to conv4 layer.
            >>> dl = LayerDeepLift(net, net.conv4)
            >>> input = torch.randn(1, 3, 32, 32, requires_grad=True)
            >>> # Computes deeplift attribution scores for conv4 layer and class 3.
            >>> attribution = dl.attribute(input, target=1)
        """
        inputs = _format_input(inputs)
        baselines = _format_baseline(baselines, inputs)
        gradient_mask = apply_gradient_requirements(inputs)

        # --- make sure that inputs and baselines fit each other ---
        _validate_input(inputs, baselines)

        baselines = _tensorize_baseline(inputs, baselines)
        # --- add pre hook on the shell of the model for combining inputs with baselines and additional inputs.
        main_model_pre_hook = self._pre_hook_main_model()

        self.model.apply(self._register_hooks)

        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        # input_base_additional_args = _expand_additional_forward_args(
        #     additional_forward_args, 2, ExpansionTypes.repeat
        # )

        # --- my expansion of additional args ---
        if len(additional_forward_args[0].shape) == 1:
            # --- batch ---
            input_base_additional_args = torch.cat([additional_forward_args[0],
                                                    additional_forward_args[0] + 1], dim=0)
        elif len(additional_forward_args[0].shape) == 2:
            # --- edge_index ---
            input_base_additional_args = torch.cat([additional_forward_args[0],
                                                    additional_forward_args[0] + inputs[0].shape[0]],
                                                   dim=1)
        input_base_additional_args = tuple([input_base_additional_args])
        # --- my end

        expanded_target = _expand_target(
            target, 2, expansion_type=ExpansionTypes.repeat
        )
        wrapped_forward_func = self._construct_forward_func(
            self.model,
            (inputs, baselines),
            expanded_target,
            input_base_additional_args,
        )

        def chunk_output_fn(out: TensorOrTupleOfTensorsGeneric,) -> Sequence:
            if isinstance(out, Tensor):
                return out.chunk(2)
            return tuple(out_sub.chunk(2) for out_sub in out)

        (all_gradients, attrs, is_layer_tuple) = gu.compute_layer_gradients_and_eval(
            wrapped_forward_func,
            self.layer,
            inputs,
            model=self.model,
            pre_hook=main_model_pre_hook,
            target=target,
            additional_forward_args=input_base_additional_args,
            attribute_to_layer_input=attribute_to_layer_input,
            output_fn=lambda out: chunk_output_fn(out),
        )

        attr_inputs = tuple(map(lambda attr: attr[0], attrs))
        attr_baselines = tuple(map(lambda attr: attr[1], attrs))
        if type(all_gradients) is not list:
            gradients = tuple(map(lambda grad: grad[0], all_gradients))

            if custom_attribution_func is None:
                attributions = tuple(
                    (input - baseline) * gradient
                    for input, baseline, gradient in zip(
                        attr_inputs, attr_baselines, gradients
                    )
                )
            else:
                attributions = _call_custom_attribution_func(
                    custom_attribution_func, gradients, attr_inputs, attr_baselines
                )
        else:
            all_attributions = []
            for gradients in all_gradients:
                gradients = tuple(map(lambda grad: grad[0], gradients))

                if custom_attribution_func is None:
                    attributions = tuple(
                        (input - baseline) * gradient
                        for input, baseline, gradient in zip(
                            attr_inputs, attr_baselines, gradients
                        )
                    )
                else:
                    attributions = _call_custom_attribution_func(
                        custom_attribution_func, gradients, attr_inputs, attr_baselines
                    )
                all_attributions.append(attributions)
            attributions = tuple(all_attributions)
            s = 0
            for attribution in attributions:
                s += attribution[0].sum()
        # remove hooks from all activations
        main_model_pre_hook.remove()
        self._remove_hooks()

        undo_gradient_requirements(inputs, gradient_mask)
        return _compute_conv_delta_and_format_attrs(
            self,
            return_convergence_delta,
            attributions,
            baselines,
            inputs,
            additional_forward_args,
            target,
            is_layer_tuple,
        )


