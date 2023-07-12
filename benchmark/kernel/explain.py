"""
FileName: explain.py
Description: 
Time: 2020/8/11 15:39
Project: GNN_benchmark
Author: Shurui Gui
"""
import torch
from benchmark.args import x_args
from benchmark.kernel.utils import Metric
from benchmark.kernel.evaluation import acc_score
from torch_geometric.utils.loop import add_self_loops
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Union
from torch import Tensor
import copy
import time




class XCollector(object):

    def __init__(self, model, loader):
        self.new()
        # self.__related_preds, self.__targets = {'zero': [], 'masked': [], 'maskout': [], 'origin': []}, []
        # self.masks: Union[List, List[List[Tensor]]] = []
        # self.data_list = []
        # self.hard_masks: Union[List, List[List[Tensor]]] = []
        #
        # self.__fidelity, self.__regular_fidelity, self.__contrastivity, self.__sparsity, self.__infidelity, self.__regular_infidelity = None, None, None, None, None, None
        # self.__acc = None
        # self.__score = None
        self.model, self.loader = model, loader

    @property
    def maskout_preds(self) -> list:
        return self.__related_preds.get('maskout')

    @property
    def targets(self) -> list:
        return self.__targets

    def metrics_exist(self):
        for value in self.metrics.values():
            if value is not None:
                return True
        return False

    def metrics_clear(self):
        for key in self.metrics.keys():
            self.metrics['key'] = None


    def new(self):
        self.__related_preds, self.__targets = {'zero': [], 'masked': [], 'maskout': [], 'origin': [], 'sparsity': []}, []
        self.masks: Union[List, List[List[Tensor]]] = []
        self.data_list = []
        self.hard_masks: Union[List, List[List[Tensor]]] = []

        self.metrics = {'fidelity': None, 'regular_fidelity': None, 'infidelity': None, 'regular_infidelity': None, 'contrastivity': None, 'sparsity': None, 'regular_infidelity': None, 'acc': None, 'score': None}

    def collect_data(self,
                     masks: List[Tensor],
                     related_preds: dir,
                     label: int,
                     data) -> None:

        if self.metrics_exist():
            self.metrics_clear()
            print(f'#W#Called collect_data() after calculate explainable metrics.')

        if not np.isnan(label):
            for key, value in related_preds[label].items():
                self.__related_preds[key].append(value)

            self.__targets.append(label)
            self.masks.append(masks)
            self.data_list.append(data)

            # # make the masks binary
            # hard_masks = copy.deepcopy(masks)
            #
            # for i, hard_mask in enumerate(hard_masks):
            #     hard_mask[hard_mask >= 0] = 1
            #     hard_mask[hard_mask < 0] = 0
            #
            #     hard_mask = hard_mask.to(torch.int64)
            #
            # self.hard_masks.append(hard_masks)

    @property
    def score(self):
        if self.metrics['score']:
            return self.metrics['score']
        else:

            self.metrics['socre'] = acc_score(self.model, self.loader)
            return self.metrics['score']

    @property
    def infidelity(self):
        if self.metrics['infidelity']:
            return self.metrics['infidelity']
        else:
            zero_mask_preds, mask_out_preds, masked_preds, one_mask_preds = \
                torch.tensor(self.__related_preds['zero']), torch.tensor(self.__related_preds['maskout']), \
                torch.tensor(self.__related_preds['masked']), torch.tensor(self.__related_preds['origin'])

            # Do Shapley Algorithm
            # mask_out_contribution = ((mask_out_preds - zero_mask_preds) + (one_mask_preds - masked_preds)) / 2
            contribution = one_mask_preds - masked_preds

            # for negative prediction's mask, the negative contribution is better.
            # So we multiply it with -1 to make it positive
            # target_ = torch.tensor(self.__targets)
            # target_[target_ == 0] = -1
            #
            # Discard above scripts:
            # because __related_pred only contains the prediction
            # probabilities of the correct labels. Thus higher is better.
            #
            # self.__Infidelity = (masked_contribution - mask_out_contribution).mean().item()
            self.metrics['infidelity'] = contribution.mean().item()

            return self.metrics['infidelity']

    @property
    def regular_infidelity(self):
        if self.metrics['regular_infidelity']:
            return self.metrics['regular_infidelity']
        else:
            zero_mask_preds, mask_out_preds, masked_preds, one_mask_preds = \
                torch.tensor(self.__related_preds['zero']), torch.tensor(self.__related_preds['maskout']), \
                torch.tensor(self.__related_preds['masked']), torch.tensor(self.__related_preds['origin'])

            # Do Shapley Algorithm
            # mask_out_contribution = ((mask_out_preds - zero_mask_preds) + (one_mask_preds - masked_preds)) / 2
            contribution = (one_mask_preds - masked_preds) / \
                           torch.exp(torch.abs(one_mask_preds - zero_mask_preds))

            self.metrics['regular_infidelity'] = contribution.mean().item()

            return self.metrics['regular_infidelity']


    @property
    def fidelity(self):
        if False or self.metrics['fidelity']:
            return self.metrics['fidelity']
        else:

            zero_mask_preds, mask_out_preds, masked_preds, one_mask_preds = \
                torch.tensor(self.__related_preds['zero']), torch.tensor(self.__related_preds['maskout']), \
                torch.tensor(self.__related_preds['masked']), torch.tensor(self.__related_preds['origin'])

            drop_probability = one_mask_preds - mask_out_preds

            self.metrics['fidelity'] = drop_probability.mean().item()
            return self.metrics['fidelity']

    @property
    def regular_fidelity(self):
        if self.metrics['regular_fidelity']:
            return self.metrics['regular_fidelity']
        else:

            # score of output deriving from masked inputs
            # try:
            #     # self.__related_preds contains the preds that are corresponding to the y_true,
            #     # so closer to 1 is better
            #     # maskout_score = accuracy_score(torch.ones(len(self.__targets)), torch.tensor(self.__related_preds['maskout']).round())
            #
            # except ValueError as e:
            #     logger.warning(e)

            zero_mask_preds, mask_out_preds, masked_preds, one_mask_preds = \
                torch.tensor(self.__related_preds['zero']), torch.tensor(self.__related_preds['maskout']), \
                torch.tensor(self.__related_preds['masked']), torch.tensor(self.__related_preds['origin'])

            drop_probability = (one_mask_preds - mask_out_preds) / \
                               torch.exp(torch.abs(one_mask_preds - zero_mask_preds))

            # score of origin model output
            # origin_score = self.score

            # self.__fidelity = origin_score - maskout_score    # higher is better
            self.metrics['regular_fidelity'] = drop_probability.mean().item()
            return self.metrics['regular_fidelity']

    @property
    def acc(self):
        if self.metrics['acc']:
            return self.metrics['acc']
        else:
            if not hasattr(self.data_list[0], 'class_mask'):
                self.metrics['acc'] = 0.
                return self.metrics['acc']
            acc = []
            for graph_idx, (data, raw_masks, target) in enumerate(zip(self.data_list, self.masks, self.__targets)):
                gt_edges = data.class_mask[0][target]
                edge_index_with_loop, _ = add_self_loops(data.edge_index, num_nodes=data.x.shape[0])
                choosen_edges = edge_index_with_loop.T[raw_masks[target] > 0].tolist()
                total_edge_num = gt_edges.__len__()
                hit_num = 0

                for gt_edge in gt_edges:
                    if (gt_edge in choosen_edges) or ([gt_edge[1], gt_edge[0]] in choosen_edges):
                        hit_num += 1

                acc.append(float(hit_num) / float(total_edge_num))

            self.metrics['acc'] = np.mean(acc)

            return self.metrics['acc']

    # @property
    # def contrastivity(self):
    #     if self.__contrastivity:
    #         return self.__contrastivity
    #     else:
    #         contrastivity = []
    #         for i in range(len(self.hard_masks)):
    #             for cur_label in range(len(self.hard_masks[0])):
    #                 if cur_label == self.__targets[i]:
    #                     continue
    #
    #                 distance_hamington = \
    #                     (self.hard_masks[i][self.__targets[i]] != self.hard_masks[i][cur_label]).int().sum().item()
    #
    #                 union = \
    #                     ((self.hard_masks[i][self.__targets[i]] + self.hard_masks[i][cur_label]) > 0).int().sum().item()
    #
    #                 if union == 0:
    #                     continue
    #
    #                 contrastivity.append(distance_hamington / union)
    #
    #         self.__contrastivity = np.mean(contrastivity)
    #
    #         return self.__contrastivity

    @property
    def sparsity(self):
        if self.__related_preds['sparsity']:
            # automatically calculated sparsity [only for SubgraphX]
            self.metrics['sparsity'] = np.mean(self.__related_preds['sparsity'])
        else:
            self.metrics['sparsity'] = x_args.sparsity

        return self.metrics['sparsity']




def sample_explain(explainer, data, x_collector: XCollector, **kwargs):
    data.to(x_args.device)
    node_idx = kwargs.get('node_idx')
    y_idx = 0 if node_idx is None else node_idx  # if graph level: y_idx = 0 if node level: y_idx = node_idx

    if torch.isnan(data.y[y_idx].squeeze()):
        return

    explain_tik = time.time()
    walks, masks, related_preds = \
        explainer(data.x, data.edge_index, **kwargs)
    explain_tok = time.time()
    print(f"#D#Explainer prediction time: {explain_tok - explain_tik:.4f}")

    orig_pred = [related_pred['origin'] for related_pred in related_preds]

    gt_label = data.y[y_idx].squeeze().long().item()
    pred_label = np.argmax(orig_pred)
    if x_args.explain_pred_label:
        target_label = pred_label
    else:
        target_label = gt_label
    x_collector.collect_data(masks,
                             related_preds,
                             target_label,
                             data=data)
    # print(x_collector.fidelity)

    try:
        if x_args.vis or x_args.save_fig:
            if x_args.walk:
                labeled_walks = walks
                labeled_walks['score'] = labeled_walks['score'][:, target_label]
                ax, G = explainer.visualize_walks(node_idx=0 if node_idx is None else node_idx, edge_index=data.edge_index,
                                                  walks=labeled_walks, edge_mask=masks[target_label],
                                                  y=data.x[:, 0] if node_idx is None else data.y, num_nodes=data.x.shape[0])
            else:
                ax, G = explainer.visualize_graph(node_idx=0 if node_idx is None else node_idx, edge_index=data.edge_index,
                                                 edge_mask=masks[target_label],
                                                 y=data.x[:, 0] if node_idx is None else data.y, num_nodes=data.x.shape[0],
                                                  sentence=data.sentence[0] if hasattr(data, 'sentence') else None,
                                                  data=data)
            ax.set_title(f'{x_args.explainer}\nF: {x_collector.fidelity:.4f}  I: {x_collector.infidelity:.4f}  S: {x_collector.sparsity:.4f}')
            if x_args.save_fig:
                from definitions import ROOT_DIR
                import os
                index = kwargs.get('index')
                assert index is not None
                fig = os.path.join(ROOT_DIR, 'visual_results', f'{x_args.dataset_name}', f'{x_args.model_name}',
                                   f'{x_args.sparsity}', ''f'{index}', f'{explainer.__class__.__name__}.png')
                fig_path = os.path.dirname(fig)
                if not os.path.exists(fig_path):
                    os.makedirs(fig_path)
                print('save fig as:', fig)
                plt.savefig(fig, dpi=300)
                plt.cla()
            else:
                plt.show()

    except Exception as e:
        print(f'#C#Visualization error: {e}')



