import os
from typing import List, Tuple, Union, Dict

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.utils.loop import add_self_loops, remove_self_loops

from .ExplainerBase import FlowBase
from benchmark import data_args
from benchmark.args import x_args
from benchmark.kernel.utils import Metric
from benchmark.models.utils import subgraph
from definitions import ROOT_DIR

EPS = 1e-15


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
