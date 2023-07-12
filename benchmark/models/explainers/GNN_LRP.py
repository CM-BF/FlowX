import copy
import os

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.utils.loop import add_self_loops

from .ExplainerBase import FlowBase
from benchmark import data_args
from benchmark.args import x_args
from benchmark.models.models import GraphSequential
from benchmark.models.utils import subgraph
from definitions import ROOT_DIR

EPS = 1e-15

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
