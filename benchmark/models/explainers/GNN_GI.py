import os

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.utils.loop import add_self_loops

from .ExplainerBase import FlowBase
from benchmark import data_args
from benchmark.args import x_args
from benchmark.models.utils import subgraph
from definitions import ROOT_DIR

EPS = 1e-15

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
