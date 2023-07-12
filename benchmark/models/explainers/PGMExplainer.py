import os
from typing import List, Tuple, Union, Dict

import numpy as np
import torch
from torch import Tensor
from torch_geometric.utils.loop import add_self_loops

import benchmark.models.ext.PGM_Graph.pgm_explainer_graph as pgmg
import benchmark.models.ext.PGM_Node.Explain_GNN.pgm_explainer as pgmn
from .ExplainerBase import NodeBase
from benchmark import data_args
from benchmark.args import x_args
from benchmark.models.utils import subgraph
from definitions import ROOT_DIR

EPS = 1e-15

class PGMExplainer(NodeBase):

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
