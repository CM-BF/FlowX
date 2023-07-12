import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.utils.loop import add_self_loops

from .ExplainerBase import NodeBase
from benchmark import data_args
from benchmark.models.ext.deeplift.layer_deep_lift import DeepLift
from benchmark.models.utils import subgraph

EPS = 1e-15

class DeepLIFT(NodeBase):

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