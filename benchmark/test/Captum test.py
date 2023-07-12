"""
FileName: Captum test.py
Description: 
Time: 2020/9/9 12:51
Project: GNN_benchmark
Author: Shurui Gui
"""

from benchmark.models.models import GCN, GAT, GIN
from benchmark.data.dataset_manager import data_args
import torch
from torch_geometric.data import Data
from benchmark.models.explainers import GraphLayerGradCam

data_args.dim_node = 20
data_args.num_classes = 1

model = GCN()
print(model)

x = torch.rand((4, 20), requires_grad=True)
edge_index = torch.tensor([[0, 1, 2, 1, 3, 1], [1, 0, 1, 2, 1, 3]])
# out = model(x=x, edge_index=edge_index)
# data = Data(x=x, edge_index=edge_index, y=torch.tensor([[1.]]))
explainer = GraphLayerGradCam(model, model.convs[-1])

attr = explainer.attribute(x, 0, additional_forward_args=edge_index, relu_attributions=True)





