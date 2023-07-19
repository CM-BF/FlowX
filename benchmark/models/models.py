"""
FileName: models.py
Description: GNN models' set
Time: 2020/7/30 9:01
Project: GNN_benchmark
Author: Shurui Gui
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
import torch_geometric.nn as gnn
from torch_geometric.utils.loop import add_self_loops, remove_self_loops
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from benchmark.args import data_args
from torch_geometric.data.batch import Batch
from benchmark.models.utils import ReadOut

from typing import Callable, Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch import Tensor

from torch_sparse import SparseTensor


class GNNBasic(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def arguments_read(self, *args, **kwargs):

        data: Batch = kwargs.get('data') or None

        if not data:
            if not args:
                assert 'x' in kwargs
                assert 'edge_index' in kwargs
                x, edge_index = kwargs['x'], kwargs['edge_index'],
                batch = kwargs.get('batch')
                if batch is None:
                    batch = torch.zeros(kwargs['x'].shape[0], dtype=torch.int64, device=torch.device('cuda'))
            elif len(args) == 2:
                x, edge_index, batch = args[0], args[1], \
                                       torch.zeros(args[0].shape[0], dtype=torch.int64, device=torch.device('cuda'))
            elif len(args) == 3:
                x, edge_index, batch = args[0], args[1], args[2]
            else:
                raise ValueError(f"forward's args should take 2 or 3 arguments but got {len(args)}")
        else:
            x, edge_index, batch = data.x, data.edge_index, data.batch

        return x, edge_index, batch

class GCN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(data_args.dim_node, data_args.dim_hidden)
        self.convs = nn.ModuleList(
            [
                GCNConv(data_args.dim_hidden, data_args.dim_hidden)
                for _ in range(2)
             ]
        )
        self.readout = GlobalMeanPool()

        self.ffn = nn.Sequential(*(
                [nn.Dropout(), nn.Linear(data_args.dim_hidden, data_args.dim_ffn)] +
                [nn.ReLU(), nn.Dropout(), nn.Linear(data_args.dim_ffn, data_args.dim_ffn)] +
                [nn.ReLU(), nn.Dropout(), nn.Linear(data_args.dim_ffn, data_args.num_classes)]
        ))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        :param Required[data]: Batch - input data
        :return:
        """
        data: Batch = kwargs.get('data') or None

        if not data:
            if not args:
                assert 'x' in kwargs
                assert 'edge_index' in kwargs
                x, edge_index, batch = kwargs['x'], kwargs['edge_index'], \
                                       torch.zeros(kwargs['x'].shape[0], dtype=torch.int64, device=torch.device('cuda'))
            elif len(args) == 2:
                x, edge_index, batch = args[0], args[1], \
                                       torch.zeros(args[0].shape[0], dtype=torch.int64, device=torch.device('cuda'))
            else:
                raise ValueError(f"forward's args should take 2 arguments but got {len(args)}")
        else:
            x, edge_index, batch = data.x, data.edge_index, data.batch

        post_conv = self.conv1(x, edge_index)
        post_conv = self.dropout(self.relu(post_conv))
        for conv in self.convs:
            post_conv = self.dropout(self.relu(conv(post_conv, edge_index)))


        out_readout = self.readout(post_conv, batch)

        out = self.ffn(out_readout)
        return out


class GCN_3l(GNNBasic):

    def __init__(self):
        super().__init__()
        num_layer = 3

        self.conv1 = GCNConv(data_args.dim_node, data_args.dim_hidden)
        self.convs = nn.ModuleList(
            [
                GCNConv(data_args.dim_hidden, data_args.dim_hidden)
                for _ in range(num_layer - 1)
             ]
        )
        self.relu1 = nn.ReLU()
        self.relus = nn.ModuleList(
            [
                nn.ReLU()
                for _ in range(num_layer - 1)
            ]
        )
        if data_args.model_level == 'node':
            self.readout = IdenticalPool()
        else:
            self.readout = GlobalMeanPool()

        self.ffn = nn.Sequential(*(
                [nn.Linear(data_args.dim_hidden, data_args.dim_hidden)] +
                [nn.ReLU(), nn.Dropout(), nn.Linear(data_args.dim_hidden, data_args.num_classes)]
        ))

        self.dropout = nn.Dropout()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        :param Required[data]: Batch - input data
        :return:
        """
        x, edge_index, batch = self.arguments_read(*args, **kwargs)


        post_conv = self.relu1(self.conv1(x, edge_index))
        for conv, relu in zip(self.convs, self.relus):
            post_conv = relu(conv(post_conv, edge_index))


        out_readout = self.readout(post_conv, batch, kwargs.get('batch_size'))

        out = self.ffn(out_readout)
        return out

    def get_emb(self, *args, **kwargs) -> torch.Tensor:
        x, edge_index, batch = self.arguments_read(*args, **kwargs)

        post_conv = self.relu1(self.conv1(x, edge_index))
        for conv, relu in zip(self.convs, self.relus):
            post_conv = relu(conv(post_conv, edge_index))
        return post_conv, batch

    def get_graph_rep(self, *args, **kwargs) -> torch.Tensor:
        return self.readout(*self.get_emb(*args, **kwargs), kwargs.get('batch_size'))


class GCN_2l(GNNBasic):

    def __init__(self):
        super().__init__()
        num_layer = 2

        self.conv1 = GCNConv(data_args.dim_node, data_args.dim_hidden)
        self.convs = nn.ModuleList(
            [
                GCNConv(data_args.dim_hidden, data_args.dim_hidden)
                for _ in range(num_layer - 1)
            ]
        )
        self.relu1 = nn.ReLU()
        self.relus = nn.ModuleList(
            [
                nn.ReLU()
                for _ in range(num_layer - 1)
            ]
        )
        if data_args.model_level == 'node':
            self.readout = IdenticalPool()
        else:
            self.readout = GlobalMeanPool()

        self.ffn = nn.Sequential(*(
                [nn.Linear(data_args.dim_hidden, data_args.num_classes)]
        ))

        self.dropout = nn.Dropout()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        :param Required[data]: Batch - input data
        :return:
        """
        x, edge_index, batch = self.arguments_read(*args, **kwargs)

        post_conv = self.relu1(self.conv1(x, edge_index))
        for conv, relu in zip(self.convs, self.relus):
            post_conv = relu(conv(post_conv, edge_index))

        out_readout = self.readout(post_conv, batch)

        out = self.ffn(out_readout)

        return out

    def get_emb(self, *args, **kwargs) -> torch.Tensor:
        x, edge_index, batch = self.arguments_read(*args, **kwargs)
        post_conv = self.conv1(x, edge_index)
        for conv in self.convs:
            post_conv = conv(post_conv, edge_index)
        return post_conv


class GIN_3l(GNNBasic):

    def __init__(self):
        super().__init__()
        num_layer = 3

        self.conv1 = GINConv(nn.Sequential(nn.Linear(data_args.dim_node, data_args.dim_hidden), nn.ReLU(),
                                           nn.Linear(data_args.dim_hidden, data_args.dim_hidden), nn.ReLU()))#,
                                           # nn.BatchNorm1d(data_args.dim_hidden)))
        self.convs = nn.ModuleList(
            [
                GINConv(nn.Sequential(nn.Linear(data_args.dim_hidden, data_args.dim_hidden), nn.ReLU(),
                                      nn.Linear(data_args.dim_hidden, data_args.dim_hidden), nn.ReLU()))#,
                                      # nn.BatchNorm1d(data_args.dim_hidden)))
                for _ in range(num_layer - 1)
             ]
        )
        self.relu1 = nn.ReLU()
        self.relus = nn.ModuleList(
            [
                nn.ReLU()
                for _ in range(num_layer - 1)
            ]
        )
        if data_args.model_level == 'node':
            self.readout = IdenticalPool()
        else:
            self.readout = GlobalMeanPool()

        self.ffn = nn.Sequential(*(
                [nn.Linear(data_args.dim_hidden, data_args.dim_hidden)] +
                [nn.ReLU(), nn.Dropout(), nn.Linear(data_args.dim_hidden, data_args.num_classes)]
        ))

        self.dropout = nn.Dropout()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        :param Required[data]: Batch - input data
        :return:
        """
        x, edge_index, batch = self.arguments_read(*args, **kwargs)


        post_conv = self.conv1(x, edge_index)
        for conv in self.convs:
            post_conv = conv(post_conv, edge_index)


        out_readout = self.readout(post_conv, batch, kwargs.get('batch_size'))

        out = self.ffn(out_readout)
        return out

    def get_emb(self, *args, **kwargs) -> torch.Tensor:
        x, edge_index, batch = self.arguments_read(*args, **kwargs)
        post_conv = self.conv1(x, edge_index)
        for conv in self.convs:
            post_conv = conv(post_conv, edge_index)
        return post_conv, batch

    def get_graph_rep(self, *args, **kwargs) -> torch.Tensor:
        return self.readout(*self.get_emb(*args, **kwargs), kwargs.get('batch_size'))

class GIN_2l(GNNBasic):

    def __init__(self):
        super().__init__()
        num_layer = 2

        self.conv1 = GINConv(nn.Sequential(nn.Linear(data_args.dim_node, data_args.dim_hidden), nn.ReLU(),
                                           nn.Linear(data_args.dim_hidden, data_args.dim_hidden), nn.ReLU()))#,
                                           # nn.BatchNorm1d(data_args.dim_hidden)))
        self.convs = nn.ModuleList(
            [
                GINConv(nn.Sequential(nn.Linear(data_args.dim_hidden, data_args.dim_hidden), nn.ReLU(),
                                      nn.Linear(data_args.dim_hidden, data_args.dim_hidden), nn.ReLU()))#,
                                      # nn.BatchNorm1d(data_args.dim_hidden)))
                for _ in range(num_layer - 1)
             ]
        )
        self.relu1 = nn.ReLU()
        self.relus = nn.ModuleList(
            [
                nn.ReLU()
                for _ in range(num_layer - 1)
            ]
        )
        if data_args.model_level == 'node':
            self.readout = IdenticalPool()
        else:
            self.readout = GlobalMeanPool()

        self.ffn = nn.Sequential(*(
                [nn.Linear(data_args.dim_hidden, data_args.dim_hidden)] +
                [nn.ReLU(), nn.Dropout(), nn.Linear(data_args.dim_hidden, data_args.num_classes)]
        ))

        self.dropout = nn.Dropout()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        :param Required[data]: Batch - input data
        :return:
        """
        x, edge_index, batch = self.arguments_read(*args, **kwargs)


        post_conv = self.conv1(x, edge_index)
        for conv in self.convs:
            post_conv = conv(post_conv, edge_index)


        out_readout = self.readout(post_conv, batch)

        out = self.ffn(out_readout)
        return out

    def get_emb(self, *args, **kwargs) -> torch.Tensor:
        x, edge_index, batch = self.arguments_read(*args, **kwargs)
        post_conv = self.conv1(x, edge_index)
        for conv in self.convs:
            post_conv = conv(post_conv, edge_index)
        return post_conv




class GAT(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = gnn.GATConv(data_args.dim_node, data_args.dim_hidden)
        self.convs = nn.ModuleList(
            [
                gnn.GATConv(data_args.dim_hidden, data_args.dim_hidden)
                for _ in range(2)
            ]
        )

        self.readout = GlobalMeanPool()

        self.ffn = nn.Sequential(*(
                [nn.Dropout(), nn.Linear(data_args.dim_hidden, data_args.dim_ffn)] +
                [nn.ReLU(), nn.Dropout(), nn.Linear(data_args.dim_ffn, data_args.dim_ffn)] +
                [nn.ReLU(), nn.Dropout(), nn.Linear(data_args.dim_ffn, data_args.num_classes)]
        ))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        :param Required[data]: Batch - input data
        :return:
        """
        data: Batch = kwargs.get('data') or None

        if not data:
            if not args:
                assert 'x' in kwargs
                assert 'edge_index' in kwargs
                x, edge_index, batch = kwargs['x'], kwargs['edge_index'], \
                                       torch.zeros(kwargs['x'].shape[0], dtype=torch.int64, device=torch.device('cuda'))
            elif len(args) == 2:
                x, edge_index, batch = args[0], args[1], \
                                       torch.zeros(args[0].shape[0], dtype=torch.int64, device=torch.device('cuda'))
            else:
                raise ValueError(f"forward's args should take 2 arguments but got {len(args)}")
        else:
            x, edge_index, batch = data.x, data.edge_index, data.batch


        post_conv = self.dropout(self.relu(self.conv1(x, edge_index)))
        for conv in self.convs:
            post_conv = self.dropout(self.relu(conv(post_conv, edge_index)))

        out_readout = self.readout(post_conv, batch)

        out = self.ffn(out_readout)
        return out


class GIN(torch.nn.Module):

    def __init__(self):
        super().__init__()

        nn1 = nn.Sequential(nn.Linear(data_args.dim_node, data_args.dim_hidden)
                            , nn.ReLU(), nn.Linear(data_args.dim_hidden, data_args.dim_hidden))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(data_args.dim_hidden)

        self.convs = nn.ModuleList(
            [
                GINConv(nn.Sequential(nn.Linear(data_args.dim_hidden, data_args.dim_hidden),
                              nn.ReLU(), nn.Linear(data_args.dim_hidden, data_args.dim_hidden)))
                for _ in range(2)
            ]
        )

        self.bns = nn.ModuleList(
            [
                nn.BatchNorm1d(data_args.dim_hidden)
                for _ in range(2)
            ]
        )

        self.readout = GlobalMeanPool()

        self.ffn = nn.Sequential(*(
                [nn.Dropout(), nn.Linear(data_args.dim_hidden, data_args.dim_ffn)] +
                [nn.ReLU(), nn.Dropout(), nn.Linear(data_args.dim_ffn, data_args.dim_ffn)] +
                [nn.ReLU(), nn.Dropout(), nn.Linear(data_args.dim_ffn, data_args.num_classes)]
        ))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        :param Required[data]: Batch - input data
        :return:
        """
        data: Batch = kwargs.get('data') or None

        if not data:
            if not args:
                assert 'x' in kwargs
                assert 'edge_index' in kwargs
                x, edge_index, batch = kwargs['x'], kwargs['edge_index'], \
                                       torch.zeros(kwargs['x'].shape[0], dtype=torch.int64, device=torch.device('cuda'))
            elif len(args) == 2:
                x, edge_index, batch = args[0], args[1], \
                                       torch.zeros(args[0].shape[0], dtype=torch.int64, device=torch.device('cuda'))
            else:
                raise ValueError(f"forward's args should take 2 arguments but got {len(args)}")
        else:
            x, edge_index, batch = data.x, data.edge_index, data.batch


        post_conv = self.relu(self.bn1(self.conv1(x, edge_index)))
        for conv, bn in zip(self.convs, self.bns):
            post_conv = conv(post_conv, edge_index)
            post_conv = self.relu(bn(post_conv))

        out_readout = self.readout(post_conv, batch)

        out = self.ffn(out_readout)
        return out


class GCNConv(gnn.GCNConv):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__explain_flow__ = False
        self.__explain__ = None
        self.edge_weight = None
        self.layer_edge_mask = None
        self.weight = nn.Parameter(self.lin.weight.data.T.clone().detach())

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.normalize and edge_weight is None:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gnn.conv.gcn_conv.gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gnn.conv.gcn_conv.gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # --- add require_grad ---
        edge_weight.requires_grad_(True)

        x = torch.matmul(x, self.weight)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        if self.bias is not None:
            out += self.bias

        # --- My: record edge_weight ---
        self.edge_weight = edge_weight

        return out

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        r"""The initial call to start propagating messages.

        Args:
            adj (Tensor or SparseTensor): A :obj:`torch.LongTensor` or a
                :obj:`torch_sparse.SparseTensor` that defines the underlying
                graph connectivity/message passing flow.
                :obj:`edge_index` holds the indices of a general (sparse)
                assignment matrix of shape :obj:`[N, M]`.
                If :obj:`edge_index` is of type :obj:`torch.LongTensor`, its
                shape must be defined as :obj:`[2, num_messages]`, where
                messages from nodes in :obj:`edge_index[0]` are sent to
                nodes in :obj:`edge_index[1]`
                (in case :obj:`flow="source_to_target"`).
                If :obj:`edge_index` is of type
                :obj:`torch_sparse.SparseTensor`, its sparse indices
                :obj:`(row, col)` should relate to :obj:`row = edge_index[1]`
                and :obj:`col = edge_index[0]`.
                The major difference between both formats is that we need to
                input the *transposed* sparse adjacency matrix into
                :func:`propagate`.
            size (tuple, optional): The size :obj:`(N, M)` of the assignment
                matrix in case :obj:`edge_index` is a :obj:`LongTensor`.
                If set to :obj:`None`, the size will be automatically inferred
                and assumed to be quadratic.
                This argument is ignored in case :obj:`edge_index` is a
                :obj:`torch_sparse.SparseTensor`. (default: :obj:`None`)
            **kwargs: Any additional data which is needed to construct and
                aggregate messages, and to update node embeddings.
        """
        size = self.__check_input__(edge_index, size)

        # Run "fused" message and aggregation (if applicable).
        if (isinstance(edge_index, SparseTensor) and self.fuse
                and not self.__explain__):
            coll_dict = self.__collect__(self.__fused_user_args__, edge_index,
                                         size, kwargs)

            msg_aggr_kwargs = self.inspector.distribute(
                'message_and_aggregate', coll_dict)
            out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)

            update_kwargs = self.inspector.distribute('update', coll_dict)
            return self.update(out, **update_kwargs)

        # Otherwise, run both functions in separation.
        elif isinstance(edge_index, Tensor) or not self.fuse:
            coll_dict = self.__collect__(self.__user_args__, edge_index, size,
                                         kwargs)

            msg_kwargs = self.inspector.distribute('message', coll_dict)
            out = self.message(**msg_kwargs)

            # For `GNNExplainer`, we require a separate message and aggregate
            # procedure since this allows us to inject the `edge_mask` into the
            # message passing computation scheme.
            if self.__explain__:
                edge_mask = self.__edge_mask__.sigmoid()
                # Some ops add self-loops to `edge_index`. We need to do the
                # same for `edge_mask` (but do not train those).
                if out.size(self.node_dim) != edge_mask.size(0):
                    loop = edge_mask.new_ones(size[0])
                    edge_mask = torch.cat([edge_mask, loop], dim=0)
                assert out.size(self.node_dim) == edge_mask.size(0)
                out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))
            elif self.__explain_flow__:

                edge_mask = self.layer_edge_mask.sigmoid()
                # Some ops add self-loops to `edge_index`. We need to do the
                # same for `edge_mask` (but do not train those).
                if out.size(self.node_dim) != edge_mask.size(0):
                    loop = edge_mask.new_ones(size[0])
                    edge_mask = torch.cat([edge_mask, loop], dim=0)
                assert out.size(self.node_dim) == edge_mask.size(0)
                out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))

            aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
            out = self.aggregate(out, **aggr_kwargs)

            update_kwargs = self.inspector.distribute('update', coll_dict)
            return self.update(out, **update_kwargs)

class GCNConv_LRP(gnn.GCNConv):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.edge_weight = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""



        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = self.gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

        # --- add require_grad ---
        edge_weight.requires_grad_(True)

        x = torch.matmul(x, self.weight)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        if self.bias is not None:
            out += -0.5 * torch.log(1 + torch.exp(-2 * self.bias))

        # --- My: record edge_weight ---
        self.edge_weight = edge_weight

        return out

    def gcn_norm(self, edge_index, edge_weight=None, num_nodes=None, improved=False,
                 add_self_loops=True, dtype=None):
        fill_value = 2. if improved else 1.

        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        return edge_index, edge_weight / 2

class GINConv(gnn.GINConv):

    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False,
                 **kwargs):
        super().__init__(nn, eps, train_eps, **kwargs)
        self.edge_weight = None
        self.fc_steps = None
        self.reweight = None
        self.__explain_flow__ = None
        self.__explain__ = None

    # def children(self):
    #     if
    #     return iter([])


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_weight: OptTensor = None, **kwargs) -> Tensor:
        """"""
        self.num_nodes = x.shape[0]
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        if edge_weight is not None:
            self.edge_weight = edge_weight
            assert edge_weight.shape[0] == edge_index.shape[1]
            self.reweight = False
        else:
            edge_index, _ = remove_self_loops(edge_index)
            self_loop_edge_index, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)
            if self_loop_edge_index.shape[1] != edge_index.shape[1]:
                edge_index = self_loop_edge_index
            self.reweight = True
        out = self.propagate(edge_index, x=x[0], size=None)

        if data_args.task == 'explain':
            layer_extractor = []
            hooks = []

            def register_hook(module: nn.Module):
                if not list(module.children()):
                    hooks.append(module.register_forward_hook(forward_hook))

            def forward_hook(module: nn.Module, input: Tuple[Tensor], output: Tensor):
                # input contains x and edge_index
                layer_extractor.append((module, input[0], output))

            # --- register hooks ---
            self.nn.apply(register_hook)

            nn_out = self.nn(out)

            for hook in hooks:
                hook.remove()

            fc_steps = []
            step = {'input': None, 'module': [], 'output': None}
            for layer in layer_extractor:
                if isinstance(layer[0], nn.Linear):
                    if step['module']:
                        fc_steps.append(step)
                    # step = {'input': layer[1], 'module': [], 'output': None}
                    step = {'input': None, 'module': [], 'output': None}
                step['module'].append(layer[0])
                if kwargs.get('probe'):
                    step['output'] = layer[2]
                else:
                    step['output'] = None

            if step['module']:
                fc_steps.append(step)
            self.fc_steps = fc_steps
        else:
            nn_out = self.nn(out)


        return nn_out

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        r"""The initial call to start propagating messages.

        Args:
            adj (Tensor or SparseTensor): A :obj:`torch.LongTensor` or a
                :obj:`torch_sparse.SparseTensor` that defines the underlying
                graph connectivity/message passing flow.
                :obj:`edge_index` holds the indices of a general (sparse)
                assignment matrix of shape :obj:`[N, M]`.
                If :obj:`edge_index` is of type :obj:`torch.LongTensor`, its
                shape must be defined as :obj:`[2, num_messages]`, where
                messages from nodes in :obj:`edge_index[0]` are sent to
                nodes in :obj:`edge_index[1]`
                (in case :obj:`flow="source_to_target"`).
                If :obj:`edge_index` is of type
                :obj:`torch_sparse.SparseTensor`, its sparse indices
                :obj:`(row, col)` should relate to :obj:`row = edge_index[1]`
                and :obj:`col = edge_index[0]`.
                The major difference between both formats is that we need to
                input the *transposed* sparse adjacency matrix into
                :func:`propagate`.
            size (tuple, optional): The size :obj:`(N, M)` of the assignment
                matrix in case :obj:`edge_index` is a :obj:`LongTensor`.
                If set to :obj:`None`, the size will be automatically inferred
                and assumed to be quadratic.
                This argument is ignored in case :obj:`edge_index` is a
                :obj:`torch_sparse.SparseTensor`. (default: :obj:`None`)
            **kwargs: Any additional data which is needed to construct and
                aggregate messages, and to update node embeddings.
        """
        size = self.__check_input__(edge_index, size)

        # Run "fused" message and aggregation (if applicable).
        if (isinstance(edge_index, SparseTensor) and self.fuse
                and not self.__explain__):
            coll_dict = self.__collect__(self.__fused_user_args__, edge_index,
                                         size, kwargs)

            msg_aggr_kwargs = self.inspector.distribute(
                'message_and_aggregate', coll_dict)
            out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)

            update_kwargs = self.inspector.distribute('update', coll_dict)
            return self.update(out, **update_kwargs)

        # Otherwise, run both functions in separation.
        elif isinstance(edge_index, Tensor) or not self.fuse:
            coll_dict = self.__collect__(self.__user_args__, edge_index, size,
                                         kwargs)

            msg_kwargs = self.inspector.distribute('message', coll_dict)
            out = self.message(**msg_kwargs)

            # For `GNNExplainer`, we require a separate message and aggregate
            # procedure since this allows us to inject the `edge_mask` into the
            # message passing computation scheme.
            if self.__explain__:
                edge_mask = self.__edge_mask__.sigmoid()
                # Some ops add self-loops to `edge_index`. We need to do the
                # same for `edge_mask` (but do not train those).
                if out.size(self.node_dim) != edge_mask.size(0):
                    loop = edge_mask.new_ones(size[0])
                    edge_mask = torch.cat([edge_mask, loop], dim=0)
                assert out.size(self.node_dim) == edge_mask.size(0)
                out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))
            elif self.__explain_flow__:

                edge_mask = self.layer_edge_mask.sigmoid()
                # Some ops add self-loops to `edge_index`. We need to do the
                # same for `edge_mask` (but do not train those).
                if out.size(self.node_dim) != edge_mask.size(0):
                    loop = edge_mask.new_ones(size[0])
                    edge_mask = torch.cat([edge_mask, loop], dim=0)
                assert out.size(self.node_dim) == edge_mask.size(0)
                out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))

            aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
            out = self.aggregate(out, **aggr_kwargs)

            update_kwargs = self.inspector.distribute('update', coll_dict)
            return self.update(out, **update_kwargs)

    def message(self, x_j: Tensor) -> Tensor:
        if self.reweight:
            edge_weight = torch.ones(x_j.shape[0], device=x_j.device)
            edge_weight.data[-self.num_nodes:] += self.eps
            edge_weight = edge_weight.detach().clone()
            edge_weight.requires_grad_(True)
            self.edge_weight = edge_weight
        return x_j * self.edge_weight.view(-1, 1)


class GNNPool(nn.Module):
    def __init__(self):
        super().__init__()

class GlobalMeanPool(GNNPool):

    def __init__(self):
        super().__init__()

    def forward(self, x, batch, size=None):
        return gnn.global_mean_pool(x, batch, size)


class IdenticalPool(GNNPool):

    def __init__(self):
        super().__init__()

    def forward(self, x, batch, size=None):
        return x


class GraphSequential(nn.Sequential):

    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, *input) -> Tensor:
        for module in self:
            if isinstance(input, tuple):
                input = module(*input)
            else:
                input = module(input)
        return input

