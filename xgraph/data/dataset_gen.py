"""
FileName: dataset_gen.py
Description: dataset generator
Time: 2020/12/28 19:16
Project: GNN_benchmark
Author: Shurui Gui
"""
import random
import torch_geometric
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import k_hop_subgraph
from xgraph.definitions import ROOT_DIR
import pickle as pkl

import os
import re
import torch
import numpy as np
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data, InMemoryDataset, \
    download_url

try:
    from rdkit import Chem
except ImportError:
    Chem = None
import os.path as osp
import zipfile
import gzip

x_map = {
    'atomic_num':
    list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
    ],
    'degree':
    list(range(0, 11)),
    'formal_charge':
    list(range(-5, 7)),
    'num_hs':
    list(range(0, 9)),
    'num_radical_electrons':
    list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

e_map = {
    'bond_type': [
        'misc',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ],
    'is_conjugated': [False, True],
}


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def random_node_splits(data, num_classes):
    # Set new random planetoid splits:
    # * 20 * num_classes labels for training
    # * 500 labels for validation
    # * 1000 labels for testing

    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)

    rest_index = torch.cat([i[20:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(rest_index[:500], size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index[500:1500], size=data.num_nodes)

    return data



class BA_LRP(InMemoryDataset):

    def __init__(self, root, name, num_per_class, transform=None, pre_transform=None):
        self.num_per_class = num_per_class
        self.name = name
        super().__init__(os.path.join(root, name), transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f'data{self.num_per_class}.pt']

    def gen_class1(self):
        x = torch.tensor([[1], [1]], dtype=torch.float)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=torch.tensor([[0]], dtype=torch.float))

        for i in range(2, 20):
            data.x = torch.cat([data.x, torch.tensor([[1]], dtype=torch.float)], dim=0)
            deg = torch.stack([(data.edge_index[0] == node_idx).float().sum() for node_idx in range(i)], dim=0)
            sum_deg = deg.sum(dim=0, keepdim=True)
            probs = (deg / sum_deg).unsqueeze(0)
            prob_dist = torch.distributions.Categorical(probs)
            node_pick = prob_dist.sample().squeeze()
            data.edge_index = torch.cat([data.edge_index,
                                         torch.tensor([[node_pick, i], [i, node_pick]], dtype=torch.long)], dim=1)

        return data

    def gen_class2(self):
        x = torch.tensor([[1], [1]], dtype=torch.float)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=torch.tensor([[1]], dtype=torch.float))
        epsilon = 1e-30

        for i in range(2, 20):
            data.x = torch.cat([data.x, torch.tensor([[1]], dtype=torch.float)], dim=0)
            deg_reciprocal = torch.stack([1 / ((data.edge_index[0] == node_idx).float().sum() + epsilon) for node_idx in range(i)], dim=0)
            sum_deg_reciprocal = deg_reciprocal.sum(dim=0, keepdim=True)
            probs = (deg_reciprocal / sum_deg_reciprocal).unsqueeze(0)
            prob_dist = torch.distributions.Categorical(probs)
            node_pick = -1
            for _ in range(1 if i % 5 != 4 else 2):
                new_node_pick = prob_dist.sample().squeeze()
                while new_node_pick == node_pick:
                    new_node_pick = prob_dist.sample().squeeze()
                node_pick = new_node_pick
                data.edge_index = torch.cat([data.edge_index,
                                             torch.tensor([[node_pick, i], [i, node_pick]], dtype=torch.long)], dim=1)

        return data

    def process(self):
        data_list = []
        for i in range(self.num_per_class):
            data_list.append(self.gen_class1())
            data_list.append(self.gen_class2())

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class BA_Shapes(InMemoryDataset):

    def __init__(self, root, num_base_node, num_shape, transform=None, pre_transform=None):
        self.num_base_node = num_base_node
        self.num_shape = num_shape
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

        indices = []
        num_classes = 4
        train_percent = 0.7
        for i in range(num_classes):
            index = (self.data.y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

        train_index = torch.cat([i[:int(len(i) * train_percent)] for i in indices], dim=0)

        rest_index = torch.cat([i[int(len(i) * train_percent):] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        self.data.train_mask = index_to_mask(train_index, size=self.data.num_nodes)
        self.data.val_mask = index_to_mask(rest_index[:len(rest_index) // 2], size=self.data.num_nodes)
        self.data.test_mask = index_to_mask(rest_index[len(rest_index) // 2:], size=self.data.num_nodes)

        self.data, self.slices = self.collate([self.data])

    @property
    def processed_file_names(self):
        return [f'data.pt']

    def gen(self):
        x = torch.tensor([[1], [1], [1], [1], [1], [1]], dtype=torch.float)
        edge_index = torch.tensor([[5, 5, 5, 5, 5, 0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5, 5, 5, 5, 5]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)

        # --- generate basic BA graph ---
        for i in range(6, self.num_base_node):
            data.x = torch.cat([data.x, torch.tensor([[1]], dtype=torch.float)], dim=0)
            deg = torch.stack([(data.edge_index[0] == node_idx).float().sum() for node_idx in range(i)], dim=0)
            sum_deg = deg.sum(dim=0, keepdim=True)
            probs = (deg / sum_deg).unsqueeze(0)
            prob_dist = torch.distributions.Categorical(probs)
            node_picks = []
            for _ in range(5):
                node_pick = prob_dist.sample().squeeze()
                while node_pick in node_picks:
                    node_pick = prob_dist.sample().squeeze()
                node_picks.append(node_pick)
                data.edge_index = torch.cat([data.edge_index,
                                             torch.tensor([[node_pick, i], [i, node_pick]], dtype=torch.long)], dim=1)

        data.y = torch.zeros(data.x.shape[0], dtype=torch.long)

        # --- add shapes ---
        house_x = torch.tensor([[1] for _ in range(5)], dtype=torch.float)
        house_y = torch.tensor([1, 2, 2, 3, 3], dtype=torch.long)
        house_edge_index = torch.tensor([[0, 1, 0, 2, 1, 2, 1, 3, 2, 4, 3, 4],
                                         [1, 0, 2, 0, 2, 1, 3, 1, 4, 2, 4, 3]], dtype=torch.long)
        house_data = Data(x=house_x, edge_index=house_edge_index, y = house_y)
        house_connect_probs = torch.tensor([[0.2 for _ in range(5)]])
        house_connect_dist = torch.distributions.Categorical(house_connect_probs)
        base_connect_probs = torch.tensor([[1.0 / self.num_base_node]]).repeat(1, self.num_base_node)
        base_connect_dist = torch.distributions.Categorical(base_connect_probs)
        for i in range(self.num_shape):
            data.edge_index = torch.cat([data.edge_index, house_data.edge_index + data.x.shape[0]], dim=1)
            house_pick = house_connect_dist.sample().squeeze() + data.x.shape[0]
            base_pick = base_connect_dist.sample().squeeze()
            data.x = torch.cat([data.x, house_data.x], dim=0)
            data.y = torch.cat([data.y, house_data.y], dim=0)
            data.edge_index = torch.cat([data.edge_index,
                                         torch.tensor([[base_pick, house_pick], [house_pick, base_pick]], dtype=torch.long)], dim=1)

        # --- add random edges ---
        probs = torch.tensor([[1.0 / data.x.shape[0]]]).repeat(2, data.x.shape[0])
        dist = torch.distributions.Categorical(probs)
        for i in range(data.x.shape[0] // 10):
            node_pair = dist.sample().squeeze()
            if node_pair[0] != node_pair[1] and \
                    (data.edge_index[1][data.edge_index[0] == node_pair[0]] == node_pair[1]).int().sum() == 0:
                data.edge_index = torch.cat([data.edge_index,
                                             torch.tensor([[node_pair[0], node_pair[1]], [node_pair[1], node_pair[0]]],
                                                          dtype=torch.long)], dim=1)

        # TODO: Add both class mask (ground truth) for both edge & node
        #   Please check XMind Flow experiments for more details
        return data

    def process(self):
        data_list = []
        data_list.append(self.gen())

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class BA_INFE(InMemoryDataset):

    def __init__(self, root, name, num_per_class, num_base_node=15, transform=None, pre_transform=None):
        self.num_per_class = num_per_class
        self.num_base_node = num_base_node
        self.name = name
        super().__init__(os.path.join(root, name), transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f'data{self.num_per_class}_ver2.pt']

    def gen_class(self, c=1):

        x = torch.tensor([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=torch.float)
        edge_index = torch.tensor([[5, 5, 5, 5, 5, 0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5, 5, 5, 5, 5]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=torch.tensor([[c - 1]], dtype=torch.float))

        # --- generate basic BA graph ---
        for i in range(6, self.num_base_node):
            data.x = torch.cat([data.x, torch.tensor([[1, 0, 0]], dtype=torch.float)], dim=0)
            deg = torch.stack([(data.edge_index[0] == node_idx).float().sum() for node_idx in range(i)], dim=0)
            sum_deg = deg.sum(dim=0, keepdim=True)
            probs = (deg / sum_deg).unsqueeze(0)
            prob_dist = torch.distributions.Categorical(probs)
            node_picks = []
            for _ in range(2):
                node_pick = prob_dist.sample().squeeze()
                while node_pick in node_picks:
                    node_pick = prob_dist.sample().squeeze()
                node_picks.append(node_pick)
                data.edge_index = torch.cat([data.edge_index,
                                             torch.tensor([[node_pick, i], [i, node_pick]], dtype=torch.long)], dim=1)

        # --- randomize the number of motifs ---
        n_base = random.randint(1, 3)
        n_extra = random.randint(1, 3)
        if c == 1:
            n_class1 = n_base + n_extra
            n_class2 = n_base
        elif c == 2:
            n_class1 = n_base
            n_class2 = n_base + n_extra
        else:
            raise Exception('Data generation class option error.')

        # --- construct motifs ---
        # class1 - motif1
        c1m1_x = torch.tensor([[0, 1, 0], [0, 0, 1]], dtype=torch.float)
        c1m1_edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        c1m1_data = Data(x=c1m1_x, edge_index=c1m1_edge_index)

        # class1 - motif2
        c1m2_x = torch.tensor([[0, 0, 1], [0, 1, 0], [0, 1, 0]], dtype=torch.float)
        c1m2_edge_index = torch.tensor([[0, 0, 1, 2],
                                        [1, 2, 0, 0]], dtype=torch.long)
        c1m2_data = Data(x=c1m2_x, edge_index=c1m2_edge_index)

        # class2 - motif1
        c2m1_x = torch.tensor([[0, 0, 1]], dtype=torch.float)
        c2m1_edge_index = torch.tensor([], dtype=torch.long)
        c2m1_data = Data(x=c2m1_x, edge_index=c2m1_edge_index)

        # class2 - motif2
        c2m2_x = torch.tensor([[0, 1, 0], [0, 0, 1], [0, 0, 1]], dtype=torch.float)
        c2m2_edge_index = torch.tensor([[0, 0, 1, 2],
                                        [1, 2, 0, 0]], dtype=torch.long)
        c2m2_data = Data(x=c2m2_x, edge_index=c2m2_edge_index)

        motif_c1 = [c1m1_data, c1m2_data]
        motif_c2 = [c2m1_data, c2m2_data]

        # --- add motifs ---
        base_connect_probs = torch.tensor([[1.0 / self.num_base_node]]).repeat(1, self.num_base_node)
        base_connect_dist = torch.distributions.Categorical(base_connect_probs)
        # class1 motifs
        class1_mask = []
        for _ in range(n_class1):
            motif = motif_c1[random.randint(0, 1)]
            data.edge_index = torch.cat([data.edge_index, motif.edge_index + data.x.shape[0]], dim=1)
            for edge in (motif.edge_index + data.x.shape[0]).T.tolist():
                if (edge not in class1_mask) and ([edge[1], edge[0]] not in class1_mask):
                    class1_mask.append(edge)
            # self loop mask
            for i in range(motif.x.shape[0]):
                class1_mask.append([i + data.x.shape[0], i + data.x.shape[0]])

            motif_pick = 0 + data.x.shape[0]
            base_pick = base_connect_dist.sample().squeeze()
            data.x = torch.cat([data.x, motif.x], dim=0)
            data.edge_index = torch.cat([data.edge_index,
                                         torch.tensor([[base_pick, motif_pick], [motif_pick, base_pick]],
                                                      dtype=torch.long)], dim=1)
            class1_mask.append([base_pick.item(), motif_pick])


        # class1 motifs
        class2_mask = []
        for _ in range(n_class2):
            motif = motif_c2[random.randint(0, 1)]
            data.edge_index = torch.cat([data.edge_index, motif.edge_index + data.x.shape[0]], dim=1)
            for edge in (motif.edge_index + data.x.shape[0]).T.tolist():
                if (edge not in class2_mask) and ([edge[1], edge[0]] not in class2_mask):
                    class2_mask.append(edge)
            # self loop mask
            for i in range(motif.x.shape[0]):
                class2_mask.append([i + data.x.shape[0], i + data.x.shape[0]])

            motif_pick = 0 + data.x.shape[0]
            base_pick = base_connect_dist.sample().squeeze()
            data.x = torch.cat([data.x, motif.x], dim=0)
            data.edge_index = torch.cat([data.edge_index,
                                         torch.tensor([[base_pick, motif_pick], [motif_pick, base_pick]],
                                                      dtype=torch.long)], dim=1)
            class2_mask.append([base_pick.item(), motif_pick])


        # TODO: Add class mask not only for edge but also for node

        data.class_mask = []
        data.class_mask.append(class1_mask)
        data.class_mask.append(class2_mask)

        return data

    def process(self):
        data_list = []
        for i in range(self.num_per_class):
            data_list.append(self.gen_class(1))
            data_list.append(self.gen_class(2))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



class BA_Shape1(InMemoryDataset):

    def __init__(self, root, num_base_node, num_shape, transform=None, pre_transform=None):
        self.num_base_node = num_base_node
        self.num_shape = num_shape
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def processed_file_names(self):
        return [f'data{"_debug2"}.pt']

    def gen(self):
        with open(os.path.join(ROOT_DIR, 'xgraph', 'dataset_loaders', 'syn1.pkl'), 'rb') as f:
            adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, edge_label_matrix = pkl.load(f)
        x = torch.from_numpy(features)
        edge_index = torch.tensor([[], []], dtype=torch.long)
        for i in range(700):
            for j in range(700):
                if np.fabs(adj[i, j] - 1) < 1e-2:
                    edge_index = torch.cat([edge_index,
                                            torch.tensor([[i], [j]], dtype=torch.long)], dim=1)
        y = torch.argmax(torch.from_numpy(y_train + y_val + y_test), dim=1)
        data = Data(x, edge_index, y=y)
        data.train_mask = torch.tensor(train_mask)
        data.val_mask = torch.tensor(val_mask)
        data.test_mask = torch.tensor(test_mask)

        return data

    def process(self):
        data_list = []
        data_list.append(self.gen())

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])




import os
import glob
import json
import torch
import pickle
import numpy as np
import os.path as osp
from torch_geometric.data import Data, InMemoryDataset
import traceback

def undirected_graph(data):
    """
    A pre_transform function that transfers the directed graph into undirected graph.
    Args:
        data (torch_geometric.data.Data): Directed graph in the format :class:`torch_geometric.data.Data`.
        where the :obj:`data.x`, :obj:`data.edge_index` are required.
    """
    data.edge_index = torch.cat([torch.stack([data.edge_index[1], data.edge_index[0]], dim=0),
                                 data.edge_index], dim=1)
    return data


def split(data, batch):
    # i-th contains elements from slice[i] to slice[i+1]
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])
    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)
    data.__num_nodes__ = np.bincount(batch).tolist()

    slices = dict()
    slices['x'] = node_slice
    slices['edge_index'] = edge_slice
    slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
    return data, slices


def read_file(folder, prefix, name):
    file_path = osp.join(folder, prefix + f'_{name}.txt')
    return np.genfromtxt(file_path, dtype=np.int64)

def read_sentigraph_data(folder: str, prefix: str):
    txt_files = glob.glob(os.path.join(folder, "{}_*.txt".format(prefix)))
    json_files = glob.glob(os.path.join(folder, "{}_*.json".format(prefix)))
    txt_names = [f.split(os.sep)[-1][len(prefix) + 1:-4] for f in txt_files]
    json_names = [f.split(os.sep)[-1][len(prefix) + 1:-5] for f in json_files]
    names = txt_names + json_names

    with open(os.path.join(folder, prefix+"_node_features.pkl"), 'rb') as f:
        x: np.array = pickle.load(f)
    x: torch.FloatTensor = torch.from_numpy(x)
    edge_index: np.array = read_file(folder, prefix, 'edge_index')
    edge_index: torch.tensor = torch.tensor(edge_index, dtype=torch.long).T
    batch: np.array = read_file(folder, prefix, 'node_indicator') - 1     # from zero
    y: np.array = read_file(folder, prefix, 'graph_labels')
    y: torch.tensor = torch.tensor(y, dtype=torch.long)

    supplement = dict()
    if 'split_indices' in names:
        split_indices: np.array = read_file(folder, prefix, 'split_indices')
        split_indices = torch.tensor(split_indices, dtype=torch.long)
        supplement['split_indices'] = split_indices
    if 'sentence_tokens' in names:
        with open(os.path.join(folder, prefix + '_sentence_tokens.json')) as f:
            sentence_tokens: dict = json.load(f)
        supplement['sentence_tokens'] = sentence_tokens

    data = Data(x=x, edge_index=edge_index, y=y)
    data, slices = split(data, batch)

    return data, slices, supplement


class SentiGraphDataset(InMemoryDataset):
    r"""
    The SentiGraph dataset_loaders from `Explainability in Graph Neural Networks: A Taxonomic Survey
    <https://arxiv.org/abs/2012.15445>`_.
    The dataset_loaders take pretrained BERT as node feature extractor
    and dependency tree as edges to transfer the text sentiment dataset_loaders into
    graph classification dataset_loaders.

    The dataset `Graph-SST2 <https://drive.google.com/file/d/1-PiLsjepzT8AboGMYLdVHmmXPpgR8eK1/view?usp=sharing>`_
    should be downloaded to the proper directory before running. All the three dataset_loaders Graph-SST2, Graph-SST5, and
    Graph-Twitter can be download in this
    `link <https://drive.google.com/drive/folders/1dt0aGMBvCEUYzaG00TYu1D03GPO7305z?usp=sharing>`_.

    Args:
        root (:obj:`str`): Root directory where the dataset_loaders are saved
        name (:obj:`str`): The name of the dataset_loaders.
        transform (:obj:`Callable`, :obj:`None`): A function/transform that takes in an
            :class:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (:obj:`Callable`, :obj:`None`):  A function/transform that takes in
            an :class:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)

    .. note:: The default parameter of pre_transform is :func:`~undirected_graph`
        which transfers the directed graph in original data into undirected graph before
        being saved to disk.
    """
    def __init__(self, root, name, transform=None, pre_transform=undirected_graph):
        self.name = name
        super(SentiGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices, self.supplement = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['node_features', 'node_indicator', 'sentence_tokens', 'edge_index',
                'graph_labels', 'split_indices']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # Read data into huge `Data` list.
        try:
            self.data, self.slices, self.supplement \
                  = read_sentigraph_data(self.raw_dir, self.name)
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            if type(e) is FileNotFoundError:
                print("Please download the required dataset_loaders file to the root directory.")
                print("The google drive link is "
                      "https://drive.google.com/drive/folders/1dt0aGMBvCEUYzaG00TYu1D03GPO7305z?usp=sharing")
            raise SystemExit()

        sentences = [sentence for sentence in self.supplement['sentence_tokens'].values()]

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            for idx in range(len(data_list)):
                data_list[idx].keys.append('sentence')
                data_list[idx].sentence = sentences[idx]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices, self.supplement), self.processed_paths[0])



class BA_Traffic(InMemoryDataset):

    def __init__(self, root, name, num_per_class, transform=None, pre_transform=None):
        self.num_per_class = num_per_class
        self.name = name
        super().__init__(os.path.join(root, name), transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f'data{self.num_per_class}.pt']

    def gen_data(self, label):
        num_node = 20   # np.random.randint(10, 20)
        data = self.create_road(num_node)
        mask_c0 = []
        mask_c1 = []

        max_deg_id = torch.stack([(data.edge_index[0] == node_idx).float().sum() for node_idx in range(num_node)], dim=0).argmax()
        data.x[max_deg_id][0] = 1
        data, mask = self.create_traffic(data, max_deg_id, traffic='bike')
        mask_c0.extend(mask)
        data, mask = self.create_traffic(data, max_deg_id, traffic='car')
        mask_c1.extend(mask)
        if label == 0:
            data, mask = self.create_traffic(data, max_deg_id, traffic='bike')
            mask_c0.extend(mask)
            data.y = torch.tensor([[0]], dtype=torch.float)
        elif label == 1:
            data, mask = self.create_traffic(data, max_deg_id, traffic='car')
            mask_c1.extend(mask)
            data.y = torch.tensor([[1]], dtype=torch.float)
        else:
            raise Exception('Label option error.')

        data.class_mask = []
        data.class_mask.append(mask_c0)
        data.class_mask.append(mask_c1)

        return data

    def create_road(self, num_node):
        x = torch.tensor([[-1, 0, 0], [-1, 0, 0]], dtype=torch.float)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=torch.tensor([[0]], dtype=torch.float))
        for i in range(2, num_node):
            data.x = torch.cat([data.x, torch.tensor([[-1, 0, 0]], dtype=torch.float)], dim=0)
            deg = torch.stack([(data.edge_index[0] == node_idx).float().sum() for node_idx in range(i)], dim=0)
            sum_deg = deg.sum(dim=0, keepdim=True)
            probs = (deg / sum_deg).unsqueeze(0)
            prob_dist = torch.distributions.Categorical(probs)
            node_pick = prob_dist.sample().squeeze()
            data.edge_index = torch.cat([data.edge_index,
                                         torch.tensor([[node_pick, i], [i, node_pick]], dtype=torch.long)], dim=1)
        return data

    def create_traffic(self, data, max_deg_id, traffic):
        # Find two different nodes in the neighborhood of max_deg_id node
        # using k-hop-subgraph function, output the node id
        neighbor_id = data.edge_index[1][data.edge_index[0] == max_deg_id]
        # Randomly choose two nodes from the neighbor_id, and they should be different
        nei_id = []
        while nei_id == []:
            chosen_id = np.random.choice(neighbor_id, 2, replace=False)
            nei_id = data.edge_index[1][data.edge_index[0] == chosen_id[1]].tolist()
            nei_id.remove(max_deg_id)
        chosen_id = np.concatenate([chosen_id, np.random.choice(nei_id, 1)])
        if traffic == 'bike':
            data.x[chosen_id[0]][1] += 1
            data.x[chosen_id[2]][1] += 1
        elif traffic == 'car':
            data.x[chosen_id[0]][2] += 1
            data.x[chosen_id[2]][2] += 1
        else:
            raise Exception('Traffic option error.')

        mask = [[chosen_id[0].item(), max_deg_id.item(), chosen_id[1].item(), chosen_id[2].item()]]
        return data, mask

    def process(self):
        random_seed = 314
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        data_list = []
        for i in range(self.num_per_class):
            data_list.append(self.gen_data(0))
            data_list.append(self.gen_data(1))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def maybe_log(path, log=True):
    if log:
        print('Extracting', path)


def extract_gz(path, folder, log=True):
    maybe_log(path, log)
    with gzip.open(path, 'r') as r:
        with open(osp.join(folder, '.'.join(os.path.basename(path).split('.')[:-1])), 'wb') as w:
            w.write(r.read())


def extract_zip(path, folder, log=True):
    r"""Extracts a zip archive to a specific folder.

    Args:
        path (string): The path to the tar archive.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """
    maybe_log(path, log)
    with zipfile.ZipFile(path, 'r') as f:
        f.extractall(folder)


class MoleculeDataset(InMemoryDataset):
    r"""
    The extension of MoleculeNet with `MUTAG <https://pubs.acs.org/doi/10.1021/jm00106a046>`_.

    The `MoleculeNet benchmark collection <http://moleculenet.ai/datasets-1>`_ from the
    `MoleculeNet: A Benchmark for Molecular Machine Learning <https://arxiv.org/abs/1703.00564>`_
    paper, containing datasets from physical chemistry, biophysics and physiology.

    The MoleculeNet datasets come with the additional node and edge features introduced by
    the `Open Graph Benchmark <https://ogb.stanford.edu/docs/graphprop/>`_, and the node features
    in MUTAG dataset are one hot features denoting the atom types.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"MUTAG"`, :obj:`"ESOL"`,
            :obj:`"FreeSolv"`, :obj:`"Lipo"`, :obj:`"PCBA"`, :obj:`"MUV"`,
            :obj:`"HIV"`, :obj:`"BACE"`, :obj:`"BBPB"`, :obj:`"Tox21"`,
            :obj:`"ToxCast"`, :obj:`"SIDER"`, :obj:`"ClinTox"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/{}'
    mutag_url = 'https://github.com/divelab/DIG_storage/raw/main/xgraph/datasets/{}'

    # Format: name: [display_name, url_name, filename, smiles_idx, y_idx]
    names = {
        'mutag': ['MUTAG', 'MUTAG.zip', None, None],
        'esol': ['ESOL', 'delaney-processed.csv', 'delaney-processed.csv', -1, -2],
        'freesolv': ['FreeSolv', 'SAMPL.csv', 'SAMPL.csv', 1, 2],
        'lipo': ['Lipophilicity', 'Lipophilicity.csv', 'Lipophilicity.csv', 2, 1],
        'pcba': ['PCBA', 'pcba.csv.gz', 'pcba.csv', -1,
                 slice(0, 128)],
        'muv': ['MUV', 'muv.csv.gz', 'muv.csv', -1,
                slice(0, 17)],
        'hiv': ['HIV', 'HIV.csv', 'HIV.csv', 0, -1],
        'bace': ['BACE', 'bace.csv', 'bace.csv', 0, 2],
        'bbbp': ['BBBP', 'BBBP.csv', 'BBBP.csv', -1, -2],
        'tox21': ['Tox21', 'tox21.csv.gz', 'tox21.csv', -1,
                  slice(0, 12)],
        'toxcast':
        ['ToxCast', 'toxcast_data.csv.gz', 'toxcast_data.csv', 0,
         slice(1, 618)],
        'sider': ['SIDER', 'sider.csv.gz', 'sider.csv', 0,
                  slice(1, 28)],
        'clintox': ['ClinTox', 'clintox.csv.gz', 'clintox.csv', 0,
                    slice(1, 3)],
    }

    def __init__(self, root, name, transform=None, pre_transform=None,
                 pre_filter=None):

        if Chem is None:
            raise ImportError('`MoleculeNet` requires `rdkit`.')

        self.name = name.lower()
        assert self.name in self.names.keys()
        super(MoleculeDataset, self).__init__(root, transform, pre_transform,
                                              pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        if self.name.lower() == 'MUTAG'.lower():
            return ['MUTAG_A.txt', 'MUTAG_graph_labels.txt', 'MUTAG_graph_indicator.txt',
                    'MUTAG_node_labels.txt', 'README.txt']
        else:
            return self.names[self.name][2]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        if self.name.lower() == 'MUTAG'.lower():
            url = self.mutag_url.format(self.names[self.name][1])
        else:
            url = self.url.format(self.names[self.name][1])
        path = download_url(url, self.raw_dir)
        if self.names[self.name][1][-2:] == 'gz':
            extract_gz(path, self.raw_dir)
            os.unlink(path)
        elif self.names[self.name][1][-3:] == 'zip':
            extract_zip(path, self.raw_dir)
            os.unlink(path)

    def process(self):
        if self.name.lower() == 'MUTAG'.lower():
            with open(os.path.join(self.raw_dir, 'MUTAG_node_labels.txt'), 'r') as f:
                nodes_all_temp = f.read().splitlines()
                nodes_all = [int(i) for i in nodes_all_temp]

            adj_all = np.zeros((len(nodes_all), len(nodes_all)))
            with open(os.path.join(self.raw_dir, 'MUTAG_A.txt'), 'r') as f:
                adj_list = f.read().splitlines()
            for item in adj_list:
                lr = item.split(', ')
                l = int(lr[0])
                r = int(lr[1])
                adj_all[l - 1, r - 1] = 1

            with open(os.path.join(self.raw_dir, 'MUTAG_graph_indicator.txt'), 'r') as f:
                graph_indicator_temp = f.read().splitlines()
                graph_indicator = [int(i) for i in graph_indicator_temp]
                graph_indicator = np.array(graph_indicator)

            with open(os.path.join(self.raw_dir, 'MUTAG_graph_labels.txt'), 'r') as f:
                graph_labels_temp = f.read().splitlines()
                graph_labels = [int(i) for i in graph_labels_temp]

            data_list = []
            for i in range(1, 189):
                idx = np.where(graph_indicator == i)
                graph_len = len(idx[0])
                adj = adj_all[idx[0][0]:idx[0][0] + graph_len, idx[0][0]:idx[0][0] + graph_len]
                label = int(graph_labels[i - 1] == 1)
                feature = nodes_all[idx[0][0]:idx[0][0] + graph_len]
                nb_clss = 7
                targets = np.array(feature).reshape(-1)
                one_hot_feature = np.eye(nb_clss)[targets]
                data_example = Data(x=torch.from_numpy(one_hot_feature).float(),
                                    edge_index=dense_to_sparse(torch.from_numpy(adj))[0],
                                    y=label)
                data_list.append(data_example)
        else:
            with open(self.raw_paths[0], 'r') as f:
                dataset = f.read().split('\n')[1:-1]
                dataset = [x for x in dataset if len(x) > 0]  # Filter empty lines.

            data_list = []
            for line in dataset:
                line = re.sub(r'\".*\"', '', line)  # Replace ".*" strings.
                line = line.split(',')

                smiles = line[self.names[self.name][3]]
                ys = line[self.names[self.name][4]]
                ys = ys if isinstance(ys, list) else [ys]

                ys = [float(y) if len(y) > 0 else float('NaN') for y in ys]
                y = torch.tensor(ys, dtype=torch.float).view(1, -1)

                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue

                xs = []
                for atom in mol.GetAtoms():
                    x = []
                    x.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
                    x.append(x_map['chirality'].index(str(atom.GetChiralTag())))
                    x.append(x_map['degree'].index(atom.GetTotalDegree()))
                    x.append(x_map['formal_charge'].index(atom.GetFormalCharge()))
                    x.append(x_map['num_hs'].index(atom.GetTotalNumHs()))
                    x.append(x_map['num_radical_electrons'].index(
                        atom.GetNumRadicalElectrons()))
                    x.append(x_map['hybridization'].index(
                        str(atom.GetHybridization())))
                    x.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
                    x.append(x_map['is_in_ring'].index(atom.IsInRing()))
                    xs.append(x)

                x = torch.tensor(xs, dtype=torch.long).view(-1, 9)

                edge_indices, edge_attrs = [], []
                for bond in mol.GetBonds():
                    i = bond.GetBeginAtomIdx()
                    j = bond.GetEndAtomIdx()

                    e = []
                    e.append(e_map['bond_type'].index(str(bond.GetBondType())))
                    e.append(e_map['stereo'].index(str(bond.GetStereo())))
                    e.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))

                    edge_indices += [[i, j], [j, i]]
                    edge_attrs += [e, e]

                edge_index = torch.tensor(edge_indices)
                edge_index = edge_index.t().to(torch.long).view(2, -1)
                edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 3)

                # Sort indices.
                if edge_index.numel() > 0:
                    perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
                    edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                            smiles=smiles)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.names[self.name][0], len(self))