"""
FileName: dataset_manager.py
Description: Dataset utils
Time: 2020/7/28 11:48
Project: GNN_benchmark
Author: Shurui Gui
"""
from torch_geometric.data import DataLoader
import sys
from benchmark import data_args
from benchmark.kernel.utils import Metric
import benchmark.data.dataset_loaders as benchmark_datasets



def load_dataset(name: str) -> dir:
    """
    Load dataset.
    :param name: dataset's name. Possible options:("ESOL", "FreeSolv", "Lipo", "PCBA", "MUV", "HIV",
    "BACE", "BBBP", "Tox21", "ToxCast", "SIDER", "ClinTox")
    :return: torch_geometric.dataset object
    """
    molecule_set = ["ESOL", "FreeSolv", "Lipo", "PCBA", "MUV", "HIV",
                    "BACE", "BBBP", "Tox21", "ToxCast", "SIDER", "ClinTox"]
    molecule_set = [x.lower() for x in molecule_set]
    name = name.lower()

    # set Metrics: loss and score based on dataset's name
    Metric.set_loss_func(name)
    Metric.set_score_func(name)


    # To Do: use transform to argument data
    if name in molecule_set:
        return benchmark_datasets.molecule_datasets(name)
    elif name == 'ba_lrp':
        return benchmark_datasets.ba_lrp(name)
    elif name == 'ba_infe':
        return benchmark_datasets.ba_infe(name)
    elif name == 'ba_shapes':
        return benchmark_datasets.ba_shapes(name)
    elif name == 'graph_sst2':
        return benchmark_datasets.graph_sst2(name)
    print(f'#E#Dataset {name} does not exist.')
    sys.exit(1)


def create_dataloader(dataset):

    if data_args.model_level == 'node':
        loader = {'train': DataLoader(dataset['train'], batch_size=1, shuffle=True),
                  'val': DataLoader(dataset['val'], batch_size=1, shuffle=True),
                  'test': DataLoader(dataset['test'], batch_size=1, shuffle=False),
                  'explain': DataLoader(dataset['test'], batch_size=1, shuffle=False)}
    else:
        loader = {'train': DataLoader(dataset['train'], batch_size=data_args.train_bs, shuffle=True),
                  'val': DataLoader(dataset['val'], batch_size=data_args.val_bs, shuffle=True),
                  'test': DataLoader(dataset['test'], batch_size=data_args.test_bs, shuffle=False),
                  'explain': DataLoader(dataset['test'], batch_size=data_args.x_bs, shuffle=False)}

    return loader