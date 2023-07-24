"""
FileName: launcher.py
Description: 
Time: 2020/9/8 9:56
Project: GNN_benchmark
Author: Shurui Gui
"""
import subprocess, time, signal, copy
import pynvml

# parse one CLI argument: gpu: list[int]

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
args = parser.parse_args()



task = 'explain'

# POLITE_MODE = True
# POLITE_NUM = 5
# allow_auto_emit_detection = False

args_group = [f'xgraphtg --task {task} --model_name {model_dataset[0]} --dataset_name {model_dataset[1]} ' \
              f'--target_idx {model_dataset[2]} --explainer {explainer} --sparsity {sparsity} ' \
              f'--log_file {task}_{model_dataset[1]}_{model_dataset[0]}_{explainer}_{sparsity}.log ' \
              # f''
              for model_series in ['GCN', 'GIN']
                  for sparsity in ['0.5', '0.6', '0.7', '0.8', '0.9']
                      for model_dataset in [(f'{model_series}_3l', 'clintox', 0), (f'{model_series}_3l', 'ba_lrp', 0),
                                            (f'{model_series}_3l', 'bbbp', 0), (f'{model_series}_3l', 'tox21', 2),
                                            (f'{model_series}_3l', 'bace', 0), (f'{model_series}_3l', 'graph_sst2', 0),
                                            (f'{model_series}_3l', 'ba_infe', 0)]#(f'{model_series}_3l', 'clintox', 0), (f'{model_series}_3l', 'ba_lrp', 0),
                                            #  (f'{model_series}_3l', 'tox21', 2),  # (f'{model_series}_3l', 'ba_infe', 0),
                                            # (f'{model_series}_3l', 'bbbp', 0), (f'{model_series}_3l', 'bace', 0),
                                            # (f'{model_series}_3l', 'graph_sst2', 0)] (f'{model_series}_2l', 'ba_shapes', 0)
                          for explainer in ['VGIB', 'RC_Explainer_Batch_star'] # ['GradCAM', 'PGMExplainer', 'DeepLIFT', 'GNNExplainer', 'PGExplainer', 'GNN_GI', 'GNN_LRP', 'FlowShap_orig', 'FlowShap_plus', ]

]

from .launchers import AdaLauncher

launcher = AdaLauncher(args.gpu)
launcher(args_group)
