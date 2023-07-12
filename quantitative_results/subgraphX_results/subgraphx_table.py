import json

import cilog
import os
from definitions import ROOT_DIR
from os.path import join as opj
import re

cilog.create_logger(sub_print=True)

table_format = [['GradCAM', 'DeepLIFT', 'GNNExplainer', 'PGExplainer', 'GNN_GI', 'GNN_LRP', 'FlowShap_orig', 'FlowShap_plus', 'FlowMask', 'PGMExplainer', 'SubgraphX'],
              ['ba_shapes', 'ba_lrp', 'clintox', 'tox21', 'bbbp', 'bace', 'graph_sst2', 'ba_infe'],
              ['0.1S', '0.2S', '0.3S', '0.4S', '0.5S', '0.6S', '0.7S', '0.8S', '0.9S']]


subgraphx_root = opj(ROOT_DIR, 'quantitative_results', 'subgraphX_results', 'results')
for json_file in os.listdir(subgraphx_root):
    dataset_name, model_name = re.match('(.*)_(G[CI]N_[23]l)_result.json', json_file).groups()
    print(f'#IN#{dataset_name} : {model_name}')
    result_file = opj(subgraphx_root, json_file)
    with open(result_file, mode='r') as f:
        results_json = json.load(f)
        # print(results)
        results_dict = {}
        for item in results_json['subgraphx'].values():
            results_dict[item['sparsity']] = f"{item['fidelity']:.4f}/{item['fidelity_inv']:.4f}"
        results_round = {key / 10 : [2, -1] for key in range(0, 10)}
        for sparsity, value in results_dict.items():
            round_sparsity = round(sparsity * 10) / 10
            if abs(results_round[round_sparsity][0] - round_sparsity) > abs(sparsity - round_sparsity):
                results_round[round_sparsity][0] = sparsity
                results_round[round_sparsity][1] = value
        format_results = {key / 10 : results_round[key / 10][1] for key in range(5, 10)}
        print(f'#IN#{format_results}')


        xlsx_name = None
        if 'GCN' in model_name:
            xlsx_name = 'GCN'
        elif 'GIN' in model_name:
            xlsx_name = 'GIN'
        assert xlsx_name is not None

        if True:
            xlsx_name += '_PL'

        for sparsity, value in format_results.items():
            file = os.path.join(ROOT_DIR, 'quantitative_results', f'{xlsx_name}_prob.xlsx')
            # print(file)
            if value == -1:
                cilog.fill_table(file,
                                 value='',
                                 x='SubgraphX', y=dataset_name, z=f'{sparsity}S',
                                 table_format=table_format)
            else:
                cilog.fill_table(file,
                                 value=value,
                                 x='SubgraphX', y=dataset_name, z=f'{sparsity}S',
                                 table_format=table_format)