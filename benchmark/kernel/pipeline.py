"""
FileName: pipeline.py
Description: Application kernel pipeline [Entry point]
Time: 2020/7/28 10:51
Project: GNN_benchmark
Author: Shurui Gui
"""

import os
import torch

from benchmark.kernel.table_utils import output_table
from benchmark.kernel.train import dataset_method_train, train_batch
from definitions import ROOT_DIR
import time
from benchmark.data import load_dataset, create_dataloader
from benchmark.models import load_model, config_model, load_explainer
from benchmark.kernel import val, test, init, set_train_utils
from benchmark.kernel.utils import save_epoch, argus_parse, detail_results
from benchmark.kernel.explain import XCollector, sample_explain
from benchmark.kernel.train_utils import TrainUtils as tr_utils
from benchmark.args import data_args
from tqdm import tqdm
from cilog import fill_table
from benchmark.models.explainers_backup import Gem
from benchmark.models.explainers.PGExplainer import PGExplainer
from benchmark.models.explainers.VGIB import VGIB
from typing import Union

# Parse arguments
print('#D#Parse arguments.')
args = argus_parse()

print(f'#IN#\n-----------------------------------\n    Task: {args["common"].task}\n'
      f'{time.asctime(time.localtime(time.time()))}')

# Initial
init(args['common'])

# Load dataset
print(f'#IN#Load Dataset {args["common"].dataset_name}')
dataset = load_dataset(args['common'].dataset_name)
print('#D#', dataset['train'][0])

# pack in batches
# data_args.model_level = 'node'
loader = create_dataloader(dataset)

# Load model
print('#IN#Loading model...')
model = load_model(args['common'].model_name)

if args['common'].task == 'train':
    # config model
    print('#D#Config model')
    config_model(model, args['train'], 'train')

    # Load training utils
    print('#D#Load training utils')
    set_train_utils(model, args['train'])

    # train the model
    for epoch in range(args['train'].ctn_epoch, args['train'].epoch):

        print(f'#IN#Epoch {epoch}:')
        for index, data in tqdm(enumerate(loader['train'])):
            # train a batch
            train_batch(model, data, args['train'])

            # middle val
            if (index + 1) % args['train'].val_gap == 0:
                stat = val(model, loader['val'])

        # Epoch val
        stat = val(model, loader['val'])
        # vis(stat)

        # checkpoints save
        save_epoch(model, args['train'], epoch, stat)

        # --- scheduler step ---
        tr_utils.scheduler.step()

    print('#IN#Training end.')


elif args['common'].task == 'test':

    # config model
    print('#D#Config model...')
    config_model(model, args['test'], 'test')

    # test the GNN model
    _ = test(model, loader['test'])

    print('#IN#Test end.')


elif args['common'].task == 'explain':

    # config model
    print('#IN#Config the model used to be explained...')
    config_model(model, args['explain'], 'explain')

    # create explainer
    print(f'#IN#Create explainer: {args["explain"].explainer}...')
    explainer = load_explainer(args['explain'].explainer, model, args['explain'])

    if isinstance(explainer, (PGExplainer, Gem, VGIB)):
        dataset_method_train(explainer, args, loader, dataset, model)

    # begin explain
    explain_collector = XCollector(model, loader['explain'])
    print(f'#IN#Begin explain')
    ex_target_idx = 0 if args['common'].target_idx != -1 else args['common'].explain_idx

    explain_tik = torch.cuda.Event(enable_timing=True)
    explain_tok = torch.cuda.Event(enable_timing=True)
    explain_tik.record()
    DEBUG = args['explain'].debug
    if DEBUG:
        if data_args.model_level == 'node':
            data = list(loader['explain'])[0]
            # node_idx = 300
            node_idx = 301
            sample_explain(explainer, data, explain_collector, sparsity=args['explain'].sparsity,
                           node_idx=node_idx)
        else:
            index = 65 # 65
            print(f'#IN#explain graph line {loader["explain"].dataset.indices[index] + 2}')
            data = list(loader['explain'])[index] # clintox 104 tox21 100(14, 45, 61) ba_lrp 103
            sample_explain(explainer, data, explain_collector, sparsity=args['explain'].sparsity, index=index)
    else:
        if data_args.model_level == 'node':
            index = -1
            for i, data in enumerate(loader['explain']):
                for j, node_idx in tqdm(enumerate(torch.where(data.mask == True)[0].tolist())):
                    # if node_idx < 300:
                    #     continue
                    index += 1
                    print(f'#IN#explain graph {i} node {node_idx}')
                    sample_explain(explainer, data, explain_collector, sparsity=args['explain'].sparsity, node_idx=node_idx,
                                   index=index)

                    detail_results(args, explain_collector, index)

                    if index >= args['explain'].num_explain - 1:
                        break
                if index >= args['explain'].num_explain - 1:
                    break
        else:
            for index, data in tqdm(enumerate(loader['explain'])):
                print(f'#IN#explain graph line {loader["explain"].dataset.indices[index] + 2}')
                sample_explain(explainer, data, explain_collector, sparsity=args['explain'].sparsity, index=index)

                detail_results(args, explain_collector, index)

                # print(index, ' '.join(data.sentence[0]))
                if index >= args['explain'].num_explain - 1:
                    break

    explain_tok.record()
    torch.cuda.synchronize()


    print(f'#IM#Explain method {args["explain"].explainer}\'s performance on {args["common"].model_name} with {args["common"].dataset_name}:\n'
          f'Fidelity: {explain_collector.fidelity:.4f}\n'
          f'Infidelity: {explain_collector.infidelity:.4f}\n'
          f'RegularFidelity: {explain_collector.regular_fidelity:.4f}\n'
          f'RegularInfidelity: {explain_collector.regular_infidelity:.4f}\n'
          # f'Acc: {explain_collector.acc:.4f}\n'
          f'Sparsity: {explain_collector.sparsity:.4f}\n'
          f'Explain total time: {explain_tik.elapsed_time(explain_tok):.0f}ms')



    if DEBUG:
        exit(0)

    if args['explain'].block_table or args['explain'].list_sample \
            or ((args['explain'].vis or args['explain'].save_fig) and not args['explain'].debug):
        print(f'#W#Block table output.')
        exit(0)

    output_table(args, explain_collector)

elif args['common'].task == 'table':
    xlsx_name = None
    if 'GCN' in args['common'].model_name:
        xlsx_name = 'GCN'
    elif 'GIN' in args['common'].model_name:
        xlsx_name = 'GIN'
    assert xlsx_name is not None
    fill_table(os.path.join(ROOT_DIR, 'quantitative_results', f'{xlsx_name}_test.xlsx'),
               value=f'{1:.4f}/{0:.4f}',
               x=args['explain'].explainer, y=args['explain'].dataset_name, z=f'{args["explain"].sparsity}S',
               table_format=args['explain'].table_format)



if args['common'].email:
    print('#mail#Task finished!')
