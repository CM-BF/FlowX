"""
FileName: launcher.py
Description: 
Time: 2020/9/8 9:56
Project: GNN_benchmark
Author: Shurui Gui
"""
import subprocess, shlex, time, os, signal, sys, copy
from definitions import ROOT_DIR
import pynvml

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

def signal_process():
    # os.setsid()
    signal.signal(signal.SIGHUP, signal.SIG_IGN)

conda_env = '/data/shurui.gui/anaconda3/envs/torch_lts/bin/python' #
# conda_env = r'D:\Users\shurui.gui\Anaconda3\envs\torch_v1.9\python.exe'
# epoch = 1000
task = 'explain'

POLITE_MODE = True
POLITE_NUM = 1
allow_auto_emit_detection = False

args_group = [f'-m benchmark.kernel.pipeline --task {task} --model_name {model_dataset[0]} --dataset_name {model_dataset[1]} ' \
              f'--target_idx {model_dataset[2]} --explainer {explainer} --sparsity {sparsity} ' \
              f'--log_file {task}_{model_dataset[1]}_{model_dataset[0]}_{explainer}_{sparsity}.log' \
              # f' --save_fig --nolabel'
              for model_series in ['GCN', 'GIN']
                  for sparsity in ['0.5', '0.6', '0.7', '0.8', '0.9']
                      for model_dataset in [(f'{model_series}_3l', 'clintox', 0), (f'{model_series}_3l', 'ba_lrp', 0),
                                            (f'{model_series}_3l', 'bbbp', 0), (f'{model_series}_3l', 'tox21', 2),
                                            (f'{model_series}_3l', 'bace', 0)]#(f'{model_series}_3l', 'clintox', 0), (f'{model_series}_3l', 'ba_lrp', 0),
                                            #  (f'{model_series}_3l', 'tox21', 2),  # (f'{model_series}_3l', 'ba_infe', 0),
                                            # (f'{model_series}_3l', 'bbbp', 0), (f'{model_series}_3l', 'bace', 0),
                                            # (f'{model_series}_3l', 'graph_sst2', 0)] (f'{model_series}_2l', 'ba_shapes', 0)
                          for explainer in ['Gem'] # ['GradCAM', 'PGMExplainer', 'DeepLIFT', 'GNNExplainer', 'PGExplainer', 'GNN_GI', 'GNN_LRP', 'FlowShap_orig', 'FlowShap_plus', ]

]

# args_group = [f'-m benchmark.kernel.pipeline --task {task} --model_name {model_dataset[0]} --dataset_name {model_dataset[1]} ' \
#               f'--target_idx {model_dataset[2]} --explainer {explainer} --sparsity {sparsity} ' \
#               f'--log_file {task}_{model_dataset[1]}_{model_dataset[0]}_{explainer}_{sparsity}.log' \
#               f' --save_fig --nolabel --debug'
#               for model_series in ['GCN']
#                   for sparsity in ['0.8']
#                       for model_dataset in [(f'{model_series}_3l', 'graph_sst2', 0)]
#                           for explainer in ['GradCAM', 'DeepLIFT', 'GNNExplainer', 'PGExplainer', 'GNN_GI', 'GNN_LRP', 'FlowShap_plus'] # ['GradCAM', 'DeepLIFT', 'GNNExplainer', 'PGExplainer', 'GNN_GI', 'GNN_LRP', 'FlowShap_orig', 'FlowShap_plus']
#
# ]

# epoch = 1000
# task = 'test'
#
# args_group = [f'-m benchmark.kernel.pipeline --task {task} --model_name {model[0]} --dataset_name {dataset[0]} ' \
#               f'--target_idx {dataset[1]} --epoch {epoch} --lr {model[1]} --log_file {task}_{dataset[0]}_{model[0]}.log'
#               for model in [('GAT', 5e-3), ('GCN', 5e-3), ('GIN', 1e-4)] #
#                   for dataset in [('tox21', 2), ('clintox', 0)] #
#                       # for explainer in ['GNNExplainer', 'GradCAM']
# ]

cmd_args_group = [' '.join([conda_env, args]) for args in args_group]

process_pool = {}
while 1:
    # process_pool: in progress process (this program thinks)
    # cmd_args_group: processes that didn't finish successfully
    #   including in progress Process and failure processes
    #   when it is empty: all processes finished successfully
    # failure process: not succeed and not running.
    polite_count = 0
    for cmd_args in cmd_args_group:

        if cmd_args in process_pool.keys():
            continue

        # fork children
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        if meminfo.used / meminfo.total < 0.9:
            print(f'Emit process:\n{cmd_args}')

            process = subprocess.Popen(cmd_args.split(' '), preexec_fn=signal_process, close_fds=True,
                                       stdout=open('/dev/null', 'a'), stderr=subprocess.STDOUT)
            process_pool[cmd_args] = process

            # Don't Emit to fast
            if task == 'table':
                time.sleep(1)
            else:
                time.sleep(5)

        if POLITE_MODE:
            polite_count += 1
            if polite_count >= POLITE_NUM:
                print(f'Be polite!')
                break


    ori_cmd_args_group = copy.deepcopy(cmd_args_group)
    while cmd_args_group == ori_cmd_args_group:
        # re-emit failure process when any process successful finished.
        ceased_processes = []
        for cmd_args, process in process_pool.items():

            return_code = process.poll()
            if return_code is not None:

                # process ceased
                ceased_processes.append(cmd_args)

                if return_code == 0:
                    print(f'Finished:{cmd_args}')
                    cmd_args_group.remove(cmd_args)
                else:
                    print(f'Abort process:{cmd_args}')

        for ceased_process in ceased_processes:
            process_pool.pop(ceased_process)

        time.sleep(1)

        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        if meminfo.used / meminfo.total < 0.1 and allow_auto_emit_detection:
            break

        if not cmd_args_group:
            break


    if not cmd_args_group:
        break

    #

