import os
import shutil

from xgraph.definitions import ROOT_DIR
from pathlib import Path
# from openpyxl import Workbook, load_workbook
import pandas as pd
from filelock import FileLock, Timeout
from pathlib import Path
import time

from xgraph.definitions import ROOT_DIR


def output_table(args, explain_collector):
    file = Path(ROOT_DIR, 'quantitative_results', f'GCN_GIN_PL.xlsx')

    lock = FileLock(file.with_suffix('.xlsx.lock'), timeout=10)
    with lock:
        sheet = args['common'].model_name.split('_')[0]
        result_df = pd.read_excel(Path(ROOT_DIR, 'quantitative_results', 'GCN_GIN_PL.xlsx'),
                                  sheet_name=sheet, index_col=[0, 1], header=[0, 1])
        result_df = expand_table(args, explain_collector, result_df)

        update_table(args, explain_collector, result_df)
        # replace one excel sheet
        with pd.ExcelWriter(Path(ROOT_DIR, 'quantitative_results', 'GCN_GIN_PL.xlsx'), mode='a', if_sheet_exists='replace') as writer:
            result_df.to_excel(writer, sheet_name=sheet, float_format='%.4f')
        # backup file, if dir not exists, create it
        os.makedirs(Path(ROOT_DIR, 'quantitative_results', '.excel_bak'), exist_ok=True)
        shutil.copy(Path(ROOT_DIR, 'quantitative_results', 'GCN_GIN_PL.xlsx'),
                    Path(ROOT_DIR, 'quantitative_results', '.excel_bak', f'GCN_GIN_PL{time.asctime(time.localtime(time.time()))}.xlsx'))


def update_table(args, explain_collector, result_df):
    for metric_name, metric_value in zip(['Fidelity+', 'Fidelity-', 'Accuracy'],
                                         [explain_collector.fidelity, explain_collector.infidelity, explain_collector.acc]):
        if metric_value is not None:
            result_df.loc[(args["explain"].sparsity, args['explain'].explainer), (args['common'].dataset_name, metric_name)] = metric_value


def expand_table(args, explain_collector, result_df):

    # --- expand table ---
    # --- For new method ---
    if (0.5, args['explain'].explainer) not in result_df.index:
        result_df = pd.concat([result_df, pd.DataFrame(index=pd.MultiIndex.from_product(
            [[0.5, 0.6, 0.7, 0.8, 0.9], [args['explain'].explainer]], names=result_df.index.names
        ))]).sort_index()

    # --- For new sparsity ---
    if (explain_collector.sparsity, args['explain'].explainer) not in result_df.index:
        result_df = pd.concat([result_df, pd.DataFrame(index=pd.MultiIndex.from_product(
            [[explain_collector.sparsity], [args['explain'].explainer]], names=result_df.index.names
        ))]).sort_index()

    # --- For new dataset ---
    if (args['common'].dataset_name, 'Fidelity+') not in result_df.columns:
        result_df = pd.concat([result_df, pd.DataFrame(columns=pd.MultiIndex.from_product(
            [[args['common'].dataset_name], ['Fidelity+', 'Fidelity-']], names=result_df.columns.names
        ))]).sort_index(axis=1)

    # --- For new metric ---
    if (args['common'].dataset_name, 'Accuracy') not in result_df.columns and explain_collector.acc is not None:
        result_df = pd.concat([result_df, pd.DataFrame(columns=pd.MultiIndex.from_product(
            [[args['common'].dataset_name], ['Accuracy']], names=result_df.columns.names
        ))]).sort_index(axis=1)
    return result_df
