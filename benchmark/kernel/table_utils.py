import os
import shutil
import time

from cilog import fill_table

from benchmark.kernel.pipeline import args, explain_collector
from definitions import ROOT_DIR


def output_table():
    xlsx_name = None
    if 'GCN' in args['common'].model_name:
        xlsx_name = 'GCN'
    elif 'GIN' in args['common'].model_name:
        xlsx_name = 'GIN'
    assert xlsx_name is not None
    if args['explain'].explain_pred_label:
        xlsx_name += '_PL'
    try_failure = 0
    while try_failure < 3:
        try:
            file = os.path.join(ROOT_DIR, 'quantitative_results', f'{xlsx_name}.xlsx')
            print(file)
            value = f'{explain_collector.fidelity:.4f}/{explain_collector.infidelity:.4f}'
            fill_table(file,
                       value=value,
                       x=args['explain'].explainer, y=args['explain'].dataset_name, z=f'{explain_collector.sparsity}S',
                       table_format=args['explain'].table_format)
            shutil.copyfile(os.path.join(ROOT_DIR, 'quantitative_results', f'{xlsx_name}.xlsx'),
                            os.path.join(ROOT_DIR, 'quantitative_results', f'{xlsx_name}_prob.xlsx'))
            break
        except:
            try_failure += 1
            time.sleep(10)
